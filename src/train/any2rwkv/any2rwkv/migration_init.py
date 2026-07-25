from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import sqrt

import torch
from safetensors import safe_open
from torch import Tensor

from .checkpoint import CheckpointManifest, sha256_file
from .errors import ContractError
from .export import BF16_VALUE_RESIDUAL_DISABLED_LOGIT, initialize_tensor
from .mapping import (
    MappingLedger,
    SourceDisposition,
    SourceEntry,
    TargetEntry,
    TargetProvenance,
)
from .layer_store import LayerTensorStore
from .migration import build_group_map, kv_expand, kv_repeat
from .target import TensorSpec, canonical_text_name, is_sequence_mixer, layer_index


class WarmStartVariant(StrEnum):
    RANDOM = "random"
    NAIVE_COPY = "naive_copy"
    GDN_CONSTRAINED = "gdn_constrained"
    KV_REPEAT = "kv_repeat"
    KV_EXPAND = "kv_expand"
    MAPPED = "mapped"


class TensorOperation(StrEnum):
    INITIALIZE = "initialize"
    ZERO = "zero"
    COPY = "copy"
    RESHAPE = "reshape"
    SLICE = "slice"
    HEADWISE_QUERY_SLICE = "headwise_query_slice"
    HEADWISE_QUERY_GATE_SUBSAMPLE = "headwise_query_gate_subsample"
    HEADWISE_QUERY_GATE_RECONSTRUCTION = "headwise_query_gate_reconstruction"
    PAD_ROWS = "pad_rows"
    HEAD_SCALAR_EXPAND = "head_scalar_expand"
    GDN_ERASE_LINEAR_DOWN = "gdn_erase_linear_down"
    GDN_ERASE_BIAS = "gdn_erase_bias"
    GDN_CONV_TIME_MIX = "gdn_conv_time_mix"
    EVEN_ROW_SUBSAMPLE = "even_row_subsample"
    DECAY_LINEAR_UP = "decay_linear_up"
    DECAY_BIAS = "decay_bias"
    TILE = "tile"
    SCALE = "scale"
    KV_REPEAT = "kv_repeat"
    KV_EXPAND = "kv_expand"
    DISABLED_SIGMOID_BIAS = "disabled_sigmoid_bias"


@dataclass(frozen=True)
class WarmStartEntry:
    target: str
    source: str | None
    target_shape: tuple[int, ...]
    source_shape: tuple[int, ...] | None
    provenance: TargetProvenance
    operation: TensorOperation
    evidence: str
    source_start: int | None = None
    source_stop: int | None = None
    num_query_heads: int | None = None
    num_kv_heads: int | None = None
    scale: float | None = None
    is_semantically_lossless: bool = False
    auxiliary_sources: tuple[str, ...] = ()
    local_trainable: bool | None = None


@dataclass(frozen=True)
class WarmStartError:
    layer_index: int
    mixer_kind: str
    target: str
    code: str
    message: str
    head_index: int | None
    group_index: int | None
    source_shape: tuple[int, ...] | None
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class WarmStartPlan:
    variant: WarmStartVariant
    entries: tuple[WarmStartEntry, ...]
    errors: tuple[WarmStartError, ...]
    source_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "variant": self.variant.value,
            "source_hashes": list(self.source_hashes),
            "entries": [asdict(entry) for entry in self.entries],
            "errors": [asdict(error) for error in self.errors],
        }


class WarmStartTensorProvider:
    """Materialize one planned target tensor without retaining the full mapping."""

    def __init__(
        self,
        source: CheckpointManifest,
        target_specs: tuple[TensorSpec, ...],
        plan: WarmStartPlan,
        *,
        seed: int = 20260714,
    ) -> None:
        self.source = source
        self.specs = {spec.name: spec for spec in target_specs}
        self.entries = {entry.target: entry for entry in plan.entries}
        self.seed = seed
        if len(self.specs) != len(target_specs) or len(self.entries) != len(plan.entries):
            raise ContractError("warm-start provider received duplicate target tensors")
        if set(self.specs) != set(self.entries):
            raise ContractError("warm-start plan does not cover provider target specs")
        current_hashes = tuple(sorted(sha256_file(shard) for shard in source.shards))
        if current_hashes != plan.source_hashes:
            raise ContractError("source checkpoint hashes changed after warm-start planning")
        self.tensor_store = LayerTensorStore(source)
        self._cached_layer_index: int | None = None
        self._cached_layer_tensors: dict[str, Tensor] = {}

    def __call__(self, spec: TensorSpec) -> Tensor:
        expected = self.specs.get(spec.name)
        if expected != spec:
            raise ContractError(f"warm-start provider received an unknown spec: {spec.name}")
        entry = self.entries[spec.name]
        source_values: dict[str, Tensor] = {}
        requested_sources = tuple(
            name for name in (entry.source, *entry.auxiliary_sources) if name is not None
        )
        if requested_sources:
            source_layer_index = layer_index(requested_sources[0])
            if source_layer_index is None:
                source_values = self.tensor_store.load_named_tensors(requested_sources)
            else:
                if source_layer_index != self._cached_layer_index:
                    self._cached_layer_tensors = self.tensor_store.load_layer(
                        source_layer_index
                    )
                    self._cached_layer_index = source_layer_index
                source_values = {
                    name: self._cached_layer_tensors[name] for name in requested_sources
                }
        return _materialize_entry(spec, entry, source_values, seed=self.seed)


@dataclass(frozen=True)
class _HeadGeometry:
    num_heads: int
    head_dim: int


def plan_warm_start(
    source: CheckpointManifest,
    target_specs: tuple[TensorSpec, ...],
    *,
    variant: WarmStartVariant | str,
) -> WarmStartPlan:
    """Plan deterministic tensor-level initialization without claiming model equivalence."""
    try:
        selected = WarmStartVariant(variant)
    except ValueError as error:
        raise ContractError(f"unsupported warm-start variant: {variant}") from error

    target_names = [spec.name for spec in target_specs]
    if len(set(target_names)) != len(target_names):
        raise ContractError("warm-start target specs contain duplicate tensor names")
    source_shapes = _read_source_shapes(source)
    specs_by_layer: dict[int, list[TensorSpec]] = {}
    entries: list[WarmStartEntry] = []
    errors: list[WarmStartError] = []
    for spec in sorted(target_specs, key=lambda item: item.name):
        index = layer_index(spec.name)
        if index is None:
            entries.append(_initialized_entry(spec, "target tensor has no sequence-layer owner"))
        else:
            specs_by_layer.setdefault(index, []).append(spec)

    source_config = source.config.get("text_config", source.config)
    if not isinstance(source_config, dict):
        raise ContractError("source text_config must be an object")
    layer_types = source_config.get("layer_types")
    if not isinstance(layer_types, list):
        raise ContractError("source config layer_types must be a list")
    for index in sorted(specs_by_layer):
        if index >= len(layer_types):
            raise ContractError(f"target layer {index} has no source layer type")
        specs = specs_by_layer[index]
        geometry = _target_geometry(specs, index)
        mixer_kind = str(layer_types[index])
        if mixer_kind == "linear_attention":
            layer_entries, layer_errors = _plan_gdn_layer(
                index, specs, geometry, source_config, source_shapes, selected
            )
        elif mixer_kind == "full_attention":
            layer_entries, layer_errors = _plan_full_attention_layer(
                index, specs, geometry, source_config, source_shapes, selected
            )
        else:
            raise ContractError(f"unsupported source mixer at layer {index}: {mixer_kind}")
        # Warm-start provenance constrains only the zero-step value. Once this
        # layer becomes active, distillation must be able to use every native
        # RWKV7 degree of freedom to absorb the residual architecture gap.
        layer_entries = [
            replace(entry, local_trainable=True) for entry in layer_entries
        ]
        entries.extend(layer_entries)
        errors.extend(layer_errors)

    return WarmStartPlan(
        selected,
        tuple(sorted(entries, key=lambda item: item.target)),
        tuple(
            sorted(
                errors,
                key=lambda item: (
                    item.layer_index,
                    item.target,
                    -1 if item.head_index is None else item.head_index,
                    -1 if item.group_index is None else item.group_index,
                    item.code,
                ),
            )
        ),
        tuple(sorted(source.file_hashes[name] for name in source.file_hashes if name.endswith(".safetensors"))),
    )


def materialize_warm_start(
    source: CheckpointManifest,
    target_specs: tuple[TensorSpec, ...],
    plan: WarmStartPlan,
    *,
    seed: int = 20260714,
) -> dict[str, Tensor]:
    """Execute a warm-start plan on CPU and reject stale or malformed plans."""
    specs = {spec.name: spec for spec in target_specs}
    if len(specs) != len(target_specs):
        raise ContractError("warm-start target specs contain duplicate tensor names")
    if set(specs) != {entry.target for entry in plan.entries}:
        raise ContractError("warm-start plan does not cover the requested target specs exactly")
    current_hashes = tuple(sorted(sha256_file(shard) for shard in source.shards))
    if current_hashes != plan.source_hashes:
        raise ContractError("source checkpoint hashes changed after warm-start planning")
    tensors = _read_source_tensors(
        source,
        {
            name
            for entry in plan.entries
            for name in (entry.source, *entry.auxiliary_sources)
            if name is not None
        },
    )
    result: dict[str, Tensor] = {}
    for entry in plan.entries:
        spec = specs[entry.target]
        result[entry.target] = _materialize_entry(
            spec,
            entry,
            {
                name: tensors[name]
                for name in (entry.source, *entry.auxiliary_sources)
                if name is not None
            },
            seed=seed,
        )
    return result


def _materialize_entry(
    spec: TensorSpec,
    entry: WarmStartEntry,
    source_values: dict[str, Tensor],
    *,
    seed: int,
) -> Tensor:
    if tuple(spec.shape) != entry.target_shape:
        raise ContractError(f"stale target shape in warm-start plan: {entry.target}")
    if entry.operation == TensorOperation.INITIALIZE:
        value = initialize_tensor(spec, base_seed=seed)
    elif entry.operation == TensorOperation.ZERO:
        value = torch.zeros(spec.shape, dtype=_torch_dtype(spec.dtype))
    elif entry.operation == TensorOperation.DISABLED_SIGMOID_BIAS:
        value = torch.full(
            entry.target_shape,
            BF16_VALUE_RESIDUAL_DISABLED_LOGIT,
            dtype=torch.float32,
        )
    else:
        source_value = source_values.get(entry.source) if entry.source is not None else None
        if entry.source is None or source_value is None:
            raise ContractError(f"warm-start source tensor is unavailable: {entry.source}")
        value = source_value
        if tuple(value.shape) != entry.source_shape:
            raise ContractError(f"source shape changed after planning: {entry.source}")
        if entry.source_start is not None:
            value = value[entry.source_start : entry.source_stop]
        if entry.operation == TensorOperation.RESHAPE:
            value = value.reshape(entry.target_shape)
        elif entry.operation == TensorOperation.PAD_ROWS:
            if value.ndim != 2 or value.shape[1] != entry.target_shape[1]:
                raise ContractError(f"cannot row-pad warm-start tensor: {entry.source}")
            padded = torch.zeros(entry.target_shape, dtype=value.dtype)
            padded[: value.shape[0]].copy_(value)
            value = padded
        elif entry.operation == TensorOperation.HEAD_SCALAR_EXPAND:
            if value.ndim != 2 or entry.target_shape[1] < value.shape[0]:
                raise ContractError(f"cannot expand GDN head scalar map: {entry.source}")
            expanded = torch.zeros(entry.target_shape, dtype=value.dtype)
            channels_per_head = entry.target_shape[0] // value.shape[0]
            if channels_per_head * value.shape[0] != entry.target_shape[0]:
                raise ContractError(f"GDN head scalar expansion is not divisible: {entry.source}")
            for head in range(value.shape[0]):
                expanded[
                    head * channels_per_head : (head + 1) * channels_per_head,
                    head,
                ] = 1
            value = expanded
        elif entry.operation == TensorOperation.GDN_ERASE_LINEAR_DOWN:
            if len(entry.auxiliary_sources) != 3:
                raise ContractError(
                    f"GDN erase linearization requires decay input, A_log, and dt_bias: {entry.target}"
                )
            gradient, _ = _gdn_erase_logit_linearization(
                source_value.float(),
                source_values[entry.auxiliary_sources[0]].float(),
                source_values[entry.auxiliary_sources[1]].float(),
                source_values[entry.auxiliary_sources[2]].float(),
            )
            if gradient.ndim != 2 or gradient.shape[1] != entry.target_shape[1]:
                raise ContractError(
                    f"GDN erase gradient cannot fit target a_lora down projection: {entry.target}"
                )
            value = torch.zeros(entry.target_shape, dtype=gradient.dtype)
            value[: gradient.shape[0]].copy_(gradient)
        elif entry.operation == TensorOperation.GDN_ERASE_BIAS:
            if len(entry.auxiliary_sources) != 1:
                raise ContractError(f"GDN erase bias requires dt_bias: {entry.target}")
            _, per_head_bias = _gdn_erase_logit_linearization(
                None,
                None,
                source_value.float(),
                source_values[entry.auxiliary_sources[0]].float(),
            )
            repeats = entry.target_shape[0] // per_head_bias.numel()
            if repeats * per_head_bias.numel() != entry.target_shape[0]:
                raise ContractError(
                    f"GDN erase bias cannot expand to target recurrent width: {entry.target}"
                )
            value = per_head_bias.repeat_interleave(repeats)
        elif entry.operation == TensorOperation.GDN_CONV_TIME_MIX:
            if len(entry.auxiliary_sources) != 1:
                raise ContractError(
                    f"GDN conv time-mix requires conv1d weights: {entry.target}"
                )
            conv = source_values[entry.auxiliary_sources[0]]
            if entry.source_start is not None:
                conv = conv[entry.source_start : entry.source_stop]
            value = _gdn_conv_time_mix(
                value.float(),
                conv.float(),
                target_shape=entry.target_shape,
            )
        elif entry.operation == TensorOperation.EVEN_ROW_SUBSAMPLE:
            if value.ndim != 2 or tuple(value.shape[1:]) != entry.target_shape[1:]:
                raise ContractError(
                    f"cannot row-subsample warm-start tensor: {entry.source}"
                )
            target_rows = entry.target_shape[0]
            indices = torch.linspace(
                0,
                value.shape[0] - 1,
                target_rows,
                dtype=torch.float64,
            ).round().to(torch.long)
            value = value.index_select(0, indices)
        elif entry.operation in (
            TensorOperation.DECAY_LINEAR_UP,
            TensorOperation.DECAY_BIAS,
        ):
            if len(entry.auxiliary_sources) != 1:
                raise ContractError(f"GDN decay mapping lacks dt_bias: {entry.target}")
            dt_bias = source_values[entry.auxiliary_sources[0]].float()
            a_scale = source_value.float().exp()
            if a_scale.shape != dt_bias.shape or a_scale.ndim != 1:
                raise ContractError(f"GDN decay vectors have incompatible shapes: {entry.target}")
            rate = a_scale * torch.nn.functional.softplus(dt_bias)
            raw_probability = rate / 0.606531
            feasible = (raw_probability > 1e-4) & (raw_probability < 1 - 1e-4)
            probability = raw_probability.clamp(1e-4, 1 - 1e-4)
            if entry.operation == TensorOperation.DECAY_BIAS:
                per_head = torch.logit(probability)
                repeats = entry.target_shape[0] // per_head.numel()
                if repeats * per_head.numel() != entry.target_shape[0]:
                    raise ContractError(f"GDN decay bias cannot expand to target: {entry.target}")
                value = per_head.repeat_interleave(repeats)
            else:
                derivative = (
                    a_scale
                    * torch.sigmoid(dt_bias)
                    / (0.606531 * probability * (1 - probability))
                ) * feasible
                expanded = torch.zeros(entry.target_shape, dtype=derivative.dtype)
                channels_per_head = entry.target_shape[0] // derivative.numel()
                if channels_per_head * derivative.numel() != entry.target_shape[0]:
                    raise ContractError(f"GDN decay slope cannot expand to target: {entry.target}")
                for head, coefficient in enumerate(derivative):
                    expanded[
                        head * channels_per_head : (head + 1) * channels_per_head,
                        head,
                    ] = coefficient
                value = expanded
        elif entry.operation == TensorOperation.TILE:
            if value.numel() == 0 or entry.target_shape[0] % value.numel():
                raise ContractError(f"cannot tile warm-start tensor: {entry.source}")
            value = value.reshape(-1).repeat(entry.target_shape[0] // value.numel())
        elif entry.operation == TensorOperation.HEADWISE_QUERY_SLICE:
            query_heads = int(entry.num_query_heads)
            head_dim = entry.target_shape[0] // query_heads
            value = value.reshape(query_heads, head_dim * 2, *value.shape[1:])[:, :head_dim]
            value = value.flatten(0, 1)
        elif entry.operation == TensorOperation.HEADWISE_QUERY_GATE_SUBSAMPLE:
            query_heads = int(entry.num_query_heads)
            value = _packed_query_gate_rows(
                value,
                query_heads=query_heads,
                source_name=entry.source,
            )
            target_rows = entry.target_shape[0]
            indices = torch.linspace(
                0,
                value.shape[0] - 1,
                target_rows,
                dtype=torch.float64,
            ).round().to(torch.long)
            value = value.index_select(0, indices)
        elif entry.operation == TensorOperation.HEADWISE_QUERY_GATE_RECONSTRUCTION:
            query_heads = int(entry.num_query_heads)
            source_gate = _packed_query_gate_rows(
                value,
                query_heads=query_heads,
                source_name=entry.source,
            )
            if entry.target_shape[0] != source_gate.shape[0]:
                raise ContractError(
                    "native gate output width must equal the source gate width: "
                    f"{entry.target}"
                )
            value = _constrained_source_row_reconstruction(
                source_gate,
                basis_rows=entry.target_shape[1],
            )
        elif entry.operation == TensorOperation.SCALE:
            value = value * float(entry.scale)
        elif entry.operation == TensorOperation.KV_REPEAT:
            value = kv_repeat(
                value,
                num_query_heads=int(entry.num_query_heads),
                num_kv_heads=int(entry.num_kv_heads),
            )
            if entry.scale is not None:
                value = value * entry.scale
        elif entry.operation == TensorOperation.KV_EXPAND:
            value = kv_expand(
                value,
                num_query_heads=int(entry.num_query_heads),
                num_kv_heads=int(entry.num_kv_heads),
            )
        elif entry.operation not in (TensorOperation.COPY, TensorOperation.SLICE):
            raise ContractError(f"unsupported tensor operation: {entry.operation}")
    if tuple(value.shape) != entry.target_shape:
        raise ContractError(
            f"materialized shape mismatch for {entry.target}: "
            f"got={tuple(value.shape)} expected={entry.target_shape}"
        )
    return value.to(dtype=_torch_dtype(spec.dtype)).contiguous()


def _packed_query_gate_rows(
    packed_query: Tensor,
    *,
    query_heads: int,
    source_name: str | None,
) -> Tensor:
    if packed_query.ndim != 2 or packed_query.shape[0] % (query_heads * 2):
        raise ContractError(
            "packed query/gate rows are incompatible with source heads: "
            f"{source_name}"
        )
    head_dim = packed_query.shape[0] // (query_heads * 2)
    return (
        packed_query.reshape(query_heads, head_dim * 2, packed_query.shape[1])[
            :, head_dim:
        ]
        .flatten(0, 1)
        .float()
    )


def _constrained_source_row_reconstruction(
    source_gate: Tensor,
    *,
    basis_rows: int,
) -> Tensor:
    """Reconstruct source gate rows from a deterministic source-row basis.

    The row-sum constraint preserves the source gate value at zero input:
    ``A @ sigmoid(B @ 0) == 0.5``.  Selected source rows are restored to exact
    one-hot coefficients, while all other rows use a scale-normalized ridge
    solution.  Activation ridge fitting remains responsible for adapting this
    algebraic weight-space start to the real hidden-state distribution.
    """
    if source_gate.ndim != 2 or basis_rows <= 0:
        raise ContractError("source-row gate reconstruction has invalid geometry")
    indices = torch.linspace(
        0,
        source_gate.shape[0] - 1,
        basis_rows,
        dtype=torch.float64,
    ).round().to(torch.long)
    basis = source_gate.index_select(0, indices).float()
    gram = basis @ basis.T
    ridge = (gram.diagonal().mean() * 1e-3).clamp_min(
        torch.finfo(gram.dtype).eps
    )
    system = gram + torch.eye(basis_rows, dtype=gram.dtype) * ridge
    cross = source_gate.float() @ basis.T
    unconstrained = torch.linalg.solve(system, cross.T).T
    ones = torch.ones(basis_rows, dtype=system.dtype)
    constraint_direction = torch.linalg.solve(system, ones)
    constraint_denominator = torch.dot(ones, constraint_direction)
    if not torch.isfinite(constraint_denominator) or constraint_denominator <= 0:
        raise ContractError("source-row gate reconstruction constraint is singular")
    correction = (1.0 - unconstrained.sum(dim=1)) / constraint_denominator
    reconstruction = (
        unconstrained + correction[:, None] * constraint_direction[None, :]
    )
    for source_row in torch.unique(indices, sorted=True).tolist():
        basis_column = int((indices == source_row).nonzero()[0].item())
        reconstruction[source_row].zero_()
        reconstruction[source_row, basis_column] = 1
    if not torch.isfinite(reconstruction).all():
        raise ContractError("source-row gate reconstruction produced non-finite weights")
    return reconstruction


def _gdn_erase_logit_linearization(
    beta_weight: Tensor | None,
    decay_input_weight: Tensor | None,
    decay_log_scale: Tensor,
    dt_bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Linearize ``logit(beta * decay)`` at the source GDN zero input."""
    if decay_log_scale.ndim != 1 or dt_bias.shape != decay_log_scale.shape:
        raise ContractError("GDN erase linearization received incompatible decay vectors")
    rate_scale = decay_log_scale.exp()
    zero_decay = torch.exp(
        -rate_scale * torch.nn.functional.softplus(dt_bias)
    )
    zero_erase = (0.5 * zero_decay).clamp(1e-6, 1.0 - 1e-6)
    bias = torch.logit(zero_erase)
    if beta_weight is None or decay_input_weight is None:
        return torch.empty(0, dtype=bias.dtype), bias
    if (
        beta_weight.ndim != 2
        or decay_input_weight.shape != beta_weight.shape
        or beta_weight.shape[0] != rate_scale.numel()
    ):
        raise ContractError("GDN erase linearization received incompatible projection rows")
    denominator = 1.0 - zero_erase
    beta_coefficient = 0.5 / denominator
    decay_coefficient = -rate_scale * torch.sigmoid(dt_bias) / denominator
    gradient = (
        beta_coefficient.unsqueeze(-1) * beta_weight
        + decay_coefficient.unsqueeze(-1) * decay_input_weight
    )
    if not torch.isfinite(gradient).all() or not torch.isfinite(bias).all():
        raise ContractError("GDN erase linearization produced non-finite parameters")
    return gradient, bias


def _gdn_conv_time_mix(
    projection: Tensor,
    conv: Tensor,
    *,
    target_shape: tuple[int, ...],
) -> Tensor:
    """Project Qwen's depthwise causal conv onto RWKV7's two-tap time mix.

    The source first applies one shared input projection per output channel and
    then a channel-wise causal convolution.  Native RWKV7 can retain only the
    current and immediately previous normalized hidden state, with one mixing
    coefficient per input channel.  For every input channel we therefore find
    the best non-negative two-tap direction in least squares over all output
    rows.  Older source taps and the SiLU non-linearity remain explicitly for
    activation fitting and layer-wise distillation.
    """
    if projection.ndim != 2:
        raise ContractError("GDN conv time-mix projection must be rank two")
    if conv.ndim == 3 and conv.shape[1] == 1:
        conv = conv[:, 0]
    if conv.ndim != 2 or conv.shape[0] != projection.shape[0]:
        raise ContractError(
            "GDN conv time-mix requires one depthwise kernel per projection row"
        )
    if conv.shape[1] < 1 or target_shape != (1, 1, projection.shape[1]):
        raise ContractError("GDN conv time-mix target geometry is incompatible")
    current = conv[:, -1].unsqueeze(1) * projection
    previous = (
        torch.zeros_like(current)
        if conv.shape[1] == 1
        else conv[:, -2].unsqueeze(1) * projection
    )
    current_energy = current.square().sum(dim=0)
    previous_energy = previous.square().sum(dim=0)
    cross = (current * previous).sum(dim=0)

    # A fixed grid makes the materialization bitwise deterministic across CPU
    # BLAS implementations and enforces the native [0, 1] interpolation prior.
    candidates = torch.linspace(
        0.0, 1.0, 257, dtype=projection.dtype, device=projection.device
    ).unsqueeze(1)
    current_fraction = 1.0 - candidates
    denominator = current_fraction.square() + candidates.square()
    explained = (
        current_fraction.square() * current_energy.unsqueeze(0)
        + candidates.square() * previous_energy.unsqueeze(0)
        + 2.0 * current_fraction * candidates * cross.unsqueeze(0)
    ) / denominator
    best = explained.argmax(dim=0)
    value = candidates.squeeze(1).index_select(0, best)
    return value.reshape(target_shape)


def apply_warm_start_plan(ledger: MappingLedger, plan: WarmStartPlan) -> None:
    """Replace structural placeholder provenance with the materialized plan."""
    canonical_sources: dict[str, str] = {}
    for source_name in ledger.sources:
        canonical = canonical_text_name(source_name)
        if canonical in canonical_sources:
            raise ContractError(
                f"ambiguous source after text-prefix normalization: {canonical}"
            )
        canonical_sources[canonical] = source_name
    selected_targets: dict[str, list[str]] = {name: [] for name in ledger.sources}
    for entry in plan.entries:
        previous = ledger.targets.get(entry.target)
        if previous is None:
            raise ContractError(
                f"warm-start target is absent from mapping ledger: {entry.target}"
            )
        requested_sources = tuple(
            dict.fromkeys(
                source
                for source in (entry.source, *entry.auxiliary_sources)
                if source is not None
            )
        )
        resolved_sources: list[str] = []
        for source in requested_sources:
            raw_source = canonical_sources.get(source)
            if raw_source is None:
                raise ContractError(
                    f"warm-start source is absent from mapping ledger: {source}"
                )
            resolved_sources.append(raw_source)
            selected_targets[raw_source].append(entry.target)
        sources = tuple(resolved_sources)
        ledger.targets[entry.target] = TargetEntry(
            entry.target,
            entry.provenance,
            sources,
            entry.target_shape,
            previous.dtype,
            entry.evidence,
            plan.source_hashes,
        )
    for source_name in tuple(ledger.sources):
        if not is_sequence_mixer(source_name):
            continue
        targets = tuple(sorted(selected_targets[source_name]))
        ledger.sources[source_name] = SourceEntry(
            source_name,
            (
                SourceDisposition.CONSUMED
                if targets
                else SourceDisposition.INTENTIONALLY_UNMAPPED
            ),
            targets,
            (
                "selected by deterministic warm-start materialization"
                if targets
                else "deferred to activation fitting; zero-step has no shape-safe tensor transfer"
            ),
        )


def _plan_gdn_layer(
    index: int,
    specs: list[TensorSpec],
    target: _HeadGeometry,
    config: dict[str, object],
    source_shapes: dict[str, tuple[int, ...]],
    variant: WarmStartVariant,
) -> tuple[list[WarmStartEntry], list[WarmStartError]]:
    key_heads = int(config["linear_num_key_heads"])
    value_heads = int(config["linear_num_value_heads"])
    key_dim = int(config["linear_key_head_dim"])
    value_dim = int(config["linear_value_head_dim"])
    if value_heads % key_heads:
        raise ContractError(
            f"ambiguous GDN head expansion at layer {index}: "
            f"key_heads={key_heads} value_heads={value_heads}"
        )
    prefix = f"model.layers.{index}.linear_attn"
    packed = _first_existing(
        source_shapes,
        f"{prefix}.in_proj_qkv.weight",
        f"{prefix}.in_proj_qkvz.weight",
        f"{prefix}.in_proj_qkvzba.weight",
    )
    output = f"{prefix}.out_proj.weight"
    q_size = key_heads * key_dim
    k_size = q_size
    v_size = value_heads * value_dim
    _validate_packed_projection(packed, source_shapes, q_size + k_size + v_size)
    geometry_matches = (
        value_heads == target.num_heads
        and key_dim == target.head_dim
        and value_dim == target.head_dim
    )
    entries: list[WarmStartEntry] = []
    errors: list[WarmStartError] = []
    for spec in specs:
        native_extra = _plan_source_compatible_native_extra(spec)
        if native_extra is not None:
            entries.append(native_extra)
            continue
        structured = _plan_gdn_structured_tensor(
            spec,
            prefix=prefix,
            packed=packed,
            source_shapes=source_shapes,
            value_heads=value_heads,
            value_dim=value_dim,
            q_size=q_size,
            k_size=k_size,
            v_size=v_size,
            variant=variant,
        )
        if structured is not None:
            entries.append(structured)
            if (
                spec.name.endswith(".g_norm.weight")
                and structured.provenance == TargetProvenance.INITIALIZED
            ):
                errors.extend(
                    _partition_errors(
                        index,
                        "gdn",
                        spec,
                        "normalization_geometry_mismatch",
                        "source per-head RMSNorm cannot be tiled into target GroupNorm geometry; activation fitting is required",
                        source_shapes.get(f"{prefix}.norm.weight"),
                        target.num_heads,
                    )
                )
            continue
        role = _projection_role(spec.name)
        if role is None or variant in (
            WarmStartVariant.RANDOM,
            WarmStartVariant.KV_REPEAT,
            WarmStartVariant.KV_EXPAND,
        ):
            entries.append(_initialized_entry(spec, f"{variant.value} does not map this GDN tensor"))
            continue
        if role == "o":
            entry = _direct_entry(
                spec,
                output,
                source_shapes,
                TargetProvenance.COPIED,
                "direct GDN output projection copy with declared target dtype cast; "
                "peripheral dynamics remain non-equivalent",
                is_lossless=False,
            )
        else:
            start, stop = {
                "r": (0, q_size),
                "k": (q_size, q_size + k_size),
                "v": (q_size + k_size, q_size + k_size + v_size),
            }[role]
            needs_source_head_repeat = role in ("r", "k") and key_heads != value_heads
            if variant in (WarmStartVariant.GDN_CONSTRAINED, WarmStartVariant.MAPPED) and needs_source_head_repeat:
                entry = _gdn_repeat_entry(
                    spec,
                    packed,
                    source_shapes,
                    start,
                    stop,
                    role,
                    key_heads,
                    value_heads,
                    key_dim,
                )
            else:
                provenance = (
                    TargetProvenance.ALGEBRAIC
                    if variant in (WarmStartVariant.GDN_CONSTRAINED, WarmStartVariant.MAPPED) and role == "r"
                    else TargetProvenance.COPIED
                )
                scale = 1.0 / sqrt(key_dim) if provenance == TargetProvenance.ALGEBRAIC else None
                entry = _slice_entry(
                    spec,
                    packed,
                    source_shapes,
                    start,
                    stop,
                    provenance,
                    (
                        "GDN read projection scaled by 1/sqrt(source_key_dim); beta/decay peripherals require fitting"
                        if scale is not None
                        else f"direct {role} slice from packed GDN qkv tensor with declared target dtype cast"
                    ),
                    scale=scale,
                    is_lossless=False,
                )
        if entry is None:
            entries.append(_initialized_entry(spec, "source projection shape requires fitting"))
            errors.extend(
                _partition_errors(
                    index,
                    "gdn",
                    spec,
                    "projection_shape_mismatch",
                    "projection cannot be copied or uniquely reshaped; fitted or initialized required",
                    source_shapes.get(packed if role != "o" else output),
                    target.num_heads,
                )
            )
        else:
            if variant == WarmStartVariant.MAPPED and not geometry_matches:
                entry = replace(
                    entry,
                    evidence=(
                        entry.evidence
                        + "; copied/algebraic value is only the starting point for teacher-trace fitting under changed head/state geometry"
                    ),
                )
            entries.append(entry)
            if not geometry_matches and role is not None:
                errors.extend(
                    _partition_errors(
                        index,
                        "gdn",
                        spec,
                        "semantic_geometry_mismatch",
                        "flat projection channels transfer deterministically, but source and target head/state geometry differ; trace fitting is required and the transfer is not lossless",
                        entry.source_shape,
                        target.num_heads,
                    )
                )
    return entries, errors


def _plan_gdn_structured_tensor(
    spec: TensorSpec,
    *,
    prefix: str,
    packed: str,
    source_shapes: dict[str, tuple[int, ...]],
    value_heads: int,
    value_dim: int,
    q_size: int,
    k_size: int,
    v_size: int,
    variant: WarmStartVariant,
) -> WarmStartEntry | None:
    if variant not in (WarmStartVariant.GDN_CONSTRAINED, WarmStartVariant.MAPPED):
        return None
    time_mix_role = next(
        (
            role
            for role in ("r", "k", "v", "w", "a", "g")
            if spec.name.endswith(f".x_{role}")
        ),
        None,
    )
    if time_mix_role in {"w", "a", "g"}:
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.ZERO,
            "source GDN decay, erase, and gate projections consume only the current normalized token",
            is_semantically_lossless=True,
            local_trainable=True,
        )
    if time_mix_role in {"r", "k", "v"}:
        projection_shape = source_shapes.get(packed)
        conv = f"{prefix}.conv1d.weight"
        conv_shape = source_shapes.get(conv)
        start, stop = {
            "r": (0, q_size),
            "k": (q_size, q_size + k_size),
            "v": (q_size + k_size, q_size + k_size + v_size),
        }[time_mix_role]
        if (
            projection_shape is None
            or projection_shape[0] < stop
            or projection_shape[1] != spec.shape[-1]
            or conv_shape is None
            or conv_shape[0] < stop
        ):
            raise ContractError(
                f"GDN {time_mix_role} conv/projection geometry cannot initialize native time mix"
            )
        return WarmStartEntry(
            spec.name,
            packed,
            spec.shape,
            projection_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.GDN_CONV_TIME_MIX,
            "least-squares projection of the source depthwise causal conv current/previous taps into native RWKV7 time mix; older taps and SiLU remain for activation fitting",
            source_start=start,
            source_stop=stop,
            auxiliary_sources=(conv,),
            is_semantically_lossless=False,
            local_trainable=True,
        )
    if spec.name.endswith(".k_k"):
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.INITIALIZE,
            "fix native normalized-key scale to one for the source GDN key coordinates",
            is_semantically_lossless=True,
            local_trainable=True,
        )
    if spec.name.endswith(".k_a"):
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.ZERO,
            "disable native write-key interpolation so the normalized source GDN key is used unchanged for writing",
            is_semantically_lossless=True,
            local_trainable=True,
        )
    beta = f"{prefix}.in_proj_b.weight"
    decay_input = f"{prefix}.in_proj_a.weight"
    decay_scale = f"{prefix}.A_log"
    decay_bias = f"{prefix}.dt_bias"
    gate = f"{prefix}.in_proj_z.weight"
    norm = f"{prefix}.norm.weight"
    if spec.name.endswith(".w_lora.lora.0.weight"):
        source_shape = source_shapes.get(decay_input)
        if source_shape != (value_heads, spec.shape[1]) or spec.shape[0] < value_heads:
            raise ContractError(f"GDN decay projection cannot fit target w_lora rank: {decay_input}")
        return WarmStartEntry(
            spec.name,
            decay_input,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.PAD_ROWS,
            "embed source GDN input-dependent decay logits in target w_lora rank space",
            is_semantically_lossless=True,
            local_trainable=True,
        )
    if spec.name.endswith(".w_lora.lora.2.weight"):
        source_shape = source_shapes.get(decay_scale)
        dt_shape = source_shapes.get(decay_bias)
        if source_shape != (value_heads,) or dt_shape != (value_heads,):
            raise ContractError(f"GDN decay vectors are incompatible: {decay_scale}")
        return WarmStartEntry(
            spec.name,
            decay_scale,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.DECAY_LINEAR_UP,
            "first-order map of source exp(A_log)*softplus(a+dt_bias) into native RWKV7 decay logits",
            is_semantically_lossless=False,
            auxiliary_sources=(decay_bias,),
        )
    if spec.name.endswith(".w_lora.lora.2.bias"):
        source_shape = source_shapes.get(decay_scale)
        dt_shape = source_shapes.get(decay_bias)
        if source_shape != (value_heads,) or dt_shape != (value_heads,):
            raise ContractError(f"GDN decay vectors are incompatible: {decay_scale}")
        return WarmStartEntry(
            spec.name,
            decay_scale,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.DECAY_BIAS,
            "map source zero-input GDN decay into the bounded native RWKV7 decay parameterization",
            is_semantically_lossless=False,
            auxiliary_sources=(decay_bias,),
        )
    if spec.name.endswith(".a_lora.lora.0.weight"):
        source_shape = source_shapes.get(beta)
        if (
            source_shape != (value_heads, spec.shape[1])
            or source_shapes.get(decay_input) != source_shape
            or source_shapes.get(decay_scale) != (value_heads,)
            or source_shapes.get(decay_bias) != (value_heads,)
            or spec.shape[0] < value_heads
        ):
            raise ContractError(
                f"GDN beta/decay projections cannot fit target a_lora down projection: {beta}"
            )
        return WarmStartEntry(
            spec.name,
            beta,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.GDN_ERASE_LINEAR_DOWN,
            "first-order zero-input map of source logit(beta * decay) into the native erase network",
            is_semantically_lossless=False,
            auxiliary_sources=(decay_input, decay_scale, decay_bias),
        )
    if spec.name.endswith(".a_lora.lora.2.weight"):
        source_shape = source_shapes.get(beta)
        if (
            source_shape is None
            or source_shape[0] != value_heads
            or spec.shape[1] < value_heads
        ):
            raise ContractError(f"GDN beta head geometry is incompatible with target channels: {beta}")
        if spec.shape[0] != value_heads * value_dim:
            return _initialized_entry(
                spec,
                "source GDN beta channel expansion does not match the target recurrent width; "
                "defer to activation fitting",
            )
        return WarmStartEntry(
            spec.name,
            beta,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.HEAD_SCALAR_EXPAND,
            "repeat each first-order source GDN erase logit across its value-head channels",
            is_semantically_lossless=False,
        )
    if spec.name.endswith(".a_lora.lora.2.bias"):
        source_shape = source_shapes.get(decay_scale)
        if source_shape != (value_heads,) or source_shapes.get(decay_bias) != (value_heads,):
            raise ContractError(
                f"GDN decay vectors cannot initialize target erase bias: {decay_scale}"
            )
        return WarmStartEntry(
            spec.name,
            decay_scale,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.GDN_ERASE_BIAS,
            "install zero-input logit(beta * decay) as the native erase bias",
            is_semantically_lossless=False,
            auxiliary_sources=(decay_bias,),
        )
    if spec.name.endswith(".g_lora.lora.0.weight"):
        source_shape = source_shapes.get(gate)
        if (
            source_shape is None
            or len(source_shape) != 2
            or source_shape[1] != spec.shape[1]
        ):
            raise ContractError(
                f"GDN gate projection cannot initialize target g_lora basis: {gate}"
            )
        return WarmStartEntry(
            spec.name,
            gate,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.EVEN_ROW_SUBSAMPLE,
            "deterministically sample source GDN gate preactivation directions (with repetition only when the native rank is wider) as the native low-rank gate basis; activation fitting learns their output combination",
            is_semantically_lossless=False,
            local_trainable=True,
        )
    if spec.name.endswith(".g_norm.weight"):
        source_shape = source_shapes.get(norm)
        if source_shape != (value_dim,):
            raise ContractError(f"GDN gated norm source shape is malformed: {norm}")
        if spec.shape != (value_heads * value_dim,):
            return _initialized_entry(
                spec,
                "source per-head RMSNorm and target GroupNorm channel geometry differ; defer to activation fitting",
            )
        return WarmStartEntry(
            spec.name,
            norm,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.TILE,
            "tile the shared source GDN per-head RMSNorm scale across value heads; target group boundaries remain lossy",
            is_semantically_lossless=False,
        )
    return None


def _plan_full_attention_layer(
    index: int,
    specs: list[TensorSpec],
    target: _HeadGeometry,
    config: dict[str, object],
    source_shapes: dict[str, tuple[int, ...]],
    variant: WarmStartVariant,
) -> tuple[list[WarmStartEntry], list[WarmStartError]]:
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    group_map = build_group_map(query_heads, kv_heads)
    prefix = f"model.layers.{index}.self_attn"
    names = {role: f"{prefix}.{name}_proj.weight" for role, name in (("r", "q"), ("k", "k"), ("v", "v"), ("o", "o"))}
    geometry_matches = query_heads == target.num_heads and head_dim == target.head_dim
    entries: list[WarmStartEntry] = []
    errors: list[WarmStartError] = []
    for spec in specs:
        if variant in (
            WarmStartVariant.NAIVE_COPY,
            WarmStartVariant.KV_REPEAT,
            WarmStartVariant.KV_EXPAND,
            WarmStartVariant.MAPPED,
        ) and any(
            spec.name.endswith(f".x_{role}")
            for role in ("r", "w", "k", "v", "a", "g")
        ):
            entries.append(
                WarmStartEntry(
                    spec.name,
                    None,
                    spec.shape,
                    None,
                    TargetProvenance.ALGEBRAIC,
                    TensorOperation.ZERO,
                    "source full-attention projections consume only the current normalized token",
                    is_semantically_lossless=True,
                    local_trainable=True,
                )
            )
            continue
        source_compatible = _plan_full_attention_native_extra(
            spec,
            packed_query=names["r"],
            source_shapes=source_shapes,
            query_heads=query_heads,
            head_dim=head_dim,
            variant=variant,
        )
        if source_compatible is not None:
            entries.append(source_compatible)
            continue
        native_extra = _plan_source_compatible_native_extra(spec)
        if native_extra is not None:
            entries.append(native_extra)
            continue
        role = _projection_role(spec.name)
        if role is None or variant in (WarmStartVariant.RANDOM, WarmStartVariant.GDN_CONSTRAINED):
            entries.append(_initialized_entry(spec, f"{variant.value} does not map this full-attention tensor"))
            continue
        source_name = names[role]
        if role == "r":
            q_size = query_heads * head_dim
            source_shape = source_shapes.get(source_name)
            if source_shape is None:
                raise ContractError(f"source projection is missing: {source_name}")
            if source_shape[0] == q_size:
                entry = _slice_entry(
                    spec,
                    source_name,
                    source_shapes,
                    0,
                    q_size,
                    TargetProvenance.COPIED,
                    "direct full-attention query projection copy without a packed query gate",
                    is_lossless=False,
                )
            elif source_shape[0] == q_size * 2:
                entry = _headwise_query_entry(
                    spec,
                    source_name,
                    source_shapes,
                    query_heads,
                    head_dim,
                )
            else:
                entry = None
        elif role in ("k", "v") and variant in (
            WarmStartVariant.KV_REPEAT,
            WarmStartVariant.KV_EXPAND,
            WarmStartVariant.MAPPED,
        ):
            operation = (
                TensorOperation.KV_REPEAT
                if variant in (WarmStartVariant.KV_REPEAT, WarmStartVariant.MAPPED)
                else TensorOperation.KV_EXPAND
            )
            provenance = (
                TargetProvenance.ALGEBRAIC
                if operation == TensorOperation.KV_REPEAT
                else TargetProvenance.INITIALIZED
            )
            entry = _kv_entry(
                spec,
                source_name,
                source_shapes,
                operation,
                provenance,
                query_heads,
                kv_heads,
                (
                    "deterministic GQA KV-group repetition; tensor mapping is exact but "
                    "recurrent state semantics are not lossless"
                    if operation == TensorOperation.KV_REPEAT
                    else "deterministic kv_expand ablation: KV repetition with per-query-head scale separation"
                ),
            )
        else:
            entry = _direct_entry(
                spec,
                source_name,
                source_shapes,
                TargetProvenance.COPIED,
                f"direct full-attention {role} projection copy with declared target dtype cast",
                is_lossless=False,
            )
        if entry is None:
            entries.append(_initialized_entry(spec, "source projection shape requires fitting"))
            errors.extend(
                _partition_errors(
                    index,
                    "full_attention",
                    spec,
                    "projection_shape_mismatch",
                    "projection cannot be copied, uniquely reshaped, or mapped by the selected GQA operation",
                    source_shapes.get(source_name),
                    target.num_heads,
                    group_map.query_to_kv,
                )
            )
        else:
            if variant == WarmStartVariant.MAPPED:
                entry = replace(
                    entry,
                    evidence=(
                        entry.evidence
                        + "; full-attention recurrence semantics require teacher-trace fitting"
                    ),
                )
            entries.append(entry)
            if not geometry_matches:
                errors.extend(
                    _partition_errors(
                        index,
                        "full_attention",
                        spec,
                        "head_geometry_mismatch",
                        (
                            f"flat projection transfer succeeded, but source query_heads={query_heads} "
                            f"head_dim={head_dim} differs from target heads={target.num_heads} "
                            f"head_dim={target.head_dim}; recurrent trace fitting is required"
                        ),
                        entry.source_shape,
                        target.num_heads,
                        group_map.query_to_kv,
                    )
                )
            if role in ("k", "v") and variant in (
                WarmStartVariant.KV_REPEAT,
                WarmStartVariant.KV_EXPAND,
                WarmStartVariant.MAPPED,
            ):
                errors.extend(
                    _partition_errors(
                        index,
                        "full_attention",
                        spec,
                        "recurrent_state_semantics_changed",
                        "GQA group expansion is deterministic, but full-attention KV cache and "
                        "RWKV7 recurrent state are not lossless equivalents",
                        entry.source_shape,
                        target.num_heads,
                        group_map.query_to_kv,
                    )
                )
    return entries, errors


def _plan_full_attention_native_extra(
    spec: TensorSpec,
    *,
    packed_query: str,
    source_shapes: dict[str, tuple[int, ...]],
    query_heads: int,
    head_dim: int,
    variant: WarmStartVariant,
) -> WarmStartEntry | None:
    """Initialize the non-isomorphic RWKV7 core as causal linear attention.

    Full attention has no exact finite-state RWKV7 parameter map.  The least
    destructive zero-step boundary keeps the source Q/K/V coordinates, uses an
    accumulating state (near-unit decay and near-zero erase), and disables the
    native key interpolation and bonus until teacher-trace fitting can justify
    them.
    """
    if variant not in (
        WarmStartVariant.NAIVE_COPY,
        WarmStartVariant.KV_REPEAT,
        WarmStartVariant.KV_EXPAND,
        WarmStartVariant.MAPPED,
    ):
        return None
    if spec.name.endswith(".k_k"):
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.INITIALIZE,
            "use unit normalized-key scale for the source attention coordinates",
            is_semantically_lossless=False,
            local_trainable=True,
        )
    if spec.name.endswith(".k_a") or spec.name.endswith(".r_k"):
        role = "write-key interpolation" if spec.name.endswith(".k_a") else "bonus"
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.ZERO,
            f"disable native {role} absent from source full attention",
            is_semantically_lossless=False,
            local_trainable=True,
        )
    if spec.name.endswith((".w_lora.lora.2.weight", ".a_lora.lora.2.weight")):
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.ZERO,
            "start from input-independent recurrent controls while retaining a trainable latent basis",
            is_semantically_lossless=False,
            local_trainable=True,
        )
    if spec.name.endswith((".w_lora.lora.2.bias", ".a_lora.lora.2.bias")):
        control = "decay rate" if ".w_lora." in spec.name else "erase rate"
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.DISABLED_SIGMOID_BIAS,
            f"set the source-absent native {control} to BF16-epsilon squared",
            is_semantically_lossless=False,
            local_trainable=True,
        )
    if spec.name.endswith(".g_lora.lora.0.weight"):
        source_shape = source_shapes.get(packed_query)
        if source_shape != (
            query_heads * head_dim * 2,
            spec.shape[1],
        ):
            raise ContractError(
                "packed full-attention query/gate projection cannot initialize "
                f"native gate basis: {packed_query}"
            )
        return WarmStartEntry(
            spec.name,
            packed_query,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.HEADWISE_QUERY_GATE_SUBSAMPLE,
            "extract Qwen3.5 q_proj gate channels per head and deterministically sample them as the native low-rank gate basis",
            num_query_heads=query_heads,
            is_semantically_lossless=False,
            local_trainable=True,
        )
    if spec.name.endswith(".g_lora.lora.2.weight"):
        source_shape = source_shapes.get(packed_query)
        if source_shape != (
            query_heads * head_dim * 2,
            spec.shape[0],
        ):
            raise ContractError(
                "packed full-attention query/gate projection cannot reconstruct "
                f"native gate output: {packed_query}"
            )
        return WarmStartEntry(
            spec.name,
            packed_query,
            spec.shape,
            source_shape,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.HEADWISE_QUERY_GATE_RECONSTRUCTION,
            "reconstruct all Qwen3.5 sigmoid gate channels from the deterministic source-row basis while preserving the 0.5 zero-input gate",
            num_query_heads=query_heads,
            is_semantically_lossless=False,
            local_trainable=True,
        )
    return None


def _target_geometry(specs: list[TensorSpec], index: int) -> _HeadGeometry:
    r_k = next((spec for spec in specs if spec.name.endswith(".r_k")), None)
    if r_k is None or len(r_k.shape) != 2 or min(r_k.shape) <= 0:
        raise ContractError(f"target layer {index} lacks an unambiguous r_k [heads,head_dim] spec")
    return _HeadGeometry(int(r_k.shape[0]), int(r_k.shape[1]))


def _projection_role(name: str) -> str | None:
    for role in ("r", "k", "v", "o"):
        if name.endswith(f".{role}_proj.weight"):
            return role
    return None


def _initialized_entry(spec: TensorSpec, evidence: str) -> WarmStartEntry:
    return WarmStartEntry(
        spec.name,
        None,
        spec.shape,
        None,
        TargetProvenance.INITIALIZED,
        TensorOperation.INITIALIZE,
        evidence,
        local_trainable=True,
    )


def _plan_source_compatible_native_extra(
    spec: TensorSpec,
) -> WarmStartEntry | None:
    if ".v_lora." not in spec.name:
        return None
    if spec.name.endswith(".v_lora.lora.0.weight"):
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.INITIALIZED,
            TensorOperation.INITIALIZE,
            "retain a deterministic latent basis while the source-incompatible native value-residual gate is disabled",
            local_trainable=True,
        )
    if spec.name.endswith(".v_lora.lora.2.weight"):
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.ZERO,
            "remove input-dependent layer-0 value mixing because the source mixer has no cross-layer value residual",
            is_semantically_lossless=True,
            local_trainable=True,
        )
    if spec.name.endswith(".v_lora.lora.2.bias"):
        return WarmStartEntry(
            spec.name,
            None,
            spec.shape,
            None,
            TargetProvenance.ALGEBRAIC,
            TensorOperation.INITIALIZE,
            "set the native value-residual coefficient to the logit of BF16 eps squared instead of the source-incompatible sigmoid(0)=0.5 default",
            is_semantically_lossless=False,
            local_trainable=True,
        )
    raise ContractError(f"unknown native value-residual tensor: {spec.name}")


def _direct_entry(
    spec: TensorSpec,
    source: str,
    shapes: dict[str, tuple[int, ...]],
    provenance: TargetProvenance,
    evidence: str,
    *,
    is_lossless: bool,
) -> WarmStartEntry | None:
    shape = shapes.get(source)
    if shape is None:
        raise ContractError(f"source projection is missing: {source}")
    if shape == spec.shape:
        operation = TensorOperation.COPY
    elif len(shape) == 3 and shape[0] * shape[1] == spec.shape[0] and shape[2:] == spec.shape[1:]:
        operation = TensorOperation.RESHAPE
    else:
        return None
    return WarmStartEntry(
        spec.name,
        source,
        spec.shape,
        shape,
        provenance,
        operation,
        evidence,
        is_semantically_lossless=is_lossless,
    )


def _slice_entry(
    spec: TensorSpec,
    source: str,
    shapes: dict[str, tuple[int, ...]],
    start: int,
    stop: int,
    provenance: TargetProvenance,
    evidence: str,
    *,
    scale: float | None = None,
    is_lossless: bool,
) -> WarmStartEntry | None:
    shape = shapes.get(source)
    if shape is None:
        raise ContractError(f"source projection is missing: {source}")
    sliced_shape = (stop - start, *shape[1:])
    if sliced_shape != spec.shape:
        return None
    operation = TensorOperation.SCALE if scale is not None else TensorOperation.SLICE
    return WarmStartEntry(
        spec.name,
        source,
        spec.shape,
        shape,
        provenance,
        operation,
        evidence,
        source_start=start,
        source_stop=stop,
        scale=scale,
        is_semantically_lossless=is_lossless,
    )


def _kv_entry(
    spec: TensorSpec,
    source: str,
    shapes: dict[str, tuple[int, ...]],
    operation: TensorOperation,
    provenance: TargetProvenance,
    query_heads: int,
    kv_heads: int,
    evidence: str,
) -> WarmStartEntry | None:
    shape = shapes.get(source)
    if shape is None:
        raise ContractError(f"source projection is missing: {source}")
    if not shape or shape[0] % kv_heads:
        return None
    repeated_shape = (shape[0] // kv_heads * query_heads, *shape[1:])
    if repeated_shape != spec.shape:
        return None
    return WarmStartEntry(
        spec.name,
        source,
        spec.shape,
        shape,
        provenance,
        operation,
        evidence,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        is_semantically_lossless=False,
    )


def _headwise_query_entry(
    spec: TensorSpec,
    source: str,
    shapes: dict[str, tuple[int, ...]],
    query_heads: int,
    head_dim: int,
) -> WarmStartEntry | None:
    shape = shapes[source]
    if shape != (query_heads * head_dim * 2, *spec.shape[1:]):
        return None
    return WarmStartEntry(
        spec.name,
        source,
        spec.shape,
        shape,
        TargetProvenance.COPIED,
        TensorOperation.HEADWISE_QUERY_SLICE,
        "Qwen3.5 q_proj reshaped as [heads,2*head_dim,input] and sliced per head; query gate excluded",
        num_query_heads=query_heads,
        is_semantically_lossless=False,
    )


def _gdn_repeat_entry(
    spec: TensorSpec,
    source: str,
    shapes: dict[str, tuple[int, ...]],
    start: int,
    stop: int,
    role: str,
    key_heads: int,
    value_heads: int,
    source_key_dim: int,
) -> WarmStartEntry | None:
    shape = shapes[source]
    sliced_shape = (stop - start, *shape[1:])
    repeated_shape = (sliced_shape[0] // key_heads * value_heads, *sliced_shape[1:])
    if repeated_shape != spec.shape:
        return None
    scale = 1.0 / sqrt(source_key_dim) if role == "r" else None
    return WarmStartEntry(
        spec.name,
        source,
        spec.shape,
        shape,
        TargetProvenance.ALGEBRAIC,
        TensorOperation.KV_REPEAT,
        (
            f"source GDN {role} heads repeated by the source-defined value_heads/key_heads ratio"
            + (" and scaled by 1/sqrt(source_key_dim)" if scale is not None else "")
        ),
        source_start=start,
        source_stop=stop,
        num_query_heads=value_heads,
        num_kv_heads=key_heads,
        scale=scale,
        is_semantically_lossless=False,
    )


def _partition_errors(
    index: int,
    mixer_kind: str,
    spec: TensorSpec,
    code: str,
    message: str,
    source_shape: tuple[int, ...] | None,
    num_heads: int,
    groups: tuple[int, ...] | None = None,
) -> list[WarmStartError]:
    return [
        WarmStartError(
            index,
            mixer_kind,
            spec.name,
            code,
            message,
            head,
            None if groups is None or head >= len(groups) else groups[head],
            source_shape,
            spec.shape,
        )
        for head in range(num_heads)
    ]


def _first_existing(shapes: dict[str, tuple[int, ...]], *names: str) -> str:
    matches = [name for name in names if name in shapes]
    if len(matches) != 1:
        raise ContractError(f"GDN packed projection must have one unambiguous source, found={matches}")
    return matches[0]


def _validate_packed_projection(name: str, shapes: dict[str, tuple[int, ...]], required_rows: int) -> None:
    shape = shapes[name]
    requires_exact_rows = name.endswith(".in_proj_qkv.weight")
    if len(shape) != 2 or shape[0] < required_rows or (requires_exact_rows and shape[0] != required_rows):
        raise ContractError(
            f"GDN packed projection has invalid shape: tensor={name} shape={shape} required_rows={required_rows}"
        )


def _read_source_shapes(source: CheckpointManifest) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in source.shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                canonical = canonical_text_name(name)
                if canonical in shapes:
                    raise ContractError(
                        f"duplicate tensor after text-prefix normalization: {canonical}"
                    )
                shapes[canonical] = tuple(handle.get_slice(name).get_shape())
    return shapes


def _read_source_tensors(
    source: CheckpointManifest, wanted: set[str]
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for shard in source.shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                canonical = canonical_text_name(name)
                if canonical not in wanted:
                    continue
                if canonical in tensors:
                    raise ContractError(
                        f"duplicate tensor after text-prefix normalization: {canonical}"
                    )
                tensors[canonical] = handle.get_tensor(name)
    missing = sorted(wanted - tensors.keys())
    if missing:
        raise ContractError(f"warm-start source tensors are missing: {missing}")
    return tensors


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as error:
        raise ContractError(f"unsupported target tensor dtype: {name}") from error
