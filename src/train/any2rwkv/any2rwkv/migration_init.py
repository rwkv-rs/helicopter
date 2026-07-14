from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import sqrt

import torch
from safetensors import safe_open
from torch import Tensor

from .checkpoint import CheckpointManifest, sha256_file
from .errors import ContractError
from .export import initialize_tensor
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
    COPY = "copy"
    RESHAPE = "reshape"
    SLICE = "slice"
    HEADWISE_QUERY_SLICE = "headwise_query_slice"
    SCALE = "scale"
    KV_REPEAT = "kv_repeat"
    KV_EXPAND = "kv_expand"


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
        source_value = None
        if entry.source is not None:
            source_layer_index = layer_index(entry.source)
            if source_layer_index is None:
                source_value = self.tensor_store.load_named_tensors((entry.source,))[
                    entry.source
                ]
            else:
                if source_layer_index != self._cached_layer_index:
                    self._cached_layer_tensors = self.tensor_store.load_layer(
                        source_layer_index
                    )
                    self._cached_layer_index = source_layer_index
                source_value = self._cached_layer_tensors[entry.source]
        return _materialize_entry(spec, entry, source_value, seed=self.seed)


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
        {entry.source for entry in plan.entries if entry.source is not None},
    )
    result: dict[str, Tensor] = {}
    for entry in plan.entries:
        spec = specs[entry.target]
        source_value = None if entry.source is None else tensors.get(entry.source)
        result[entry.target] = _materialize_entry(
            spec, entry, source_value, seed=seed
        )
    return result


def _materialize_entry(
    spec: TensorSpec,
    entry: WarmStartEntry,
    source_value: Tensor | None,
    *,
    seed: int,
) -> Tensor:
    if tuple(spec.shape) != entry.target_shape:
        raise ContractError(f"stale target shape in warm-start plan: {entry.target}")
    if entry.operation == TensorOperation.INITIALIZE:
        value = initialize_tensor(spec, base_seed=seed)
    else:
        if entry.source is None or source_value is None:
            raise ContractError(f"warm-start source tensor is unavailable: {entry.source}")
        value = source_value
        if tuple(value.shape) != entry.source_shape:
            raise ContractError(f"source shape changed after planning: {entry.source}")
        if entry.source_start is not None:
            value = value[entry.source_start : entry.source_stop]
        if entry.operation == TensorOperation.RESHAPE:
            value = value.reshape(entry.target_shape)
        elif entry.operation == TensorOperation.HEADWISE_QUERY_SLICE:
            query_heads = int(entry.num_query_heads)
            head_dim = entry.target_shape[0] // query_heads
            value = value.reshape(query_heads, head_dim * 2, *value.shape[1:])[:, :head_dim]
            value = value.flatten(0, 1)
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
        if entry.source is None:
            sources: tuple[str, ...] = ()
        else:
            raw_source = canonical_sources.get(entry.source)
            if raw_source is None:
                raise ContractError(
                    f"warm-start source is absent from mapping ledger: {entry.source}"
                )
            sources = (raw_source,)
            selected_targets[raw_source].append(entry.target)
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
    )


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
