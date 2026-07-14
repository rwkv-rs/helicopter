from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import ContractError
from .mapping import MappingLedger, SourceDisposition, SourceEntry, TargetEntry, TargetProvenance


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    initialization: str


def rwkv7_mixer_specs(
    layer_index: int,
    *,
    hidden_size: int,
    head_dim: int = 64,
    decay_rank: int = 64,
    a_rank: int = 64,
    gate_rank: int = 128,
    value_rank: int = 32,
) -> tuple[TensorSpec, ...]:
    heads = hidden_size // head_dim
    prefix = f"model.layers.{layer_index}.attn"
    specs = [
        *(TensorSpec(f"{prefix}.{name}", (1, 1, hidden_size), "bfloat16", "source-statistical-time-mix-v1") for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")),
        TensorSpec(f"{prefix}.k_k", (hidden_size,), "bfloat16", "normalized-key-scale-v1"),
        TensorSpec(f"{prefix}.k_a", (hidden_size,), "bfloat16", "erase-interpolation-v1"),
        TensorSpec(f"{prefix}.r_k", (heads, head_dim), "bfloat16", "zero-bonus-v1"),
        *(TensorSpec(f"{prefix}.{name}_proj.weight", (hidden_size, hidden_size), "bfloat16", "teacher-trace-ridge-v1") for name in ("r", "k", "v", "o")),
        TensorSpec(f"{prefix}.w_lora.lora.0.weight", (decay_rank, hidden_size), "bfloat16", "teacher-trace-ridge-v1"),
        TensorSpec(f"{prefix}.w_lora.lora.2.weight", (hidden_size, decay_rank), "bfloat16", "teacher-trace-ridge-v1"),
        TensorSpec(f"{prefix}.w_lora.lora.2.bias", (hidden_size,), "bfloat16", "native-decay-inverse-v1"),
        TensorSpec(f"{prefix}.a_lora.lora.0.weight", (a_rank, hidden_size), "bfloat16", "teacher-trace-ridge-v1"),
        TensorSpec(f"{prefix}.a_lora.lora.2.weight", (hidden_size, a_rank), "bfloat16", "teacher-trace-ridge-v1"),
        TensorSpec(f"{prefix}.a_lora.lora.2.bias", (hidden_size,), "bfloat16", "erase-gate-v1"),
        TensorSpec(f"{prefix}.g_lora.lora.0.weight", (gate_rank, hidden_size), "bfloat16", "teacher-trace-ridge-v1"),
        TensorSpec(f"{prefix}.g_lora.lora.2.weight", (hidden_size, gate_rank), "bfloat16", "teacher-trace-ridge-v1"),
        TensorSpec(f"{prefix}.g_norm.weight", (hidden_size,), "bfloat16", "source-output-statistics-v1"),
        TensorSpec(f"{prefix}.g_norm.bias", (hidden_size,), "bfloat16", "zero-v1"),
    ]
    if layer_index:
        specs.extend((
            TensorSpec(f"{prefix}.v_lora.lora.0.weight", (value_rank, hidden_size), "bfloat16", "teacher-trace-ridge-v1"),
            TensorSpec(f"{prefix}.v_lora.lora.2.weight", (hidden_size, value_rank), "bfloat16", "teacher-trace-ridge-v1"),
            TensorSpec(f"{prefix}.v_lora.lora.2.bias", (hidden_size,), "bfloat16", "zero-v1"),
        ))
    return tuple(specs)


def is_sequence_mixer(name: str) -> bool:
    # Only the 60 text-backbone mixers are conversion targets. Qwen3.5 MTP
    # contains its own full-attention decoder layer and must remain bitwise
    # preserved, even though its parameter names also contain ``self_attn``.
    canonical = canonical_text_name(name)
    return canonical.startswith("model.layers.") and (
        ".linear_attn." in canonical or ".self_attn." in canonical
    )


def is_vision_tensor(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("visual", "vision", "image", "patch_embed", "merger"))


def canonical_text_name(name: str) -> str:
    """Map the multimodal Qwen wrapper prefix to the causal-LM text shell."""
    for prefix, target in (
        ("model.language_model.lm_head.", "lm_head."),
        ("model.language_model.mtp.", "mtp."),
        ("language_model.lm_head.", "lm_head."),
        ("language_model.mtp.", "mtp."),
    ):
        if name.startswith(prefix):
            return target + name.removeprefix(prefix)
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name.startswith("language_model."):
        return "model." + name.removeprefix("language_model.")
    return name


def layer_index(name: str) -> int | None:
    parts = name.split(".")
    try:
        position = parts.index("layers")
        return int(parts[position + 1])
    except (ValueError, IndexError):
        return None


def build_zero_step_ledger(
    source_names: Iterable[str],
    *,
    layer_count: int,
    hidden_size: int,
    source_shard_hashes: tuple[str, ...],
) -> tuple[MappingLedger, tuple[TensorSpec, ...], tuple[str, ...]]:
    source_names = tuple(sorted(source_names))
    specs = tuple(
        spec
        for index in range(layer_count)
        for spec in rwkv7_mixer_specs(index, hidden_size=hidden_size)
    )
    target_names = {spec.name for spec in specs}
    preserved_pairs = tuple(
        (name, canonical_text_name(name))
        for name in source_names
        if not is_sequence_mixer(name) and not is_vision_tensor(name)
    )
    preserved_targets = [target for _, target in preserved_pairs]
    if len(set(preserved_targets)) != len(preserved_targets):
        raise ContractError("text-backbone prefix normalization produced duplicate target tensors")
    preserved = tuple(preserved_targets)
    target_names.update(preserved)
    ledger = MappingLedger()
    sources_by_layer: dict[int, list[str]] = {index: [] for index in range(layer_count)}
    for name in source_names:
        index = layer_index(name)
        if is_vision_tensor(name):
            entry = SourceEntry(name, SourceDisposition.INTENTIONALLY_UNMAPPED, (), "text-backbone-only scope")
        elif is_sequence_mixer(name):
            if index is None or index not in sources_by_layer:
                entry = SourceEntry(name, SourceDisposition.REJECTED, (), "mixer tensor has ambiguous layer ownership")
            else:
                sources_by_layer[index].append(name)
                entry = SourceEntry(
                    name,
                    SourceDisposition.INTENTIONALLY_UNMAPPED,
                    (),
                    "structural zero-step defers sequence-mixer provenance to warm-start/fitting",
                )
        else:
            entry = SourceEntry(
                name,
                SourceDisposition.PRESERVED,
                (canonical_text_name(name),),
                "non-mixer tensor copied without semantic change; multimodal text prefix normalized if present",
            )
        ledger.add_source(entry)
    for source_name, target_name in preserved_pairs:
        ledger.add_target(TargetEntry(target_name, TargetProvenance.COPIED, (source_name,), (), "source", "bitwise-copy-v1", source_shard_hashes))
    for spec in specs:
        ledger.add_target(TargetEntry(spec.name, TargetProvenance.INITIALIZED, (), spec.shape, spec.dtype, spec.initialization, source_shard_hashes))
    ledger.validate(source_names, target_names)
    return ledger, specs, tuple(sorted(target_names))
