from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ContractError

SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5_moe_text", "qwen3_5", "qwen3_5_moe"})
SUPPORTED_ARCHITECTURES = frozenset({
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
})
SUPPORTED_LAYER_TYPES = frozenset({"linear_attention", "full_attention"})
FINAL_LAYER_COUNT = 60


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("text_config")
    return value if isinstance(value, Mapping) else config


@dataclass(frozen=True)
class SourceContract:
    model_type: str
    architecture: str
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    has_moe: bool
    mtp_num_hidden_layers: int
    rope_theta: float
    partial_rotary_factor: float
    extracted_text_backbone: bool


@dataclass(frozen=True)
class TargetContract:
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    model_type: str = "any2rwkv_qwen35_rwkv7"
    architecture: str = "Any2RWKV7ForCausalLM"

    @property
    def final(self) -> bool:
        return self.num_hidden_layers == FINAL_LAYER_COUNT and all(
            layer == "rwkv7" for layer in self.layer_types
        )


def validate_source_config(
    config: Mapping[str, Any],
    *,
    require_final_layers: bool = True,
    text_backbone_only: bool = False,
) -> SourceContract:
    model_type = str(config.get("model_type", ""))
    architectures = config.get("architectures", [])
    architecture = str(architectures[0]) if isinstance(architectures, list) and architectures else ""
    if model_type not in SUPPORTED_MODEL_TYPES or architecture not in SUPPORTED_ARCHITECTURES:
        raise ContractError(
            f"unsupported source architecture model_type={model_type!r} architecture={architecture!r}; "
            f"supported model_types={sorted(SUPPORTED_MODEL_TYPES)} architectures={sorted(SUPPORTED_ARCHITECTURES)}"
        )
    has_vision = "vision_config" in config or "vision_start_token_id" in config
    if has_vision and not text_backbone_only:
        raise ContractError(
            "multimodal Qwen3.5 input requires the explicit text-backbone-only contract; "
            "vision tensors must be recorded as intentionally-unmapped"
        )

    text = _text_config(config)
    layer_count = int(text.get("num_hidden_layers", 0))
    if require_final_layers and layer_count != FINAL_LAYER_COUNT:
        raise ContractError(f"expected {FINAL_LAYER_COUNT} decoder layers, found {layer_count}")
    layer_types = tuple(str(value) for value in text.get("layer_types", ()))
    if len(layer_types) != layer_count or set(layer_types) - SUPPORTED_LAYER_TYPES:
        raise ContractError(
            "layer_types must uniquely describe every decoder layer as linear_attention or full_attention"
        )
    heads = int(text.get("num_attention_heads", 0))
    kv_heads = int(text.get("num_key_value_heads", heads))
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ContractError(f"invalid attention head layout heads={heads} kv_heads={kv_heads}")
    rope = text.get("rope_parameters", text.get("rope_scaling", {}))
    rope = rope if isinstance(rope, Mapping) else {}
    return SourceContract(
        model_type=model_type,
        architecture=architecture,
        num_hidden_layers=layer_count,
        layer_types=layer_types,
        hidden_size=int(text.get("hidden_size", 0)),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        has_moe="moe" in model_type or int(text.get("num_experts", 0)) > 0,
        mtp_num_hidden_layers=int(text.get("mtp_num_hidden_layers", config.get("mtp_num_hidden_layers", 0))),
        rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10_000.0))),
        partial_rotary_factor=float(
            rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
        ),
        extracted_text_backbone=has_vision,
    )


def build_target_config(
    source_config: Mapping[str, Any],
    *,
    converted_layers: int | None = None,
    require_final_layers: bool = True,
) -> dict[str, Any]:
    source = validate_source_config(
        source_config,
        require_final_layers=require_final_layers,
        text_backbone_only=True,
    )
    converted = source.num_hidden_layers if converted_layers is None else int(converted_layers)
    if not 0 <= converted <= source.num_hidden_layers:
        raise ContractError(f"converted_layers must be in [0,{source.num_hidden_layers}], got {converted}")
    final = converted == source.num_hidden_layers == FINAL_LAYER_COUNT
    layout = ["rwkv7" if index < converted else source.layer_types[index] for index in range(source.num_hidden_layers)]
    source_text = _text_config(source_config)
    target = dict(source_text)
    fully_recurrent_proxy = converted == source.num_hidden_layers and not final
    target["model_type"] = "any2rwkv_qwen35_rwkv7" if final else ("any2rwkv_proxy" if fully_recurrent_proxy else "any2rwkv_hybrid")
    architecture = (
        "Any2RWKV7ForCausalLM"
        if final
        else (
            "Any2RWKVProxyForCausalLM"
            if fully_recurrent_proxy
            else "Any2RWKVHybridForCausalLM"
        )
    )
    target["architectures"] = [architecture]
    config_class = (
        "Any2RWKV7Config"
        if final
        else ("Any2RWKVProxyConfig" if fully_recurrent_proxy else "Any2RWKVHybridConfig")
    )
    target["auto_map"] = {
        "AutoConfig": f"configuration_any2rwkv.{config_class}",
        "AutoModelForCausalLM": f"modeling_any2rwkv.{architecture}",
    }
    target["num_hidden_layers"] = source.num_hidden_layers
    target["layer_types"] = layout
    target["head_dim"] = 64
    target["head_size"] = 64
    if source.hidden_size % 64:
        raise ContractError(f"native RWKV7 requires hidden_size divisible by 64, found {source.hidden_size}")
    target["num_heads"] = source.hidden_size // 64
    target["num_attention_heads"] = source.hidden_size // 64
    target["rope_theta"] = source.rope_theta
    target["partial_rotary_factor"] = source.partial_rotary_factor
    target["source_config_metadata"] = {
        "model_type": source_config.get("model_type"),
        "architectures": source_config.get("architectures"),
        "text_config_model_type": source_text.get("model_type"),
    }
    target["any2rwkv"] = {
        "source_model_type": source.model_type,
        "source_architecture": source.architecture,
        "source_was_multimodal": source.extracted_text_backbone,
        "source_layer_types": list(source.layer_types),
        "converted_layers": converted,
        "final_recurrent": final,
        "fully_recurrent_proxy": fully_recurrent_proxy,
        "preserved": ["moe", "mtp", "embedding", "norm", "rope", "lm_head", "tokenizer"],
        "rope_boundary": "source_projection_then_native_rwkv7_mixer",
        "recurrence": "native_rwkv7",
        "source_text_config": dict(source_text),
    }
    if final and not TargetContract(source.num_hidden_layers, tuple(layout)).final:
        raise ContractError("final export requires all 60 layers to be recurrent RWKV7")
    return target
