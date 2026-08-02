from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
PRIVATE_ANY2RWKV_ARTIFACT_CONTRACT = "private-any2rwkv-qwen-shell-v1"


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


@dataclass(frozen=True)
class RecurrentHeadGeometry:
    num_heads: int
    head_size: int
    recurrent_width: int
    source: str
    gdn_state_geometry_preserved: bool


def derive_recurrent_head_geometry(
    source_config: Mapping[str, Any],
) -> RecurrentHeadGeometry:
    """Preserve the source GDN state partition whenever one is present."""
    text = _text_config(source_config)
    hidden_size = int(text.get("hidden_size", 0))
    layer_types = tuple(str(value) for value in text.get("layer_types", ()))
    if hidden_size <= 0:
        raise ContractError("source hidden_size must be positive")
    if "linear_attention" in layer_types:
        key_heads = int(text.get("linear_num_key_heads", 0))
        value_heads = int(text.get("linear_num_value_heads", 0))
        key_head_size = int(text.get("linear_key_head_dim", 0))
        value_head_size = int(text.get("linear_value_head_dim", 0))
        if (
            key_heads <= 0
            or value_heads <= 0
            or key_head_size <= 0
            or value_head_size <= 0
        ):
            raise ContractError(
                "linear-attention source must declare positive key/value head geometry"
            )
        if key_head_size != value_head_size or value_heads % key_heads:
            raise ContractError(
                "native RWKV7 exact GDN migration requires equal key/value head "
                "size and an integral key-to-value head repeat"
            )
        recurrent_width = value_heads * value_head_size
        attention_heads = int(text.get("num_attention_heads", 0))
        attention_head_size = int(text.get("head_dim", 0))
        if (
            attention_heads <= 0
            or attention_head_size <= 0
            or attention_heads * attention_head_size != recurrent_width
        ):
            raise ContractError(
                "source GDN width (value width) and full-attention query width must match "
                "so every mixer preserves one recurrent width"
            )
        return RecurrentHeadGeometry(
            num_heads=value_heads,
            head_size=value_head_size,
            recurrent_width=recurrent_width,
            source="linear_attention_value_state",
            gdn_state_geometry_preserved=True,
        )

    attention_heads = int(text.get("num_attention_heads", 0))
    attention_head_size = int(text.get("head_dim", 0))
    if (
        attention_heads <= 0
        or attention_head_size <= 0
    ):
        raise ContractError(
            "attention-only source must declare a positive query-head geometry"
        )
    return RecurrentHeadGeometry(
        num_heads=attention_heads,
        head_size=attention_head_size,
        recurrent_width=attention_heads * attention_head_size,
        source="full_attention_query_layout",
        gdn_state_geometry_preserved=False,
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
    recurrent_geometry = derive_recurrent_head_geometry(source_config)
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
    target["head_dim"] = recurrent_geometry.head_size
    target["head_size"] = recurrent_geometry.head_size
    target["num_heads"] = recurrent_geometry.num_heads
    target["num_attention_heads"] = recurrent_geometry.num_heads
    target["attention_hidden_size"] = recurrent_geometry.recurrent_width
    target["rope_theta"] = source.rope_theta
    target["partial_rotary_factor"] = source.partial_rotary_factor
    raw_rope_parameters = source_text.get("rope_parameters", {})
    if isinstance(raw_rope_parameters, Mapping):
        target["rope_parameters"] = {
            key: value
            for key, value in raw_rope_parameters.items()
            if key not in {"mrope_section", "mrope_interleaved"}
        }
    target["source_config_metadata"] = {
        "model_type": source_config.get("model_type"),
        "architectures": source_config.get("architectures"),
        "text_config_model_type": source_text.get("model_type"),
    }
    target["any2rwkv"] = {
        "artifact_contract": PRIVATE_ANY2RWKV_ARTIFACT_CONTRACT,
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
        "recurrent_head_geometry": {
            "num_heads": recurrent_geometry.num_heads,
            "head_size": recurrent_geometry.head_size,
            "recurrent_width": recurrent_geometry.recurrent_width,
            "source": recurrent_geometry.source,
            "gdn_state_geometry_preserved": (
                recurrent_geometry.gdn_state_geometry_preserved
            ),
        },
        "source_text_config": dict(source_text),
        "ignored_multimodal_rope_fields": [
            key
            for key in ("mrope_section", "mrope_interleaved")
            if isinstance(raw_rope_parameters, Mapping)
            and key in raw_rope_parameters
        ],
    }
    if final and not TargetContract(source.num_hidden_layers, tuple(layout)).final:
        raise ContractError("final export requires all 60 layers to be recurrent RWKV7")
    return target
