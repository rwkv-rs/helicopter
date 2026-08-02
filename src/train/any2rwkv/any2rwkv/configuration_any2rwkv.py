from __future__ import annotations

from transformers import PretrainedConfig

ANY_TO_RWKV_MODEL_TYPE = "any_to_rwkv"
ANY_TO_RWKV_ARCHITECTURE = "AnyToRWKVForCausalLM"


class AnyToRWKVConfigBase(PretrainedConfig):
    """Shared fields for independent Any-to-RWKV model artifacts."""

    model_type = "any_to_rwkv_base"

    def __init__(self, **kwargs):
        if kwargs.get("auto_map") is not None:
            raise ValueError(
                "Any-to-RWKV artifacts use a registered model family; auto_map is forbidden"
            )
        if "any2rwkv" in kwargs:
            raise ValueError(
                "legacy any2rwkv metadata is not an Any-to-RWKV model identity"
            )
        requested_mixers = kwargs.pop("mixer_types", None)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = int(kwargs.get("vocab_size", 248320))
        self.hidden_size = int(kwargs.get("hidden_size", 4096))
        self.intermediate_size = int(
            kwargs.get("intermediate_size", self.hidden_size * 4)
        )
        self.num_hidden_layers = int(kwargs.get("num_hidden_layers", 60))
        self.head_dim = int(kwargs.get("head_dim", 64))
        self.head_size = int(kwargs.get("head_size", self.head_dim))
        requested_attention_width = int(
            kwargs.get("attention_hidden_size", self.hidden_size)
        )
        if requested_attention_width <= 0 or requested_attention_width % self.head_dim:
            raise ValueError(
                "attention_hidden_size must be positive and divisible by head_dim"
            )
        self.num_heads = int(
            kwargs.get("num_heads", requested_attention_width // self.head_dim)
        )
        self.attention_hidden_size = requested_attention_width
        if self.num_heads * self.head_dim != self.attention_hidden_size:
            raise ValueError("attention_hidden_size must equal num_heads * head_dim")
        self.num_attention_heads = self.num_heads
        self.mixer_types = list(requested_mixers or ["rwkv7"] * self.num_hidden_layers)
        if len(self.mixer_types) != self.num_hidden_layers:
            raise ValueError("mixer_types must contain exactly one entry per layer")
        unsupported_mixers = sorted(
            set(self.mixer_types) - {"rwkv7", "linear_attention", "full_attention"}
        )
        if unsupported_mixers:
            raise ValueError(f"unsupported mixer_types: {unsupported_mixers}")
        self.decay_low_rank_dim = int(kwargs.get("decay_low_rank_dim", 64))
        self.gate_low_rank_dim = int(kwargs.get("gate_low_rank_dim", 128))
        self.a_low_rank_dim = int(kwargs.get("a_low_rank_dim", 64))
        self.v_low_rank_dim = int(kwargs.get("v_low_rank_dim", 32))
        self.use_native_mm8 = bool(kwargs.get("use_native_mm8", False))
        self.native_mm8_min_params = int(kwargs.get("native_mm8_min_params", 8_000_000))
        self.native_mm8_policy = str(kwargs.get("native_mm8_policy", "memory"))
        self.use_native_mm4 = bool(kwargs.get("use_native_mm4", False))
        self.native_mm4_min_params = int(kwargs.get("native_mm4_min_params", 8_000_000))
        self.native_mm4_policy = str(kwargs.get("native_mm4_policy", "memory"))
        self.rms_norm_eps = float(kwargs.get("rms_norm_eps", 1e-6))
        self.use_cache = bool(kwargs.get("use_cache", True))
        self.any_to_rwkv = dict(kwargs.get("any_to_rwkv", {}))
        self.rope_parameters = dict(kwargs.get("rope_parameters", {}))
        self.mtp_num_hidden_layers = int(kwargs.get("mtp_num_hidden_layers", 0))
        self.mtp_use_dedicated_embeddings = bool(
            kwargs.get("mtp_use_dedicated_embeddings", False)
        )
        self.hidden_act = str(kwargs.get("hidden_act", "silu"))
        self.num_experts = int(kwargs.get("num_experts", 0))
        self.num_experts_per_tok = int(kwargs.get("num_experts_per_tok", 0))
        self.moe_intermediate_size = int(
            kwargs.get("moe_intermediate_size", self.intermediate_size)
        )
        self.shared_expert_intermediate_size = int(
            kwargs.get("shared_expert_intermediate_size", self.intermediate_size)
        )


class AnyToRWKVConfig(AnyToRWKVConfigBase):
    """Independent, fully recurrent Any-to-RWKV model family."""

    model_type = ANY_TO_RWKV_MODEL_TYPE

    def __init__(self, **kwargs):
        kwargs.setdefault("architectures", [ANY_TO_RWKV_ARCHITECTURE])
        super().__init__(**kwargs)


class AnyToRWKVProxyConfig(AnyToRWKVConfigBase):
    """Independent fully recurrent pilot identity."""

    model_type = "any_to_rwkv_proxy"

    def __init__(self, **kwargs):
        kwargs.setdefault("architectures", ["AnyToRWKVProxyForCausalLM"])
        super().__init__(**kwargs)


class AnyToRWKVHybridConfig(AnyToRWKVConfigBase):
    """Progressive conversion checkpoint identity, never a final model."""

    model_type = "any_to_rwkv_hybrid"

    def __init__(self, **kwargs):
        kwargs.setdefault("architectures", ["AnyToRWKVHybridForCausalLM"])
        super().__init__(**kwargs)


__all__ = [
    "ANY_TO_RWKV_ARCHITECTURE",
    "ANY_TO_RWKV_MODEL_TYPE",
    "AnyToRWKVConfig",
    "AnyToRWKVConfigBase",
    "AnyToRWKVHybridConfig",
    "AnyToRWKVProxyConfig",
]
