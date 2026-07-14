from __future__ import annotations

from transformers import PretrainedConfig


class Any2RWKVConfigBase(PretrainedConfig):
    """Shared fields for final, proxy, and hybrid Any2RWKV checkpoints."""

    model_type = "any2rwkv_base"

    def __init__(self, **kwargs):
        # ``PretrainedConfig`` is wrapped by huggingface_hub's strict
        # dataclass validator.  Its validator runs before this subclass can
        # install custom fields and rejects the RWKV-specific ``rwkv7`` layer
        # marker.  Keep the marker out of the parent kwargs and restore it
        # immediately after the parent initialization.
        requested_layers = kwargs.pop("layer_types", None)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = int(kwargs.get("vocab_size", 248320))
        self.hidden_size = int(kwargs.get("hidden_size", 4096))
        self.intermediate_size = int(kwargs.get("intermediate_size", self.hidden_size * 4))
        self.num_hidden_layers = int(kwargs.get("num_hidden_layers", 60))
        self.head_dim = int(kwargs.get("head_dim", 64))
        self.head_size = int(kwargs.get("head_size", self.head_dim))
        self.num_heads = int(kwargs.get("num_heads", self.hidden_size // self.head_dim))
        self.num_attention_heads = self.num_heads
        self.layer_types = list(requested_layers or ["rwkv7"] * self.num_hidden_layers)
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
        self.any2rwkv = dict(kwargs.get("any2rwkv", {}))
        self.rope_parameters = dict(kwargs.get("rope_parameters", {}))
        self.mtp_num_hidden_layers = int(kwargs.get("mtp_num_hidden_layers", 0))
        self.mtp_use_dedicated_embeddings = bool(kwargs.get("mtp_use_dedicated_embeddings", False))
        if getattr(self, "auto_map", None) is None:
            self.auto_map = {
                "AutoConfig": f"configuration_any2rwkv.{type(self).__name__}",
                "AutoModelForCausalLM": "modeling_any2rwkv.Any2RWKV7ForCausalLM",
            }


class Any2RWKV7Config(Any2RWKVConfigBase):
    """Final 60-layer, fully recurrent Qwen3.5 text-backbone identity."""

    # FLA registers model_type="rwkv7" locally, so the final artifact needs a
    # unique identity to force Transformers through this checkpoint's code.
    model_type = "any2rwkv_qwen35_rwkv7"

    def __init__(self, **kwargs):
        # Transformers 5's strict config machinery synthesizes an initializer
        # for a subclass that does not define one.  That synthesized method
        # skips fields owned by our base class, so keep this explicit delegate.
        kwargs.setdefault("architectures", ["Any2RWKV7ForCausalLM"])
        super().__init__(**kwargs)


class Any2RWKVProxyConfig(Any2RWKVConfigBase):
    """Fully recurrent but non-60-layer experimental proxy identity."""

    model_type = "any2rwkv_proxy"

    def __init__(self, **kwargs):
        kwargs.setdefault("architectures", ["Any2RWKVProxyForCausalLM"])
        super().__init__(**kwargs)
        self.auto_map["AutoModelForCausalLM"] = (
            "modeling_any2rwkv.Any2RWKVProxyForCausalLM"
        )


class Any2RWKVHybridConfig(Any2RWKVConfigBase):
    """Partially replaced teacher/student checkpoint identity."""

    model_type = "any2rwkv_hybrid"

    def __init__(self, **kwargs):
        kwargs.setdefault("architectures", ["Any2RWKVHybridForCausalLM"])
        super().__init__(**kwargs)
        self.auto_map["AutoModelForCausalLM"] = (
            "modeling_any2rwkv.Any2RWKVHybridForCausalLM"
        )
