from __future__ import annotations

from collections.abc import Mapping

import torch

from .errors import ContractError

TRANSFORMERS_RWKV7_ARCHITECTURE = "Rwkv7ForCausalLM"
TRANSFORMERS_RWKV7_MODEL_TYPE = "rwkv7"
TRANSFORMERS_RWKV7_ARTIFACT_CONTRACT = "transformers-rwkv7-v1"

_SUPPORTED_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_KEY_SUMMARY_PRIORITY = (
    "model.embeddings.weight",
    "model.blocks.0.att.r_k",
    "model.blocks.0.ffn.key.weight",
    "model.ln_out.weight",
    "model.ln_out.bias",
    "head.weight",
)

_CONFIG_FIELDS = {
    "architectures",
    "bos_token_id",
    "context_length",
    "dtype",
    "embedding_layer_norm_fused",
    "eos_token_id",
    "group_norm_epsilon",
    "head_size",
    "hidden_size",
    "intermediate_size",
    "layer_norm_epsilon",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "rescale_every",
    "tie_word_embeddings",
    "use_cache",
    "vocab_size",
    "wkv_backend",
    "wkv_state_dtype",
}


def build_transformers_rwkv7_config(
    *,
    vocab_size: int,
    context_length: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    head_size: int,
    dtype: str,
    layer_norm_epsilon: float = 1e-5,
    group_norm_epsilon: float = 64e-5,
    bos_token_id: int | None = 0,
    eos_token_id: int | list[int] | None = 0,
) -> dict[str, object]:
    """Build metadata for the public Transformers RWKV-7 model contract.

    This function projects explicit pure-RWKV geometry only. It does not
    reinterpret the private Any2RWKV Qwen-shell target or claim that its
    preserved Qwen FFN/MoE tensors satisfy the community model.
    """
    config = {
        "model_type": TRANSFORMERS_RWKV7_MODEL_TYPE,
        "architectures": [TRANSFORMERS_RWKV7_ARCHITECTURE],
        "vocab_size": vocab_size,
        "context_length": context_length,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "head_size": head_size,
        "num_attention_heads": hidden_size // head_size if head_size else 0,
        "layer_norm_epsilon": layer_norm_epsilon,
        "group_norm_epsilon": group_norm_epsilon,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "rescale_every": 0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "embedding_layer_norm_fused": False,
        "wkv_backend": "auto",
        "wkv_state_dtype": "float32",
        "dtype": dtype,
    }
    return validate_transformers_rwkv7_config(config)


def validate_transformers_rwkv7_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate metadata without accepting private loader or hybrid aliases."""
    if not isinstance(config, Mapping):
        raise ContractError("Transformers RWKV-7 config must be an object")
    payload = dict(config)
    if payload.get("model_type") != TRANSFORMERS_RWKV7_MODEL_TYPE:
        raise ContractError("Transformers RWKV-7 export requires model_type='rwkv7'")
    if payload.get("architectures") != [TRANSFORMERS_RWKV7_ARCHITECTURE]:
        raise ContractError(
            "Transformers RWKV-7 export requires architectures=['Rwkv7ForCausalLM']"
        )
    if payload.get("auto_map") is not None:
        raise ContractError(
            "Transformers RWKV-7 export must use the registered community model, "
            "not checkpoint-local auto_map code"
        )
    unexpected_fields = sorted(payload.keys() - _CONFIG_FIELDS)
    if unexpected_fields:
        raise ContractError(
            "Transformers RWKV-7 config contains unsupported fields: "
            f"{unexpected_fields}"
        )

    positive_integer_fields = (
        "vocab_size",
        "context_length",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "head_size",
        "num_attention_heads",
    )
    for name in positive_integer_fields:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(
                f"Transformers RWKV-7 config field {name!r} must be a positive integer"
            )
    hidden_size = int(payload["hidden_size"])
    head_size = int(payload["head_size"])
    num_attention_heads = int(payload["num_attention_heads"])
    if hidden_size % head_size or num_attention_heads != hidden_size // head_size:
        raise ContractError(
            "Transformers RWKV-7 requires num_attention_heads == "
            "hidden_size // head_size"
        )

    for name in ("layer_norm_epsilon", "group_norm_epsilon"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ContractError(
                f"Transformers RWKV-7 config field {name!r} must be positive"
            )
    if payload.get("wkv_state_dtype") != "float32":
        raise ContractError("Transformers RWKV-7 requires wkv_state_dtype='float32'")
    dtype = payload.get("dtype")
    if dtype not in _SUPPORTED_DTYPES:
        raise ContractError(
            f"Transformers RWKV-7 dtype must be one of {sorted(_SUPPORTED_DTYPES)}, "
            f"got {dtype!r}"
        )
    if payload.get("tie_word_embeddings") is not False:
        raise ContractError(
            "Transformers RWKV-7 artifact seam requires an explicit untied head.weight"
        )
    rescale_every = payload.get("rescale_every")
    if (
        isinstance(rescale_every, bool)
        or not isinstance(rescale_every, int)
        or rescale_every < 0
    ):
        raise ContractError(
            "Transformers RWKV-7 rescale_every must be a non-negative integer"
        )
    if not isinstance(payload.get("use_cache"), bool):
        raise ContractError("Transformers RWKV-7 use_cache must be boolean")
    if not isinstance(payload.get("embedding_layer_norm_fused"), bool):
        raise ContractError(
            "Transformers RWKV-7 embedding_layer_norm_fused must be boolean"
        )
    if not isinstance(payload.get("wkv_backend"), str) or not payload["wkv_backend"]:
        raise ContractError("Transformers RWKV-7 wkv_backend must be non-empty")
    return payload


def transformers_rwkv7_state_shapes(
    config: Mapping[str, object],
) -> dict[str, tuple[int, ...]]:
    """Resolve the expected public state-dict directly from the community model."""
    payload = validate_transformers_rwkv7_config(config)
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
        from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7ForCausalLM
    except ImportError as error:
        raise ContractError(
            "Transformers RWKV-7 artifact export requires a Transformers build "
            "that registers Rwkv7Config and Rwkv7ForCausalLM"
        ) from error

    if (
        Rwkv7Config.model_type != TRANSFORMERS_RWKV7_MODEL_TYPE
        or Rwkv7ForCausalLM.__name__ != TRANSFORMERS_RWKV7_ARCHITECTURE
        or Rwkv7ForCausalLM.base_model_prefix != "model"
    ):
        raise ContractError(
            "installed Transformers RWKV-7 classes do not satisfy the expected "
            "model_type/class/base_model_prefix contract"
        )
    try:
        auto_config_fields = dict(payload)
        del auto_config_fields["model_type"]
        runtime_config = AutoConfig.for_model(
            TRANSFORMERS_RWKV7_MODEL_TYPE,
            **auto_config_fields,
        )
        if type(runtime_config) is not Rwkv7Config:
            raise ContractError(
                "installed Transformers AutoConfig registry does not resolve "
                "model_type='rwkv7' to Rwkv7Config"
            )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(runtime_config)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ContractError(
            f"Transformers RWKV-7 rejected export config: {error}"
        ) from error
    if type(model) is not Rwkv7ForCausalLM:
        raise ContractError(
            "installed Transformers AutoModelForCausalLM registry does not resolve "
            "Rwkv7Config to Rwkv7ForCausalLM"
        )
    shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    required_layout = {
        "model.embeddings.weight",
        "model.blocks.0.att.r_k",
        "model.blocks.0.ffn.key.weight",
        "model.ln_out.weight",
        "model.ln_out.bias",
        "head.weight",
    }
    if not required_layout <= shapes.keys():
        raise ContractError(
            "installed Transformers RWKV-7 state-dict layout drifted from "
            "model.embeddings/model.blocks/model.ln_out/head"
        )
    return shapes


def validate_transformers_rwkv7_state_dict(
    config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, tuple[int, ...]]:
    """Fail closed unless every public RWKV-7 key, shape, and dtype is exact."""
    if not isinstance(state_dict, Mapping):
        raise ContractError("Transformers RWKV-7 state_dict must be an object")
    invalid_entries = [
        name
        for name, tensor in state_dict.items()
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
    ]
    if invalid_entries:
        raise ContractError(
            "Transformers RWKV-7 state_dict must directly map string names to "
            f"tensors; invalid={invalid_entries[:20]}"
        )

    expected = transformers_rwkv7_state_shapes(config)
    actual_names = set(state_dict)
    missing = sorted(expected.keys() - actual_names)
    unexpected = sorted(actual_names - expected.keys())
    if missing or unexpected:
        raise ContractError(
            "Transformers RWKV-7 state_dict keys differ from the community model: "
            f"missing={_summarize_names(missing)} "
            f"unexpected={_summarize_names(unexpected)}"
        )

    shape_errors = [
        f"{name}: expected={expected[name]} actual={tuple(state_dict[name].shape)}"
        for name in sorted(expected)
        if tuple(state_dict[name].shape) != expected[name]
    ]
    if shape_errors:
        raise ContractError(
            "Transformers RWKV-7 state_dict shapes differ from the community model: "
            + "; ".join(shape_errors[:20])
        )

    payload = validate_transformers_rwkv7_config(config)
    expected_dtype = _SUPPORTED_DTYPES[str(payload["dtype"])]
    dtype_errors = [
        f"{name}: expected={expected_dtype} actual={state_dict[name].dtype}"
        for name in sorted(expected)
        if state_dict[name].dtype != expected_dtype
    ]
    if dtype_errors:
        raise ContractError(
            "Transformers RWKV-7 state_dict dtypes differ from config: "
            + "; ".join(dtype_errors[:20])
        )
    meta_tensors = sorted(
        name for name, tensor in state_dict.items() if tensor.device.type == "meta"
    )
    if meta_tensors:
        raise ContractError(
            "Transformers RWKV-7 export cannot serialize meta tensors: "
            f"{_summarize_names(meta_tensors)}"
        )
    return expected


def _summarize_names(names: list[str]) -> str:
    prioritized = [name for name in _KEY_SUMMARY_PRIORITY if name in names]
    visible = (
        prioritized
        + [name for name in names if name not in prioritized][: 20 - len(prioritized)]
    )
    suffix = (
        "" if len(names) <= len(visible) else f" (+{len(names) - len(visible)} more)"
    )
    return f"{visible}{suffix}"


__all__ = [
    "TRANSFORMERS_RWKV7_ARCHITECTURE",
    "TRANSFORMERS_RWKV7_ARTIFACT_CONTRACT",
    "TRANSFORMERS_RWKV7_MODEL_TYPE",
    "build_transformers_rwkv7_config",
    "transformers_rwkv7_state_shapes",
    "validate_transformers_rwkv7_config",
    "validate_transformers_rwkv7_state_dict",
]
