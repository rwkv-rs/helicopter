from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .errors import ContractError

TRANSFORMERS_RWKV7_ARCHITECTURE = "Rwkv7ForCausalLM"
TRANSFORMERS_RWKV7_MODEL_TYPE = "rwkv7"
TRANSFORMERS_RWKV7_ARTIFACT_CONTRACT = "transformers-rwkv7-v1"


def _community_classes():
    from .preflight import require_rwkv7_runtime

    require_rwkv7_runtime()
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
        from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7ForCausalLM
    except ImportError as error:
        raise ContractError(
            "Transformers RWKV-7 export requires the public "
            "Rwkv7Config/Rwkv7ForCausalLM interface"
        ) from error

    if (
        Rwkv7Config.model_type != TRANSFORMERS_RWKV7_MODEL_TYPE
        or Rwkv7ForCausalLM.__name__ != TRANSFORMERS_RWKV7_ARCHITECTURE
        or Rwkv7ForCausalLM.base_model_prefix != "model"
    ):
        raise ContractError(
            "installed Transformers RWKV-7 classes do not satisfy the expected "
            "model_type/class/base_model_prefix interface"
        )
    return AutoConfig, AutoModelForCausalLM, Rwkv7Config, Rwkv7ForCausalLM


def _runtime_config(config: Mapping[str, object]):
    if not isinstance(config, Mapping):
        raise ContractError("Transformers RWKV-7 config must be an object")
    payload = dict(config)
    if payload.pop("model_type", None) != TRANSFORMERS_RWKV7_MODEL_TYPE:
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
    payload.pop("auto_map", None)

    AutoConfig, _, Rwkv7Config, Rwkv7ForCausalLM = _community_classes()
    try:
        runtime_config = Rwkv7Config(**payload)
        runtime_config.validate_architecture()
    except Exception as error:
        raise ContractError(
            f"public Rwkv7Config rejected export config: {error}"
        ) from error

    serialized = runtime_config.to_diff_dict()
    auto_fields = dict(serialized)
    auto_fields.pop("model_type", None)
    try:
        auto_config = AutoConfig.for_model(
            TRANSFORMERS_RWKV7_MODEL_TYPE,
            **auto_fields,
        )
    except Exception as error:
        raise ContractError(
            f"Transformers AutoConfig rejected public RWKV-7 config: {error}"
        ) from error
    if type(auto_config) is not Rwkv7Config:
        raise ContractError(
            "installed Transformers AutoConfig registry does not resolve "
            "model_type='rwkv7' to Rwkv7Config"
        )
    return runtime_config, Rwkv7ForCausalLM


def _runtime_model(config: Mapping[str, object]):
    runtime_config, Rwkv7ForCausalLM = _runtime_config(config)
    _, AutoModelForCausalLM, _, _ = _community_classes()
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(runtime_config)
    except Exception as error:
        raise ContractError(
            f"Transformers AutoModelForCausalLM rejected public RWKV-7 config: {error}"
        ) from error
    if type(model) is not Rwkv7ForCausalLM:
        raise ContractError(
            "installed Transformers AutoModelForCausalLM registry does not resolve "
            "Rwkv7Config to Rwkv7ForCausalLM"
        )
    return runtime_config, model


def _serialized_config(runtime_config) -> dict[str, object]:
    payload = runtime_config.to_diff_dict()
    if payload.get("model_type") != TRANSFORMERS_RWKV7_MODEL_TYPE:
        raise ContractError("public Rwkv7Config serialized an unexpected model_type")
    if payload.get("architectures") != [TRANSFORMERS_RWKV7_ARCHITECTURE]:
        raise ContractError("public Rwkv7Config serialized an unexpected architecture")
    if payload.get("auto_map") is not None:
        raise ContractError("public Rwkv7Config must not serialize checkpoint-local code")
    return payload


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
    """Build the public configuration through ``Rwkv7Config`` itself."""
    _, _, Rwkv7Config, _ = _community_classes()
    try:
        runtime_config = Rwkv7Config(
            architectures=[TRANSFORMERS_RWKV7_ARCHITECTURE],
            vocab_size=vocab_size,
            context_length=context_length,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            head_size=head_size,
            dtype=dtype,
            layer_norm_epsilon=layer_norm_epsilon,
            group_norm_epsilon=group_norm_epsilon,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        runtime_config.validate_architecture()
    except Exception as error:
        raise ContractError(
            f"public Rwkv7Config rejected builder inputs: {error}"
        ) from error
    return validate_transformers_rwkv7_config(runtime_config.to_diff_dict())


def validate_transformers_rwkv7_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Normalize and validate metadata through the public configuration class."""
    runtime_config, _ = _runtime_config(config)
    return _serialized_config(runtime_config)


def materialize_transformers_rwkv7_model(
    config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, object], nn.Module, dict[str, tuple[int, ...]]]:
    """Strict-load real tensors into a meta-built public RWKV-7 model."""
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
    meta_tensors = sorted(
        name for name, tensor in state_dict.items() if tensor.device.type == "meta"
    )
    if meta_tensors:
        raise ContractError(
            "Transformers RWKV-7 export cannot serialize meta tensors: "
            f"{meta_tensors[:20]}"
        )

    runtime_config, model = _runtime_model(config)
    expected_state = model.state_dict()
    dtype_errors = [
        f"{name}: expected={expected_state[name].dtype} actual={tensor.dtype}"
        for name, tensor in state_dict.items()
        if name in expected_state and tensor.dtype != expected_state[name].dtype
    ]
    if dtype_errors:
        raise ContractError(
            "Transformers RWKV-7 state_dict dtypes differ from the public model: "
            + "; ".join(dtype_errors[:20])
        )
    try:
        model.load_state_dict(dict(state_dict), strict=True, assign=True)
    except RuntimeError as error:
        raise ContractError(
            f"public Rwkv7ForCausalLM strict state_dict load failed: {error}"
        ) from error
    remaining_meta = [
        name for name, tensor in model.state_dict().items() if tensor.device.type == "meta"
    ]
    if remaining_meta:
        raise ContractError(
            "public Rwkv7ForCausalLM strict load left meta tensors: "
            f"{remaining_meta[:20]}"
        )
    shapes = {
        name: tuple(tensor.shape)
        for name, tensor in expected_state.items()
    }
    return _serialized_config(runtime_config), model, shapes


__all__ = [
    "TRANSFORMERS_RWKV7_ARCHITECTURE",
    "TRANSFORMERS_RWKV7_ARTIFACT_CONTRACT",
    "TRANSFORMERS_RWKV7_MODEL_TYPE",
    "build_transformers_rwkv7_config",
    "materialize_transformers_rwkv7_model",
    "validate_transformers_rwkv7_config",
]
