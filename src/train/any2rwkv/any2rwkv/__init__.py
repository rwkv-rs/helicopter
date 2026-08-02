"""Any-to-RWKV architecture conversion and independent model family."""

from .contract import (
    SourceContract,
    TargetContract,
    build_target_config,
    validate_source_config,
)
from .mapping import MappingLedger, SourceDisposition, TargetProvenance


def register_any_to_rwkv_auto_classes() -> None:
    """Register the installed Any-to-RWKV family without checkpoint auto_map."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .configuration_any2rwkv import (
        AnyToRWKVConfig,
        AnyToRWKVHybridConfig,
        AnyToRWKVProxyConfig,
    )
    from .modeling_any2rwkv import (
        AnyToRWKVForCausalLM,
        AnyToRWKVHybridForCausalLM,
        AnyToRWKVProxyForCausalLM,
    )

    registrations = (
        (AnyToRWKVConfig, AnyToRWKVForCausalLM),
        (AnyToRWKVProxyConfig, AnyToRWKVProxyForCausalLM),
        (AnyToRWKVHybridConfig, AnyToRWKVHybridForCausalLM),
    )
    for config_class, model_class in registrations:
        AutoConfig.register(
            config_class.model_type,
            config_class,
            exist_ok=True,
        )
        AutoModelForCausalLM.register(
            config_class,
            model_class,
            exist_ok=True,
        )


__all__ = [
    "MappingLedger",
    "SourceContract",
    "SourceDisposition",
    "TargetContract",
    "TargetProvenance",
    "build_target_config",
    "register_any_to_rwkv_auto_classes",
    "validate_source_config",
]

__version__ = "0.1.0"
