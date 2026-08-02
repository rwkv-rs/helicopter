from __future__ import annotations

from ..adapters.qwen35 import Qwen35SourceAdapter
from ..adapters.rwkv7 import NativeRWKV7TargetAdapter
from ..core import AdapterRecipeRegistry, ResolvedRecipe
from .qwen35_to_rwkv7 import Qwen35ToRWKV7Recipe


def default_registry() -> AdapterRecipeRegistry:
    registry = AdapterRecipeRegistry()
    registry.register_source(Qwen35SourceAdapter())
    registry.register_target(NativeRWKV7TargetAdapter())
    registry.register_recipe(Qwen35ToRWKV7Recipe())
    return registry


def resolve_recipe(
    recipe_id: str,
    *,
    source_adapter_id: str | None = None,
    target_adapter_id: str | None = None,
) -> ResolvedRecipe:
    return default_registry().resolve(
        recipe_id,
        source_adapter_id=source_adapter_id,
        target_adapter_id=target_adapter_id,
    )


__all__ = ["default_registry", "resolve_recipe"]
