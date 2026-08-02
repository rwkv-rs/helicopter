from __future__ import annotations

from dataclasses import dataclass

from ..errors import ContractError
from .contracts import (
    DistillationRecipe,
    SourceArchitectureAdapter,
    TargetArchitectureAdapter,
)


@dataclass(frozen=True)
class ResolvedRecipe:
    recipe: DistillationRecipe
    source: SourceArchitectureAdapter
    target: TargetArchitectureAdapter


class AdapterRecipeRegistry:
    """Resolve explicit source/target adapters without guessing architectures."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceArchitectureAdapter] = {}
        self._targets: dict[str, TargetArchitectureAdapter] = {}
        self._recipes: dict[str, DistillationRecipe] = {}

    def register_source(self, adapter: SourceArchitectureAdapter) -> None:
        self._require_methods(adapter, ("inspect_checkpoint", "load_checkpoint"), "source adapter")
        self._register(self._sources, adapter.adapter_id, adapter, "source adapter")

    def register_target(self, adapter: TargetArchitectureAdapter) -> None:
        self._require_methods(adapter, ("build_target_config", "validate_training_environment"), "target adapter")
        self._register(self._targets, adapter.adapter_id, adapter, "target adapter")

    def register_recipe(self, recipe: DistillationRecipe) -> None:
        self._require_methods(recipe, ("validate_source", "run_layerwise_distillation"), "recipe")
        self._register(self._recipes, recipe.recipe_id, recipe, "recipe")

    def resolve(
        self,
        recipe_id: str,
        *,
        source_adapter_id: str | None = None,
        target_adapter_id: str | None = None,
    ) -> ResolvedRecipe:
        try:
            recipe = self._recipes[recipe_id]
        except KeyError as error:
            raise ContractError(
                f"unknown distillation recipe {recipe_id!r}; "
                f"registered={sorted(self._recipes)}"
            ) from error
        self._require_compatible_adapter_id(
            self._sources,
            requested_id=source_adapter_id,
            required_id=recipe.source_adapter_id,
            recipe_id=recipe_id,
            kind="source",
        )
        self._require_compatible_adapter_id(
            self._targets,
            requested_id=target_adapter_id,
            required_id=recipe.target_adapter_id,
            recipe_id=recipe_id,
            kind="target",
        )
        try:
            source = self._sources[recipe.source_adapter_id]
        except KeyError as error:
            raise ContractError(
                f"recipe {recipe_id!r} requires missing source adapter "
                f"{recipe.source_adapter_id!r}; registered={sorted(self._sources)}"
            ) from error
        try:
            target = self._targets[recipe.target_adapter_id]
        except KeyError as error:
            raise ContractError(
                f"recipe {recipe_id!r} requires missing target adapter "
                f"{recipe.target_adapter_id!r}; registered={sorted(self._targets)}"
            ) from error
        return ResolvedRecipe(recipe, source, target)

    @staticmethod
    def _require_compatible_adapter_id(
        registry: dict[str, object],
        *,
        requested_id: str | None,
        required_id: str,
        recipe_id: str,
        kind: str,
    ) -> None:
        if requested_id is None:
            return
        if requested_id not in registry:
            raise ContractError(
                f"unknown {kind} adapter {requested_id!r}; registered={sorted(registry)}"
            )
        if requested_id != required_id:
            raise ContractError(
                f"recipe {recipe_id!r} is incompatible with {kind} adapter "
                f"{requested_id!r}; requires {required_id!r}"
            )

    @staticmethod
    def _register(registry: dict[str, object], identifier: str, value: object, kind: str) -> None:
        if not identifier or identifier.strip() != identifier:
            raise ContractError(f"{kind} id must be a non-empty canonical string")
        if identifier in registry:
            raise ContractError(f"duplicate {kind} id: {identifier}")
        registry[identifier] = value

    @staticmethod
    def _require_methods(value: object, names: tuple[str, ...], kind: str) -> None:
        missing = [name for name in names if not callable(getattr(value, name, None))]
        if missing:
            raise ContractError(f"{kind} does not implement required methods: {missing}")
