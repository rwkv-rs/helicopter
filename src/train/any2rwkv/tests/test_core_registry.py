from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from any2rwkv.core import AdapterRecipeRegistry, ArchitectureInspection
from any2rwkv.errors import ContractError


@dataclass(frozen=True)
class SyntheticSource:
    adapter_id: str = "synthetic_source"

    def inspect_checkpoint(self, checkpoint_dir: Path, *, require_final_layout: bool):
        return ArchitectureInspection(self.adapter_id, 3, 8, {"kind": "synthetic"})

    def load_checkpoint(self, checkpoint_dir: Path, *, require_final_layout: bool):
        return {"path": checkpoint_dir, "final": require_final_layout}


@dataclass(frozen=True)
class SyntheticTarget:
    adapter_id: str = "synthetic_target"

    def build_target_config(self, source_checkpoint, *, require_final_layout: bool):
        return {"source": source_checkpoint, "final": require_final_layout}

    def validate_training_environment(self, *, head_size: int):
        return {}


@dataclass(frozen=True)
class SyntheticRecipe:
    recipe_id: str = "synthetic_to_synthetic"
    source_adapter_id: str = "synthetic_source"
    target_adapter_id: str = "synthetic_target"

    def validate_source(self, inspection: ArchitectureInspection) -> None:
        assert inspection.adapter_id == self.source_adapter_id

    def run_layerwise_distillation(self, request):
        return {"status": "synthetic", "request": request}


def test_core_registry_resolves_synthetic_recipe_without_model_dependencies() -> None:
    registry = AdapterRecipeRegistry()
    registry.register_source(SyntheticSource())
    registry.register_target(SyntheticTarget())
    registry.register_recipe(SyntheticRecipe())
    resolved = registry.resolve("synthetic_to_synthetic")
    inspection = resolved.source.inspect_checkpoint(Path("unused"), require_final_layout=False)
    resolved.recipe.validate_source(inspection)
    assert inspection.num_layers == 3
    assert resolved.target.adapter_id == "synthetic_target"


def test_core_registry_rejects_unknown_and_duplicate_ids_precisely() -> None:
    registry = AdapterRecipeRegistry()
    registry.register_source(SyntheticSource())
    with pytest.raises(ContractError, match="duplicate source adapter"):
        registry.register_source(SyntheticSource())
    with pytest.raises(ContractError, match="unknown distillation recipe"):
        registry.resolve("missing")


def test_recipe_resolution_fails_if_declared_target_is_not_registered() -> None:
    registry = AdapterRecipeRegistry()
    registry.register_source(SyntheticSource())
    registry.register_recipe(SyntheticRecipe())
    with pytest.raises(ContractError, match="missing target adapter"):
        registry.resolve("synthetic_to_synthetic")


def test_registry_rejects_recipe_that_only_declares_ids() -> None:
    @dataclass(frozen=True)
    class IncompleteRecipe:
        recipe_id: str = "incomplete"
        source_adapter_id: str = "synthetic_source"
        target_adapter_id: str = "synthetic_target"

    registry = AdapterRecipeRegistry()
    with pytest.raises(ContractError, match="run_layerwise_distillation"):
        registry.register_recipe(IncompleteRecipe())
