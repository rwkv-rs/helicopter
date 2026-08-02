from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ArchitectureInspection:
    adapter_id: str
    num_layers: int
    hidden_size: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class DistillationExecutionRequest:
    source_checkpoint: object
    run_dir: Path
    zero_step_dir: Path
    token_rows: tuple[tuple[int, ...], ...]
    validation_rows: tuple[tuple[int, ...], ...]
    train_row_source_sample_ids: tuple[tuple[str, ...], ...]
    validation_row_source_sample_ids: tuple[tuple[str, ...], ...]
    plan: object
    training_config: Path
    dataset_manifest: Path
    resume: Path | None
    stop_after_optimizer_steps: int | None = None
    progress_callback: Callable[[str, Path], None] | None = None


@dataclass(frozen=True)
class PerformanceProfileCacheRequest:
    source_checkpoint: object
    run_dir: Path
    zero_step_dir: Path
    token_rows: tuple[tuple[int, ...], ...]
    validation_rows: tuple[tuple[int, ...], ...]
    plan: object
    training_config: Path
    dataset_manifest: Path
    row_selection: dict[str, object] | None = None


@dataclass(frozen=True)
class GQAZeroStepValidationRequest:
    source_checkpoint: object
    run_dir: Path
    evidence_dir: Path
    zero_step_dir: Path
    plan: object
    training_config: Path
    dataset_manifest: Path
    layer_index: int
    train_row_source_sample_ids: tuple[tuple[str, ...], ...]
    validation_row_source_sample_ids: tuple[tuple[str, ...], ...]


@runtime_checkable
class SourceArchitectureAdapter(Protocol):
    adapter_id: str

    def inspect_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        require_final_layout: bool,
    ) -> ArchitectureInspection: ...

    def load_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        require_final_layout: bool,
    ) -> object: ...


@runtime_checkable
class TargetArchitectureAdapter(Protocol):
    adapter_id: str

    def build_target_config(
        self,
        source_checkpoint: object,
        *,
        require_final_layout: bool,
    ) -> dict[str, Any]: ...

    def validate_training_environment(
        self, *, head_size: int
    ) -> Mapping[str, str]: ...


@runtime_checkable
class DistillationRecipe(Protocol):
    recipe_id: str
    source_adapter_id: str
    target_adapter_id: str

    def validate_source(self, inspection: ArchitectureInspection) -> None: ...

    def run_layerwise_distillation(
        self, request: DistillationExecutionRequest
    ) -> dict[str, object]: ...

    def run_corrective_distillation(
        self, request: DistillationExecutionRequest
    ) -> dict[str, object]: ...

    def prepare_performance_profile_caches(
        self, request: PerformanceProfileCacheRequest
    ) -> dict[str, object]: ...

    def run_gqa_zero_step_validation(
        self, request: GQAZeroStepValidationRequest
    ) -> dict[str, object]: ...
