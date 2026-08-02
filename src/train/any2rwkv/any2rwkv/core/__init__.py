"""Architecture-independent heterogeneous distillation contracts."""

from importlib import import_module
from typing import TYPE_CHECKING

from .contracts import (
    ArchitectureInspection,
    DistillationExecutionRequest,
    DistillationRecipe,
    GQAZeroStepValidationRequest,
    PerformanceProfileCacheRequest,
    SourceArchitectureAdapter,
    TargetArchitectureAdapter,
)
from .experiment_tracking import ExperimentTracker, write_experiment_report
from .layer_major_contract import (
    LayerMajorResumeContract,
    activate_layer_major_training,
    assert_layer_major_isolation,
    canonical_digest,
)
from .mapping_contract import (
    CalibrationDevelopmentFinalSplit,
    CandidateSelection,
    MaterializedTarget,
    SourceConsumption,
    StrictMappingLedger,
)
from .registry import AdapterRecipeRegistry, ResolvedRecipe

if TYPE_CHECKING:
    from .layer_input_cache import (
        LayerInputBatch,
        LayerInputCacheEstimate,
        LayerInputCacheReader,
        estimate_layer_input_cache_bytes,
        prepare_distributed_layer_input_cache,
        publish_distributed_layer_input_cache,
        require_layer_input_cache_capacity,
        write_distributed_layer_input_cache_partition,
        write_layer_input_cache,
    )
    from .migration_baselines import (
        build_migration_baseline_matrix,
        read_migration_baseline_stage,
        write_migration_baseline_stage,
    )


_LAZY_EXPORT_MODULES = {
    "LayerInputBatch": ".layer_input_cache",
    "LayerInputCacheEstimate": ".layer_input_cache",
    "LayerInputCacheReader": ".layer_input_cache",
    "estimate_layer_input_cache_bytes": ".layer_input_cache",
    "prepare_distributed_layer_input_cache": ".layer_input_cache",
    "publish_distributed_layer_input_cache": ".layer_input_cache",
    "require_layer_input_cache_capacity": ".layer_input_cache",
    "write_distributed_layer_input_cache_partition": ".layer_input_cache",
    "write_layer_input_cache": ".layer_input_cache",
    "build_migration_baseline_matrix": ".migration_baselines",
    "read_migration_baseline_stage": ".migration_baselines",
    "write_migration_baseline_stage": ".migration_baselines",
    "run_tiny_pipeline": ".tiny_pipeline",
}


def __getattr__(name: str) -> object:
    try:
        module_name = _LAZY_EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "AdapterRecipeRegistry",
    "ArchitectureInspection",
    "CalibrationDevelopmentFinalSplit",
    "CandidateSelection",
    "DistillationExecutionRequest",
    "DistillationRecipe",
    "ExperimentTracker",
    "GQAZeroStepValidationRequest",
    "LayerInputBatch",
    "LayerInputCacheEstimate",
    "LayerInputCacheReader",
    "LayerMajorResumeContract",
    "MaterializedTarget",
    "PerformanceProfileCacheRequest",
    "ResolvedRecipe",
    "SourceArchitectureAdapter",
    "SourceConsumption",
    "StrictMappingLedger",
    "TargetArchitectureAdapter",
    "activate_layer_major_training",
    "assert_layer_major_isolation",
    "build_migration_baseline_matrix",
    "canonical_digest",
    "estimate_layer_input_cache_bytes",
    "prepare_distributed_layer_input_cache",
    "publish_distributed_layer_input_cache",
    "read_migration_baseline_stage",
    "require_layer_input_cache_capacity",
    "run_tiny_pipeline",
    "write_distributed_layer_input_cache_partition",
    "write_experiment_report",
    "write_layer_input_cache",
    "write_migration_baseline_stage",
]
