"""Architecture-independent heterogeneous distillation contracts."""

from .contracts import (
    ArchitectureInspection,
    DistillationExecutionRequest,
    DistillationRecipe,
    PerformanceProfileCacheRequest,
    SourceArchitectureAdapter,
    TargetArchitectureAdapter,
)
from .registry import AdapterRecipeRegistry, ResolvedRecipe
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
from .experiment_tracking import ExperimentTracker, write_experiment_report
from .migration_baselines import (
    build_migration_baseline_matrix,
    read_migration_baseline_stage,
    write_migration_baseline_stage,
)

__all__ = [
    "AdapterRecipeRegistry",
    "ArchitectureInspection",
    "DistillationExecutionRequest",
    "ExperimentTracker",
    "DistillationRecipe",
    "PerformanceProfileCacheRequest",
    "LayerInputBatch",
    "LayerInputCacheEstimate",
    "LayerInputCacheReader",
    "ResolvedRecipe",
    "SourceArchitectureAdapter",
    "TargetArchitectureAdapter",
    "estimate_layer_input_cache_bytes",
    "prepare_distributed_layer_input_cache",
    "publish_distributed_layer_input_cache",
    "require_layer_input_cache_capacity",
    "write_distributed_layer_input_cache_partition",
    "write_layer_input_cache",
    "write_experiment_report",
    "build_migration_baseline_matrix",
    "read_migration_baseline_stage",
    "write_migration_baseline_stage",
]
