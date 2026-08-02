from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from .errors import ContractError


@dataclass(frozen=True)
class LayerConvergenceState:
    completed_epochs: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    best_training_metric: float | None = None
    best_training_epoch: int | None = None
    bad_epochs: int = 0


@dataclass(frozen=True)
class LayerConvergenceDecision:
    state: LayerConvergenceState
    improved: bool
    training_curve_improved: bool
    converged: bool
    exhausted: bool
    reason: str | None


def epoch_permutation(*, row_count: int, seed: int, layer: int, epoch: int) -> tuple[tuple[int, ...], str]:
    if row_count <= 0 or seed < 0 or layer < 0 or epoch < 0:
        raise ContractError("epoch permutation arguments are invalid")
    material = f"any2rwkv-layer-major-v1\0{seed}\0{layer}\0{epoch}"
    local_seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    rows = list(range(row_count))
    random.Random(local_seed).shuffle(rows)
    digest = hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()
    return tuple(rows), digest


def update_convergence(
    state: LayerConvergenceState,
    *,
    metric: float,
    min_epochs: int,
    max_epochs: int,
    min_delta: float,
    patience: int,
) -> LayerConvergenceDecision:
    if (
        min_epochs < 1
        or max_epochs < min_epochs
        or min_delta < 0
        or patience < 1
    ):
        raise ContractError("layer convergence limits are invalid")
    epoch = state.completed_epochs
    # Checkpoint selection and plateau detection are different decisions.
    # Preserve every strictly better held-out generation; ``min_delta`` only
    # controls whether the optimizer curve made enough progress to reset patience.
    improved = state.best_metric is None or metric < state.best_metric
    best_metric = metric if improved else state.best_metric
    best_epoch = epoch if improved else state.best_epoch
    training_curve_improved = (
        state.best_training_metric is None
        or metric < state.best_training_metric - min_delta
    )
    best_training_metric = (
        metric if training_curve_improved else state.best_training_metric
    )
    best_training_epoch = (
        epoch if training_curve_improved else state.best_training_epoch
    )
    bad_epochs = 0 if training_curve_improved else state.bad_epochs + 1
    completed = epoch + 1
    converged_by_plateau = completed >= min_epochs and bad_epochs >= patience
    converged = converged_by_plateau
    exhausted = completed >= max_epochs and not converged
    return LayerConvergenceDecision(
        LayerConvergenceState(
            completed_epochs=completed,
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_training_metric=best_training_metric,
            best_training_epoch=best_training_epoch,
            bad_epochs=bad_epochs,
        ),
        improved,
        training_curve_improved,
        converged,
        exhausted,
        "validation-plateau" if converged_by_plateau else None,
    )
