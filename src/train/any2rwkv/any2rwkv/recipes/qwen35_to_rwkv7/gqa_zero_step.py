from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor

from ...mixer import ProjectionBoundaryRWKV7Attention, apply_partial_rope
from ...recurrent import rwkv7_step
from ...zero_step_probe import (
    BiasFreeProjection,
    LowRankProjection,
    SelectedBiasFreeProjection,
    fit_low_rank_projection,
    fit_streamed_affine_model,
    iter_streamed_native_two_state_steps,
    native_signal_output_rollout,
    rope_aligned_two_state_bases_streamed,
    select_bias_free_projection,
    streamed_hazard_attention,
    tensor_metrics,
    tensor_sha256,
)


@dataclass(frozen=True)
class GQANativeFitTrace:
    """Post-RMSNorm source signals for one GQA layer.

    Row-aligned tensors contain a calibration prefix followed by an adaptive
    development suffix.  They never contain the frozen installation split.
    """

    mixer_input: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    grouped_key: Tensor
    grouped_value: Tensor
    gate: Tensor
    mixer_output: Tensor
    query_weight: Tensor
    key_weight: Tensor
    value_weight: Tensor
    output_weight: Tensor
    output_bias: Tensor | None


@dataclass(frozen=True)
class GQANativeFitConfig:
    calibration_rows: int
    positions: Tensor
    source_head_dim: int
    rotary_dim: int
    rope_theta: float
    supervised_token_start: int = 0
    observable_fit_steps: int = 64
    observable_fit_learning_rate: float = 0.05
    row_chunk_size: int = 1
    global_row_indices: tuple[int, ...] | None = None
    reduce_sum: Callable[[Tensor], Tensor] | None = None
    reduce_max: Callable[[Tensor], Tensor] | None = None


@dataclass(frozen=True)
class GQANativeFitResult:
    """A complete native-shaped candidate and its observable error ledger."""

    parameters: Mapping[str, Tensor]
    report: Mapping[str, object]


@dataclass(frozen=True)
class _NativeProjectionFit:
    read: SelectedBiasFreeProjection
    key: SelectedBiasFreeProjection
    value: SelectedBiasFreeProjection
    decay: LowRankProjection
    erase: LowRankProjection
    gate: LowRankProjection
    output: SelectedBiasFreeProjection
    report: Mapping[str, object]


@dataclass(frozen=True)
class _StreamedNativeSignals:
    affine_output: Tensor
    compressed_output: Tensor
    read: Tensor
    requested_decay: Tensor
    decay: Tensor
    erase: Tensor
    key: Tensor
    value: Tensor
    free_running_output: Tensor


@dataclass
class _SourceSufficientStatistics:
    count: Tensor
    sum_source: Tensor
    source_gram: Tensor

    @classmethod
    def zeros(
        cls,
        source_width: int,
        *,
        device: torch.device,
    ) -> "_SourceSufficientStatistics":
        return cls(
            count=torch.zeros((), dtype=torch.float64, device=device),
            sum_source=torch.zeros(
                source_width,
                dtype=torch.float64,
                device=device,
            ),
            source_gram=torch.zeros(
                source_width,
                source_width,
                dtype=torch.float64,
                device=device,
            ),
        )

    def add(self, source: Tensor) -> None:
        source = source.reshape(-1, source.shape[-1]).float()
        block_count = source.shape[0]
        block_sum = source.double().sum(dim=0)
        block_mean = block_sum / block_count
        block_centered = source.double() - block_mean
        old_mean = self.sum_source / self.count.clamp_min(1)
        delta = block_mean - old_mean
        correction = (
            self.count
            * block_count
            / (self.count + block_count)
        ) * torch.outer(delta, delta)
        self.source_gram.add_(correction)
        self.source_gram.add_(block_centered.T @ block_centered)
        self.count.add_(block_count)
        self.sum_source.add_(block_sum)

    def raw_gram(self) -> Tensor:
        return self.source_gram.double() + torch.outer(
            self.sum_source,
            self.sum_source,
        ) / self.count.clamp_min(1)

    def merge_(self, other: "_SourceSufficientStatistics") -> None:
        if self.source_gram.shape != other.source_gram.shape:
            raise ValueError("source statistics do not align for merging")
        combined_count = self.count + other.count
        self_mean = self.sum_source / self.count.clamp_min(1)
        other_mean = other.sum_source / other.count.clamp_min(1)
        correction = (
            self.count
            * other.count
            / combined_count.clamp_min(1)
        ) * torch.outer(other_mean - self_mean, other_mean - self_mean)
        self.source_gram.add_(other.source_gram)
        self.source_gram.add_(correction)
        self.count.copy_(combined_count)
        self.sum_source.add_(other.sum_source)

    def reduce_(self, reduce_sum: Callable[[Tensor], Tensor] | None) -> None:
        if reduce_sum is None:
            return
        raw_gram = reduce_sum(self.raw_gram())
        self.count = reduce_sum(self.count)
        self.sum_source = reduce_sum(self.sum_source)
        centered_gram = raw_gram - torch.outer(
            self.sum_source,
            self.sum_source,
        ) / self.count.clamp_min(1)
        self.source_gram.copy_(centered_gram)

    @classmethod
    def combine(
        cls,
        parts: tuple["_SourceSufficientStatistics", ...],
    ) -> "_SourceSufficientStatistics":
        if not parts:
            raise ValueError("cannot combine an empty source-statistic set")
        combined = cls.zeros(
            parts[0].source_gram.shape[0],
            device=parts[0].source_gram.device,
        )
        raw_gram = torch.zeros_like(
            combined.source_gram,
            dtype=torch.float64,
        )
        for part in parts:
            combined.count.add_(part.count)
            combined.sum_source.add_(part.sum_source)
            raw_gram.add_(part.raw_gram())
        raw_gram.sub_(
            torch.outer(
                combined.sum_source,
                combined.sum_source,
            )
            / combined.count.clamp_min(1)
        )
        combined.source_gram.copy_(raw_gram)
        return combined


@dataclass
class _AffineSufficientStatistics:
    source: _SourceSufficientStatistics
    target_count: Tensor
    target_sum_source: Tensor
    sum_target: Tensor
    source_target: Tensor
    target_square: Tensor

    @classmethod
    def zeros(
        cls,
        source_width: int,
        target_width: int,
        *,
        device: torch.device,
        source: _SourceSufficientStatistics | None = None,
    ) -> "_AffineSufficientStatistics":
        return cls(
            source=source
            if source is not None
            else _SourceSufficientStatistics.zeros(
                source_width,
                device=device,
            ),
            target_count=torch.zeros(
                (),
                dtype=torch.float64,
                device=device,
            ),
            target_sum_source=torch.zeros(
                source_width,
                dtype=torch.float64,
                device=device,
            ),
            sum_target=torch.zeros(
                target_width,
                dtype=torch.float64,
                device=device,
            ),
            source_target=torch.zeros(
                source_width,
                target_width,
                dtype=torch.float64,
                device=device,
            ),
            target_square=torch.zeros(
                (),
                dtype=torch.float64,
                device=device,
            ),
        )

    @property
    def count(self) -> Tensor:
        return self.source.count

    @property
    def sum_source(self) -> Tensor:
        return self.source.sum_source

    @property
    def source_gram(self) -> Tensor:
        return self.source.source_gram

    def add(
        self,
        source: Tensor,
        target: Tensor,
        *,
        update_source: bool = True,
    ) -> None:
        source = source.reshape(-1, source.shape[-1]).float()
        target = target.reshape(-1, target.shape[-1]).float()
        if source.shape[0] != target.shape[0]:
            raise ValueError("streamed sufficient-statistic rows do not align")
        block_count = source.shape[0]
        block_source_sum = source.double().sum(dim=0)
        block_target_sum = target.double().sum(dim=0)
        block_source_mean = block_source_sum / block_count
        block_target_mean = block_target_sum / block_count
        centered_source = source.double() - block_source_mean
        centered_target = target.double() - block_target_mean
        source_delta = (
            block_source_mean
            - self.target_sum_source / self.target_count.clamp_min(1)
        )
        target_delta = (
            block_target_mean
            - self.sum_target / self.target_count.clamp_min(1)
        )
        correction_scale = (
            self.target_count
            * block_count
            / (self.target_count + block_count)
        )
        self.source_target.add_(
            correction_scale
            * torch.outer(source_delta, target_delta)
        )
        if update_source:
            self.source.add(source)
        self.target_count.add_(block_count)
        self.target_sum_source.add_(block_source_sum)
        self.sum_target.add_(block_target_sum)
        self.source_target.add_(
            centered_source.T @ centered_target
        )
        self.target_square.add_(target.double().square().sum())

    def raw_source_target(self) -> Tensor:
        return self.source_target.double() + torch.outer(
            self.target_sum_source,
            self.sum_target,
        ) / self.target_count.clamp_min(1)

    def merge_target_(self, other: "_AffineSufficientStatistics") -> None:
        if self.source_target.shape != other.source_target.shape:
            raise ValueError("affine target statistics do not align for merging")
        combined_count = self.target_count + other.target_count
        self_source_mean = (
            self.target_sum_source / self.target_count.clamp_min(1)
        )
        other_source_mean = (
            other.target_sum_source / other.target_count.clamp_min(1)
        )
        self_target_mean = self.sum_target / self.target_count.clamp_min(1)
        other_target_mean = other.sum_target / other.target_count.clamp_min(1)
        correction = (
            self.target_count
            * other.target_count
            / combined_count.clamp_min(1)
        ) * torch.outer(
            other_source_mean - self_source_mean,
            other_target_mean - self_target_mean,
        )
        self.source_target.add_(other.source_target)
        self.source_target.add_(correction)
        self.target_count.copy_(combined_count)
        self.target_sum_source.add_(other.target_sum_source)
        self.sum_target.add_(other.sum_target)
        self.target_square.add_(other.target_square)

    def reduce_target_(
        self,
        reduce_sum: Callable[[Tensor], Tensor] | None,
    ) -> None:
        if reduce_sum is None:
            return
        raw_source_target = reduce_sum(self.raw_source_target())
        self.target_count = reduce_sum(self.target_count)
        self.target_sum_source = reduce_sum(self.target_sum_source)
        self.sum_target = reduce_sum(self.sum_target)
        centered_source_target = raw_source_target - torch.outer(
            self.target_sum_source,
            self.sum_target,
        ) / self.target_count.clamp_min(1)
        self.source_target.copy_(centered_source_target)
        self.target_square = reduce_sum(self.target_square)

    def reduce_(
        self,
        reduce_sum: Callable[[Tensor], Tensor] | None,
    ) -> None:
        self.source.reduce_(reduce_sum)
        self.reduce_target_(reduce_sum)

    @classmethod
    def combine(
        cls,
        parts: tuple["_AffineSufficientStatistics", ...],
        *,
        source: _SourceSufficientStatistics | None = None,
    ) -> "_AffineSufficientStatistics":
        if not parts:
            raise ValueError("cannot combine an empty affine-statistic set")
        combined = cls.zeros(
            parts[0].source_gram.shape[0],
            parts[0].source_target.shape[1],
            device=parts[0].source_gram.device,
            source=source
            if source is not None
            else _SourceSufficientStatistics.combine(
                tuple(part.source for part in parts)
            ),
        )
        raw_source_target = torch.zeros_like(
            combined.source_target,
            dtype=torch.float64,
        )
        for part in parts:
            combined.target_count.add_(part.target_count)
            combined.target_sum_source.add_(part.target_sum_source)
            combined.sum_target.add_(part.sum_target)
            raw_source_target.add_(part.raw_source_target())
            combined.target_square.add_(part.target_square)
        raw_source_target.sub_(
            torch.outer(
                combined.target_sum_source,
                combined.sum_target,
            )
            / combined.target_count.clamp_min(1)
        )
        combined.source_target.copy_(raw_source_target)
        return combined


@dataclass
class _MetricSums:
    squared_error: Tensor
    target_square: Tensor
    prediction_square: Tensor
    dot: Tensor

    @classmethod
    def zeros(cls, *, device: torch.device) -> "_MetricSums":
        values = torch.zeros(4, dtype=torch.float64, device=device)
        return cls(*values.unbind())

    def add(self, prediction: Tensor, target: Tensor) -> None:
        prediction = prediction.double()
        target = target.double()
        difference = prediction - target
        self.squared_error.add_(difference.square().sum())
        self.target_square.add_(target.square().sum())
        self.prediction_square.add_(prediction.square().sum())
        self.dot.add_((prediction * target).sum())

    def reduce_(
        self,
        reduce_sum: Callable[[Tensor], Tensor] | None,
    ) -> None:
        if reduce_sum is None:
            return
        self.squared_error = reduce_sum(self.squared_error)
        self.target_square = reduce_sum(self.target_square)
        self.prediction_square = reduce_sum(self.prediction_square)
        self.dot = reduce_sum(self.dot)

    def metrics(self) -> dict[str, float]:
        target_square = self.target_square.clamp_min(1e-30)
        return {
            "nmse": float(self.squared_error / target_square),
            "relative_l2": float(
                torch.sqrt(self.squared_error / target_square)
            ),
            "cosine": float(
                self.dot
                / torch.sqrt(
                    self.prediction_square.clamp_min(1e-30)
                    * target_square
                )
            ),
        }


def _solve_bias_free_statistics(
    statistics: _AffineSufficientStatistics,
    *,
    ridge: float,
    prior_weight: Tensor | None = None,
) -> Tensor:
    source_width = statistics.source_gram.shape[0]
    target_width = statistics.source_target.shape[1]
    if prior_weight is None:
        prior = torch.zeros(
            target_width,
            source_width,
            dtype=torch.float64,
            device=statistics.source_gram.device,
        )
    else:
        if prior_weight.shape != (target_width, source_width):
            raise ValueError("streamed prior weight has invalid shape")
        prior = prior_weight.double()
    gram = statistics.source.raw_gram()
    residual_rhs = statistics.raw_source_target() - gram @ prior.T
    regularized = gram + torch.eye(
        source_width,
        dtype=torch.float64,
        device=gram.device,
    ) * ridge
    return (
        prior + torch.linalg.solve(regularized, residual_rhs).T
    ).float()


def _solve_affine_statistics(
    statistics: _AffineSufficientStatistics,
    *,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    if float(statistics.count) <= 0:
        raise ValueError("cannot solve empty streamed affine statistics")
    count = statistics.target_count
    source_mean = statistics.target_sum_source / count
    target_mean = statistics.sum_target / count
    centered_gram = statistics.source_gram.double()
    centered_rhs = statistics.source_target.double()
    regularized = centered_gram + torch.eye(
        centered_gram.shape[0],
        dtype=torch.float64,
        device=centered_gram.device,
    ) * ridge
    weight = torch.linalg.solve(regularized, centered_rhs).T
    bias = target_mean - source_mean @ weight.T
    return weight.float(), bias.float()


def _statistics_metrics(
    statistics: _AffineSufficientStatistics,
    weight: Tensor,
    *,
    bias: Tensor | None = None,
) -> dict[str, float]:
    weight = weight.double()
    if bias is None:
        bias_value = torch.zeros(
            weight.shape[0],
            dtype=torch.float64,
            device=weight.device,
        )
    else:
        bias_value = bias.double()
    prediction_square = torch.einsum(
        "oi,ij,oj->",
        weight,
        statistics.source.raw_gram(),
        weight,
    )
    prediction_square = (
        prediction_square
        + 2
        * torch.dot(
            bias_value,
            weight @ statistics.sum_source.double(),
        )
        + statistics.count * bias_value.square().sum()
    )
    prediction_target = (
        (weight.T * statistics.raw_source_target()).sum()
        + torch.dot(bias_value, statistics.sum_target.double())
    )
    squared_error = (
        prediction_square
        - 2 * prediction_target
        + statistics.target_square
    ).clamp_min(0)
    target_square = statistics.target_square.clamp_min(1e-30)
    return {
        "nmse": float(squared_error / target_square),
        "relative_l2": float(torch.sqrt(squared_error / target_square)),
        "cosine": float(
            prediction_target
            / torch.sqrt(
                prediction_square.clamp_min(1e-30) * target_square
            )
        ),
    }


def _select_bias_free_statistics(
    *,
    fit: _AffineSufficientStatistics,
    selection: _AffineSufficientStatistics,
    calibration: _AffineSufficientStatistics,
    ridges: tuple[float, ...],
    prior_weight: Tensor | None = None,
) -> SelectedBiasFreeProjection:
    if not ridges or any(ridge <= 0 for ridge in ridges):
        raise ValueError("ridge candidates must be non-empty and positive")
    ridge_scale = float(
        torch.trace(fit.source.raw_gram())
        / max(1, fit.source_gram.shape[0])
    )
    ridge_scale = max(ridge_scale, torch.finfo(torch.float64).tiny)
    selection_nmse: dict[float, float] = {}
    for multiplier in ridges:
        weight = _solve_bias_free_statistics(
            fit,
            ridge=multiplier * ridge_scale,
            prior_weight=prior_weight,
        )
        selection_nmse[multiplier] = _statistics_metrics(
            selection,
            weight,
        )["nmse"]
    selected_multiplier = min(
        ridges,
        key=lambda value: (selection_nmse[value], value),
    )
    selected_ridge = selected_multiplier * ridge_scale
    weight = _solve_bias_free_statistics(
        calibration,
        ridge=selected_ridge,
        prior_weight=prior_weight,
    )
    return SelectedBiasFreeProjection(
        projection=BiasFreeProjection(
            weight=weight,
            prediction=torch.empty(
                0,
                dtype=torch.float32,
                device=weight.device,
            ),
        ),
        ridge=selected_multiplier,
        absolute_ridge=selected_ridge,
        ridge_scale=ridge_scale,
        selection_nmse=selection_nmse,
    )


def _select_shared_bias_free_statistics(
    candidates: Mapping[
        str,
        tuple[
            _AffineSufficientStatistics,
            _AffineSufficientStatistics,
            _AffineSufficientStatistics | None,
            Tensor | None,
        ],
    ],
    *,
    ridges: tuple[float, ...],
    selection_groups: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, SelectedBiasFreeProjection]:
    """Select many projections sharing one source Gram with factored solves."""

    if not candidates:
        raise ValueError("shared ridge selection requires candidates")
    first_fit = next(iter(candidates.values()))[0]
    source_width = first_fit.source_gram.shape[0]
    ridge_scale = float(
        torch.trace(first_fit.source.raw_gram()) / max(1, source_width)
    )
    ridge_scale = max(ridge_scale, torch.finfo(torch.float64).tiny)

    def factor(
        statistics: _AffineSufficientStatistics,
        ridge: float,
    ) -> Tensor:
        regularized = statistics.source.raw_gram() + torch.eye(
            source_width,
            dtype=torch.float64,
            device=statistics.source_gram.device,
        ) * ridge
        return torch.linalg.cholesky(regularized)

    def solve(
        statistics: _AffineSufficientStatistics,
        cholesky: Tensor,
        prior_weight: Tensor | None,
    ) -> Tensor:
        gram = statistics.source.raw_gram()
        rhs = statistics.raw_source_target()
        prior = None if prior_weight is None else prior_weight.double()
        if prior is not None:
            rhs.sub_(gram @ prior.T)
        solution = torch.cholesky_solve(rhs, cholesky)
        if prior is not None:
            solution.T.add_(prior)
        return solution.T.float()

    selection_nmse = {name: {} for name in candidates}
    for multiplier in ridges:
        cholesky = factor(first_fit, multiplier * ridge_scale)
        for name, (fit, selection, _, prior) in candidates.items():
            weight = solve(fit, cholesky, prior)
            selection_nmse[name][multiplier] = _statistics_metrics(
                selection,
                weight,
            )["nmse"]

    selected = {
        name: min(
            ridges,
            key=lambda value: (selection_nmse[name][value], value),
        )
        for name in candidates
    }
    retained_names = set(candidates)
    if selection_groups is not None:
        grouped_names = {
            name
            for names in selection_groups.values()
            for name in names
        }
        if grouped_names != retained_names:
            raise ValueError(
                "shared ridge selection groups must cover every candidate once"
            )
        if sum(map(len, selection_groups.values())) != len(grouped_names):
            raise ValueError(
                "shared ridge selection groups must not overlap"
            )
        retained_names = {
            min(
                names,
                key=lambda name: (
                    selection_nmse[name][selected[name]],
                    name,
                ),
            )
            for names in selection_groups.values()
        }
    calibration_statistics = {}
    merged_targets = set()
    merged_sources = set()
    for name, (fit, selection, calibration, _) in candidates.items():
        if calibration is not None:
            calibration_statistics[name] = calibration
            continue
        target_identity = id(fit)
        if target_identity not in merged_targets:
            fit.merge_target_(selection)
            merged_targets.add(target_identity)
        source_identity = id(fit.source)
        if source_identity not in merged_sources:
            fit.source.merge_(selection.source)
            merged_sources.add(source_identity)
        calibration_statistics[name] = fit
    results = {}
    calibration = next(iter(calibration_statistics.values()))
    for multiplier in sorted(
        {selected[name] for name in retained_names}
    ):
        cholesky = factor(
            calibration,
            multiplier * ridge_scale,
        )
        for name, (_, _, _, prior) in candidates.items():
            if name not in retained_names or selected[name] != multiplier:
                continue
            candidate_calibration = calibration_statistics[name]
            weight = solve(
                candidate_calibration,
                cholesky,
                prior,
            )
            results[name] = SelectedBiasFreeProjection(
                projection=BiasFreeProjection(
                    weight=weight,
                    prediction=torch.empty(
                        0,
                        dtype=torch.float32,
                        device=candidate_calibration.source_gram.device,
                    ),
                ),
                ridge=multiplier,
                absolute_ridge=multiplier * ridge_scale,
                ridge_scale=ridge_scale,
                selection_nmse=selection_nmse[name],
            )
    return results


def estimate_gqa_native_streamed_peak_bytes(
    *,
    fit_rows: int,
    context_length: int,
    hidden_size: int,
    query_heads: int,
    key_value_heads: int,
    source_head_dim: int,
    native_head_dim: int,
    row_chunk_size: int,
    decay_rank: int | None = None,
    erase_rank: int | None = None,
    gate_rank: int | None = None,
) -> dict[str, int]:
    """Conservative peak estimate for the streamed formal GQA solver."""

    values = (
        fit_rows,
        context_length,
        hidden_size,
        query_heads,
        key_value_heads,
        source_head_dim,
        native_head_dim,
        row_chunk_size,
    )
    if any(value <= 0 for value in values):
        raise ValueError("streamed GQA peak geometry must be positive")
    if source_head_dim != 2 * native_head_dim:
        raise ValueError("streamed GQA peak requires source_dim=2*native_dim")
    decay_rank = native_head_dim if decay_rank is None else decay_rank
    erase_rank = native_head_dim if erase_rank is None else erase_rank
    gate_rank = native_head_dim if gate_rank is None else gate_rank
    if min(decay_rank, erase_rank, gate_rank) <= 0:
        raise ValueError("streamed GQA low-rank geometry must be positive")
    calibration_rows = max(2, fit_rows // 2)
    trace_width = 2 * hidden_size + 4 * query_heads * source_head_dim
    fp32_trace = 4 * fit_rows * context_length * trace_width
    hazard_output_bytes = (
        4
        * fit_rows
        * context_length
        * 2
        * query_heads
        * source_head_dim
    )
    affine_closure = (
        4
        * calibration_rows
        * key_value_heads
        * source_head_dim
        * source_head_dim
        * 3
    )
    affine_vector_workspace = (
        4
        * calibration_rows
        * context_length
        * (
            2 * query_heads * source_head_dim
            + key_value_heads * (source_head_dim + 1)
        )
    )
    streamed_native_workspace = (
        4
        * row_chunk_size
        * query_heads
        * 2
        * native_head_dim
        * native_head_dim
        * 8
    )
    observable_grams = (
        4
        * query_heads
        * 3
        * source_head_dim
        * source_head_dim
    )
    maximum_target_width = max(
        hidden_size,
        query_heads * source_head_dim,
    )
    target_width = query_heads * source_head_dim
    direct_statistics = (
        8
        * 2
        * (
            hidden_size * hidden_size
            + 5 * hidden_size * target_width
        )
    )
    output_statistics = (
        8
        * 3
        * (
            target_width * target_width
            + target_width * hidden_size
        )
    )
    normal_equations = (
        8 * 7 * hidden_size * maximum_target_width
        + 4 * 3 * hidden_size * maximum_target_width
    )
    projection_priors = (
        4 * 3 * hidden_size * target_width
    )
    auxiliary_statistics = 8 * (
        3 * (gate_rank * gate_rank + gate_rank * target_width)
        + decay_rank * decay_rank
        + decay_rank * target_width
        + erase_rank * erase_rank
        + erase_rank * target_width
    )
    solver_safety_margin = max(
        1024 * 1024,
        normal_equations // 2,
    )
    hazard_block_workspace = (
        4
        * 64
        * context_length
        * 6
    )
    estimated_peak = (
        fp32_trace
        + hazard_output_bytes
        + affine_closure
        + affine_vector_workspace
        + streamed_native_workspace
        + observable_grams
        + max(direct_statistics, output_statistics)
        + normal_equations
        + projection_priors
        + auxiliary_statistics
        + solver_safety_margin
        + hazard_block_workspace
    )
    return {
        "estimated_peak_bytes": int(estimated_peak),
        "fp32_trace_bytes": int(fp32_trace),
        "hazard_output_bytes": int(hazard_output_bytes),
        "affine_closure_bytes": int(affine_closure),
        "affine_vector_workspace_bytes": int(
            affine_vector_workspace
        ),
        "streamed_native_workspace_bytes": int(
            streamed_native_workspace
        ),
        "observable_gram_bytes": int(observable_grams),
        "direct_sufficient_statistic_bytes": int(direct_statistics),
        "output_sufficient_statistic_bytes": int(output_statistics),
        "normal_equation_bytes": int(normal_equations),
        "projection_prior_bytes": int(projection_priors),
        "auxiliary_sufficient_statistic_bytes": int(
            auxiliary_statistics
        ),
        "solver_safety_margin_bytes": int(solver_safety_margin),
        "hazard_block_workspace_bytes": int(hazard_block_workspace),
    }


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _per_head_group_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    key_value_heads: int,
    reduce_sum: Callable[[Tensor], Tensor] | None = None,
) -> dict[str, list[dict[str, float]]]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "per-head GQA metrics require aligned [batch,time,head,feature] tensors"
        )
    query_heads = prediction.shape[2]
    if key_value_heads <= 0 or query_heads % key_value_heads:
        raise ValueError("per-head GQA metrics do not align with KV groups")
    group_width = query_heads // key_value_heads
    def reduced_metrics(predicted: Tensor, expected: Tensor):
        sums = _MetricSums.zeros(device=prediction.device)
        sums.add(predicted, expected)
        sums.reduce_(reduce_sum)
        return sums.metrics()

    per_query_head = [
        reduced_metrics(
            prediction[:, :, head],
            target[:, :, head],
        )
        for head in range(query_heads)
    ]
    per_key_value_group = [
        reduced_metrics(
            prediction[
                :,
                :,
                group * group_width : (group + 1) * group_width,
            ],
            target[
                :,
                :,
                group * group_width : (group + 1) * group_width,
            ],
        )
        for group in range(key_value_heads)
    ]
    return {
        "per_query_head": per_query_head,
        "per_key_value_group": per_key_value_group,
    }


def fit_gqa_native_zero_step(
    module: ProjectionBoundaryRWKV7Attention,
    trace: GQANativeFitTrace,
    config: GQANativeFitConfig,
) -> GQANativeFitResult:
    """Fit the exact-hazard/observable-compression/native-projection chain.

    Only ``trace[:calibration_rows]`` participates in solving.  The remaining
    rows are used for adaptive-development diagnostics.  Installation on the
    separately frozen validation cache is intentionally owned by the caller.
    """

    _validate_trace(module, trace, config)
    calibration_rows = config.calibration_rows
    global_row_indices = _fit_global_row_indices(
        trace.query.shape[0],
        config,
    )
    local_calibration_rows = bisect_left(
        global_row_indices,
        calibration_rows,
    )
    maximum_row = torch.tensor(
        global_row_indices[-1],
        dtype=torch.int64,
        device=trace.query.device,
    )
    if config.reduce_max is not None:
        maximum_row = config.reduce_max(maximum_row)
    global_trace_rows = int(maximum_row.item()) + 1
    calibration_query = trace.query[:local_calibration_rows]
    head_sum = calibration_query.double().sum(dim=(0, 1))
    head_count = torch.tensor(
        local_calibration_rows * trace.query.shape[1],
        dtype=torch.float64,
        device=trace.query.device,
    )
    if config.reduce_sum is not None:
        head_sum = config.reduce_sum(head_sum)
        head_count = config.reduce_sum(head_count)
    head_center = (head_sum / head_count.clamp_min(1)).float()
    hazard_attention = streamed_hazard_attention(
        trace.query,
        trace.key,
        trace.value,
        head_center,
        development_row_start=local_calibration_rows,
        supervised_token_start=config.supervised_token_start,
        reduce_sum=config.reduce_sum,
    )

    groups = trace.grouped_key.shape[2]
    query_heads = trace.query.shape[2]
    group_width = query_heads // groups
    group_sum = torch.stack(
        [
            calibration_query[
                :,
                :,
                group * group_width : (group + 1) * group_width,
            ].double().sum(dim=(0, 1, 2))
            for group in range(groups)
        ]
    )
    group_count = torch.tensor(
        local_calibration_rows
        * trace.query.shape[1]
        * group_width,
        dtype=torch.float64,
        device=trace.query.device,
    )
    if config.reduce_sum is not None:
        group_sum = config.reduce_sum(group_sum)
        group_count = config.reduce_sum(group_count)
    group_center = (group_sum / group_count.clamp_min(1)).float()
    affine = fit_streamed_affine_model(
        calibration_query,
        trace.grouped_key[:local_calibration_rows],
        trace.grouped_value[:local_calibration_rows],
        group_center,
        calibration_batches=local_calibration_rows,
        row_chunk_size=config.row_chunk_size,
        reduce_sum=config.reduce_sum,
    )
    basis = rope_aligned_two_state_bases_streamed(
        calibration_query,
        trace.grouped_key[:local_calibration_rows],
        trace.grouped_value[:local_calibration_rows],
        affine,
        native_dim=module.head_dim,
        rotary_dim=config.rotary_dim,
        steps=config.observable_fit_steps,
        learning_rate=config.observable_fit_learning_rate,
        reduce_sum=config.reduce_sum,
    )
    dc_indices = (module.head_dim - 1, module.head_dim - 1)
    projection_fit = _fit_native_projection_streamed(
        module=module,
        trace=trace,
        affine=affine,
        basis=basis,
        config=config,
        dc_indices=dc_indices,
    )
    parameters = _native_parameter_set(
        module,
        projection_fit,
        source_value_weight=trace.value_weight,
    )

    development = slice(local_calibration_rows, None)
    supervised = slice(config.supervised_token_start, None)
    solver_config = {
        "calibration_rows": calibration_rows,
        "supervised_token_start": config.supervised_token_start,
        "source_head_dim": config.source_head_dim,
        "rotary_dim": config.rotary_dim,
        "rope_theta": config.rope_theta,
        "observable_fit_steps": config.observable_fit_steps,
        "observable_fit_learning_rate": config.observable_fit_learning_rate,
        "row_chunk_size": config.row_chunk_size,
        "global_row_indices_sha256": _json_sha256(
            list(global_row_indices)
        ),
        "positions_sha256": tensor_sha256(config.positions),
    }
    trace_sha256 = {
        name: tensor_sha256(value)
        for name, value in {
            "mixer_input": trace.mixer_input,
            "query": trace.query,
            "key": trace.key,
            "value": trace.value,
            "grouped_key": trace.grouped_key,
            "grouped_value": trace.grouped_value,
            "gate": trace.gate,
            "mixer_output": trace.mixer_output,
        }.items()
    }
    source_weight_sha256 = {
        name: tensor_sha256(value)
        for name, value in {
            "query_weight": trace.query_weight,
            "key_weight": trace.key_weight,
            "value_weight": trace.value_weight,
            "output_weight": trace.output_weight,
        }.items()
    }
    exact_development = hazard_attention.exact_output[development, supervised]
    bounded_development = hazard_attention.bounded_output[
        development,
        supervised,
    ]
    bounded_output_sums = _MetricSums.zeros(device=trace.query.device)
    bounded_output_sums.add(
        bounded_development,
        exact_development,
    )
    bounded_output_sums.reduce_(config.reduce_sum)
    decomposition = _streamed_decomposition_report(
        trace=trace,
        affine=affine,
        basis=basis,
        exact_output=hazard_attention.exact_output,
        calibration_rows=local_calibration_rows,
        supervised_token_start=config.supervised_token_start,
        native_dim=module.head_dim,
        dc_indices=dc_indices,
        reduce_sum=config.reduce_sum,
    )
    report = {
        "schema_version": 1,
        "boundary": (
            "gqa-exact-hazard-bounded-surrogate-observable-two-state-"
            "native-projection-v1"
        ),
        "fit_rows": calibration_rows,
        "development_rows": int(global_trace_rows - calibration_rows),
        "split_contract": (
            "calibration prefix solves/selects; adaptive-development suffix "
            "reports decomposition; frozen validation is caller-owned"
        ),
        "source_geometry": {
            "query_heads": query_heads,
            "key_value_heads": groups,
            "source_head_dim": config.source_head_dim,
        },
        "target_geometry": {
            "native_heads": module.num_heads,
            "native_head_dim": module.head_dim,
        },
        "solver_config": solver_config,
        "solver_config_sha256": _json_sha256(solver_config),
        "trace_sha256": trace_sha256,
        "trace_aggregate_sha256": _json_sha256(trace_sha256),
        "source_weight_sha256": source_weight_sha256,
        "source_weight_aggregate_sha256": _json_sha256(
            source_weight_sha256
        ),
        "exact_prefix_hazard_oracle": {
            "attention_output": hazard_attention.exact_rollout_metrics,
            **_per_head_group_metrics(
                hazard_attention.exact_output[development, supervised],
                exact_development,
                key_value_heads=groups,
                reduce_sum=config.reduce_sum,
            ),
        },
        "bounded_hazard_surrogate": {
            "link": "sigmoid",
            "hazard": hazard_attention.bounded_hazard_metrics,
            "attention_output": bounded_output_sums.metrics(),
            **_per_head_group_metrics(
                bounded_development,
                exact_development,
                key_value_heads=groups,
                reduce_sum=config.reduce_sum,
            ),
        },
        "observable_state_compression": {
            "budget": {
                "states_per_source_head": 2,
                "dc_per_state": 1,
                "query_features_per_state": module.head_dim - 1,
                "shared_input_subspace": False,
            },
            "partial_rope": {
                "rotary_coordinates_in_first_state": config.rotary_dim,
                "first_state_invariant_features": (
                    module.head_dim - 1 - config.rotary_dim
                ),
                "second_state_invariant_features": module.head_dim - 1,
            },
            **decomposition["observable_state_compression"],
        },
        "native_transition": decomposition["native_transition"],
        "streaming_workspace": {
            "time_major_operator_state": False,
            "square_hazard_grid": False,
            "row_chunk_size": config.row_chunk_size,
            "affine_peak_state_elements": affine.peak_state_elements,
        },
        "native_parameter_projection": projection_fit.report,
        "parameter_sha256": {
            name: tensor_sha256(value) for name, value in sorted(parameters.items())
        },
    }
    return GQANativeFitResult(parameters=parameters, report=report)


def validate_gqa_native_fit_trace(
    module: ProjectionBoundaryRWKV7Attention,
    trace: GQANativeFitTrace,
    config: GQANativeFitConfig,
) -> None:
    """Validate all rank-local inputs before a distributed fit is entered."""

    _validate_trace(module, trace, config)


def _collect_streamed_native_signals(
    query: Tensor,
    grouped_key: Tensor,
    grouped_value: Tensor,
    affine,
    basis: Tensor,
    *,
    native_dim: int,
    dc_indices: tuple[int, int],
) -> _StreamedNativeSignals:
    """Collect only vector signals required by the native parameter solve."""

    batch, tokens, query_heads, source_dim = query.shape
    native_heads = query_heads * 2
    device = query.device
    vector_shape = (batch, tokens, native_heads, native_dim)
    source_shape = (batch, tokens, query_heads, source_dim)
    affine_output = torch.empty(
        source_shape,
        dtype=torch.float32,
        device=device,
    )
    compressed_output = torch.empty_like(affine_output)
    read = torch.empty(vector_shape, dtype=torch.float32, device=device)
    requested_decay = torch.empty(
        (batch, tokens, native_heads),
        dtype=torch.float32,
        device=device,
    )
    decay = torch.empty_like(requested_decay)
    erase = torch.empty_like(read)
    key = torch.empty_like(read)
    value = torch.empty_like(read)
    free_running_output = torch.empty_like(affine_output)
    for step in iter_streamed_native_two_state_steps(
        query,
        grouped_key,
        grouped_value,
        affine,
        basis,
        native_dim=native_dim,
        dc_indices=dc_indices,
    ):
        rows = slice(step.row_start, step.row_stop)
        index = step.time_index
        affine_output[rows, index] = step.affine_output
        compressed_output[rows, index] = step.compressed_output
        read[rows, index] = step.read
        requested_decay[rows, index] = step.requested_decay
        decay[rows, index] = step.decay
        erase[rows, index] = step.erase
        key[rows, index] = step.key
        value[rows, index] = step.value
        free_running_output[rows, index] = step.free_running_output
    return _StreamedNativeSignals(
        affine_output=affine_output,
        compressed_output=compressed_output,
        read=read,
        requested_decay=requested_decay,
        decay=decay,
        erase=erase,
        key=key,
        value=value,
        free_running_output=free_running_output,
    )


def _streamed_decomposition_report(
    *,
    trace: GQANativeFitTrace,
    affine,
    basis: Tensor,
    exact_output: Tensor,
    calibration_rows: int,
    supervised_token_start: int,
    native_dim: int,
    dc_indices: tuple[int, int],
    reduce_sum: Callable[[Tensor], Tensor] | None = None,
) -> dict[str, object]:
    device = trace.query.device
    query_heads = trace.query.shape[2]
    key_value_heads = trace.grouped_key.shape[2]
    group_width = query_heads // key_value_heads
    metrics = {
        name: _MetricSums.zeros(device=device)
        for name in (
            "compressed_vs_affine",
            "compressed_vs_exact",
            "native_vs_compressed",
            "native_vs_exact",
        )
    }
    compressed_per_head = [
        _MetricSums.zeros(device=device) for _ in range(query_heads)
    ]
    native_per_head = [
        _MetricSums.zeros(device=device) for _ in range(query_heads)
    ]
    compressed_per_group = [
        _MetricSums.zeros(device=device) for _ in range(key_value_heads)
    ]
    native_per_group = [
        _MetricSums.zeros(device=device) for _ in range(key_value_heads)
    ]
    clamped = torch.zeros(2, dtype=torch.float64, device=device)
    for step in iter_streamed_native_two_state_steps(
        trace.query,
        trace.grouped_key,
        trace.grouped_value,
        affine,
        basis,
        native_dim=native_dim,
        dc_indices=dc_indices,
    ):
        if (
            step.time_index < supervised_token_start
            or step.row_stop <= calibration_rows
        ):
            continue
        start = max(0, calibration_rows - step.row_start)
        rows = slice(step.row_start + start, step.row_stop)
        local = slice(start, None)
        exact = exact_output[rows, step.time_index]
        compressed = step.compressed_output[local]
        native = step.free_running_output[local]
        metrics["compressed_vs_affine"].add(
            compressed,
            step.affine_output[local],
        )
        metrics["compressed_vs_exact"].add(compressed, exact)
        metrics["native_vs_compressed"].add(native, compressed)
        metrics["native_vs_exact"].add(native, exact)
        for head in range(query_heads):
            compressed_per_head[head].add(
                compressed[:, head],
                exact[:, head],
            )
            native_per_head[head].add(native[:, head], exact[:, head])
        for group in range(key_value_heads):
            head_slice = slice(
                group * group_width,
                (group + 1) * group_width,
            )
            compressed_per_group[group].add(
                compressed[:, head_slice],
                exact[:, head_slice],
            )
            native_per_group[group].add(
                native[:, head_slice],
                exact[:, head_slice],
            )
        clamped[0].add_(
            (step.requested_decay[local] != step.decay[local])
            .double()
            .sum()
        )
        clamped[1].add_(step.decay[local].numel())
    for sums in metrics.values():
        sums.reduce_(reduce_sum)
    for sums in (
        *compressed_per_head,
        *native_per_head,
        *compressed_per_group,
        *native_per_group,
    ):
        sums.reduce_(reduce_sum)
    if reduce_sum is not None:
        clamped = reduce_sum(clamped)
    return {
        "observable_state_compression": {
            "affine_output": metrics["compressed_vs_affine"].metrics(),
            "exact_attention_output": metrics[
                "compressed_vs_exact"
            ].metrics(),
            "per_query_head": [
                item.metrics() for item in compressed_per_head
            ],
            "per_key_value_group": [
                item.metrics() for item in compressed_per_group
            ],
        },
        "native_transition": {
            "free_running_vs_compressed": metrics[
                "native_vs_compressed"
            ].metrics(),
            "free_running_vs_exact_attention": metrics[
                "native_vs_exact"
            ].metrics(),
            "per_query_head": [
                item.metrics() for item in native_per_head
            ],
            "per_key_value_group": [
                item.metrics() for item in native_per_group
            ],
            "requested_decay_clamped_fraction": float(
                clamped[0] / clamped[1].clamp_min(1)
            ),
        },
    }


def _fit_global_row_indices(
    batch: int,
    config: GQANativeFitConfig,
) -> tuple[int, ...]:
    row_indices = (
        tuple(range(batch))
        if config.global_row_indices is None
        else tuple(int(value) for value in config.global_row_indices)
    )
    if len(row_indices) != batch:
        raise ValueError("GQA global row identities do not align with trace")
    if tuple(sorted(row_indices)) != row_indices:
        raise ValueError("GQA global row identities must be sorted")
    if len(set(row_indices)) != len(row_indices) or min(row_indices) < 0:
        raise ValueError("GQA global row identities must be unique and non-negative")
    return row_indices


def _validate_trace(
    module: ProjectionBoundaryRWKV7Attention,
    trace: GQANativeFitTrace,
    config: GQANativeFitConfig,
) -> None:
    row_aligned = (
        trace.mixer_input,
        trace.query,
        trace.key,
        trace.value,
        trace.grouped_key,
        trace.grouped_value,
        trace.gate,
        trace.mixer_output,
    )
    batch, tokens = trace.mixer_input.shape[:2]
    if any(value.shape[:2] != (batch, tokens) for value in row_aligned):
        raise ValueError("GQA fit trace batch/time dimensions do not align")
    global_row_indices = _fit_global_row_indices(batch, config)
    calibration_count = bisect_left(
        global_row_indices,
        config.calibration_rows,
    )
    if (
        config.calibration_rows <= 1
        or calibration_count <= 0
        or calibration_count >= batch
    ):
        raise ValueError(
            "GQA fit requires local calibration and development contributions"
        )
    if trace.query.shape != trace.key.shape or trace.query.shape != trace.value.shape:
        raise ValueError("repeated GQA query/key/value traces must align")
    query_heads, source_head_dim = trace.query.shape[2:]
    if source_head_dim != config.source_head_dim:
        raise ValueError("GQA source head dimension differs from fit config")
    if trace.grouped_key.shape != trace.grouped_value.shape:
        raise ValueError("grouped GQA key/value traces must align")
    if query_heads % trace.grouped_key.shape[2]:
        raise ValueError("GQA KV groups must divide query heads")
    if source_head_dim != 2 * module.head_dim:
        raise ValueError("GQA two-state fit requires source_dim=2*native_dim")
    if module.num_heads != query_heads * 2:
        raise ValueError("GQA two-state fit requires two native heads per query head")
    if config.positions.shape != (batch, tokens):
        raise ValueError("GQA position ids do not align with fit rows")
    if not 0 <= config.supervised_token_start < tokens:
        raise ValueError("GQA supervised token start is outside the trace")
    if config.observable_fit_steps < 0:
        raise ValueError("GQA observable fit steps must be non-negative")
    if config.observable_fit_learning_rate <= 0:
        raise ValueError("GQA observable fit learning rate must be positive")
    if config.row_chunk_size <= 0:
        raise ValueError("GQA streamed row chunk size must be positive")
    if trace.gate.shape != trace.query.shape:
        raise ValueError("GQA gate must align with repeated query heads")
    if trace.mixer_output.shape[-1] != module.hidden_size:
        raise ValueError("GQA mixer output width differs from target residual width")
    source_hidden = trace.mixer_input.shape[-1]
    groups = trace.grouped_key.shape[2]
    expected_weights = {
        "query_weight": (
            query_heads * source_head_dim * 2,
            source_hidden,
        ),
        "key_weight": (groups * source_head_dim, source_hidden),
        "value_weight": (groups * source_head_dim, source_hidden),
        "output_weight": (module.hidden_size, query_heads * source_head_dim),
    }
    for name, expected in expected_weights.items():
        if tuple(getattr(trace, name).shape) != expected:
            raise ValueError(
                f"GQA {name} shape differs from source geometry: "
                f"got={tuple(getattr(trace, name).shape)} expected={expected}"
            )
    if trace.output_bias is not None and tuple(trace.output_bias.shape) != (
        module.hidden_size,
    ):
        raise ValueError("GQA output bias shape differs from target residual width")
    if trace.output_bias is not None and bool(trace.output_bias.ne(0).any()):
        raise ValueError(
            "GQA source output bias is not representable by bias-free o_proj"
        )
    finite_inputs = (
        *row_aligned,
        trace.query_weight,
        trace.key_weight,
        trace.value_weight,
        trace.output_weight,
        config.positions,
    )
    if trace.output_bias is not None:
        finite_inputs = (*finite_inputs, trace.output_bias)
    for value in finite_inputs:
        if value.is_floating_point() and not bool(torch.isfinite(value.float()).all()):
            raise ValueError("GQA fit trace contains non-finite values")


def _rotate_native(
    value: Tensor,
    positions: Tensor,
    *,
    source_head_dim: int,
    rotary_dim: int,
    rope_theta: float,
    inverse: bool,
) -> Tensor:
    batch, tokens = value.shape[:2]
    flat_width = value.shape[2] * value.shape[3]
    if flat_width % source_head_dim:
        raise ValueError("native projection width does not fit source RoPE heads")
    source_view = value.reshape(
        batch,
        tokens,
        flat_width // source_head_dim,
        source_head_dim,
    )
    rotated = apply_partial_rope(
        source_view,
        -positions if inverse else positions,
        rotary_dim=rotary_dim,
        theta=rope_theta,
    )
    return rotated.reshape_as(value)


def _fit_native_projection_streamed(
    *,
    module: ProjectionBoundaryRWKV7Attention,
    trace: GQANativeFitTrace,
    affine,
    basis: Tensor,
    config: GQANativeFitConfig,
    dc_indices: tuple[int, int],
) -> _NativeProjectionFit:
    """Fit native parameters from replayed additive sufficient statistics."""

    calibration_rows = config.calibration_rows
    fit_rows = max(1, calibration_rows // 2)
    global_row_indices = _fit_global_row_indices(
        trace.mixer_input.shape[0],
        config,
    )
    local_calibration_rows = bisect_left(
        global_row_indices,
        calibration_rows,
    )
    source_hidden = trace.mixer_input.shape[-1]
    source_heads = trace.query.shape[2]
    native_head_dim = module.head_dim
    native_heads = source_heads * 2
    target_width = native_heads * native_head_dim
    device = trace.mixer_input.device
    ridges = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    def split_statistics(
        source_width: int,
        output_width: int,
        *,
        include_development: bool = False,
        shared_sources: Mapping[
            str, _SourceSufficientStatistics
        ] | None = None,
    ) -> dict[str, _AffineSufficientStatistics]:
        return {
            name: _AffineSufficientStatistics.zeros(
                source_width,
                output_width,
                device=device,
                source=(
                    None
                    if shared_sources is None
                    else shared_sources[name]
                ),
            )
            for name in (
                "fit",
                "selection",
                *(("development",) if include_development else ()),
            )
        }

    direct_sources = {
        name: _SourceSufficientStatistics.zeros(
            source_hidden,
            device=device,
        )
        for name in ("fit", "selection")
    }
    statistics = {
        name: split_statistics(
            source_hidden,
            target_width,
            shared_sources=direct_sources,
        )
        for name in ("read", "key", "value", "decay", "erase")
    }
    query_weight = trace.query_weight.reshape(
        source_heads,
        config.source_head_dim * 2,
        source_hidden,
    )[:, : config.source_head_dim]
    read_prior = torch.zeros(
        source_heads,
        2,
        native_head_dim,
        source_hidden,
        dtype=torch.float32,
        device=device,
    )
    for head in range(source_heads):
        for state_index, dc_index in enumerate(dc_indices):
            feature_indices = [
                index
                for index in range(native_head_dim)
                if index != dc_index
            ]
            read_prior[head, state_index, feature_indices] = (
                basis[head, state_index].T @ query_weight[head]
            )
    read_prior = read_prior.flatten(0, 2)
    kv_heads = trace.key_weight.shape[0] // config.source_head_dim
    group_repeat = source_heads // kv_heads

    def repeated_kv_prior(weight: Tensor) -> Tensor:
        return (
            weight.reshape(kv_heads, config.source_head_dim, source_hidden)
            .repeat_interleave(group_repeat, dim=0)
            .flatten(0, 1)
        )

    gate_rank = int(module.g_lora.lora[0].out_features)
    source_gate_weight = trace.query_weight.reshape(
        source_heads,
        config.source_head_dim * 2,
        source_hidden,
    )[:, config.source_head_dim :].flatten(0, 1)
    effective_gate_rank = min(
        gate_rank,
        source_gate_weight.shape[0],
        source_hidden,
    )
    gate_indices = torch.linspace(
        0,
        source_gate_weight.shape[0] - 1,
        effective_gate_rank,
        dtype=torch.float64,
        device=device,
    ).round().to(torch.long)
    gate_down = source_gate_weight.index_select(0, gate_indices)
    gate_statistics = split_statistics(effective_gate_rank, target_width)

    def split_segments(
        row_start: int,
        row_stop: int,
        *,
        include_development: bool,
    ) -> tuple[tuple[str, int, int], ...]:
        fit_stop = bisect_left(
            global_row_indices,
            fit_rows,
            row_start,
            row_stop,
        )
        calibration_stop = bisect_left(
            global_row_indices,
            calibration_rows,
            row_start,
            row_stop,
        )
        bounds = (
            ("fit", row_start, fit_stop),
            (
                "selection",
                fit_stop,
                calibration_stop,
            ),
            *(
                (
                    (
                        "development",
                        calibration_stop,
                        row_stop,
                    ),
                )
                if include_development
                else ()
            ),
        )
        return tuple(
            (name, start - row_start, stop - row_start)
            for name, start, stop in bounds
            if start < stop
        )

    def add_row_splits(
        target_statistics: dict[str, _AffineSufficientStatistics],
        source: Tensor,
        target: Tensor,
        *,
        row_start: int,
        row_stop: int,
        include_development: bool = False,
    ) -> None:
        for name, start, stop in split_segments(
            row_start,
            row_stop,
            include_development=include_development,
        ):
            target_statistics[name].add(
                source[start:stop],
                target[start:stop],
            )

    block_size = 64
    current_chunk: tuple[int, int] | None = None
    block_sources: list[Tensor] = []
    block_gate_sources: list[Tensor] = []
    block_targets: dict[str, list[Tensor]] = {
        name: [] for name in statistics
    }
    block_gate_targets: list[Tensor] = []

    def flush_direct_block() -> None:
        nonlocal block_sources
        nonlocal block_gate_sources
        nonlocal block_targets
        nonlocal block_gate_targets
        if current_chunk is None or not block_sources:
            return
        row_start, row_stop = current_chunk
        source_block = torch.stack(block_sources, dim=0)
        gate_source_block = torch.stack(block_gate_sources, dim=0)
        target_blocks = {
            name: torch.stack(values, dim=0)
            for name, values in block_targets.items()
        }
        gate_target_block = torch.stack(block_gate_targets, dim=0)
        for split_name, start, stop in split_segments(
            row_start,
            row_stop,
            include_development=False,
        ):
            split_source = source_block[:, start:stop].flatten(0, 1)
            for index, name in enumerate(statistics):
                statistics[name][split_name].add(
                    split_source,
                    target_blocks[name][:, start:stop].flatten(0, 1),
                    update_source=index == 0,
                )
            gate_statistics[split_name].add(
                gate_source_block[:, start:stop].flatten(0, 1),
                gate_target_block[:, start:stop].flatten(0, 1),
            )
        block_sources = []
        block_gate_sources = []
        block_targets = {name: [] for name in statistics}
        block_gate_targets = []

    for step in iter_streamed_native_two_state_steps(
        trace.query,
        trace.grouped_key,
        trace.grouped_value,
        affine,
        basis,
        native_dim=native_head_dim,
        dc_indices=dc_indices,
    ):
        chunk = (step.row_start, step.row_stop)
        if current_chunk != chunk:
            flush_direct_block()
            current_chunk = chunk
        rows = slice(step.row_start, step.row_stop)
        source = trace.mixer_input[rows, step.time_index].float()
        positions = config.positions[
            rows,
            step.time_index : step.time_index + 1,
        ]
        read_target = _rotate_native(
            step.read.unsqueeze(1),
            positions,
            source_head_dim=config.source_head_dim,
            rotary_dim=config.rotary_dim,
            rope_theta=config.rope_theta,
            inverse=True,
        ).squeeze(1)
        key_target = _rotate_native(
            step.key.unsqueeze(1),
            positions,
            source_head_dim=config.source_head_dim,
            rotary_dim=config.rotary_dim,
            rope_theta=config.rope_theta,
            inverse=True,
        ).squeeze(1)
        decay_probability = (
            -torch.log(
                step.decay.unsqueeze(-1)
                .expand(-1, -1, native_head_dim)
                .clamp_min(1e-12)
            )
            / math.exp(-0.5)
        ).clamp(1e-6, 1 - 1e-6)
        targets = {
            "read": read_target.flatten(1),
            "key": key_target.flatten(1),
            "value": step.value.flatten(1),
            "decay": torch.logit(decay_probability).flatten(1),
            "erase": torch.logit(
                step.erase.clamp(1e-6, 1 - 1e-6)
            ).flatten(1),
        }
        gate_features = torch.sigmoid(source @ gate_down.T)
        gate_target = trace.gate[
            rows,
            step.time_index,
        ].reshape(step.row_stop - step.row_start, target_width)
        block_sources.append(source)
        for name, target in targets.items():
            block_targets[name].append(target)
        block_gate_sources.append(gate_features)
        block_gate_targets.append(gate_target)
        if len(block_sources) == block_size:
            flush_direct_block()
    flush_direct_block()

    for source_statistics in direct_sources.values():
        source_statistics.reduce_(config.reduce_sum)
    for target_statistics in statistics.values():
        for split_name in ("fit", "selection"):
            target_statistics[split_name].reduce_target_(
                config.reduce_sum
            )
    for split_name in ("fit", "selection"):
        gate_statistics[split_name].reduce_(config.reduce_sum)
    gate_statistics["calibration"] = _AffineSufficientStatistics.combine(
        (gate_statistics["fit"], gate_statistics["selection"])
    )

    direct_priors = {
        "read": read_prior,
        "key": repeated_kv_prior(trace.key_weight),
        "value": repeated_kv_prior(trace.value_weight),
    }
    direct_candidate_groups = {
        name: (
            f"{name}:zero",
            f"{name}:source-compatible",
        )
        for name in direct_priors
    }
    shared_direct = _select_shared_bias_free_statistics(
        {
            f"{name}:{center}": (
                statistics[name]["fit"],
                statistics[name]["selection"],
                None,
                None if center == "zero" else prior,
            )
            for name, prior in direct_priors.items()
            for center in ("zero", "source-compatible")
        },
        ridges=ridges,
        selection_groups=direct_candidate_groups,
    )
    for target_statistics in statistics.values():
        target_statistics["calibration"] = target_statistics["fit"]

    def selected_direct(
        name: str,
    ) -> tuple[SelectedBiasFreeProjection, str]:
        selected_name = next(
            candidate
            for candidate in direct_candidate_groups[name]
            if candidate in shared_direct
        )
        return (
            shared_direct[selected_name],
            selected_name.rsplit(":", 1)[1],
        )

    read, read_center = selected_direct("read")
    key, key_center = selected_direct("key")
    value, value_center = selected_direct("value")
    ridge_scale = float(
        torch.trace(
            statistics["decay"]["calibration"].source.raw_gram()
        )
        / max(1, source_hidden)
    )
    low_rank_ridge = max(
        ridge_scale * 1e-3,
        torch.finfo(torch.float64).tiny,
    )

    def low_rank_down(
        name: str,
        *,
        native_rank: int,
        hidden_activation: str,
    ) -> Tensor:
        affine_weight, _ = _solve_affine_statistics(
            statistics[name]["calibration"],
            ridge=low_rank_ridge,
        )
        effective_rank = min(native_rank, source_hidden, target_width)
        right_gram = affine_weight.double().T @ affine_weight.double()
        _, eigenvectors = torch.linalg.eigh(right_gram)
        down = eigenvectors.flip(-1).T[:effective_rank].float().contiguous()
        if hidden_activation == "tanh":
            calibration_source = trace.mixer_input[
                :local_calibration_rows
            ].reshape(-1, source_hidden).float()
            maximum = (
                calibration_source @ down.T
            ).abs().max().clamp_min(1e-6)
            if config.reduce_max is not None:
                maximum = config.reduce_max(maximum)
            down = down * (0.25 / maximum)
        return down

    decay_rank = int(module.w_lora.lora[0].out_features)
    erase_rank = int(module.a_lora.lora[0].out_features)
    decay_down = low_rank_down(
        "decay",
        native_rank=decay_rank,
        hidden_activation="tanh",
    )
    erase_down = low_rank_down(
        "erase",
        native_rank=erase_rank,
        hidden_activation="identity",
    )
    decay_up_statistics = _AffineSufficientStatistics.zeros(
        decay_down.shape[0],
        target_width,
        device=device,
    )
    erase_up_statistics = _AffineSufficientStatistics.zeros(
        erase_down.shape[0],
        target_width,
        device=device,
    )
    for step in iter_streamed_native_two_state_steps(
        trace.query,
        trace.grouped_key,
        trace.grouped_value,
        affine,
        basis,
        native_dim=native_head_dim,
        dc_indices=dc_indices,
    ):
        if step.row_start >= local_calibration_rows:
            break
        row_stop = min(step.row_stop, local_calibration_rows)
        rows = slice(step.row_start, row_stop)
        source = trace.mixer_input[rows, step.time_index].float()
        local_rows = row_stop - step.row_start
        decay_probability = (
            -torch.log(
                step.decay[:local_rows]
                .unsqueeze(-1)
                .expand(-1, -1, native_head_dim)
                .clamp_min(1e-12)
            )
            / math.exp(-0.5)
        ).clamp(1e-6, 1 - 1e-6)
        decay_up_statistics.add(
            torch.tanh(source @ decay_down.T),
            torch.logit(decay_probability).flatten(1),
        )
        erase_up_statistics.add(
            source @ erase_down.T,
            torch.logit(
                step.erase[:local_rows].clamp(1e-6, 1 - 1e-6)
            ).flatten(1),
        )
    decay_up_statistics.reduce_(config.reduce_sum)
    erase_up_statistics.reduce_(config.reduce_sum)
    decay_up, decay_bias = _solve_affine_statistics(
        decay_up_statistics,
        ridge=low_rank_ridge,
    )
    erase_up, erase_bias = _solve_affine_statistics(
        erase_up_statistics,
        ridge=low_rank_ridge,
    )
    decay = _pad_low_rank_projection(
        LowRankProjection(
            down_weight=decay_down,
            up_weight=decay_up,
            bias=decay_bias,
            prediction=torch.empty(0, device=device),
            hidden_activation="tanh",
            output_bias=True,
        ),
        native_rank=decay_rank,
    )
    erase = _pad_low_rank_projection(
        LowRankProjection(
            down_weight=erase_down,
            up_weight=erase_up,
            bias=erase_bias,
            prediction=torch.empty(0, device=device),
            hidden_activation="identity",
            output_bias=True,
        ),
        native_rank=erase_rank,
    )
    gate_up = _select_bias_free_statistics(
        fit=gate_statistics["fit"],
        selection=gate_statistics["selection"],
        calibration=gate_statistics["calibration"],
        ridges=ridges,
    )
    gate = _pad_low_rank_projection(
        LowRankProjection(
            down_weight=gate_down,
            up_weight=gate_up.projection.weight,
            bias=torch.zeros(target_width, dtype=torch.float32, device=device),
            prediction=torch.empty(0, device=device),
            hidden_activation="sigmoid",
            output_bias=False,
        ),
        native_rank=gate_rank,
    )
    ridge_multiplier = {
        "r_proj": read.ridge,
        "k_proj": key.ridge,
        "v_proj": value.ridge,
        "g_lora_up": gate_up.ridge,
    }
    absolute_ridge = {
        "r_proj": read.absolute_ridge,
        "k_proj": key.absolute_ridge,
        "v_proj": value.absolute_ridge,
        "g_lora_up": gate_up.absolute_ridge,
        "low_rank": low_rank_ridge,
    }
    del (
        statistics,
        gate_statistics,
        decay_up_statistics,
        erase_up_statistics,
        gate_up,
    )

    output_statistics = split_statistics(
        target_width,
        module.hidden_size,
        include_development=True,
    )
    signal_metrics = {
        name: _MetricSums.zeros(device=device)
        for name in ("read", "decay", "erase", "key", "value", "gate")
    }
    current_chunk = None
    recurrent_state = None
    output_block_source: list[Tensor] = []
    output_block_target: list[Tensor] = []
    output_block_times: list[int] = []

    def flush_output_block() -> None:
        nonlocal output_block_source
        nonlocal output_block_target
        nonlocal output_block_times
        if current_chunk is None or not output_block_source:
            return
        row_start, row_stop = current_chunk
        source_block = torch.stack(output_block_source, dim=0)
        target_block = torch.stack(output_block_target, dim=0)
        for split_name, start, stop in split_segments(
            row_start,
            row_stop,
            include_development=True,
        ):
            split_source = source_block[:, start:stop]
            split_target = target_block[:, start:stop]
            if split_name == "development":
                supervised_indices = [
                    index
                    for index, time_index in enumerate(output_block_times)
                    if time_index >= config.supervised_token_start
                ]
                if not supervised_indices:
                    continue
                time_indices = torch.tensor(
                    supervised_indices,
                    dtype=torch.long,
                    device=device,
                )
                split_source = split_source.index_select(0, time_indices)
                split_target = split_target.index_select(0, time_indices)
            output_statistics[split_name].add(
                split_source.flatten(0, 1),
                split_target.flatten(0, 1),
            )
        output_block_source = []
        output_block_target = []
        output_block_times = []

    for step in iter_streamed_native_two_state_steps(
        trace.query,
        trace.grouped_key,
        trace.grouped_value,
        affine,
        basis,
        native_dim=native_head_dim,
        dc_indices=dc_indices,
    ):
        chunk = (step.row_start, step.row_stop)
        if chunk != current_chunk:
            flush_output_block()
            recurrent_state = torch.zeros(
                step.row_stop - step.row_start,
                native_heads,
                native_head_dim,
                native_head_dim,
                dtype=torch.float32,
                device=device,
            )
            current_chunk = chunk
        rows = slice(step.row_start, step.row_stop)
        source = trace.mixer_input[rows, step.time_index].float()
        positions = config.positions[
            rows,
            step.time_index : step.time_index + 1,
        ]
        predicted_read = _rotate_native(
            (source @ read.projection.weight.T)
            .reshape(-1, native_heads, native_head_dim)
            .unsqueeze(1),
            positions,
            source_head_dim=config.source_head_dim,
            rotary_dim=config.rotary_dim,
            rope_theta=config.rope_theta,
            inverse=False,
        ).squeeze(1)
        predicted_key = _rotate_native(
            (source @ key.projection.weight.T)
            .reshape(-1, native_heads, native_head_dim)
            .unsqueeze(1),
            positions,
            source_head_dim=config.source_head_dim,
            rotary_dim=config.rotary_dim,
            rope_theta=config.rope_theta,
            inverse=False,
        ).squeeze(1)
        predicted_value = (
            source @ value.projection.weight.T
        ).reshape(-1, native_heads, native_head_dim)
        decay_logits = (
            torch.tanh(source @ decay.down_weight.T)
            @ decay.up_weight.T
            + decay.bias
        ).reshape(-1, native_heads, native_head_dim)
        predicted_decay = torch.exp(
            -math.exp(-0.5) * torch.sigmoid(decay_logits)
        )
        predicted_erase = torch.sigmoid(
            (
                source @ erase.down_weight.T
            ) @ erase.up_weight.T
            + erase.bias
        ).reshape(-1, native_heads, native_head_dim)
        predicted_gate = (
            torch.sigmoid(source @ gate.down_weight.T)
            @ gate.up_weight.T
        ).reshape(-1, native_heads, native_head_dim)
        normalized_key = functional.normalize(predicted_key, dim=-1)
        recurrent_output, recurrent_state = rwkv7_step(
            recurrent_state,
            predicted_read,
            predicted_decay,
            predicted_key,
            predicted_value,
            -normalized_key,
            normalized_key * predicted_erase,
        )
        flat_recurrent = recurrent_output.flatten(1)
        normalized = functional.group_norm(
            flat_recurrent,
            num_groups=native_heads,
            weight=None,
            bias=None,
            eps=native_head_dim * 1e-5,
        )
        pre_output = normalized * predicted_gate.flatten(1)
        output_target = trace.mixer_output[rows, step.time_index].float()
        output_block_source.append(pre_output)
        output_block_target.append(output_target)
        output_block_times.append(step.time_index)
        if len(output_block_source) == block_size:
            flush_output_block()
        if (
            step.time_index >= config.supervised_token_start
            and step.row_stop > local_calibration_rows
        ):
            start = max(0, local_calibration_rows - step.row_start)
            development = slice(start, None)
            signal_metrics["read"].add(
                predicted_read[development],
                step.read[development],
            )
            signal_metrics["decay"].add(
                predicted_decay[development],
                step.decay[development].unsqueeze(-1).expand_as(
                    predicted_decay[development]
                ),
            )
            signal_metrics["erase"].add(
                predicted_erase[development],
                step.erase[development],
            )
            signal_metrics["key"].add(
                predicted_key[development],
                step.key[development],
            )
            signal_metrics["value"].add(
                predicted_value[development],
                step.value[development],
            )
            signal_metrics["gate"].add(
                predicted_gate[development],
                trace.gate[
                    rows,
                    step.time_index,
                ][development].reshape_as(predicted_gate[development]),
            )
    flush_output_block()

    for target_statistics in output_statistics.values():
        target_statistics.reduce_(config.reduce_sum)
    for sums in signal_metrics.values():
        sums.reduce_(config.reduce_sum)
    output_candidates = _select_shared_bias_free_statistics(
        {
            center: (
                output_statistics["fit"],
                output_statistics["selection"],
                None,
                None if center == "zero" else trace.output_weight,
            )
            for center in ("zero", "source-compatible")
        },
        ridges=ridges,
        selection_groups={
            "o_proj": ("zero", "source-compatible"),
        },
    )
    output_statistics["calibration"] = output_statistics["fit"]
    output_center = next(iter(output_candidates))
    output = output_candidates[output_center]
    ridge_multiplier["o_proj"] = output.ridge
    absolute_ridge["o_proj"] = output.absolute_ridge
    report = {
        "ridge_center": {
            "r_proj": read_center,
            "k_proj": key_center,
            "v_proj": value_center,
            "o_proj": output_center,
        },
        "native_rank": {
            "w_lora": decay_rank,
            "a_lora": erase_rank,
            "g_lora": gate_rank,
        },
        "effective_rank": {
            "w_lora": decay_down.shape[0],
            "a_lora": erase_down.shape[0],
            "g_lora": effective_gate_rank,
        },
        "ridge_multiplier": ridge_multiplier,
        "absolute_ridge": absolute_ridge,
        "pre_post_rope_read_projection": signal_metrics["read"].metrics(),
        "decay_signal": signal_metrics["decay"].metrics(),
        "erase_signal": signal_metrics["erase"].metrics(),
        "write_key_signal": signal_metrics["key"].metrics(),
        "write_value_signal": signal_metrics["value"].metrics(),
        "gate_signal": signal_metrics["gate"].metrics(),
        "complete_free_running_mixer": _statistics_metrics(
            output_statistics["development"],
            output.projection.weight,
        ),
        "solver_storage": "additive-sufficient-statistics",
    }
    return _NativeProjectionFit(
        read=read,
        key=key,
        value=value,
        decay=decay,
        erase=erase,
        gate=gate,
        output=output,
        report=report,
    )


def _fit_native_projection(
    *,
    module: ProjectionBoundaryRWKV7Attention,
    trace: GQANativeFitTrace,
    target_read: Tensor,
    native_transition,
    config: GQANativeFitConfig,
    native_head_dim: int,
    query_basis: Tensor,
    dc_indices: tuple[int, int],
) -> _NativeProjectionFit:
    calibration_rows = config.calibration_rows
    ridges = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    target_pre_rope_read = _rotate_native(
        target_read,
        config.positions,
        source_head_dim=config.source_head_dim,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        inverse=True,
    )
    target_pre_rope_key = _rotate_native(
        native_transition.key,
        config.positions,
        source_head_dim=config.source_head_dim,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        inverse=True,
    )
    source_heads = query_basis.shape[0]
    source_hidden = trace.mixer_input.shape[-1]
    query_weight = trace.query_weight.reshape(
        source_heads,
        config.source_head_dim * 2,
        source_hidden,
    )[:, : config.source_head_dim]
    read_prior = torch.zeros(
        source_heads,
        2,
        native_head_dim,
        source_hidden,
        dtype=torch.float32,
        device=trace.mixer_input.device,
    )
    for head in range(source_heads):
        for state_index, dc_index in enumerate(dc_indices):
            feature_indices = [
                index for index in range(native_head_dim) if index != dc_index
            ]
            read_prior[head, state_index, feature_indices] = (
                query_basis[head, state_index].T @ query_weight[head]
            )
    read_prior = read_prior.flatten(0, 2)
    kv_heads = trace.key_weight.shape[0] // config.source_head_dim
    group_repeat = source_heads // kv_heads

    def repeated_kv_prior(weight: Tensor) -> Tensor:
        return (
            weight.reshape(kv_heads, config.source_head_dim, source_hidden)
            .repeat_interleave(group_repeat, dim=0)
            .flatten(0, 1)
        )

    def select_direct(target: Tensor, prior: Tensor):
        candidates = {
            "zero": select_bias_free_projection(
                trace.mixer_input,
                target,
                calibration_batches=calibration_rows,
                ridges=ridges,
            ),
            "source-compatible": select_bias_free_projection(
                trace.mixer_input,
                target,
                calibration_batches=calibration_rows,
                ridges=ridges,
                prior_weight=prior,
            ),
        }
        selected_name = min(
            candidates,
            key=lambda name: (
                candidates[name].selection_nmse[candidates[name].ridge],
                name,
            ),
        )
        return candidates[selected_name], selected_name

    read, read_center = select_direct(target_pre_rope_read, read_prior)
    key, key_center = select_direct(
        target_pre_rope_key,
        repeated_kv_prior(trace.key_weight),
    )
    value, value_center = select_direct(
        native_transition.value,
        repeated_kv_prior(trace.value_weight),
    )
    calibration_source = trace.mixer_input[:calibration_rows].reshape(
        -1, source_hidden
    ).float()
    ridge_scale = float(
        calibration_source.square().sum() / max(1, calibration_source.shape[-1])
    )
    low_rank_ridge = max(
        ridge_scale * 1e-3,
        torch.finfo(torch.float32).tiny,
    )
    native_heads = native_transition.key.shape[2]
    decay_channels = native_transition.decay.unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        native_head_dim,
    )
    decay_probability = (
        -torch.log(decay_channels.clamp_min(1e-12)) / math.exp(-0.5)
    ).clamp(1e-6, 1 - 1e-6)
    decay_logits = torch.logit(decay_probability)
    erase_logits = torch.logit(
        native_transition.erase.clamp(1e-6, 1 - 1e-6)
    )
    decay_rank = int(module.w_lora.lora[0].out_features)
    erase_rank = int(module.a_lora.lora[0].out_features)
    gate_rank = int(module.g_lora.lora[0].out_features)
    decay = _fit_native_low_rank_projection(
        trace.mixer_input,
        decay_logits,
        calibration_batches=calibration_rows,
        native_rank=decay_rank,
        ridge=low_rank_ridge,
        hidden_activation="tanh",
    )
    erase = _fit_native_low_rank_projection(
        trace.mixer_input,
        erase_logits,
        calibration_batches=calibration_rows,
        native_rank=erase_rank,
        ridge=low_rank_ridge,
        hidden_activation="identity",
    )
    gate_target = trace.gate.reshape(
        *trace.gate.shape[:2],
        native_heads,
        native_head_dim,
    )
    source_gate_weight = trace.query_weight.reshape(
        source_heads,
        config.source_head_dim * 2,
        source_hidden,
    )[:, config.source_head_dim :].flatten(0, 1)
    effective_gate_rank = min(
        gate_rank,
        source_gate_weight.shape[0],
        source_hidden,
    )
    gate_indices = torch.linspace(
        0,
        source_gate_weight.shape[0] - 1,
        effective_gate_rank,
        dtype=torch.float64,
        device=trace.mixer_input.device,
    ).round().to(torch.long)
    gate_down = source_gate_weight.index_select(0, gate_indices)
    gate_features = torch.sigmoid(trace.mixer_input @ gate_down.T)
    gate_up = select_bias_free_projection(
        gate_features,
        gate_target,
        calibration_batches=calibration_rows,
        ridges=ridges,
    )
    gate = _pad_low_rank_projection(
        LowRankProjection(
            down_weight=gate_down,
            up_weight=gate_up.projection.weight,
            bias=torch.zeros(
                native_heads * native_head_dim,
                dtype=torch.float32,
                device=trace.mixer_input.device,
            ),
            prediction=gate_up.projection.prediction,
            hidden_activation="sigmoid",
            output_bias=False,
        ),
        native_rank=gate_rank,
    )

    predicted_read = _rotate_native(
        read.projection.prediction,
        config.positions,
        source_head_dim=config.source_head_dim,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        inverse=False,
    )
    predicted_key = _rotate_native(
        key.projection.prediction,
        config.positions,
        source_head_dim=config.source_head_dim,
        rotary_dim=config.rotary_dim,
        rope_theta=config.rope_theta,
        inverse=False,
    )
    predicted_decay = torch.exp(
        -math.exp(-0.5) * torch.sigmoid(decay.prediction)
    )
    predicted_erase = torch.sigmoid(erase.prediction)
    rollout_output = native_signal_output_rollout(
        predicted_read,
        predicted_decay,
        predicted_key,
        value.projection.prediction,
        predicted_erase,
    )
    flat_recurrent = rollout_output.flatten(2)
    normalized = functional.group_norm(
        flat_recurrent.reshape(-1, flat_recurrent.shape[-1]),
        num_groups=native_heads,
        weight=None,
        bias=None,
        eps=native_head_dim * 1e-5,
    ).reshape_as(flat_recurrent)
    pre_output = normalized * gate.prediction.flatten(2)
    output_candidates = {
        "zero": select_bias_free_projection(
            pre_output,
            trace.mixer_output,
            calibration_batches=calibration_rows,
            ridges=ridges,
        ),
        "source-compatible": select_bias_free_projection(
            pre_output,
            trace.mixer_output,
            calibration_batches=calibration_rows,
            ridges=ridges,
            prior_weight=trace.output_weight,
        ),
    }
    output_center = min(
        output_candidates,
        key=lambda name: (
            output_candidates[name].selection_nmse[
                output_candidates[name].ridge
            ],
            name,
        ),
    )
    output = output_candidates[output_center]
    development = slice(calibration_rows, None)
    supervised = slice(config.supervised_token_start, None)
    report = {
        "ridge_center": {
            "r_proj": read_center,
            "k_proj": key_center,
            "v_proj": value_center,
            "o_proj": output_center,
        },
        "native_rank": {
            "w_lora": decay_rank,
            "a_lora": erase_rank,
            "g_lora": gate_rank,
        },
        "effective_rank": {
            "w_lora": min(
                decay_rank,
                source_hidden,
                math.prod(decay_logits.shape[2:]),
            ),
            "a_lora": min(
                erase_rank,
                source_hidden,
                math.prod(erase_logits.shape[2:]),
            ),
            "g_lora": effective_gate_rank,
        },
        "ridge_multiplier": {
            "r_proj": read.ridge,
            "k_proj": key.ridge,
            "v_proj": value.ridge,
            "g_lora_up": gate_up.ridge,
            "o_proj": output.ridge,
        },
        "absolute_ridge": {
            "r_proj": read.absolute_ridge,
            "k_proj": key.absolute_ridge,
            "v_proj": value.absolute_ridge,
            "g_lora_up": gate_up.absolute_ridge,
            "o_proj": output.absolute_ridge,
            "low_rank": low_rank_ridge,
        },
        "pre_post_rope_read_projection": tensor_metrics(
            predicted_read[development, supervised],
            target_read[development, supervised],
        ),
        "decay_signal": tensor_metrics(
            predicted_decay[development, supervised],
            native_transition.decay[development, supervised]
            .unsqueeze(-1)
            .expand_as(predicted_decay[development, supervised]),
        ),
        "erase_signal": tensor_metrics(
            predicted_erase[development, supervised],
            native_transition.erase[development, supervised],
        ),
        "write_key_signal": tensor_metrics(
            predicted_key[development, supervised],
            native_transition.key[development, supervised],
        ),
        "write_value_signal": tensor_metrics(
            value.projection.prediction[development, supervised],
            native_transition.value[development, supervised],
        ),
        "gate_signal": tensor_metrics(
            gate.prediction[development, supervised],
            gate_target[development, supervised],
        ),
        "complete_free_running_mixer": tensor_metrics(
            output.projection.prediction[development, supervised],
            trace.mixer_output[development, supervised],
        ),
    }
    return _NativeProjectionFit(
        read=read,
        key=key,
        value=value,
        decay=decay,
        erase=erase,
        gate=gate,
        output=output,
        report=report,
    )


def _fit_native_low_rank_projection(
    source: Tensor,
    target: Tensor,
    *,
    calibration_batches: int,
    native_rank: int,
    ridge: float,
    hidden_activation: str,
) -> LowRankProjection:
    effective_rank = min(
        native_rank,
        source.shape[-1],
        math.prod(target.shape[2:]),
    )
    fitted = fit_low_rank_projection(
        source,
        target,
        calibration_batches=calibration_batches,
        rank=effective_rank,
        ridge=ridge,
        hidden_activation=hidden_activation,
    )
    return _pad_low_rank_projection(fitted, native_rank=native_rank)


def _pad_low_rank_projection(
    projection: LowRankProjection,
    *,
    native_rank: int,
) -> LowRankProjection:
    fitted_rank = projection.down_weight.shape[0]
    if native_rank < fitted_rank:
        raise ValueError("native low-rank width is smaller than fitted rank")
    if native_rank == fitted_rank:
        return projection
    down_weight = functional.pad(
        projection.down_weight,
        (0, 0, 0, native_rank - fitted_rank),
    )
    up_weight = functional.pad(
        projection.up_weight,
        (0, native_rank - fitted_rank),
    )
    return LowRankProjection(
        down_weight=down_weight,
        up_weight=up_weight,
        bias=projection.bias,
        prediction=projection.prediction,
        hidden_activation=projection.hidden_activation,
        output_bias=projection.output_bias,
    )


def _native_parameter_set(
    module: ProjectionBoundaryRWKV7Attention,
    fit: _NativeProjectionFit,
    *,
    source_value_weight: Tensor,
) -> dict[str, Tensor]:
    parameters = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in module.named_parameters()
    }
    parameters["k_k"].fill_(1)
    parameters["g_norm.weight"].fill_(1)
    parameters["r_proj.weight"].copy_(fit.read.projection.weight)
    parameters["k_proj.weight"].copy_(fit.key.projection.weight)
    parameters["v_proj.weight"].copy_(fit.value.projection.weight)
    parameters["o_proj.weight"].copy_(fit.output.projection.weight)
    parameters["w_lora.lora.0.weight"].copy_(fit.decay.down_weight)
    parameters["w_lora.lora.2.weight"].copy_(fit.decay.up_weight)
    parameters["w_lora.lora.2.bias"].copy_(fit.decay.bias)
    parameters["a_lora.lora.0.weight"].copy_(fit.erase.down_weight)
    parameters["a_lora.lora.2.weight"].copy_(fit.erase.up_weight)
    parameters["a_lora.lora.2.bias"].copy_(fit.erase.bias)
    parameters["g_lora.lora.0.weight"].copy_(fit.gate.down_weight)
    parameters["g_lora.lora.2.weight"].copy_(fit.gate.up_weight)
    if module.layer_idx:
        value_rank = parameters["v_lora.lora.0.weight"].shape[0]
        basis_indices = torch.linspace(
            0,
            source_value_weight.shape[0] - 1,
            value_rank,
            dtype=torch.float64,
            device=source_value_weight.device,
        ).round().to(torch.long)
        parameters["v_lora.lora.0.weight"].copy_(
            source_value_weight.index_select(0, basis_indices)
        )
        parameters["v_lora.lora.2.bias"].fill_(
            torch.logit(
                torch.tensor(
                    torch.finfo(torch.bfloat16).eps**2,
                    device=source_value_weight.device,
                )
            )
        )
    return parameters
