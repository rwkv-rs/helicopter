from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class QualityMetrics:
    ppl_ratio: float
    mean_token_kl: float
    layer_cosines: tuple[float, ...]
    layer_normalized_mse: tuple[float, ...]
    smoke_pass_rate: float
    ruler_ci_lower_ratio: float
    ruler_bucket_min_ratio: float
    downstream_ci_lower_ratio: float
    downstream_max_drop_points: float


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    failures: tuple[str, ...]


P0_REQUIRED = (
    "canonical_state",
    "mapping_coverage",
    "kernel_oracle",
    "gdn_oracle",
    "full_attention_fixture",
    "gqa_fixture",
    "loss_bridge",
    "active_layer_invariant",
    "resume_parity",
    "hf_roundtrip",
)


def p0_gate(evidence: Mapping[str, bool]) -> GateResult:
    failures = tuple(name for name in P0_REQUIRED if evidence.get(name) is not True)
    return GateResult("P0", not failures, failures)


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def quality_gate(metrics: QualityMetrics, *, level: str) -> GateResult:
    failures: list[str] = []
    if metrics.smoke_pass_rate < 0.90:
        failures.append("smoke_pass_rate<0.90")
    if level == "P1":
        if metrics.ppl_ratio > 1.5:
            failures.append("ppl_ratio>1.5")
        if metrics.mean_token_kl > 1.0:
            failures.append("mean_token_kl>1.0")
        if median(metrics.layer_normalized_mse) > 0.25:
            failures.append("layer_mse_median>0.25")
        if median(metrics.layer_cosines) < 0.85 or min(metrics.layer_cosines) < 0.70:
            failures.append("layer_cosine_floor")
    elif level == "P2":
        if metrics.ppl_ratio > 1.20 or metrics.mean_token_kl > 0.25:
            failures.append("ppl_or_kl")
        if median(metrics.layer_cosines) < 0.95 or percentile(metrics.layer_cosines, 0.05) < 0.90:
            failures.append("layer_cosine_quantiles")
        if median(metrics.layer_normalized_mse) > 0.10 or percentile(metrics.layer_normalized_mse, 0.95) > 0.25:
            failures.append("layer_mse_quantiles")
        if metrics.ruler_ci_lower_ratio < 0.85 or metrics.ruler_bucket_min_ratio < 0.80:
            failures.append("ruler")
        if metrics.downstream_ci_lower_ratio < 0.90 or metrics.downstream_max_drop_points > 5:
            failures.append("downstream")
    else:
        raise ValueError(f"unknown quality gate: {level}")
    return GateResult(level, not failures, tuple(failures))


def migration_gate(baselines: Mapping[str, float]) -> GateResult:
    required = {"random", "naive_copy", "mapped", "activation_fitted", "layerwise_distilled"}
    missing = sorted(required - baselines.keys())
    failures = [f"missing:{name}" for name in missing]
    if not missing and not (baselines["mapped"] < baselines["random"] and baselines["mapped"] < baselines["naive_copy"]):
        failures.append("mapped initialization does not beat random and naive_copy")
    if not missing and baselines["layerwise_distilled"] > min(baselines["random"], baselines["naive_copy"]):
        failures.append("distilled result does not beat fixed-token baselines")
    return GateResult("migration", not failures, tuple(failures))


def paired_bootstrap_ratio_ci(
    student: Sequence[float],
    teacher: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 20260714,
) -> tuple[float, float]:
    if len(student) != len(teacher) or not student:
        raise ValueError("paired bootstrap requires equal non-empty sample arrays")
    student_tensor = torch.tensor(student, dtype=torch.float64)
    teacher_tensor = torch.tensor(teacher, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    ratios: list[float] = []
    for _ in range(samples):
        indices = torch.randint(len(student), (len(student),), generator=generator)
        denominator = teacher_tensor[indices].mean()
        ratios.append(float(student_tensor[indices].mean() / torch.clamp(denominator, min=1e-30)))
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def validate_disjoint_splits(splits: Mapping[str, Sequence[str]]) -> dict[str, int]:
    owners: dict[str, str] = {}
    duplicates: list[str] = []
    for split, sample_ids in splits.items():
        for sample_id in sample_ids:
            previous = owners.setdefault(sample_id, split)
            if previous != split:
                duplicates.append(f"{sample_id}:{previous}:{split}")
    if duplicates:
        raise ValueError(f"quality/calibration splits overlap: {sorted(duplicates)}")
    required = {"distill_train", "nvfp4_calibration", "validation", "ruler", "downstream", "smoke"}
    missing = required - splits.keys()
    if missing:
        raise ValueError(f"missing frozen splits: {sorted(missing)}")
    return {name: len(values) for name, values in splits.items()}
