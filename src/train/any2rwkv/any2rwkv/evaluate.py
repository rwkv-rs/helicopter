from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
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


@dataclass(frozen=True)
class QualityThresholdProfile:
    profile_id: str
    profile_sha256: str
    calibration_artifact_sha256: str
    values: Mapping[str, Mapping[str, float]]
    evidence_verified: bool = False
    calibration_student_sha256s: tuple[str, ...] = ()


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


_REQUIRED_THRESHOLDS = {
    "P1": {
        "smoke_pass_rate_min", "ppl_ratio_max", "mean_token_kl_max",
        "layer_mse_median_max", "layer_cosine_median_min", "layer_cosine_min",
    },
    "P2": {
        "smoke_pass_rate_min", "ppl_ratio_max", "mean_token_kl_max",
        "layer_cosine_median_min", "layer_cosine_p05_min",
        "layer_mse_median_max", "layer_mse_p95_max",
        "ruler_ci_lower_ratio_min", "ruler_bucket_min_ratio_min",
        "downstream_ci_lower_ratio_min", "downstream_max_drop_points_max",
    },
}


def derive_empirical_quality_thresholds(
    per_seed_metrics: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    """Derive a conservative envelope from independently eligible pilot runs.

    The rule is deliberately parameter-free: a lower-bound metric uses the
    worst (minimum) eligible seed and an upper-bound metric uses the worst
    (maximum) eligible seed. Eligibility is structural (P0, convergence and
    mapped-vs-baseline superiority), not another hand-written quality margin.
    Candidate evaluation is a separate run, so the envelope cannot be adjusted
    after observing the candidate.
    """
    if len(per_seed_metrics) < 3:
        raise ValueError("quality threshold derivation requires at least three seeds")
    derived: dict[str, dict[str, float]] = {}
    for level, required in _REQUIRED_THRESHOLDS.items():
        level_values: dict[str, list[float]] = {name: [] for name in required}
        for row in per_seed_metrics:
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError("calibration seed row has no metrics mapping")
            level_metrics = metrics.get(level)
            if not isinstance(level_metrics, Mapping) or set(level_metrics) != required:
                raise ValueError(f"calibration seed row has invalid {level} metrics")
            for name in required:
                value = float(level_metrics[name])
                if not math.isfinite(value):
                    raise ValueError(f"calibration metric is not finite: {level}.{name}")
                level_values[name].append(value)
        derived[level] = {
            name: (min(values) if name.endswith("_min") else max(values))
            for name, values in level_values.items()
        }
    return derived


def derive_empirical_quality_threshold_provenance(
    per_seed_metrics: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    """Record the observations and run identities supporting every threshold."""
    thresholds = derive_empirical_quality_thresholds(per_seed_metrics)
    provenance: dict[str, dict[str, dict[str, object]]] = {}
    for level, names in _REQUIRED_THRESHOLDS.items():
        provenance[level] = {}
        for name in names:
            observations = [
                {
                    "seed": int(row["seed"]),
                    "run_id": str(row["run_id"]),
                    "seed_result_sha256": str(row["seed_result_sha256"]),
                    "value": float(row["metrics"][level][name]),
                }
                for row in per_seed_metrics
            ]
            limit = thresholds[level][name]
            provenance[level][name] = {
                "numeric_evidence_type": "independent-multi-seed-calibration",
                "direction": "lower-bound" if name.endswith("_min") else "upper-bound",
                "rule": "worst-eligible-seed-envelope",
                "threshold": limit,
                "limiting_run_ids": [
                    row["run_id"] for row in observations if row["value"] == limit
                ],
                "observations": observations,
            }
    return provenance


def _validate_embedded_eligibility(row: Mapping[str, object], derivation_protocol_sha: str) -> bool:
    evidence = row.get("eligibility_evidence")
    if not isinstance(evidence, Mapping):
        return False
    baselines = evidence.get("migration_baselines")
    if not isinstance(baselines, Mapping) or not isinstance(baselines.get("values"), Mapping):
        return False
    values = baselines["values"]
    if set(values) != {"mapped", "random", "naive"}:
        return False
    mapped, random, naive = (float(values[name]) for name in ("mapped", "random", "naive"))
    direction = baselines.get("direction")
    superior = (
        mapped < random and mapped < naive
        if direction == "lower"
        else mapped > random and mapped > naive if direction == "higher" else False
    )
    return bool(
        evidence.get("schema_version") == 1
        and evidence.get("status") == "complete"
        and evidence.get("protocol_sha256") == derivation_protocol_sha
        and evidence.get("run_id") == row.get("run_id")
        and evidence.get("p0_failures") == []
        and float(evidence.get("converged_layer_fraction", -1.0)) == 1.0
        and evidence.get("baseline_curve_sha256") == row.get("baseline_curve_sha256")
        and len(str(row.get("eligibility_evidence_sha256", ""))) == 64
        and superior
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_evidence_reference(
    row: Mapping[str, object], *, name: str, artifact_root: Path
) -> bool:
    reference = row.get(f"{name}_reference")
    embedded = row.get(name)
    expected_sha = str(row.get(f"{name}_sha256", ""))
    if not isinstance(reference, Mapping) or not isinstance(embedded, Mapping):
        return False
    evidence_path = Path(str(reference.get("artifact", "")))
    if not evidence_path.is_absolute():
        evidence_path = (artifact_root / evidence_path).resolve()
    if (
        not evidence_path.is_file()
        or str(reference.get("artifact_sha256", "")) != expected_sha
        or _sha256(evidence_path) != expected_sha
    ):
        return False
    try:
        return json.loads(evidence_path.read_text(encoding="utf-8")) == embedded
    except (OSError, json.JSONDecodeError):
        return False


def _verify_top_evidence_reference(
    evidence: Mapping[str, object], *, name: str, artifact_root: Path
) -> bool:
    reference = evidence.get(f"{name}_reference")
    embedded = evidence.get(name)
    if not isinstance(reference, Mapping) or not isinstance(embedded, Mapping):
        return False
    evidence_path = Path(str(reference.get("artifact", "")))
    if not evidence_path.is_absolute():
        evidence_path = (artifact_root / evidence_path).resolve()
    expected_sha = str(reference.get("artifact_sha256", ""))
    if not evidence_path.is_file() or _sha256(evidence_path) != expected_sha:
        return False
    try:
        return json.loads(evidence_path.read_text(encoding="utf-8")) == embedded
    except (OSError, json.JSONDecodeError):
        return False


def read_quality_threshold_profile(path: Path) -> QualityThresholdProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration = payload.get("calibration")
    values = payload.get("thresholds")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "calibrated"
        or not isinstance(calibration, dict)
        or not isinstance(values, dict)
    ):
        raise ValueError("quality threshold profile is not calibrated")
    artifact = Path(str(calibration.get("artifact", "")))
    artifact = artifact if artifact.is_absolute() else (path.parent / artifact).resolve()
    expected_artifact_sha = str(calibration.get("artifact_sha256", ""))
    if not artifact.is_file() or _sha256(artifact) != expected_artifact_sha:
        raise ValueError("quality threshold calibration artifact SHA-256 mismatch")
    evidence = json.loads(artifact.read_text(encoding="utf-8"))
    seeds = evidence.get("seeds")
    bindings = evidence.get("bindings")
    derivation = evidence.get("threshold_derivation")
    per_seed_metrics = evidence.get("per_seed_metrics")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("status") != "complete"
        or evidence.get("protocol") != "teacher-paired"
        or evidence.get("precision") != "bf16"
        or not isinstance(seeds, list)
        or len(seeds) < 3
        or len(set(seeds)) != len(seeds)
        or not isinstance(bindings, dict)
        or any(
            not str(bindings.get(name, ""))
            for name in (
                "teacher_checkpoint_sha256",
                "dataset_sha256",
                "code_commit",
                "recipe_id",
                "metric_definition",
                "distillation_plan_sha256",
                "training_control_evidence_sha256",
            )
        )
        or not isinstance(per_seed_metrics, list)
        or len(per_seed_metrics) != len(seeds)
        or {row.get("seed") for row in per_seed_metrics if isinstance(row, dict)}
        != set(seeds)
        or not isinstance(derivation, dict)
        or any(
            not isinstance(row, dict)
            or not str(row.get("run_id", ""))
            or len(str(row.get("sample_metrics_sha256", ""))) != 64
            or len(str(row.get("baseline_curve_sha256", ""))) != 64
            or len(str(row.get("student_checkpoint_sha256", ""))) != 64
            or len(str(row.get("seed_result_sha256", ""))) != 64
            or not _validate_embedded_eligibility(
                row, str(derivation.get("protocol_sha256", ""))
            )
            for row in per_seed_metrics
        )
        or derivation.get("method") != "multi-seed-empirical-calibration"
        or derivation.get("version") != 1
        or derivation.get("rule") != "worst-eligible-seed-envelope"
        or derivation.get("candidate_evaluation_is_independent") is not True
        or int(derivation.get("bootstrap_samples", 0)) < 10_000
        or int(derivation.get("bootstrap_seed", -1)) < 0
        or float(derivation.get("confidence", 0.0)) != 0.95
        or len(str(derivation.get("protocol_sha256", ""))) != 64
        or derivation.get("source_run_ids")
        != [row.get("run_id") for row in per_seed_metrics]
    ):
        raise ValueError(
            "quality threshold calibration must be complete, multi-seed, "
            "teacher-paired, independently evaluated, reproducibly derived, "
            "and hash-bound"
        )
    from .core.quality_calibration import derive_quality_metrics_from_samples
    from .core.training_control_calibration import training_control_fingerprint
    from .core.dataset_evidence import read_quality_dataset_evidence

    if not all(
        _verify_top_evidence_reference(evidence, name=name, artifact_root=artifact.parent)
        for name in ("training_control_evidence", "distillation_plan", "dataset_manifest")
    ):
        raise ValueError("quality threshold top-level evidence reference is invalid")
    read_quality_dataset_evidence(
        evidence["dataset_manifest_reference"],
        owner_path=artifact,
        expected_sha256=str(bindings["dataset_sha256"]),
    )
    control = evidence["training_control_evidence"]
    plan = evidence["distillation_plan"]
    shared = ("teacher_checkpoint_sha256", "dataset_sha256", "code_commit", "recipe_id")
    if (
        not isinstance(control, Mapping)
        or not isinstance(plan, Mapping)
        or not isinstance(control.get("bindings"), Mapping)
        or control.get("precision") != "bf16"
        or any(control.get("bindings", {}).get(key) != bindings.get(key) for key in shared)
        or control.get("selected_control_fingerprint") != training_control_fingerprint(dict(plan))
    ):
        raise ValueError("quality threshold training-control evidence is misbound")

    for row in per_seed_metrics:
        if not all(
            _verify_evidence_reference(row, name=name, artifact_root=artifact.parent)
            for name in ("sample_metrics", "baseline_curve", "eligibility_evidence")
        ):
            raise ValueError("quality threshold evidence artifact reference is invalid")
        recomputed = derive_quality_metrics_from_samples(
            dict(row["sample_metrics"]),
            bootstrap_samples=int(derivation["bootstrap_samples"]),
            bootstrap_seed=int(derivation["bootstrap_seed"]),
        )
        if row.get("metrics") != recomputed:
            raise ValueError("quality threshold sample metrics do not match raw evidence")
    try:
        machine_derived = derive_empirical_quality_thresholds(per_seed_metrics)
        machine_provenance = derive_empirical_quality_threshold_provenance(
            per_seed_metrics
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"quality threshold derivation failed: {error}") from error
    if (
        derivation.get("derived_thresholds") != machine_derived
        or derivation.get("threshold_provenance") != machine_provenance
        or values != machine_derived
    ):
        raise ValueError(
            "quality threshold profile does not match the machine-derived "
            "worst-eligible-seed envelope"
        )
    normalized: dict[str, dict[str, float]] = {}
    for level, required in _REQUIRED_THRESHOLDS.items():
        row = values.get(level)
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"quality threshold profile has invalid {level} keys")
        normalized[level] = {name: float(row[name]) for name in required}
    return QualityThresholdProfile(
        profile_id=str(payload.get("profile_id", "")),
        profile_sha256=_sha256(path),
        calibration_artifact_sha256=expected_artifact_sha,
        values=normalized,
        evidence_verified=True,
        calibration_student_sha256s=tuple(
            str(row["student_checkpoint_sha256"]) for row in per_seed_metrics
        ),
    )


def quality_gate(
    metrics: QualityMetrics,
    *,
    level: str,
    thresholds: QualityThresholdProfile | None,
) -> GateResult:
    if thresholds is None or not thresholds.evidence_verified:
        return GateResult(level, False, ("uncalibrated-threshold-profile",))
    if level not in _REQUIRED_THRESHOLDS:
        raise ValueError(f"unknown quality gate: {level}")
    limit = thresholds.values[level]
    failures: list[str] = []
    if metrics.smoke_pass_rate < limit["smoke_pass_rate_min"]:
        failures.append("smoke_pass_rate")
    if level == "P1":
        if metrics.ppl_ratio > limit["ppl_ratio_max"]:
            failures.append("ppl_ratio")
        if metrics.mean_token_kl > limit["mean_token_kl_max"]:
            failures.append("mean_token_kl")
        if median(metrics.layer_normalized_mse) > limit["layer_mse_median_max"]:
            failures.append("layer_mse_median")
        if (
            median(metrics.layer_cosines) < limit["layer_cosine_median_min"]
            or min(metrics.layer_cosines) < limit["layer_cosine_min"]
        ):
            failures.append("layer_cosine_floor")
    elif level == "P2":
        if (
            metrics.ppl_ratio > limit["ppl_ratio_max"]
            or metrics.mean_token_kl > limit["mean_token_kl_max"]
        ):
            failures.append("ppl_or_kl")
        if (
            median(metrics.layer_cosines) < limit["layer_cosine_median_min"]
            or percentile(metrics.layer_cosines, 0.05) < limit["layer_cosine_p05_min"]
        ):
            failures.append("layer_cosine_quantiles")
        if (
            median(metrics.layer_normalized_mse) > limit["layer_mse_median_max"]
            or percentile(metrics.layer_normalized_mse, 0.95) > limit["layer_mse_p95_max"]
        ):
            failures.append("layer_mse_quantiles")
        if (
            metrics.ruler_ci_lower_ratio < limit["ruler_ci_lower_ratio_min"]
            or metrics.ruler_bucket_min_ratio < limit["ruler_bucket_min_ratio_min"]
        ):
            failures.append("ruler")
        if (
            metrics.downstream_ci_lower_ratio < limit["downstream_ci_lower_ratio_min"]
            or metrics.downstream_max_drop_points > limit["downstream_max_drop_points_max"]
        ):
            failures.append("downstream")
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
        raise ValueError(f"quality splits overlap: {sorted(duplicates)}")
    required = {"distill_train", "validation", "ruler", "downstream", "smoke"}
    missing = required - splits.keys()
    if missing:
        raise ValueError(f"missing frozen splits: {sorted(missing)}")
    return {name: len(values) for name, values in splits.items()}
