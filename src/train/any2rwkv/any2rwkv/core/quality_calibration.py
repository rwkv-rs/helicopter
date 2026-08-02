from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from ..artifacts import file_sha256, write_json
from ..evaluate import (
    derive_empirical_quality_threshold_provenance,
    derive_empirical_quality_thresholds,
    paired_bootstrap_ratio_ci,
    percentile,
)
from .training_control_calibration import (
    training_control_fingerprint,
    validate_training_control_artifact,
)
from .dataset_evidence import read_quality_dataset_evidence


_BINDING_KEYS = (
    "teacher_checkpoint_sha256",
    "dataset_sha256",
    "code_commit",
    "recipe_id",
    "metric_definition",
    "distillation_plan_sha256",
    "training_control_evidence_sha256",
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"quality calibration input is not an object: {path}")
    return payload


def _read_eligibility_evidence(
    reference: object,
    *,
    result_path: Path,
    protocol_sha: str,
    run_id: str,
    baseline_curve_sha: str,
    baseline_curve: dict[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    if not isinstance(reference, dict):
        raise ValueError("calibration result has no eligibility evidence reference")
    evidence_path = Path(str(reference.get("artifact", "")))
    if not evidence_path.is_absolute():
        evidence_path = (result_path.parent / evidence_path).resolve()
    expected_sha = str(reference.get("artifact_sha256", ""))
    if not evidence_path.is_file() or file_sha256(evidence_path) != expected_sha:
        raise ValueError("calibration eligibility evidence SHA-256 mismatch")
    evidence = _read_object(evidence_path)
    baselines = evidence.get("migration_baselines")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("status") != "complete"
        or evidence.get("protocol_sha256") != protocol_sha
        or evidence.get("run_id") != run_id
        or evidence.get("p0_failures") != []
        or float(evidence.get("converged_layer_fraction", -1.0)) != 1.0
        or evidence.get("baseline_curve_sha256") != baseline_curve_sha
        or not isinstance(baselines, dict)
        or baselines.get("direction") not in {"lower", "higher"}
        or set(baselines.get("values", {})) != {"mapped", "random", "naive"}
    ):
        raise ValueError("calibration eligibility evidence is incomplete")
    values = {name: float(value) for name, value in baselines["values"].items()}
    if (
        baseline_curve.get("metric") != baselines.get("metric")
        or baseline_curve.get("direction") != baselines.get("direction")
        or baseline_curve.get("values") != baselines.get("values")
    ):
        raise ValueError("eligibility evidence does not match the baseline curve")
    if baselines["direction"] == "lower":
        superior = values["mapped"] < values["random"] and values["mapped"] < values["naive"]
    else:
        superior = values["mapped"] > values["random"] and values["mapped"] > values["naive"]
    if not superior:
        raise ValueError("mapped initialization does not beat random and naive baselines")
    return evidence, expected_sha, evidence_path


def _read_bound_artifact(
    reference: object,
    *,
    owner_path: Path,
    name: str,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    if not isinstance(reference, dict):
        raise ValueError(f"calibration result has no {name} artifact reference")
    artifact_path = Path(str(reference.get("artifact", "")))
    if not artifact_path.is_absolute():
        artifact_path = (owner_path.parent / artifact_path).resolve()
    expected_sha = str(reference.get("artifact_sha256", ""))
    if not artifact_path.is_file() or file_sha256(artifact_path) != expected_sha:
        raise ValueError(f"{name} artifact SHA-256 mismatch")
    artifact = _read_object(artifact_path)
    if (
        artifact.get("schema_version") != 1
        or artifact.get("status") != "complete"
        or any(artifact.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(f"{name} artifact is incomplete or misbound")
    return artifact, expected_sha, artifact_path


def _score_pairs(rows: object, *, name: str) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"sample metrics has no {name} rows")
    teacher: list[float] = []
    student: list[float] = []
    checked: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("sample_id", "")):
            raise ValueError(f"sample metrics has an invalid {name} row")
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"sample metrics has duplicate {name} sample ids")
        sample_ids.add(sample_id)
        teacher_score = float(row.get("teacher_score", float("nan")))
        student_score = float(row.get("student_score", float("nan")))
        if not math.isfinite(teacher_score) or not math.isfinite(student_score):
            raise ValueError(f"sample metrics has non-finite {name} scores")
        teacher.append(teacher_score)
        student.append(student_score)
        checked.append(row)
    return teacher, student, checked


def _group_ratio_min(rows: list[dict[str, Any]], key: str) -> float:
    grouped: dict[str, tuple[float, float]] = {}
    for row in rows:
        group = str(row.get(key, ""))
        if not group:
            raise ValueError(f"sample metrics row has no {key}")
        teacher, student = grouped.get(group, (0.0, 0.0))
        grouped[group] = (teacher + float(row["teacher_score"]), student + float(row["student_score"]))
    ratios = [student / teacher for teacher, student in grouped.values() if teacher > 0]
    if len(ratios) != len(grouped):
        raise ValueError(f"sample metrics contains a zero-teacher {key} group")
    return min(ratios)


def _max_group_drop_points(rows: list[dict[str, Any]], key: str) -> float:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        group = str(row.get(key, ""))
        if not group:
            raise ValueError(f"sample metrics row has no {key}")
        grouped.setdefault(group, []).append(
            (float(row["teacher_score"]), float(row["student_score"]))
        )
    return max(
        100.0
        * (
            sum(teacher for teacher, _ in values) / len(values)
            - sum(student for _, student in values) / len(values)
        )
        for values in grouped.values()
    )


def derive_quality_metrics_from_samples(
    artifact: dict[str, Any], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, dict[str, float]]:
    lm_rows = artifact.get("warmed_lm_rows")
    layer_rows = artifact.get("layer_rows")
    smoke_rows = artifact.get("smoke_rows")
    if not isinstance(lm_rows, list) or not lm_rows or not isinstance(layer_rows, list) or not layer_rows:
        raise ValueError("sample metrics lacks LM or layer observations")
    token_count = sum(int(row.get("token_count", 0)) for row in lm_rows if isinstance(row, dict))
    if token_count <= 0 or any(not isinstance(row, dict) or int(row.get("token_count", 0)) <= 0 for row in lm_rows):
        raise ValueError("sample metrics has invalid LM token counts")
    teacher_nll = sum(float(row.get("teacher_nll_sum", float("nan"))) for row in lm_rows)
    student_nll = sum(float(row.get("student_nll_sum", float("nan"))) for row in lm_rows)
    token_kl = sum(float(row.get("token_kl_sum", float("nan"))) for row in lm_rows)
    if not all(math.isfinite(value) for value in (teacher_nll, student_nll, token_kl)):
        raise ValueError("sample metrics has non-finite LM observations")
    cosines = [float(row.get("cosine", float("nan"))) for row in layer_rows if isinstance(row, dict)]
    mses = [float(row.get("normalized_mse", float("nan"))) for row in layer_rows if isinstance(row, dict)]
    layer_ids = [int(row.get("layer_index", -1)) for row in layer_rows if isinstance(row, dict)]
    if len(cosines) != len(layer_rows) or len(set(layer_ids)) != len(layer_rows) or not all(
        math.isfinite(value) for value in cosines + mses
    ):
        raise ValueError("sample metrics has invalid layer observations")
    if not isinstance(smoke_rows, list) or not smoke_rows or any(
        not isinstance(row, dict) or type(row.get("passed")) is not bool for row in smoke_rows
    ):
        raise ValueError("sample metrics has invalid smoke observations")
    smoke_ids = [str(row.get("sample_id", "")) for row in smoke_rows]
    if "" in smoke_ids or len(set(smoke_ids)) != len(smoke_ids):
        raise ValueError("sample metrics has invalid smoke sample ids")
    smoke_rate = sum(int(row["passed"]) for row in smoke_rows) / len(smoke_rows)
    ppl_ratio = math.exp((student_nll - teacher_nll) / token_count)
    mean_kl = token_kl / token_count
    ruler_teacher, ruler_student, ruler_rows = _score_pairs(artifact.get("ruler_rows"), name="RULER")
    downstream_teacher, downstream_student, downstream_rows = _score_pairs(
        artifact.get("downstream_rows"), name="downstream"
    )
    ruler_lower, _ = paired_bootstrap_ratio_ci(
        ruler_student, ruler_teacher, samples=bootstrap_samples, seed=bootstrap_seed
    )
    downstream_lower, _ = paired_bootstrap_ratio_ci(
        downstream_student, downstream_teacher, samples=bootstrap_samples, seed=bootstrap_seed
    )
    p1 = {
        "smoke_pass_rate_min": smoke_rate,
        "ppl_ratio_max": ppl_ratio,
        "mean_token_kl_max": mean_kl,
        "layer_mse_median_max": percentile(mses, 0.5),
        "layer_cosine_median_min": percentile(cosines, 0.5),
        "layer_cosine_min": min(cosines),
    }
    p2 = {
        "smoke_pass_rate_min": smoke_rate,
        "ppl_ratio_max": ppl_ratio,
        "mean_token_kl_max": mean_kl,
        "layer_cosine_median_min": percentile(cosines, 0.5),
        "layer_cosine_p05_min": percentile(cosines, 0.05),
        "layer_mse_median_max": percentile(mses, 0.5),
        "layer_mse_p95_max": percentile(mses, 0.95),
        "ruler_ci_lower_ratio_min": ruler_lower,
        "ruler_bucket_min_ratio_min": _group_ratio_min(ruler_rows, "bucket"),
        "downstream_ci_lower_ratio_min": downstream_lower,
        "downstream_max_drop_points_max": _max_group_drop_points(downstream_rows, "task"),
    }
    return {"P1": p1, "P2": p2}


def build_quality_threshold_profile(
    *,
    protocol_path: Path,
    seed_result_paths: Sequence[Path],
    artifact_path: Path,
    profile_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a profile from preregistered, independently eligible seed runs."""
    protocol = _read_object(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    bindings = protocol.get("bindings")
    bootstrap = protocol.get("bootstrap")
    control_reference = protocol.get("training_control_evidence")
    plan_reference = protocol.get("distillation_plan")
    dataset_manifest, dataset_manifest_sha, dataset_manifest_path = read_quality_dataset_evidence(
        protocol.get("dataset_manifest"),
        owner_path=protocol_path,
        expected_sha256=str(bindings.get("dataset_sha256", "")) if isinstance(bindings, dict) else "",
    )
    if (
        protocol.get("schema_version") != 1
        or protocol.get("status") != "frozen"
        or protocol.get("frozen_before_calibration_runs") is not True
        or protocol.get("precision") != "bf16"
        or not str(protocol.get("profile_id", ""))
        or not isinstance(bindings, dict)
        or any(not str(bindings.get(key, "")) for key in _BINDING_KEYS)
        or not isinstance(bootstrap, dict)
        or int(bootstrap.get("samples", 0)) < 10_000
        or float(bootstrap.get("confidence", 0.0)) != 0.95
        or int(bootstrap.get("seed", -1)) < 0
        or not isinstance(control_reference, dict)
        or not isinstance(plan_reference, dict)
    ):
        raise ValueError("quality calibration protocol is not a frozen BF16 protocol")
    control_artifact_path = Path(str(control_reference.get("artifact", "")))
    if not control_artifact_path.is_absolute():
        control_artifact_path = (protocol_path.parent / control_artifact_path).resolve()
    control_sha = str(control_reference.get("artifact_sha256", ""))
    if (
        not control_artifact_path.is_file()
        or file_sha256(control_artifact_path) != control_sha
        or bindings["training_control_evidence_sha256"] != control_sha
    ):
        raise ValueError("training-control evidence artifact SHA-256 mismatch")
    control_artifact = validate_training_control_artifact(control_artifact_path)
    plan_path = Path(str(plan_reference.get("artifact", "")))
    if not plan_path.is_absolute():
        plan_path = (protocol_path.parent / plan_path).resolve()
    plan_sha = str(plan_reference.get("artifact_sha256", ""))
    if (
        not plan_path.is_file()
        or file_sha256(plan_path) != plan_sha
        or bindings["distillation_plan_sha256"] != plan_sha
    ):
        raise ValueError("distillation plan artifact SHA-256 mismatch")
    plan_payload = _read_object(plan_path)
    candidates = control_artifact.get("candidates")
    shared_binding_keys = (
        "teacher_checkpoint_sha256",
        "dataset_sha256",
        "code_commit",
        "recipe_id",
    )
    if (
        control_artifact.get("schema_version") != 1
        or control_artifact.get("status") != "complete"
        or control_artifact.get("precision") != "bf16"
        or not isinstance(candidates, list)
        or not isinstance(control_artifact.get("bindings"), dict)
        or any(
            control_artifact["bindings"].get(key) != bindings.get(key)
            for key in shared_binding_keys
        )
        or control_artifact.get("selected_control_fingerprint")
        != training_control_fingerprint(plan_payload)
    ):
        raise ValueError("training-control evidence artifact is incomplete or misbound")
    control_run_ids = {
        str(run.get("run_id", ""))
        for candidate in candidates
        if isinstance(candidate, dict)
        for run in candidate.get("runs", [])
        if isinstance(run, dict)
    }
    control_checkpoint_sha256s = {
        str(run.get("student_checkpoint_sha256", ""))
        for candidate in candidates
        if isinstance(candidate, dict)
        for run in candidate.get("runs", [])
        if isinstance(run, dict)
    }
    if (
        not control_run_ids
        or "" in control_run_ids
        or not control_checkpoint_sha256s
        or any(len(value) != 64 for value in control_checkpoint_sha256s)
    ):
        raise ValueError("training-control evidence has invalid run provenance")
    if len(seed_result_paths) < 3:
        raise ValueError("quality calibration requires at least three seed result files")

    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    seen_runs: set[str] = set()
    seen_students: set[str] = set()
    for path in seed_result_paths:
        result = _read_object(path)
        seed = int(result.get("seed", -1))
        run_id = str(result.get("run_id", ""))
        student_sha = str(result.get("student_checkpoint_sha256", ""))
        sample_metrics = result.get("sample_metrics")
        baseline_curve = result.get("baseline_curve")
        if (
            result.get("schema_version") != 1
            or result.get("status") != "complete"
            or result.get("protocol_sha256") != protocol_sha
            or result.get("precision") != "bf16"
            or result.get("bindings") != bindings
            or seed < 0
            or not run_id
            or len(student_sha) != 64
            or not isinstance(sample_metrics, dict)
            or not isinstance(baseline_curve, dict)
        ):
            raise ValueError(f"seed result is not complete and protocol-bound: {path}")
        if seed in seen_seeds or run_id in seen_runs or student_sha in seen_students:
            raise ValueError("quality calibration seed, run, and student identities must be unique")
        if run_id in control_run_ids:
            raise ValueError("quality calibration cannot reuse a training-control run")
        if student_sha in control_checkpoint_sha256s:
            raise ValueError("quality calibration cannot reuse a training-control checkpoint")
        seen_seeds.add(seed)
        seen_runs.add(run_id)
        seen_students.add(student_sha)
        expected_artifact_binding = {
            "protocol_sha256": protocol_sha,
            "precision": "bf16",
            "bindings": bindings,
            "seed": seed,
            "run_id": run_id,
            "student_checkpoint_sha256": student_sha,
        }
        baseline_payload, baseline_sha, baseline_path = _read_bound_artifact(
            baseline_curve,
            owner_path=path,
            name="baseline curve",
            expected=expected_artifact_binding,
        )
        if (
            baseline_payload.get("direction") not in {"lower", "higher"}
            or set(baseline_payload.get("values", {})) != {"mapped", "random", "naive"}
            or not str(baseline_payload.get("metric", ""))
        ):
            raise ValueError("baseline curve has invalid comparison evidence")
        sample_payload, sample_sha, sample_path = _read_bound_artifact(
            sample_metrics,
            owner_path=path,
            name="sample metrics",
            expected=expected_artifact_binding,
        )
        metrics = derive_quality_metrics_from_samples(
            sample_payload,
            bootstrap_samples=int(bootstrap["samples"]),
            bootstrap_seed=int(bootstrap["seed"]),
        )
        eligibility, eligibility_sha, eligibility_path = _read_eligibility_evidence(
            result.get("eligibility_evidence"),
            result_path=path,
            protocol_sha=protocol_sha,
            run_id=run_id,
            baseline_curve_sha=baseline_sha,
            baseline_curve=baseline_payload,
        )
        rows.append(
            {
                "seed": seed,
                "run_id": run_id,
                "student_checkpoint_sha256": student_sha,
                "sample_metrics_sha256": sample_sha,
                "sample_metrics": sample_payload,
                "sample_metrics_reference": {
                    "artifact": os.path.relpath(sample_path, artifact_path.parent.resolve()),
                    "artifact_sha256": sample_sha,
                },
                "baseline_curve_sha256": baseline_sha,
                "baseline_curve": baseline_payload,
                "baseline_curve_reference": {
                    "artifact": os.path.relpath(baseline_path, artifact_path.parent.resolve()),
                    "artifact_sha256": baseline_sha,
                },
                "metrics": metrics,
                "eligibility_evidence_sha256": eligibility_sha,
                "eligibility_evidence": eligibility,
                "eligibility_evidence_reference": {
                    "artifact": os.path.relpath(eligibility_path, artifact_path.parent.resolve()),
                    "artifact_sha256": eligibility_sha,
                },
                "seed_result_sha256": file_sha256(path),
            }
        )

    rows.sort(key=lambda row: (int(row["seed"]), str(row["run_id"])))
    thresholds = derive_empirical_quality_thresholds(rows)
    threshold_provenance = derive_empirical_quality_threshold_provenance(rows)
    artifact = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "teacher-paired",
        "precision": "bf16",
        "seeds": [row["seed"] for row in rows],
        "bindings": bindings,
        "training_control_evidence_sha256": control_sha,
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_reference": {
            "artifact": os.path.relpath(dataset_manifest_path, artifact_path.parent.resolve()),
            "artifact_sha256": dataset_manifest_sha,
        },
        "training_control_evidence": control_artifact,
        "training_control_evidence_reference": {
            "artifact": os.path.relpath(control_artifact_path, artifact_path.parent.resolve()),
            "artifact_sha256": control_sha,
        },
        "distillation_plan": plan_payload,
        "distillation_plan_reference": {
            "artifact": os.path.relpath(plan_path, artifact_path.parent.resolve()),
            "artifact_sha256": plan_sha,
        },
        "per_seed_metrics": rows,
        "threshold_derivation": {
            "method": "multi-seed-empirical-calibration",
            "version": 1,
            "rule": "worst-eligible-seed-envelope",
            "candidate_evaluation_is_independent": True,
            "bootstrap_samples": int(bootstrap["samples"]),
            "bootstrap_seed": int(bootstrap["seed"]),
            "confidence": float(bootstrap["confidence"]),
            "protocol_sha256": protocol_sha,
            "source_run_ids": [row["run_id"] for row in rows],
            "derived_thresholds": thresholds,
            "threshold_provenance": threshold_provenance,
            "method_evidence": [
                {
                    "claim": "freeze non-inferiority margins before candidate evaluation",
                    "url": "https://arxiv.org/abs/1807.03413",
                },
                {
                    "claim": "multiple seeds are required because benchmark variance changes conclusions",
                    "url": "https://arxiv.org/abs/2103.03098",
                },
                {
                    "claim": "use paired evaluation on shared examples",
                    "url": "https://aclanthology.org/2021.acl-long.179/",
                },
                {
                    "claim": "report paired differences with confidence intervals",
                    "url": "https://aclanthology.org/2022.lrec-1.640/",
                },
            ],
        },
    }
    write_json(artifact_path, artifact)
    profile = {
        "schema_version": 1,
        "status": "calibrated",
        "profile_id": str(protocol.get("profile_id", "")),
        "calibration": {
            "artifact": os.path.relpath(artifact_path.resolve(), profile_path.parent.resolve()),
            "artifact_sha256": file_sha256(artifact_path),
        },
        "thresholds": thresholds,
    }
    write_json(profile_path, profile)
    return artifact, profile
