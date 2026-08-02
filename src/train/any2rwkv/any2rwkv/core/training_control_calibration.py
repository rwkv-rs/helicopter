from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from ..artifacts import file_sha256, write_json
from .dataset_evidence import read_quality_dataset_evidence


CONTROL_KEYS = (
    "learning_rate",
    "learning_rate_schedule",
    "learning_rate_warmup_ratio",
    "min_learning_rate_ratio",
    "gradient_clip_norm",
    "max_parameter_update_relative_l2",
    "burn_in_tokens",
    "supervised_tokens",
    "accumulation_steps",
    "micro_batch_size",
    "distributed_world_size",
    "activation_fit_rows",
    "activation_fit_ridge",
    "activation_fit_functional_steps",
    "activation_fit_functional_learning_rate",
    "layer_min_epochs",
    "layer_max_epochs",
    "layer_learning_rate_schedule_epochs",
    "layer_fixed_epochs",
    "layer_min_delta",
    "layer_patience",
    "corrective_min_sweeps",
    "corrective_max_sweeps",
    "corrective_min_delta",
    "local_loss_weights",
    "global_loss_weights",
)

CONTROL_DEFAULTS = {
    "learning_rate_schedule": "constant",
    "learning_rate_warmup_ratio": 0.0,
    "min_learning_rate_ratio": 1.0,
    "gradient_clip_norm": None,
    "max_parameter_update_relative_l2": None,
    "distributed_world_size": 1,
    "layer_learning_rate_schedule_epochs": None,
    "layer_fixed_epochs": None,
    "activation_fit_rows": 0,
    "activation_fit_ridge": 1e-3,
}


def training_control_fingerprint(plan: dict[str, Any]) -> str:
    plan = dict(plan)
    legacy_aliases = {
        "activation_fit_functional_steps": "activation_fit_time_mix_steps",
        "activation_fit_functional_learning_rate": (
            "activation_fit_time_mix_learning_rate"
        ),
    }
    for canonical, legacy in legacy_aliases.items():
        if (
            canonical in plan
            and legacy in plan
            and plan[canonical] != plan[legacy]
        ):
            raise ValueError(
                "legacy time-mix activation-fit aliases conflict with "
                "functional controls"
            )
        if canonical not in plan and legacy in plan:
            plan[canonical] = plan[legacy]
    missing = [
        key for key in CONTROL_KEYS if key not in plan and key not in CONTROL_DEFAULTS
    ]
    if missing:
        raise ValueError(f"distillation plan lacks training controls: {missing}")
    payload = {key: plan.get(key, CONTROL_DEFAULTS.get(key)) for key in CONTROL_KEYS}
    if plan.get("activation_fit_attention_time_mix_ablation", False):
        payload["activation_fit_attention_time_mix_ablation"] = True
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training-control input is not an object: {path}")
    return payload


def _derive_curve_metrics(
    reference: object,
    *,
    result_path: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, float | int], str, dict[str, Any], Path]:
    if not isinstance(reference, dict):
        raise ValueError("training-control result has no validation curve reference")
    curve_path = Path(str(reference.get("artifact", "")))
    if not curve_path.is_absolute():
        curve_path = (result_path.parent / curve_path).resolve()
    curve_sha = str(reference.get("artifact_sha256", ""))
    if not curve_path.is_file() or file_sha256(curve_path) != curve_sha:
        raise ValueError("training-control validation curve SHA-256 mismatch")
    curve = _read_object(curve_path)
    layers = curve.get("layer_results")
    if (
        curve.get("schema_version") != 1
        or curve.get("status") != "complete"
        or any(curve.get(key) != value for key, value in expected.items())
        or not isinstance(layers, list)
        or int(curve.get("expected_layer_count", 0)) <= 0
    ):
        raise ValueError("training-control validation curve is incomplete or misbound")
    layer_count = int(curve["expected_layer_count"])
    if {int(row.get("layer_index", -1)) for row in layers if isinstance(row, dict)} != set(
        range(layer_count)
    ) or len(layers) != layer_count:
        raise ValueError("training-control validation curve does not cover every layer exactly once")
    final_validation = curve.get("final_validation")
    if not isinstance(final_validation, dict):
        raise ValueError("training-control curve has no final validation")
    kl_token_count = int(final_validation.get("kl_token_count", 0))
    token_kl_sum = float(final_validation.get("token_kl_sum", float("nan")))
    nll_token_count = int(final_validation.get("nll_token_count", 0))
    nll_delta_sum = float(final_validation.get("nll_delta_sum", float("nan")))
    if (
        kl_token_count <= 0
        or nll_token_count <= 0
        or not math.isfinite(token_kl_sum)
        or not math.isfinite(nll_delta_sum)
    ):
        raise ValueError("training-control curve contains invalid final validation")
    token_budget = 0
    converged = 0
    for row in layers:
        selected = row.get("selected_validation") if isinstance(row, dict) else None
        if not isinstance(selected, dict):
            raise ValueError("training-control layer has no selected validation observation")
        normalized_mse = float(selected.get("normalized_mse", float("nan")))
        cosine = float(selected.get("cosine", float("nan")))
        trained = int(row.get("trained_token_count", 0))
        if (
            trained <= 0
            or not math.isfinite(normalized_mse)
            or not math.isfinite(cosine)
        ):
            raise ValueError("training-control curve contains invalid layer evidence")
        token_budget += trained
        converged += int(row.get("converged") is True)
    metrics: dict[str, float | int] = {
        "validation_token_kl": token_kl_sum / kl_token_count,
        "ppl_ratio": math.exp(nll_delta_sum / nll_token_count),
        "converged_layer_fraction": converged / layer_count,
        "token_budget": token_budget,
    }
    return metrics, curve_sha, curve, curve_path


def build_training_control_calibration(
    *,
    protocol_path: Path,
    result_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Select controls using frozen, equal-budget, multi-seed candidate runs."""
    protocol = _read_object(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    bindings = protocol.get("bindings")
    candidates = protocol.get("candidate_plans")
    dataset_manifest, dataset_manifest_sha, dataset_manifest_path = read_quality_dataset_evidence(
        protocol.get("dataset_manifest"),
        owner_path=protocol_path,
        expected_sha256=str(bindings.get("dataset_sha256", "")) if isinstance(bindings, dict) else "",
    )
    if (
        protocol.get("schema_version") != 1
        or protocol.get("status") != "frozen"
        or protocol.get("frozen_before_runs") is not True
        or protocol.get("precision") != "bf16"
        or not isinstance(bindings, dict)
        or any(
            not str(bindings.get(key, ""))
            for key in (
                "teacher_checkpoint_sha256",
                "dataset_sha256",
                "code_commit",
                "recipe_id",
                "metric_definition",
            )
        )
        or not isinstance(candidates, list)
        or len(candidates) < 3
    ):
        raise ValueError("training-control protocol is not a frozen BF16 ablation")

    candidate_by_id: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("training-control candidate must be an object")
        candidate_id = str(candidate.get("candidate_id", ""))
        plan_path = Path(str(candidate.get("plan", "")))
        if not plan_path.is_absolute():
            plan_path = (protocol_path.parent / plan_path).resolve()
        if not candidate_id or candidate_id in candidate_by_id or not plan_path.is_file():
            raise ValueError("training-control candidate id/path is invalid")
        plan = _read_object(plan_path)
        candidate_by_id[candidate_id] = {
            "plan_sha256": file_sha256(plan_path),
            "control_fingerprint": training_control_fingerprint(plan),
        }

    rows_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_runs: set[str] = set()
    seen_checkpoints: set[str] = set()
    for path in result_paths:
        result = _read_object(path)
        candidate_id = str(result.get("candidate_id", ""))
        expected = candidate_by_id.get(candidate_id)
        run_id = str(result.get("run_id", ""))
        checkpoint_sha = str(result.get("student_checkpoint_sha256", ""))
        if (
            expected is None
            or result.get("schema_version") != 1
            or result.get("status") != "complete"
            or result.get("protocol_sha256") != protocol_sha
            or result.get("precision") != "bf16"
            or result.get("bindings") != bindings
            or result.get("plan_sha256") != expected["plan_sha256"]
            or result.get("control_fingerprint") != expected["control_fingerprint"]
            or int(result.get("seed", -1)) < 0
            or not run_id
            or run_id in seen_runs
            or len(checkpoint_sha) != 64
            or checkpoint_sha in seen_checkpoints
        ):
            raise ValueError(f"training-control result is invalid: {path}")
        metrics, curve_sha, curve, curve_path = _derive_curve_metrics(
            result.get("validation_curve"),
            result_path=path,
            expected={
                "protocol_sha256": protocol_sha,
                "precision": "bf16",
                "bindings": bindings,
                "candidate_id": candidate_id,
                "seed": int(result["seed"]),
                "run_id": run_id,
                "plan_sha256": expected["plan_sha256"],
                "control_fingerprint": expected["control_fingerprint"],
                "student_checkpoint_sha256": checkpoint_sha,
            },
        )
        if float(metrics["converged_layer_fraction"]) != 1.0:
            raise ValueError("training-control candidate did not converge every layer")
        seen_runs.add(run_id)
        seen_checkpoints.add(checkpoint_sha)
        rows_by_candidate[candidate_id].append(
            {
                "seed": int(result["seed"]),
                "run_id": run_id,
                "student_checkpoint_sha256": checkpoint_sha,
                "validation_curve_sha256": curve_sha,
                "validation_curve": curve,
                "validation_curve_reference": {
                    "artifact": os.path.relpath(curve_path, output_path.parent.resolve()),
                    "artifact_sha256": curve_sha,
                },
                "validation_token_kl": float(metrics["validation_token_kl"]),
                "ppl_ratio": float(metrics["ppl_ratio"]),
                "converged_layer_fraction": float(metrics["converged_layer_fraction"]),
                "token_budget": int(metrics["token_budget"]),
                "result_sha256": file_sha256(path),
            }
        )

    seed_sets: list[set[int]] = []
    token_budgets: set[int] = set()
    summaries: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_by_id):
        rows = sorted(rows_by_candidate[candidate_id], key=lambda row: row["seed"])
        seeds = {row["seed"] for row in rows}
        if len(rows) < 3 or len(seeds) != len(rows):
            raise ValueError("each training-control candidate requires at least three unique seeds")
        seed_sets.append(seeds)
        token_budgets.update(row["token_budget"] for row in rows)
        summaries.append(
            {
                "candidate_id": candidate_id,
                **candidate_by_id[candidate_id],
                "mean_validation_token_kl": sum(row["validation_token_kl"] for row in rows)
                / len(rows),
                "mean_ppl_ratio": sum(row["ppl_ratio"] for row in rows) / len(rows),
                "runs": rows,
            }
        )
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("training-control candidates must use the same seed set")
    if len(token_budgets) != 1:
        raise ValueError("training-control candidates must use one equal token budget")

    selected = min(
        summaries,
        key=lambda row: (
            row["mean_validation_token_kl"],
            row["mean_ppl_ratio"],
            row["control_fingerprint"],
        ),
    )
    artifact = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "precision": "bf16",
        "bindings": bindings,
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_reference": {
            "artifact": os.path.relpath(dataset_manifest_path, output_path.parent.resolve()),
            "artifact_sha256": dataset_manifest_sha,
        },
        "selection_rule": {
            "primary": "lowest-mean-validation-token-kl",
            "secondary": "lowest-mean-ppl-ratio",
            "tie_break": "control-fingerprint",
            "candidate_count_min": 3,
            "seed_count_min": 3,
            "equal_token_budget": next(iter(token_budgets)),
            "candidate_runs_independent_from_quality_calibration": True,
        },
        "candidates": summaries,
        "selected_candidate_id": selected["candidate_id"],
        "selected_plan_sha256": selected["plan_sha256"],
        "selected_control_fingerprint": selected["control_fingerprint"],
    }
    write_json(output_path, artifact)
    return artifact


def validate_training_control_artifact(
    artifact_path: Path, *, expected_plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    artifact = _read_object(artifact_path)
    bindings = artifact.get("bindings")
    candidates = artifact.get("candidates")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("status") != "complete"
        or artifact.get("precision") != "bf16"
        or not isinstance(bindings, dict)
        or not isinstance(candidates, list)
        or len(candidates) < 3
    ):
        raise ValueError("training-control artifact is incomplete")
    read_quality_dataset_evidence(
        artifact.get("dataset_manifest_reference"),
        owner_path=artifact_path,
        expected_sha256=str(bindings.get("dataset_sha256", "")),
    )
    seed_sets: list[set[int]] = []
    token_budgets: set[int] = set()
    summaries: list[tuple[float, float, str, str]] = []
    seen_runs: set[str] = set()
    seen_checkpoints: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("runs"), list):
            raise ValueError("training-control candidate evidence is invalid")
        candidate_id = str(candidate.get("candidate_id", ""))
        plan_sha = str(candidate.get("plan_sha256", ""))
        fingerprint = str(candidate.get("control_fingerprint", ""))
        rows = candidate["runs"]
        seeds = {int(row.get("seed", -1)) for row in rows if isinstance(row, dict)}
        if len(rows) < 3 or len(seeds) != len(rows) or min(seeds, default=-1) < 0:
            raise ValueError("each training-control candidate requires at least three unique seeds")
        seed_sets.append(seeds)
        recomputed_rows: list[dict[str, float | int]] = []
        for row in rows:
            run_id = str(row.get("run_id", ""))
            checkpoint_sha = str(row.get("student_checkpoint_sha256", ""))
            if (
                not run_id
                or run_id in seen_runs
                or len(checkpoint_sha) != 64
                or checkpoint_sha in seen_checkpoints
            ):
                raise ValueError("training-control run identity is invalid")
            seen_runs.add(run_id)
            seen_checkpoints.add(checkpoint_sha)
            metrics, curve_sha, curve, _ = _derive_curve_metrics(
                row.get("validation_curve_reference"),
                result_path=artifact_path,
                expected={
                    "protocol_sha256": artifact.get("protocol_sha256"),
                    "precision": "bf16",
                    "bindings": bindings,
                    "candidate_id": candidate_id,
                    "seed": int(row["seed"]),
                    "run_id": run_id,
                    "plan_sha256": plan_sha,
                    "control_fingerprint": fingerprint,
                    "student_checkpoint_sha256": checkpoint_sha,
                },
            )
            if (
                curve != row.get("validation_curve")
                or curve_sha != row.get("validation_curve_sha256")
                or any(row.get(key) != value for key, value in metrics.items())
                or float(metrics["converged_layer_fraction"]) != 1.0
            ):
                raise ValueError("training-control aggregate does not match validation curve")
            token_budgets.add(int(metrics["token_budget"]))
            recomputed_rows.append(metrics)
        mean_kl = sum(float(row["validation_token_kl"]) for row in recomputed_rows) / len(rows)
        mean_ppl = sum(float(row["ppl_ratio"]) for row in recomputed_rows) / len(rows)
        if (
            candidate.get("mean_validation_token_kl") != mean_kl
            or candidate.get("mean_ppl_ratio") != mean_ppl
        ):
            raise ValueError("training-control candidate summary is not reproducible")
        summaries.append((mean_kl, mean_ppl, fingerprint, candidate_id))
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]) or len(token_budgets) != 1:
        raise ValueError("training-control candidates do not share seeds and token budget")
    selected = min(summaries)
    if (
        artifact.get("selected_candidate_id") != selected[3]
        or artifact.get("selected_control_fingerprint") != selected[2]
        or (
            expected_plan is not None
            and artifact.get("selected_control_fingerprint")
            != training_control_fingerprint(expected_plan)
        )
    ):
        raise ValueError("training-control selected candidate is not reproducible")
    return artifact
