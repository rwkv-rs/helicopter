from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..artifacts import checkpoint_sha256, file_sha256, write_json
from .training_control_calibration import training_control_fingerprint


def export_training_control_result(
    *,
    protocol_path: Path,
    candidate_id: str,
    run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Export one completed, protocol-bound pilot run for control calibration."""
    protocol = _read_object(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    bindings = protocol.get("bindings")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("status") != "frozen"
        or protocol.get("precision") != "bf16"
        or not isinstance(bindings, dict)
    ):
        raise ValueError("training-control protocol is not a frozen BF16 ablation")
    plan_path = _candidate_plan_path(protocol, protocol_path, candidate_id)
    plan = _read_object(plan_path)
    metadata = _read_object(run_dir / "metadata.json")
    layer_progress = _read_object(run_dir / "layer-major-progress.json")
    global_progress = _read_object(run_dir / "global-corrective-progress.json")
    convergence = _read_object(run_dir / "layer-convergence.json")
    global_result = _read_object(run_dir / "global-corrective.json")

    plan_sha = file_sha256(plan_path)
    control_fingerprint = training_control_fingerprint(plan)
    _validate_run_bindings(
        bindings=bindings,
        metadata=metadata,
        layer_progress=layer_progress,
        global_progress=global_progress,
        plan_sha=plan_sha,
        control_fingerprint=control_fingerprint,
    )
    checkpoint = run_dir / "checkpoint-global-corrective"
    student_sha = checkpoint_sha256(checkpoint)
    run_id = str(metadata.get("run_id", ""))
    seed = int(plan.get("seed", -1))
    if not run_id or seed < 0:
        raise ValueError("run metadata or candidate plan lacks a valid run identity")

    curve = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "precision": "bf16",
        "bindings": bindings,
        "candidate_id": candidate_id,
        "seed": seed,
        "run_id": run_id,
        "plan_sha256": plan_sha,
        "control_fingerprint": control_fingerprint,
        "student_checkpoint_sha256": student_sha,
        "expected_layer_count": _expected_layer_count(metadata),
        "layer_results": _layer_results(convergence, plan),
        "final_validation": _final_validation(global_result),
    }
    _validate_curve_shape(curve)
    curve_path = output_path.with_name(output_path.stem + "-validation-curve.json")
    write_json(curve_path, curve)
    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "precision": "bf16",
        "bindings": bindings,
        "candidate_id": candidate_id,
        "seed": seed,
        "run_id": run_id,
        "plan_sha256": plan_sha,
        "control_fingerprint": control_fingerprint,
        "student_checkpoint_sha256": student_sha,
        "validation_curve": {
            "artifact": os.path.relpath(curve_path, output_path.parent.resolve()),
            "artifact_sha256": file_sha256(curve_path),
        },
    }
    write_json(output_path, result)
    return result


def _candidate_plan_path(protocol: dict[str, Any], protocol_path: Path, candidate_id: str) -> Path:
    for candidate in protocol.get("candidate_plans", []):
        if isinstance(candidate, dict) and candidate.get("candidate_id") == candidate_id:
            path = Path(str(candidate.get("plan", "")))
            path = path if path.is_absolute() else (protocol_path.parent / path).resolve()
            if path.is_file():
                return path
    raise ValueError(f"training-control protocol has no candidate plan: {candidate_id}")


def _validate_run_bindings(*, bindings, metadata, layer_progress, global_progress, plan_sha, control_fingerprint) -> None:
    source = metadata.get("source")
    source_files = source.get("files") if isinstance(source, dict) else None
    source_path = Path(str(source.get("path", ""))) if isinstance(source, dict) else None
    if (
        metadata.get("precision") != "bf16"
        or metadata.get("product_commit") != bindings.get("code_commit")
        or metadata.get("recipe", {}).get("id") != bindings.get("recipe_id")
        or not isinstance(source_files, dict)
        or source_path is None
        or not source_path.is_dir()
    ):
        raise ValueError("run metadata cannot prove training-control bindings")
    _verify_source_files(source_path, source_files)
    if checkpoint_sha256(source_path) != bindings.get("teacher_checkpoint_sha256"):
        raise ValueError("run source checkpoint does not match training-control protocol")
    source_sha = _sha256_json(source_files)
    for progress in (layer_progress, global_progress):
        binding = progress.get("binding")
        if not isinstance(binding, dict) or (
            binding.get("source_checkpoint_sha256") != source_sha
            or binding.get("training_config_sha256") != plan_sha
            or binding.get("dataset_manifest_sha256") != bindings.get("dataset_sha256")
        ):
            raise ValueError("run progress binding does not match candidate plan or protocol")
def _verify_source_files(source_path: Path, expected: dict[str, Any]) -> None:
    for name, digest in expected.items():
        path = source_path / str(name)
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError("run source files no longer match immutable run metadata")


def _expected_layer_count(metadata: dict[str, Any]) -> int:
    source = metadata.get("source")
    layers = source.get("layers") if isinstance(source, dict) else None
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError("run metadata lacks source layer count")
    return layers


def _layer_results(convergence: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    history = convergence.get("epochs")
    if convergence.get("schema_version") != 2 or not isinstance(history, list):
        raise ValueError("layer convergence artifact is invalid")
    supervised_tokens = int(plan.get("supervised_tokens", 0))
    if supervised_tokens <= 0:
        raise ValueError("candidate plan lacks supervised token count")
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        if not isinstance(row, dict):
            raise ValueError("layer convergence history contains a non-object row")
        by_layer[int(row.get("layer", -1))].append(row)
    results = []
    for layer_index in sorted(by_layer):
        rows = by_layer[layer_index]
        final = rows[-1]
        best_epoch = int(final.get("best_epoch", -1))
        selected_rows = [row for row in rows if int(row.get("epoch", -1)) == best_epoch]
        if (
            len(selected_rows) != 1
            or final.get("convergence_observed", final.get("converged")) is not True
        ):
            raise ValueError("layer convergence cannot identify one selected validation")
        validation = selected_rows[0].get("validation")
        if not isinstance(validation, dict):
            raise ValueError("layer convergence selected validation is missing")
        results.append(
            {
                "layer_index": layer_index,
                "converged": True,
                "trained_token_count": sum(int(row.get("train_row_count", 0)) for row in rows)
                * supervised_tokens,
                "selected_validation": {
                    "normalized_mse": validation.get("normalized_mse"),
                    "cosine": validation.get("cosine"),
                },
            }
        )
    return results


def _final_validation(global_result: dict[str, Any]) -> dict[str, Any]:
    selected = global_result.get("selected_checkpoint")
    history = global_result.get("history")
    if (
        global_result.get("schema_version") != 1
        or global_result.get("status") != "complete"
        or not isinstance(selected, str)
        or not isinstance(history, list)
    ):
        raise ValueError("global corrective artifact is incomplete")
    matches = [row for row in history if isinstance(row, dict) and row.get("end_checkpoint") == selected]
    if len(matches) != 1 or not isinstance(matches[0].get("validation"), dict):
        raise ValueError("global corrective artifact cannot bind selected final validation")
    validation = matches[0]["validation"]
    return {
        key: validation.get(key)
        for key in ("kl_token_count", "token_kl_sum", "nll_token_count", "nll_delta_sum")
    }


def _validate_curve_shape(curve: dict[str, Any]) -> None:
    expected = int(curve["expected_layer_count"])
    rows = curve["layer_results"]
    if {row["layer_index"] for row in rows} != set(range(expected)) or len(rows) != expected:
        raise ValueError("layer convergence does not cover every expected layer")
    final = curve["final_validation"]
    for key in ("kl_token_count", "nll_token_count"):
        if not isinstance(final.get(key), int) or final[key] <= 0:
            raise ValueError("global corrective final validation has invalid token counts")
    for key in ("token_kl_sum", "nll_delta_sum"):
        if not isinstance(final.get(key), (int, float)):
            raise ValueError("global corrective final validation has invalid sums")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training-control input is not an object: {path}")
    return payload


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
