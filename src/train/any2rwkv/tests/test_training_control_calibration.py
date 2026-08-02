from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from any2rwkv.artifacts import file_sha256
from any2rwkv.core.training_control_calibration import (
    build_training_control_calibration,
    training_control_fingerprint,
)
from any2rwkv.core.training_control_export import export_training_control_result


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _dataset_manifest(root: Path) -> Path:
    report = root / "deduplication-report.json"
    _write(
        report,
        {
            "exact_duplicates": {"pairs": []},
            "near_duplicates": {
                "policy": "reject",
                "pairs": [],
                "candidate_search_complete": True,
            },
        },
    )
    manifest = root / "data-splits.json"
    _write(
        manifest,
        {
            "schema_version": 1,
            "status": "prepared",
            "split_assignment": {"sample_ids_mutually_exclusive": True},
            "deduplication": {"report_path": report.name, "report_sha256": file_sha256(report)},
            "splits": {name: {} for name in ("distill_train", "validation", "ruler", "downstream", "smoke")},
        },
    )
    return manifest


def _plan(learning_rate: float) -> dict[str, object]:
    return {
        "learning_rate": learning_rate,
        "burn_in_tokens": 8,
        "supervised_tokens": 16,
        "accumulation_steps": 2,
        "micro_batch_size": 1,
        "activation_fit_functional_steps": 8,
        "activation_fit_functional_learning_rate": 0.001,
        "layer_min_epochs": 2,
        "layer_max_epochs": 4,
        "layer_min_delta": 0.001,
        "layer_patience": 1,
        "corrective_min_sweeps": 1,
        "corrective_max_sweeps": 2,
        "corrective_min_delta": 0.001,
        "local_loss_weights": {
            "mixer_mse": 1.0,
            "block_mse": 1.0,
            "cosine": 0.1,
        },
        "global_loss_weights": {"token_kl": 1.0, "shifted_ce": 0.25},
    }


def _fixture(root: Path):
    dataset_manifest = _dataset_manifest(root)
    bindings = {
        "teacher_checkpoint_sha256": "a" * 64,
        "dataset_sha256": file_sha256(dataset_manifest),
        "code_commit": "fixture",
        "recipe_id": "fixture_to_rwkv7",
        "metric_definition": "control-v1",
    }
    candidates = []
    plans = {}
    for candidate_id, lr in (("slow", 1e-5), ("best", 1e-4), ("fast", 1e-3)):
        path = root / f"{candidate_id}.json"
        plan = _plan(lr)
        _write(path, plan)
        plans[candidate_id] = (path, plan)
        candidates.append({"candidate_id": candidate_id, "plan": path.name})
    protocol = root / "protocol.json"
    _write(
        protocol,
        {
            "schema_version": 1,
            "status": "frozen",
            "frozen_before_runs": True,
            "precision": "bf16",
            "bindings": bindings,
            "candidate_plans": candidates,
            "dataset_manifest": {
                "artifact": dataset_manifest.name,
                "artifact_sha256": file_sha256(dataset_manifest),
            },
        },
    )
    protocol_sha = file_sha256(protocol)
    results = []
    base = {"slow": 0.4, "best": 0.2, "fast": 0.3}
    for candidate_id, (plan_path, plan) in plans.items():
        for seed in (1, 2, 3):
            path = root / f"{candidate_id}-{seed}.json"
            run_id = f"{candidate_id}-{seed}"
            checkpoint_sha = hashlib.sha256(f"{candidate_id}:{seed}".encode()).hexdigest()
            validation_kl = base[candidate_id] + seed * 0.001
            ppl_ratio = 1.0 + base[candidate_id]
            curve_path = root / f"{candidate_id}-{seed}-curve.json"
            _write(
                curve_path,
                {
                    "schema_version": 1,
                    "status": "complete",
                    "protocol_sha256": protocol_sha,
                    "precision": "bf16",
                    "bindings": bindings,
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "run_id": run_id,
                    "plan_sha256": file_sha256(plan_path),
                    "control_fingerprint": training_control_fingerprint(plan),
                        "student_checkpoint_sha256": checkpoint_sha,
                        "expected_layer_count": 2,
                        "final_validation": {
                            "kl_token_count": 200,
                            "token_kl_sum": validation_kl * 200,
                            "nll_token_count": 180,
                            "nll_delta_sum": math.log(ppl_ratio) * 180,
                        },
                        "layer_results": [
                        {
                            "layer_index": layer,
                            "converged": True,
                            "trained_token_count": 512,
                            "selected_validation": {
                                "normalized_mse": validation_kl,
                                "cosine": 0.9,
                            },
                        }
                        for layer in (0, 1)
                    ],
                },
            )
            _write(
                path,
                {
                    "schema_version": 1,
                    "status": "complete",
                    "protocol_sha256": protocol_sha,
                    "precision": "bf16",
                    "bindings": bindings,
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "run_id": run_id,
                    "plan_sha256": file_sha256(plan_path),
                    "control_fingerprint": training_control_fingerprint(plan),
                    "student_checkpoint_sha256": checkpoint_sha,
                    "validation_curve": {
                        "artifact": curve_path.name,
                        "artifact_sha256": file_sha256(curve_path),
                    },
                },
            )
            results.append(path)
    return protocol, results, plans


def test_selects_lowest_multi_seed_validation_kl(tmp_path: Path) -> None:
    protocol, results, plans = _fixture(tmp_path)
    output = tmp_path / "controls.json"
    artifact = build_training_control_calibration(
        protocol_path=protocol,
        result_paths=results,
        output_path=output,
    )
    assert artifact["selected_candidate_id"] == "best"
    assert artifact["selected_plan_sha256"] == file_sha256(plans["best"][0])
    assert output.is_file()


def test_training_control_fingerprint_normalizes_legacy_functional_fit_aliases() -> None:
    canonical = _plan(1e-4)
    canonical["activation_fit_functional_steps"] = 32
    canonical["activation_fit_functional_learning_rate"] = 3e-4
    legacy = _plan(1e-4)
    legacy.pop("activation_fit_functional_steps")
    legacy.pop("activation_fit_functional_learning_rate")
    legacy["activation_fit_time_mix_steps"] = 32
    legacy["activation_fit_time_mix_learning_rate"] = 3e-4
    assert training_control_fingerprint(canonical) == training_control_fingerprint(legacy)


def test_training_control_fingerprint_rejects_conflicting_functional_fit_aliases() -> None:
    plan = _plan(1e-4)
    plan["activation_fit_functional_steps"] = 32
    plan["activation_fit_time_mix_steps"] = 64
    with pytest.raises(ValueError, match="aliases conflict"):
        training_control_fingerprint(plan)

    plan = _plan(1e-4)
    plan["activation_fit_functional_learning_rate"] = 3e-4
    plan["activation_fit_time_mix_learning_rate"] = 1e-3
    with pytest.raises(ValueError, match="aliases conflict"):
        training_control_fingerprint(plan)


def test_rejects_unequal_seed_sets(tmp_path: Path) -> None:
    protocol, results, _ = _fixture(tmp_path)
    results.pop()
    with pytest.raises(ValueError, match="three unique seeds"):
        build_training_control_calibration(
            protocol_path=protocol,
            result_paths=results,
            output_path=tmp_path / "controls.json",
        )


def test_rejects_non_finite_final_validation_curve(tmp_path: Path) -> None:
    protocol, results, _ = _fixture(tmp_path)
    result = json.loads(results[0].read_text(encoding="utf-8"))
    curve_path = tmp_path / result["validation_curve"]["artifact"]
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    curve["final_validation"]["token_kl_sum"] = float("nan")
    _write(curve_path, curve)
    result["validation_curve"]["artifact_sha256"] = file_sha256(curve_path)
    _write(results[0], result)
    with pytest.raises(ValueError, match="invalid final validation"):
        build_training_control_calibration(
            protocol_path=protocol,
            result_paths=results,
            output_path=tmp_path / "controls.json",
        )


def test_exports_hash_bound_real_run_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "config.json", {"model_type": "fixture"})
    (source / "model.safetensors").write_bytes(b"fixture weights")
    source_files = {
        path.name: file_sha256(path)
        for path in sorted(source.iterdir())
        if path.is_file()
    }
    from any2rwkv.artifacts import checkpoint_sha256

    plan = {**_plan(1e-4), "seed": 7}
    plan_path = tmp_path / "candidate.json"
    _write(plan_path, plan)
    bindings = {
        "teacher_checkpoint_sha256": checkpoint_sha256(source),
        "dataset_sha256": "d" * 64,
        "code_commit": "c" * 40,
        "recipe_id": "qwen35_to_rwkv7",
        "metric_definition": "control-v2",
    }
    protocol_path = tmp_path / "protocol.json"
    _write(
        protocol_path,
        {
            "schema_version": 1,
            "status": "frozen",
            "precision": "bf16",
            "bindings": bindings,
            "candidate_plans": [{"candidate_id": "candidate", "plan": plan_path.name}],
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write(
        run_dir / "metadata.json",
        {
            "run_id": "pilot-7",
            "precision": "bf16",
            "product_commit": bindings["code_commit"],
            "recipe": {"id": bindings["recipe_id"]},
            "source": {"path": str(source), "files": source_files, "layers": 2},
        },
    )
    binding = {
        "source_checkpoint_sha256": hashlib.sha256(
            json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "training_config_sha256": file_sha256(plan_path),
        "dataset_manifest_sha256": bindings["dataset_sha256"],
    }
    _write(run_dir / "layer-major-progress.json", {"binding": binding})
    _write(run_dir / "global-corrective-progress.json", {"binding": binding})
    _write(
        run_dir / "layer-convergence.json",
        {
            "schema_version": 2,
            "epochs": [
                {
                    "layer": layer,
                    "epoch": 0,
                    "train_row_count": 5,
                    "best_epoch": 0,
                    "converged": True,
                    "validation": {"normalized_mse": 0.1 + layer, "cosine": 0.9},
                }
                for layer in range(2)
            ],
        },
    )
    _write(
        run_dir / "global-corrective.json",
        {
            "schema_version": 1,
            "status": "complete",
            "selected_checkpoint": "global-snapshots/sweep-000",
            "history": [
                {
                    "end_checkpoint": "global-snapshots/sweep-000",
                    "validation": {
                        "kl_token_count": 20,
                        "token_kl_sum": 4.0,
                        "nll_token_count": 18,
                        "nll_delta_sum": math.log(1.5) * 18,
                    },
                }
            ],
        },
    )
    checkpoint = run_dir / "checkpoint-global-corrective"
    checkpoint.mkdir()
    _write(checkpoint / "config.json", {"model_type": "fixture"})
    (checkpoint / "model.safetensors").write_bytes(b"student weights")

    result_path = tmp_path / "result.json"
    result = export_training_control_result(
        protocol_path=protocol_path,
        candidate_id="candidate",
        run_dir=run_dir,
        output_path=result_path,
    )

    curve_path = tmp_path / result["validation_curve"]["artifact"]
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    assert result_path.is_file()
    assert curve["final_validation"]["token_kl_sum"] == 4.0
    assert curve["layer_results"][0]["trained_token_count"] == 5 * plan["supervised_tokens"]
