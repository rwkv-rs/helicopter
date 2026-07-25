from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

import pytest

from any2rwkv.artifacts import file_sha256
from any2rwkv.core.quality_calibration import build_quality_threshold_profile
from any2rwkv.core.dataset_evidence import read_quality_dataset_evidence
from any2rwkv.core.training_control_calibration import training_control_fingerprint
from any2rwkv.evaluate import read_quality_threshold_profile


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _dataset_manifest(root: Path) -> Path:
    report = root / "deduplication-report.json"
    _write_json(
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
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "status": "prepared",
            "split_assignment": {"sample_ids_mutually_exclusive": True},
            "deduplication": {
                "report_path": report.name,
                "report_sha256": file_sha256(report),
            },
            "splits": {name: {} for name in ("distill_train", "validation", "ruler", "downstream", "smoke")},
        },
    )
    return manifest


def test_complete_report_policy_without_pairs_can_use_data_directly(tmp_path: Path) -> None:
    manifest = _dataset_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    report = tmp_path / payload["deduplication"]["report_path"]
    report_payload = json.loads(report.read_text())
    report_payload["near_duplicates"]["policy"] = "report"
    _write_json(report, report_payload)
    payload["deduplication"]["report_sha256"] = file_sha256(report)
    _write_json(manifest, payload)
    expected = file_sha256(manifest)
    loaded, claimed, path = read_quality_dataset_evidence(
        {"artifact": str(manifest), "artifact_sha256": expected},
        owner_path=tmp_path / "owner.json",
        expected_sha256=expected,
    )
    assert loaded["status"] == "prepared"
    assert claimed == expected
    assert path == manifest


def _fixture_inputs(root: Path) -> tuple[Path, list[Path]]:
    dataset_manifest = _dataset_manifest(root)
    dataset_sha = file_sha256(dataset_manifest)
    plan_payload = {
        "learning_rate": 1e-4,
        "burn_in_tokens": 8,
        "supervised_tokens": 16,
        "accumulation_steps": 2,
        "micro_batch_size": 1,
        "activation_fit_functional_steps": 8,
        "activation_fit_functional_learning_rate": 1e-4,
        "layer_min_epochs": 2,
        "layer_max_epochs": 4,
        "layer_min_delta": 0.001,
        "layer_patience": 1,
        "corrective_min_sweeps": 1,
        "corrective_max_sweeps": 2,
        "corrective_min_delta": 0.001,
        "local_loss_weights": {"mixer_mse": 1.0, "block_mse": 1.0, "cosine": 0.1},
        "global_loss_weights": {"token_kl": 1.0, "shifted_ce": 0.25},
    }
    plan = root / "distillation-plan.json"
    _write_json(plan, plan_payload)
    plan_sha = file_sha256(plan)
    control_bindings = {
        "teacher_checkpoint_sha256": "a" * 64,
        "dataset_sha256": dataset_sha,
        "code_commit": "fixture-commit",
        "recipe_id": "synthetic_to_synthetic",
        "metric_definition": "control-v1",
    }
    control_protocol_sha = "f" * 64
    fingerprint = training_control_fingerprint(plan_payload)
    control_candidates = []
    for candidate in (1, 2, 3):
        candidate_id = f"candidate-{candidate}"
        kl = 0.1 * candidate
        ppl = 1.0 + kl
        derived_kl = kl * 100 / 100
        derived_ppl = math.exp(math.log(ppl) * 100 / 100)
        runs = []
        for seed in (1, 2, 3):
            run_id = f"control-{candidate}-{seed}"
            checkpoint_sha = hashlib.sha256(run_id.encode()).hexdigest()
            curve = root / f"{run_id}-curve.json"
            curve_payload = {
                "schema_version": 1,
                "status": "complete",
                "protocol_sha256": control_protocol_sha,
                "precision": "bf16",
                "bindings": control_bindings,
                "candidate_id": candidate_id,
                "seed": seed,
                "run_id": run_id,
                "plan_sha256": plan_sha,
                "control_fingerprint": fingerprint,
                "student_checkpoint_sha256": checkpoint_sha,
                "expected_layer_count": 1,
                "layer_results": [
                    {
                        "layer_index": 0,
                        "converged": True,
                        "trained_token_count": 100,
                        "selected_validation": {
                            "normalized_mse": 0.1,
                            "cosine": 0.9,
                        },
                    }
                ],
                "final_validation": {
                    "kl_token_count": 100,
                    "token_kl_sum": kl * 100,
                    "nll_token_count": 100,
                    "nll_delta_sum": math.log(ppl) * 100,
                },
            }
            _write_json(curve, curve_payload)
            runs.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "student_checkpoint_sha256": checkpoint_sha,
                    "validation_curve_sha256": file_sha256(curve),
                    "validation_curve": curve_payload,
                    "validation_curve_reference": {
                        "artifact": curve.name,
                        "artifact_sha256": file_sha256(curve),
                    },
                    "validation_token_kl": derived_kl,
                    "ppl_ratio": derived_ppl,
                    "converged_layer_fraction": 1.0,
                    "token_budget": 100,
                    "result_sha256": str(candidate + seed) * 64,
                }
            )
        control_candidates.append(
            {
                "candidate_id": candidate_id,
                "plan_sha256": plan_sha,
                "control_fingerprint": fingerprint,
                "mean_validation_token_kl": sum([derived_kl] * 3) / 3,
                "mean_ppl_ratio": sum([derived_ppl] * 3) / 3,
                "runs": runs,
            }
        )
    control_artifact = root / "training-controls.json"
    _write_json(
        control_artifact,
        {
            "schema_version": 1,
            "status": "complete",
            "protocol_sha256": control_protocol_sha,
            "precision": "bf16",
            "bindings": control_bindings,
            "dataset_manifest": json.loads(dataset_manifest.read_text(encoding="utf-8")),
            "dataset_manifest_reference": {
                "artifact": dataset_manifest.name,
                "artifact_sha256": dataset_sha,
            },
            "selected_candidate_id": "candidate-1",
            "selected_plan_sha256": plan_sha,
            "selected_control_fingerprint": fingerprint,
            "candidates": control_candidates,
        },
    )
    control_sha = file_sha256(control_artifact)
    bindings = {
        "teacher_checkpoint_sha256": "a" * 64,
        "dataset_sha256": dataset_sha,
        "code_commit": "fixture-commit",
        "recipe_id": "synthetic_to_synthetic",
        "metric_definition": "quality-v1",
        "distillation_plan_sha256": plan_sha,
        "training_control_evidence_sha256": control_sha,
    }
    protocol = root / "protocol.json"
    _write_json(
        protocol,
        {
            "schema_version": 1,
            "status": "frozen",
            "frozen_before_calibration_runs": True,
            "profile_id": "fixture-profile",
            "precision": "bf16",
            "bindings": bindings,
            "bootstrap": {"samples": 10_000, "confidence": 0.95, "seed": 17},
            "training_control_evidence": {
                "artifact": control_artifact.name,
                "artifact_sha256": control_sha,
            },
            "distillation_plan": {
                "artifact": plan.name,
                "artifact_sha256": plan_sha,
            },
            "dataset_manifest": {
                "artifact": dataset_manifest.name,
                "artifact_sha256": dataset_sha,
            },
        },
    )
    protocol_sha = file_sha256(protocol)
    paths: list[Path] = []
    for seed, offset in ((3, 0.01), (1, 0.0), (2, 0.02)):
        eligibility = root / f"eligibility-{seed}.json"
        run_id = f"run-{seed}"
        student_sha = str(seed) * 64
        common_binding = {
            "protocol_sha256": protocol_sha,
            "precision": "bf16",
            "bindings": bindings,
            "seed": seed,
            "run_id": run_id,
            "student_checkpoint_sha256": student_sha,
        }
        baseline = root / f"baseline-{seed}.json"
        _write_json(
            baseline,
            {
                "schema_version": 1,
                "status": "complete",
                **common_binding,
                "metric": "validation_token_kl",
                "direction": "lower",
                "values": {"mapped": 0.2, "random": 0.8, "naive": 0.5},
            },
        )
        baseline_sha = file_sha256(baseline)
        sample_metrics = root / f"sample-metrics-{seed}.json"
        ppl_ratio = 1.01 + offset
        smoke_passes = 96 - round(offset * 100)
        _write_json(
            sample_metrics,
            {
                "schema_version": 1,
                "status": "complete",
                **common_binding,
                "warmed_lm_rows": [
                    {
                        "sample_id": "lm-0",
                        "token_count": 100,
                        "teacher_nll_sum": 100.0,
                        "student_nll_sum": 100.0 + math.log(ppl_ratio) * 100,
                        "token_kl_sum": (0.02 + offset) * 100,
                    }
                ],
                "layer_rows": [
                    {"layer_index": 0, "normalized_mse": 0.03 + offset, "cosine": 0.98 - offset},
                    {"layer_index": 1, "normalized_mse": 0.04 + offset, "cosine": 0.95 - offset},
                ],
                "smoke_rows": [
                    {"sample_id": f"smoke-{index}", "passed": index < smoke_passes}
                    for index in range(100)
                ],
                "ruler_rows": [
                    {
                        "sample_id": f"ruler-{index}",
                        "bucket": "short" if index < 5 else "long",
                        "teacher_score": 1.0,
                        "student_score": 0.95 - offset,
                    }
                    for index in range(10)
                ],
                "downstream_rows": [
                    {
                        "sample_id": f"downstream-{index}",
                        "task": "task-a" if index < 5 else "task-b",
                        "teacher_score": 1.0,
                        "student_score": 0.96 - offset,
                    }
                    for index in range(10)
                ],
            },
        )
        _write_json(
            eligibility,
            {
                "schema_version": 1,
                "status": "complete",
                "protocol_sha256": protocol_sha,
                "run_id": run_id,
                "p0_failures": [],
                "converged_layer_fraction": 1.0,
                "baseline_curve_sha256": baseline_sha,
                "migration_baselines": {
                    "metric": "validation_token_kl",
                    "direction": "lower",
                    "values": {"mapped": 0.2, "random": 0.8, "naive": 0.5},
                },
            },
        )
        path = root / f"seed-{seed}.json"
        _write_json(
            path,
            {
                "schema_version": 1,
                "status": "complete",
                "seed": seed,
                "run_id": run_id,
                "protocol_sha256": protocol_sha,
                "precision": "bf16",
                "bindings": bindings,
                "student_checkpoint_sha256": student_sha,
                "sample_metrics": {
                    "artifact": sample_metrics.name,
                    "artifact_sha256": file_sha256(sample_metrics),
                },
                "baseline_curve": {
                    "artifact": baseline.name,
                    "artifact_sha256": baseline_sha,
                },
                "eligibility_evidence": {
                    "artifact": eligibility.name,
                    "artifact_sha256": file_sha256(eligibility),
                },
            },
        )
        paths.append(path)
    return protocol, paths


def test_builder_derives_worst_eligible_seed_envelope(tmp_path: Path) -> None:
    protocol, seed_results = _fixture_inputs(tmp_path)
    artifact_path = tmp_path / "calibration.json"
    profile_path = tmp_path / "profile.json"
    artifact, profile = build_quality_threshold_profile(
        protocol_path=protocol,
        seed_result_paths=seed_results,
        artifact_path=artifact_path,
        profile_path=profile_path,
    )
    assert artifact["seeds"] == [1, 2, 3]
    assert profile["thresholds"]["P1"]["smoke_pass_rate_min"] == pytest.approx(0.94)
    assert profile["thresholds"]["P1"]["ppl_ratio_max"] == pytest.approx(1.03)
    provenance = artifact["threshold_derivation"]["threshold_provenance"]
    assert provenance["P1"]["ppl_ratio_max"]["limiting_run_ids"] == ["run-2"]
    assert len(provenance["P1"]["ppl_ratio_max"]["observations"]) == 3
    loaded = read_quality_threshold_profile(profile_path)
    assert loaded.evidence_verified
    assert loaded.calibration_student_sha256s == ("1" * 64, "2" * 64, "3" * 64)


def test_builder_rejects_duplicate_calibration_student(tmp_path: Path) -> None:
    protocol, seed_results = _fixture_inputs(tmp_path)
    duplicate = json.loads(seed_results[1].read_text(encoding="utf-8"))
    duplicate["student_checkpoint_sha256"] = "3" * 64
    _write_json(seed_results[1], duplicate)
    with pytest.raises(ValueError, match="identities must be unique"):
        build_quality_threshold_profile(
            protocol_path=protocol,
            seed_result_paths=seed_results,
            artifact_path=tmp_path / "calibration.json",
            profile_path=tmp_path / "profile.json",
        )


def test_builder_rejects_unverified_or_non_superior_eligibility(tmp_path: Path) -> None:
    protocol, seed_results = _fixture_inputs(tmp_path)
    result = json.loads(seed_results[0].read_text(encoding="utf-8"))
    eligibility_path = tmp_path / result["eligibility_evidence"]["artifact"]
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    eligibility["migration_baselines"]["values"]["mapped"] = 0.9
    _write_json(eligibility_path, eligibility)
    result["eligibility_evidence"]["artifact_sha256"] = file_sha256(eligibility_path)
    _write_json(seed_results[0], result)
    with pytest.raises(ValueError, match="does not match|does not beat"):
        build_quality_threshold_profile(
            protocol_path=protocol,
            seed_result_paths=seed_results,
            artifact_path=tmp_path / "calibration.json",
            profile_path=tmp_path / "profile.json",
        )


def test_builder_rejects_reused_training_control_run(tmp_path: Path) -> None:
    protocol, seed_results = _fixture_inputs(tmp_path)
    reused = json.loads(seed_results[0].read_text(encoding="utf-8"))
    reused["run_id"] = "control-1-1"
    _write_json(seed_results[0], reused)
    with pytest.raises(ValueError, match="cannot reuse"):
        build_quality_threshold_profile(
            protocol_path=protocol,
            seed_result_paths=seed_results,
            artifact_path=tmp_path / "calibration.json",
            profile_path=tmp_path / "profile.json",
        )


def test_profile_reload_rejects_tampered_raw_sample_metrics(tmp_path: Path) -> None:
    protocol, seed_results = _fixture_inputs(tmp_path)
    artifact_path = tmp_path / "calibration.json"
    profile_path = tmp_path / "profile.json"
    build_quality_threshold_profile(
        protocol_path=protocol,
        seed_result_paths=seed_results,
        artifact_path=artifact_path,
        profile_path=profile_path,
    )
    result = json.loads(seed_results[0].read_text(encoding="utf-8"))
    sample_path = tmp_path / result["sample_metrics"]["artifact"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample["warmed_lm_rows"][0]["token_kl_sum"] += 1.0
    _write_json(sample_path, sample)
    with pytest.raises(ValueError, match="evidence artifact reference"):
        read_quality_threshold_profile(profile_path)
