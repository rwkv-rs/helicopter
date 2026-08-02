from __future__ import annotations

import json
from pathlib import Path

import pytest

from any2rwkv.artifacts import checkpoint_sha256, file_sha256
from any2rwkv.core.migration_baselines import (
    build_migration_baseline_matrix,
    write_migration_baseline_stage,
)
from any2rwkv.distill import MIGRATION_BASELINE_STAGES
from any2rwkv.errors import ContractError
from any2rwkv.evaluator_runner import read_migration_baselines


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fixture-weights")
    return path


def _file_reference(path: Path) -> dict[str, str]:
    return {
        "path": path.name,
        "kind": "file",
        "sha256": file_sha256(path),
    }


def _write_stage_set(
    root: Path, *, changed_stage: str | None = None
) -> tuple[list[Path], Path]:
    candidate = _checkpoint(root / "candidate")
    fit_report = root / "fit-report.json"
    fit_report.write_text(
        json.dumps(
            {
                "status": "accepted",
                "selected_module_state_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    curve = root / "training-curve.json"
    curve.write_text('{"status":"complete"}\n', encoding="utf-8")
    protocol = {
        "teacher_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "split": "validation",
        "seed": 20260725,
        "burn_in_tokens": 16,
        "precision": "bf16-fp32-state",
    }
    manifests = []
    for index, stage in enumerate(MIGRATION_BASELINE_STAGES):
        sample_rows = [
            {
                "sample_id": (
                    "changed"
                    if changed_stage == stage and sample_index == 0
                    else f"sample-{sample_index}"
                ),
                "token_count": 8,
                "token_kl_sum": 0.0 if stage == "teacher" else 0.1 + index,
                "end_to_end_zero_step_nmse": 0.0 if stage == "teacher" else 0.01 + index,
                "incremental_nmse": {
                    "stage_increment": (
                        0.0 if stage == "teacher" else 0.001 + index
                    )
                },
            }
            for sample_index in range(2)
        ]
        evidence: dict[str, object] = {"artifacts": {}}
        if stage.startswith("gqa_"):
            evidence["trace_sha256"] = "d" * 64
        if stage == "gqa_exact_hazard_oracle":
            evidence["counterfactual"] = "oracle-signal"
        if stage == "activation_fitted":
            evidence.update(
                {
                    "solver_invoked": True,
                    "artifacts": {
                        "fit_report": _file_reference(fit_report),
                        "materialization": {
                            "path": candidate.name,
                            "kind": "checkpoint",
                            "sha256": checkpoint_sha256(candidate),
                        },
                    },
                }
            )
        if stage == "fully_recurrent" or stage.startswith("corrective_sweep_"):
            evidence["artifacts"] = {"training_curve": _file_reference(curve)}
        manifest = root / f"{stage}.json"
        write_migration_baseline_stage(
            manifest,
            stage=stage,
            protocol=protocol,
            sample_rows=sample_rows,
            candidate_path=candidate,
            candidate_kind="checkpoint",
            evidence=evidence,
        )
        manifests.append(manifest)
    return manifests, candidate


def test_matrix_recomputes_raw_samples_and_verifies_solver_artifacts(
    tmp_path: Path,
) -> None:
    manifests, student = _write_stage_set(tmp_path)
    output = tmp_path / "migration-baselines.json"
    result = build_migration_baseline_matrix(
        manifests,
        student_checkpoint=student,
        output=output,
    )

    assert set(result["baselines"]) == set(MIGRATION_BASELINE_STAGES)
    assert result["baselines"]["teacher"]["mean_token_kl"] == 0
    assert (
        result["baselines"]["gqa_bounded_hazard"][
            "incremental_zero_step_nmse"
        ]["stage_increment"]
        > 0
    )
    assert result["baselines"]["activation_fitted"]["solver_invoked"] is True
    assert (
        read_migration_baselines(
            output, student_sha256=result["student_sha256"]
        )["activation_fitted"]
        > 0
    )
    with pytest.raises(ValueError, match="different student"):
        read_migration_baselines(output, student_sha256="0" * 64)

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["baselines"]["gqa_bounded_hazard"][
        "end_to_end_zero_step_nmse"
    ] += 1
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="aggregate differs from raw stage evidence",
    ):
        read_migration_baselines(
            output,
            student_sha256=result["student_sha256"],
        )


def test_matrix_rejects_stage_with_different_sample_ids(tmp_path: Path) -> None:
    manifests, student = _write_stage_set(
        tmp_path, changed_stage="gqa_bounded_hazard"
    )
    with pytest.raises(ContractError, match="same sample/token protocol"):
        build_migration_baseline_matrix(
            manifests,
            student_checkpoint=student,
            output=tmp_path / "migration-baselines.json",
        )


def test_matrix_rejects_unaccepted_activation_fit_report(tmp_path: Path) -> None:
    manifests, student = _write_stage_set(tmp_path)
    fit_report = tmp_path / "fit-report.json"
    fit_report.write_text(
        json.dumps(
            {
                "status": "rejected",
                "selected_module_state_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    activation_manifest = tmp_path / "activation_fitted.json"
    payload = json.loads(activation_manifest.read_text(encoding="utf-8"))
    payload["evidence"]["artifacts"]["fit_report"]["sha256"] = file_sha256(
        fit_report
    )
    activation_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="not an accepted installed generation"):
        build_migration_baseline_matrix(
            manifests,
            student_checkpoint=student,
            output=tmp_path / "migration-baselines.json",
        )
