from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .artifacts import checkpoint_sha256, file_sha256, write_json
from .errors import ContractError
from .evaluate import P0_REQUIRED


@dataclass(frozen=True)
class P0ValidationInputs:
    run_dir: Path
    student: Path
    kernel_oracle: Path
    package_root: Path


def run_p0_validation(inputs: P0ValidationInputs) -> dict[str, object]:
    """Execute and bind every P0 invariant to one concrete student checkpoint."""
    student_sha = checkpoint_sha256(inputs.student)
    evidence_dir = inputs.run_dir / "p0-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, object]] = {}

    for name, nodeids in _PYTEST_EVIDENCE.items():
        extra = _validate_run_artifact(inputs.run_dir, name)
        result = _run_pytest(inputs.package_root, nodeids)
        artifact = _artifact_path(inputs.run_dir, evidence_dir, name)
        payload = {
            "schema_version": 1,
            "kind": name,
            "passed": result.returncode == 0,
            "student_sha256": student_sha,
            "command": result.args,
            "nodeids": list(nodeids),
            "test_source_sha256": {
                str(Path(nodeid.split("::", 1)[0])): file_sha256(
                    inputs.package_root / nodeid.split("::", 1)[0]
                )
                for nodeid in nodeids
            },
            "stdout": result.stdout,
            "run_artifact": extra,
        }
        write_json(artifact, payload)
        if result.returncode:
            raise ContractError(f"P0 {name} failed; see {artifact}")
        entries[name] = _entry(inputs.run_dir, artifact, student_sha)

    kernel_payload = json.loads(inputs.kernel_oracle.read_text(encoding="utf-8"))
    kernel_passed = (
        kernel_payload.get("status") == "pass"
        and kernel_payload.get("kernel") == "rwkv-lm/RWKV7_STATEPASSING_CLAMPW_CUDA"
        and all(kernel_payload.get("gradient_finite", ()))
    )
    kernel_artifact = inputs.run_dir / "kernel-oracle.json"
    write_json(
        kernel_artifact,
        {
            "schema_version": 1,
            "kind": "kernel_oracle",
            "passed": kernel_passed,
            "student_sha256": student_sha,
            "source_path": str(inputs.kernel_oracle),
            "source_sha256": file_sha256(inputs.kernel_oracle),
            "result": kernel_payload,
        },
    )
    if not kernel_passed:
        raise ContractError(f"native kernel oracle failed; see {kernel_artifact}")
    entries["kernel_oracle"] = _entry(inputs.run_dir, kernel_artifact, student_sha)

    roundtrip_artifact = evidence_dir / "hf_roundtrip.json"
    first = _run_roundtrip(inputs, evidence_dir / "roundtrip-first.json")
    second = _run_roundtrip(inputs, evidence_dir / "roundtrip-second.json")
    comparable = ("greedy_digest", "logits_digest", "ppl")
    roundtrip_passed = all(first[key] == second[key] for key in comparable)
    write_json(
        roundtrip_artifact,
        {
            "schema_version": 1,
            "kind": "hf_roundtrip",
            "passed": roundtrip_passed,
            "student_sha256": student_sha,
            "fresh_process_runs": [first, second],
            "compared": list(comparable),
        },
    )
    if not roundtrip_passed:
        raise ContractError(f"fresh-process HF roundtrip differs; see {roundtrip_artifact}")
    entries["hf_roundtrip"] = _entry(inputs.run_dir, roundtrip_artifact, student_sha)

    missing = sorted(set(P0_REQUIRED) - entries.keys())
    if missing:
        raise ContractError(f"P0 producer has no implementation for {missing}")
    manifest = {
        "schema_version": 1,
        "student_sha256": student_sha,
        "evidence": entries,
    }
    write_json(inputs.run_dir / "p0-evidence.json", manifest)
    return manifest


_PYTEST_EVIDENCE: dict[str, tuple[str, ...]] = {
    "canonical_state": (
        "tests/test_recurrent.py::RecurrentTests::test_full_chunked_and_decode_are_equivalent",
        "tests/test_recurrent.py::RecurrentTests::test_zero_reset_and_native_decay_inverse",
        "tests/test_modeling.py::ModelingTests::test_prefill_and_token_decode_are_equivalent",
    ),
    "mapping_coverage": (
        "tests/test_mapping_distill.py::MappingTests::test_bidirectional_coverage_and_taxonomy",
        "tests/test_target.py::TargetMappingTests::test_zero_step_ledger_preserves_shell_and_excludes_vision_explicitly",
    ),
    "gdn_oracle": (
        "tests/test_oracle.py::OracleTests::test_frozen_32_case_fp64_oracle_passes",
    ),
    "full_attention_fixture": (
        "tests/test_recurrent.py::RecurrentTests::test_attention_trace_contract_requires_position_mask_and_multiple_contexts",
        "tests/test_migration_init.py::MigrationInitializationTests::test_head_factored_projection_has_explicit_reshape_provenance",
    ),
    "gqa_fixture": (
        "tests/test_recurrent.py::RecurrentTests::test_gqa_mapping_and_baselines_are_explicit",
        "tests/test_migration_init.py::MigrationInitializationTests::test_full_attention_kv_baselines_are_numerically_distinct_and_group_aware",
        "tests/test_migration_init.py::MigrationInitializationTests::test_ambiguous_gqa_layout_is_rejected_before_materialization",
    ),
    "loss_bridge": (
        "tests/test_hybrid.py::HybridCheckpointTests::test_frozen_suffix_checkpoint_keeps_teacher_eval_and_gradient_bridge",
        "tests/test_mapping_distill.py::DistillationInvariantTests::test_frozen_teacher_suffix_keeps_global_gradient_bridge",
    ),
    "active_layer_invariant": (
        "tests/test_mapping_distill.py::DistillationInvariantTests::test_only_active_layer_gets_gradient_weight_and_optimizer_state",
        "tests/test_scale_streaming.py::LayerTensorStoreTests::test_streamed_hybrid_reloads_suffix_and_bridges_gradient_to_active_mixer",
    ),
    "resume_parity": (
        "tests/test_mapping_distill.py::DistillationInvariantTests::test_resume_matches_uninterrupted_updates",
        "tests/test_mapping_distill.py::DistillationInvariantTests::test_atomic_checkpoint_restores_scaler_rng_cursor_and_metadata",
        "tests/test_mapping_distill.py::DistillationInvariantTests::test_sharded_checkpoint_rewrites_only_active_layer_and_restores_accumulation",
        "tests/test_mapping_distill.py::DistillationInvariantTests::test_resume_preserves_mid_accumulation_gradient_and_sweep_cursor",
        "tests/test_scale_streaming.py::ActiveLayerOptimizerTests::test_mid_accumulation_snapshot_restores_without_resident_optimizer_list",
        "tests/test_scale_streaming.py::RWKV7MixerLayerStoreTests::test_sweep_snapshot_restores_selected_all_layer_overlay",
    ),
}


def _run_pytest(package_root: Path, nodeids: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *nodeids],
        cwd=package_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _validate_run_artifact(run_dir: Path, name: str) -> dict[str, object] | None:
    if name != "mapping_coverage":
        return None
    path = run_dir / "mapping-coverage.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_coverage") != 1.0 or payload.get("target_coverage") != 1.0:
        raise ContractError("actual run mapping coverage is incomplete")
    return {"path": path.name, "sha256": file_sha256(path), "payload": payload}


def _run_roundtrip(inputs: P0ValidationInputs, output: Path) -> dict[str, object]:
    script = inputs.package_root / "scripts/validate_hf_roundtrip.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(inputs.student),
            "--output",
            str(output),
        ],
        cwd=inputs.package_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ContractError(f"HF roundtrip subprocess failed: {result.stdout}\n{result.stderr}")
    return json.loads(output.read_text(encoding="utf-8"))


def _entry(run_dir: Path, artifact: Path, student_sha: str) -> dict[str, object]:
    return {
        "path": str(artifact.relative_to(run_dir)),
        "sha256": file_sha256(artifact),
        "student_sha256": student_sha,
        "passed": True,
    }


def _artifact_path(run_dir: Path, evidence_dir: Path, name: str) -> Path:
    required_names = {
        "loss_bridge": "loss-bridge.json",
        "resume_parity": "resume-parity.json",
    }
    filename = required_names.get(name)
    return evidence_dir / f"{name}.json" if filename is None else run_dir / filename
