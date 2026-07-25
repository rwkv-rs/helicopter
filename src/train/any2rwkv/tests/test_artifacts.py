from __future__ import annotations

import json
from pathlib import Path

import pytest

from any2rwkv import artifacts
from any2rwkv.artifacts import (
    default_contract_lock,
    git_sha,
    require_independent_run_output,
    verify_scale_gate,
)
from any2rwkv.cli import build_parser


def test_contract_uses_transformers_without_serving_or_quantization() -> None:
    lock = default_contract_lock()
    assert lock["inference"]["backend"] == "transformers"
    encoded = json.dumps(lock).lower()
    assert "vllm" not in encoded
    assert "nvfp4" not in encoded


def test_cli_exposes_only_bf16_conversion_training_and_evaluation() -> None:
    parser = build_parser()
    help_text = parser.format_help().lower()
    assert "quantize" not in help_text
    assert "vllm" not in help_text


def test_product_contract_has_no_serving_reference() -> None:
    product_root = Path(__file__).resolve().parents[4]
    lock = default_contract_lock(product_root)
    paths = json.dumps(lock.get("oracle", {}).get("reference_files", {})).lower()
    assert "src/infer" not in paths
    assert "vllm" not in paths


def test_scale_gate_binds_transformers_inference_without_service_artifact(
    tmp_path: Path, monkeypatch,
) -> None:
    student_sha = "a" * 64
    payloads = {
        "quality.json": {
            "gates": {
                gate: {"passed": True}
                for gate in ("P0", "migration", "P1")
            }
        },
        "p0-evidence.json": {"student_sha256": student_sha},
        "transformers-inference.json": {
            "schema_version": 1,
            "passed": True,
            "model_sha256": student_sha,
            "backend": "transformers",
            "strict_reload": True,
            "single_batch_greedy": True,
            "full_chunked_cache": True,
            "state_reset": True,
            "batch_isolation": True,
        },
        "smoke-rubric.json": {
            "student_sha256": student_sha,
            "passed": True,
        },
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(artifacts, "verify_run_bundle", lambda _output: [])

    evidence = verify_scale_gate(tmp_path)

    assert "transformers_inference_sha256" in evidence
    assert "service_sha256" not in evidence


def test_git_sha_prefers_managed_sync_manifest_for_product_and_submodule(
    tmp_path: Path,
) -> None:
    product = tmp_path / "product"
    submodule = product / "src/train/rwkv-lm"
    submodule.mkdir(parents=True)
    revisions = product / ".helicopter-dev/source-revisions.json"
    revisions.parent.mkdir()
    revisions.write_text(
        json.dumps(
            {
                "product_commit": "1" * 40,
                "submodules": {"src/train/rwkv-lm": "2" * 40},
            }
        ),
        encoding="utf-8",
    )

    assert git_sha(product) == "1" * 40
    assert git_sha(submodule) == "2" * 40


def test_run_output_must_not_overlap_read_only_source_tree(tmp_path: Path) -> None:
    source = tmp_path / "weights" / ("a" * 40)
    source.mkdir(parents=True)
    require_independent_run_output(tmp_path / "runs" / "scale-001", source)
    for output in (source, source / "run", tmp_path / "weights"):
        with pytest.raises(ValueError, match="independent directory trees"):
            require_independent_run_output(output, source)
