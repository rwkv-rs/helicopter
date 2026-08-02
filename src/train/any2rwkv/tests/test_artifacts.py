from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from any2rwkv import artifacts
from any2rwkv import cli as cli_module
from any2rwkv.artifacts import (
    default_contract_lock,
    git_sha,
    require_independent_run_output,
    verify_scale_gate,
)
from any2rwkv.cli import build_parser
from any2rwkv.errors import ContractError


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


def test_cli_exposes_formal_gqa_validation_as_an_explicit_stage() -> None:
    args = build_parser().parse_args(
        [
            "validate-gqa-zero-step",
            "--source",
            "/weights/source",
            "--recipe",
            "qwen35_to_rwkv7",
            "--output",
            "/runs/converted",
            "--evidence-output",
            "/runs/gqa-evidence",
            "--dataset-manifest",
            "/data/splits.json",
            "--training-config",
            "/plans/gqa.json",
            "--layer",
            "3",
            "--precision",
            "bf16",
            "--rwkv-hf-sha",
            "a" * 40,
            "--rwkv-lm-sha",
            "b" * 40,
        ]
    )

    assert args.action == "validate-gqa-zero-step"
    assert args.layer == 3
    assert args.evidence_output == "/runs/gqa-evidence"


def test_distill_cli_exposes_positive_resumable_optimizer_step_limit() -> None:
    argv = [
        "distill",
        "--source",
        "/weights/source",
        "--recipe",
        "qwen35_to_rwkv7",
        "--output",
        "/runs/converted",
        "--dataset-manifest",
        "/data/splits.json",
        "--training-config",
        "/plans/first-layer.json",
        "--precision",
        "fp32io16",
        "--rwkv-hf-sha",
        "a" * 40,
        "--rwkv-lm-sha",
        "b" * 40,
        "--stop-after-optimizer-steps",
        "2",
    ]

    args = build_parser().parse_args(argv)

    assert args.stop_after_optimizer_steps == 2
    invalid = [*argv[:-1], "0"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(invalid)


def test_gqa_metadata_publish_failure_propagates_without_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    metadata = {
        "precision": "bf16",
        "submodules": {
            "rwkv-hf": "a" * 40,
            "rwkv-lm": "b" * 40,
        },
        "recipe": {
            "id": "qwen35_to_rwkv7",
            "source_adapter": "qwen35",
            "target_adapter": "rwkv7",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    class FakeDistributed:
        is_primary = True

        def broadcast_object(self, value):
            return value

        def barrier(self):
            raise AssertionError("metadata failure must not enter a barrier")

        def close(self):
            return None

    monkeypatch.setattr(
        cli_module,
        "resolve_recipe",
        lambda _recipe, **_adapter_ids: SimpleNamespace(
            recipe=SimpleNamespace(recipe_id="qwen35_to_rwkv7")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "run_gqa_zero_step_validation",
        lambda **_kwargs: {"status": "accepted"},
    )
    monkeypatch.setattr(
        cli_module.DistributedContext,
        "initialize",
        lambda: FakeDistributed(),
    )
    monkeypatch.setattr(
        cli_module,
        "write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected metadata failure")
        ),
    )
    args = SimpleNamespace(
        action="validate-gqa-zero-step",
        recipe="qwen35_to_rwkv7",
        output=str(output),
        source="/weights/source",
        evidence_output="/runs/evidence",
        dataset_manifest="/data/splits.json",
        training_config="/plans/gqa.json",
        layer=3,
        precision="bf16",
        rwkv_hf_sha="a" * 40,
        rwkv_lm_sha="b" * 40,
        allow_proxy_layers=False,
    )

    with pytest.raises(
        ContractError,
        match="GQA validation metadata publish failed",
    ):
        cli_module.run_existing_stage(args)


def test_existing_stage_resolves_persisted_adapter_binding_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "submodules": {
                    "rwkv-hf": "a" * 40,
                    "rwkv-lm": "b" * 40,
                },
                "recipe": {
                    "id": "qwen35_to_rwkv7",
                    "source_adapter": "unknown-source",
                    "target_adapter": "rwkv7",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli_module,
        "run_distillation",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("binding failure must precede distillation work")
        ),
    )
    args = SimpleNamespace(
        action="distill",
        recipe="qwen35_to_rwkv7",
        output=str(output),
        rwkv_hf_sha="a" * 40,
        rwkv_lm_sha="b" * 40,
    )

    with pytest.raises(ContractError, match="unknown source adapter"):
        cli_module.run_existing_stage(args)


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
