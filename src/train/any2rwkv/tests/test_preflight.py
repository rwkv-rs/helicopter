from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.fixture import write_fixture
from any2rwkv.preflight import (
    TRANSFORMERS_REVISION,
    TRANSFORMERS_SOURCE_URL,
    _distribution_binding,
    collect_full_loop_preflight,
    collect_preflight,
)

PRODUCT_ROOT = Path(__file__).resolve().parents[4]


def _exact_distribution(
    name: str,
    *,
    expected_url: str,
    expected_revision: str,
) -> dict[str, object]:
    return {
        "name": name,
        "version": "test",
        "direct_url": expected_url,
        "direct_url_error": None,
        "vcs": "git",
        "requested_revision": expected_revision,
        "commit_id": expected_revision,
        "expected_url": expected_url,
        "expected_revision": expected_revision,
        "source_matches": True,
        "requested_revision_matches": True,
        "revision_matches": True,
        "requirement_satisfied": True,
    }


def test_preflight_calls_public_transformers_rwkv7_provenance(monkeypatch) -> None:
    calls = []
    runtime_manifest = {
        "repository": "https://github.com/rwkv-rs/fla-rwkv.git",
        "revision": "a4a8aa98df6ec5322f194a80ec57363dd045adfc",
        "flash_rwkv_revision": "866aafd2eed146b0eda1ce03444009ae030f89e3",
    }
    public_rwkv7 = SimpleNamespace(
        Rwkv7Config=SimpleNamespace(model_type="rwkv7"),
        Rwkv7ForCausalLM=SimpleNamespace(base_model_prefix="model"),
        validate_rwkv7_runtime_provenance=lambda: (
            calls.append("validated") or runtime_manifest
        ),
    )
    monkeypatch.setattr(
        "any2rwkv.preflight._distribution_binding",
        _exact_distribution,
    )
    monkeypatch.setattr(
        "any2rwkv.preflight.importlib.import_module",
        lambda name: public_rwkv7
        if name == "transformers.models.rwkv7"
        else pytest.fail(f"unexpected import: {name}"),
    )

    result = collect_preflight()

    assert calls == ["validated"]
    assert "rwkv_hf" not in result
    assert "rwkv_lm" not in result
    assert result["transformers"]["distribution"]["expected_url"] == (
        TRANSFORMERS_SOURCE_URL
    )
    assert result["transformers"]["distribution"]["commit_id"] == (
        TRANSFORMERS_REVISION
    )
    assert result["transformers"]["public_interface"] is True
    assert result["transformers"]["runtime_provenance"] == runtime_manifest
    assert result["transformers"]["runtime_provenance_error"] is None
    assert result["transformers"]["requirement_satisfied"] is True


def test_preflight_fails_closed_when_public_runtime_provenance_rejects(
    monkeypatch,
) -> None:
    def reject_runtime() -> None:
        raise RuntimeError("FlashRWKV revision provenance mismatch")

    public_rwkv7 = SimpleNamespace(
        Rwkv7Config=SimpleNamespace(model_type="rwkv7"),
        Rwkv7ForCausalLM=SimpleNamespace(base_model_prefix="model"),
        validate_rwkv7_runtime_provenance=reject_runtime,
    )
    monkeypatch.setattr(
        "any2rwkv.preflight._distribution_binding",
        _exact_distribution,
    )
    monkeypatch.setattr(
        "any2rwkv.preflight.importlib.import_module",
        lambda _name: public_rwkv7,
    )

    result = collect_preflight()

    assert result["transformers"]["runtime_provenance"] is None
    assert "FlashRWKV revision provenance mismatch" in result["transformers"][
        "runtime_provenance_error"
    ]
    assert result["transformers"]["requirement_satisfied"] is False
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("url", "requested_revision", "commit_id", "satisfied"),
    [
        (
            TRANSFORMERS_SOURCE_URL,
            TRANSFORMERS_REVISION,
            TRANSFORMERS_REVISION,
            True,
        ),
        (
            "https://github.com/huggingface/transformers.git",
            TRANSFORMERS_REVISION,
            TRANSFORMERS_REVISION,
            False,
        ),
        (TRANSFORMERS_SOURCE_URL, "feature/rwkv7", TRANSFORMERS_REVISION, False),
        (TRANSFORMERS_SOURCE_URL, TRANSFORMERS_REVISION, "0" * 40, False),
    ],
)
def test_distribution_binding_requires_exact_vcs_url_and_revision(
    monkeypatch,
    url: str,
    requested_revision: str,
    commit_id: str,
    satisfied: bool,
) -> None:
    class Distribution:
        version = "5.15.0.dev0"

        @staticmethod
        def read_text(name: str) -> str | None:
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": url,
                    "vcs_info": {
                        "vcs": "git",
                        "requested_revision": requested_revision,
                        "commit_id": commit_id,
                    },
                }
            )

    monkeypatch.setattr(
        "any2rwkv.preflight.importlib.metadata.distribution",
        lambda _name: Distribution(),
    )

    binding = _distribution_binding(
        "transformers",
        expected_url=TRANSFORMERS_SOURCE_URL,
        expected_revision=TRANSFORMERS_REVISION,
    )

    assert binding["requirement_satisfied"] is satisfied


def test_distribution_binding_diagnoses_a_missing_distribution(monkeypatch) -> None:
    def missing(_name: str):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(
        "any2rwkv.preflight.importlib.metadata.distribution",
        missing,
    )

    binding = _distribution_binding(
        "transformers",
        expected_url=TRANSFORMERS_SOURCE_URL,
        expected_revision=TRANSFORMERS_REVISION,
    )

    assert binding["version"] is None
    assert binding["direct_url_error"] == "distribution is not installed"
    assert binding["requirement_satisfied"] is False


def test_full_loop_preflight_reports_every_missing_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "any2rwkv.preflight.collect_preflight",
        lambda: {
            "torch": {"cuda_available": False},
            "transformers": {
                "distribution": {"direct_url_error": "not installed"},
                "requirement_satisfied": False,
            },
        },
    )
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        '{"remote_read_only_path":"/missing/frozen-source"}\n',
        encoding="utf-8",
    )
    raw_manifest = tmp_path / "raw.json"
    raw_manifest.write_text("{}\n", encoding="utf-8")
    result = collect_full_loop_preflight(
        recipe_id="qwen35_to_rwkv7",
        source_manifest_path=source_manifest,
        source_path=Path("/missing/frozen-source"),
        raw_data_manifest_path=raw_manifest,
        dataset_manifest_path=tmp_path / "prepared" / "data-splits.json",
        training_config_path=tmp_path / "training.json",
        lighteval_config_path=tmp_path / "lighteval.toml",
        evalscope_config_path=tmp_path / "evalscope.yaml",
        allow_proxy_layers=True,
        precision="fp32io16",
    )

    assert result["passed"] is False
    assert result["status"] == "blocked"
    blockers = "\n".join(result["blockers"])
    assert "transformers distribution does not satisfy exact requirement" in blockers
    assert "CUDA is unavailable for the Any-to-RWKV architecture conversion" in blockers
    assert "source:" in blockers
    assert "raw_data:" in blockers
    assert "training_config:" in blockers
    assert "prepared_data:" in blockers
    assert "lighteval: config is missing" in blockers
    assert "evalscope: config is missing" in blockers
    assert result["evaluation"]["checkpoints"] == (
        "source",
        "zero-step",
        "distilled",
    )
    assert result["source"]["manifest_sha256"]
    assert result["source"]["verified"] is None
    assert result["resolved_assets"]["source"]["checkpoint"] == (
        "/missing/frozen-source"
    )
    assert result["resolved_assets"]["training"]["config_sha256"] is None


def test_full_loop_preflight_accepts_portable_source_and_positive_world_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = write_fixture(tmp_path / "materialized", layers=4)
    checkpoint = read_checkpoint(source, require_final_layers=False)
    revision = "d" * 40
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "classification": "real-proxy-model-not-60-layer-isomorphic",
                "repository": "fixture/qwen35",
                "revision": revision,
                "weight_file": "model.safetensors",
                "weight_sha256": checkpoint.file_hashes["model.safetensors"],
                "files": checkpoint.file_hashes,
                "remote_read_only_path": str(tmp_path / revision),
            }
        ),
        encoding="utf-8",
    )
    training_config = tmp_path / "training.json"
    plan = json.loads(
        (
            PRODUCT_ROOT
            / "src/train/any2rwkv/manifests/qwen35-2b-v42-adamw-user-baseline-plan.json"
        ).read_text(encoding="utf-8")
    )
    plan["distributed_world_size"] = 1
    training_config.write_text(json.dumps(plan), encoding="utf-8")
    raw_manifest = tmp_path / "raw.json"
    raw_manifest.write_text("{}\n", encoding="utf-8")
    inspection = SimpleNamespace(
        adapter_id="qwen35",
        num_layers=4,
        hidden_size=64,
        metadata={"model_id": "fixture/qwen35"},
    )
    monkeypatch.setattr(
        "any2rwkv.recipes.resolve_recipe",
        lambda _: SimpleNamespace(
            source=SimpleNamespace(
                inspect_checkpoint=lambda *_args, **_kwargs: inspection
            ),
            recipe=SimpleNamespace(validate_source=lambda _inspection: None),
        ),
    )

    result = collect_full_loop_preflight(
        recipe_id="qwen35_to_rwkv7",
        source_manifest_path=source_manifest,
        source_path=source,
        raw_data_manifest_path=raw_manifest,
        dataset_manifest_path=tmp_path / "prepared.json",
        training_config_path=training_config,
        lighteval_config_path=tmp_path / "lighteval.toml",
        evalscope_config_path=tmp_path / "evalscope.yaml",
        allow_proxy_layers=True,
        precision="fp32io16",
    )

    assert result["source"]["verified"], [
        blocker for blocker in result["blockers"] if blocker.startswith("source:")
    ]
    assert result["source"]["verified"]["repository"] == "fixture/qwen35"
    assert result["source"]["verified"]["revision"] == revision
    assert result["source"]["verified"]["equivalent_materialization"] is True
    assert result["training"]["layer_major"]["world_size"] == 1
    assert not any("8 ranks" in blocker for blocker in result["blockers"])
