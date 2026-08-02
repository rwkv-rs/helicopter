from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.errors import ContractError
from any2rwkv.fixture import write_fixture
from any2rwkv.preflight import (
    TRANSFORMERS_REVISION,
    TRANSFORMERS_SOURCE_URL,
    _distribution_binding,
    _require_module_ownership,
    collect_full_loop_preflight,
    collect_preflight,
)
from any2rwkv.provenance import (
    canonical_github_repository,
    github_repository_matches,
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


def _recurrent_rwkv7(
    r,
    w,
    k,
    v,
    a,
    b,
    scale=1.0,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    state_indices=None,
    mode="fp32io16",
):
    del w, k, v, a, b, scale, output_final_state, cu_seqlens, state_indices, mode
    return r, initial_state


def test_preflight_calls_public_fla_recurrent_provenance(monkeypatch) -> None:
    calls = []
    flash_provenance = SimpleNamespace(
        repository="https://github.com/rwkv-rs/FlashRWKV.git",
        revision="866aafd2eed146b0eda1ce03444009ae030f89e3",
    )
    public_recurrent = SimpleNamespace(
        recurrent_rwkv7=_recurrent_rwkv7,
        get_last_rwkv7_provider=lambda: "flash_rwkv",
        validate_flash_rwkv_installation=lambda: (
            calls.append("validated") or flash_provenance
        ),
    )
    monkeypatch.setattr(
        "any2rwkv.preflight._distribution_binding",
        _exact_distribution,
    )
    monkeypatch.setattr(
        "any2rwkv.preflight._require_module_ownership",
        lambda *_args: {"verified": True},
    )
    monkeypatch.setattr(
        "any2rwkv.preflight.importlib.import_module",
        lambda name: (
            public_recurrent
            if name == "fla.ops.rwkv7"
            else pytest.fail(f"unexpected import: {name}")
        ),
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
    assert result["transformers"]["runtime_provenance"]["operation"] == (
        "fla.ops.rwkv7.recurrent_rwkv7"
    )
    assert result["transformers"]["runtime_provenance_error"] is None
    assert result["transformers"]["requirement_satisfied"] is True


def test_preflight_fails_closed_when_public_runtime_provenance_rejects(
    monkeypatch,
) -> None:
    def reject_runtime() -> None:
        raise RuntimeError("FlashRWKV revision provenance mismatch")

    public_recurrent = SimpleNamespace(
        recurrent_rwkv7=_recurrent_rwkv7,
        get_last_rwkv7_provider=lambda: None,
        validate_flash_rwkv_installation=reject_runtime,
    )
    monkeypatch.setattr(
        "any2rwkv.preflight._distribution_binding",
        _exact_distribution,
    )
    monkeypatch.setattr(
        "any2rwkv.preflight._require_module_ownership",
        lambda *_args: {"verified": True},
    )
    monkeypatch.setattr(
        "any2rwkv.preflight.importlib.import_module",
        lambda _name: public_recurrent,
    )

    result = collect_preflight()

    assert result["transformers"]["runtime_provenance"] is None
    assert (
        "FlashRWKV revision provenance mismatch"
        in result["transformers"]["runtime_provenance_error"]
    )
    assert result["transformers"]["requirement_satisfied"] is False
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("url", "requested_revision", "commit_id", "satisfied"),
    [
        (
            "git+https://GITHUB.COM/RWKV-RS/TRANSFORMERS-RWKV/",
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


@pytest.mark.parametrize(
    "url",
    [
        "https://user@github.com/rwkv-rs/transformers-rwkv.git",
        "https://github.com:443/rwkv-rs/transformers-rwkv.git",
        "https://github.com/rwkv‐rs/transformers-rwkv.git",
        "https://github.com/rwkv-rs%2Ffork/transformers-rwkv.git",
        "https://github.com//rwkv-rs/transformers-rwkv.git",
        "https://github.com/rwkv-rs/transformers-rwkv.git?ref=main",
        "https://github.com/rwkv-rs/transformers-rwkv.git#main",
        "https://gitlab.com/rwkv-rs/transformers-rwkv.git",
    ],
)
def test_github_repository_canonicalizer_rejects_hostile_urls(url: str) -> None:
    with pytest.raises(ValueError):
        canonical_github_repository(url)


def test_github_repository_match_rejects_foreign_or_fork_repository() -> None:
    assert not github_repository_matches(
        "https://github.com/foreign/transformers-rwkv.git",
        TRANSFORMERS_SOURCE_URL,
    )


def test_github_repository_canonicalizer_fresh_process() -> None:
    script = """
from any2rwkv.provenance import canonical_github_repository

expected = "https://github.com/rwkv-rs/fla-rwkv"
assert canonical_github_repository(
    "git+https://GITHUB.COM/RWKV-RS/FLA-RWKV.git/"
) == expected
for hostile in (
    "https://user@github.com/rwkv-rs/fla-rwkv.git",
    "https://github.com/rwkv-rs%2Ffork/fla-rwkv.git",
    "https://github.com//rwkv-rs/fla-rwkv.git",
):
    try:
        canonical_github_repository(hostile)
    except ValueError:
        pass
    else:
        raise AssertionError(hostile)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


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


def test_module_ownership_rejects_a_shadow_transformers_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    owned = tmp_path / "owned" / "transformers" / "__init__.py"
    shadow = tmp_path / "shadow" / "transformers" / "__init__.py"
    owned.parent.mkdir(parents=True)
    shadow.parent.mkdir(parents=True)
    owned.write_text("", encoding="utf-8")
    shadow.write_text("", encoding="utf-8")

    class Distribution:
        @staticmethod
        def locate_file(_relative: str) -> Path:
            return owned

    monkeypatch.setattr(
        "any2rwkv.preflight.importlib.metadata.distribution",
        lambda _name: Distribution(),
    )
    monkeypatch.setattr(
        "any2rwkv.preflight.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(shadow)),
    )

    with pytest.raises(ContractError, match="module ownership mismatch"):
        _require_module_ownership("transformers", "transformers")


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
