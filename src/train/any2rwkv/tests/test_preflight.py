from __future__ import annotations

from pathlib import Path

from any2rwkv.artifacts import git_sha
from any2rwkv.preflight import (
    collect_full_loop_preflight,
    collect_preflight,
    transformers_version_supported,
)

PRODUCT_ROOT = Path(__file__).resolve().parents[4]


def test_preflight_binds_both_native_checkouts_and_imported_adapter() -> None:
    rwkv_hf_sha = git_sha(PRODUCT_ROOT / "src/train/rwkv-hf")
    rwkv_lm_sha = "0" * 40

    result = collect_preflight(
        PRODUCT_ROOT,
        expected_rwkv_hf_sha=rwkv_hf_sha,
        expected_rwkv_lm_sha=rwkv_lm_sha,
    )

    assert result["rwkv_hf"]["checkout_commit"] == rwkv_hf_sha
    assert result["rwkv_hf"]["commit_matches"] is True
    assert result["rwkv_hf"]["module_in_checkout"] is True
    assert result["rwkv_lm"]["checkout_commit"] is None
    assert result["rwkv_lm"]["commit_matches"] is False
    assert result["rwkv_lm"]["kernel_loader_sha256"] is None
    assert result["rwkv_lm"]["kernel_source_sha256"] is None
    assert result["rwkv_lm"]["kernel_binding_sha256"] is None
    assert result["passed"] is False


def test_preflight_rejects_a_stale_rwkv_lm_commit() -> None:
    result = collect_preflight(
        PRODUCT_ROOT,
        expected_rwkv_hf_sha=git_sha(PRODUCT_ROOT / "src/train/rwkv-hf"),
        expected_rwkv_lm_sha="0" * 40,
    )

    assert result["rwkv_lm"]["commit_matches"] is False
    assert result["passed"] is False


def test_transformers_version_contract_rejects_dev_drift_and_exclusion() -> None:
    assert transformers_version_supported("5.5.3") is True
    assert transformers_version_supported("5.13.9") is True
    assert transformers_version_supported("5.5.3.dev0") is False
    assert transformers_version_supported("5.6.0") is False
    assert transformers_version_supported("5.14.0") is False
    assert transformers_version_supported("5.15.0.dev0") is False


def test_full_loop_preflight_reports_every_missing_gate(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        '{"remote_read_only_path":"/missing/frozen-source"}\n',
        encoding="utf-8",
    )
    raw_manifest = tmp_path / "raw.json"
    raw_manifest.write_text("{}\n", encoding="utf-8")
    result = collect_full_loop_preflight(
        PRODUCT_ROOT,
        recipe_id="qwen35_to_rwkv7",
        source_manifest_path=source_manifest,
        source_path=Path("/missing/frozen-source"),
        raw_data_manifest_path=raw_manifest,
        dataset_manifest_path=tmp_path / "prepared" / "data-splits.json",
        training_config_path=tmp_path / "training.json",
        lighteval_config_path=tmp_path / "lighteval.toml",
        evalscope_config_path=tmp_path / "evalscope.yaml",
        expected_rwkv_hf_sha=git_sha(PRODUCT_ROOT / "src/train/rwkv-hf"),
        expected_rwkv_lm_sha="0" * 40,
        allow_proxy_layers=True,
        precision="fp32io16",
    )

    assert result["passed"] is False
    assert result["status"] == "blocked"
    blockers = "\n".join(result["blockers"])
    assert "rwkv-lm backend is uninitialized or at the wrong commit" in blockers
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
