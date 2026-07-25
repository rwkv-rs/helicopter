from __future__ import annotations

from pathlib import Path

from any2rwkv.artifacts import git_sha
from any2rwkv.preflight import collect_preflight


PRODUCT_ROOT = Path(__file__).resolve().parents[4]


def test_preflight_binds_both_native_checkouts_and_imported_adapter() -> None:
    rwkv_hf_sha = git_sha(PRODUCT_ROOT / "src/train/rwkv-hf")
    rwkv_lm_sha = git_sha(PRODUCT_ROOT / "src/train/rwkv-lm")

    result = collect_preflight(
        PRODUCT_ROOT,
        expected_rwkv_hf_sha=rwkv_hf_sha,
        expected_rwkv_lm_sha=rwkv_lm_sha,
    )

    assert result["rwkv_hf"]["checkout_commit"] == rwkv_hf_sha
    assert result["rwkv_hf"]["commit_matches"] is True
    assert result["rwkv_hf"]["module_in_checkout"] is True
    assert result["rwkv_lm"]["checkout_commit"] == rwkv_lm_sha
    assert result["rwkv_lm"]["commit_matches"] is True
    assert result["rwkv_lm"]["kernel_loader_sha256"]
    assert result["rwkv_lm"]["kernel_source_sha256"]
    assert result["rwkv_lm"]["kernel_binding_sha256"]


def test_preflight_rejects_a_stale_rwkv_lm_commit() -> None:
    result = collect_preflight(
        PRODUCT_ROOT,
        expected_rwkv_hf_sha=git_sha(PRODUCT_ROOT / "src/train/rwkv-hf"),
        expected_rwkv_lm_sha="0" * 40,
    )

    assert result["rwkv_lm"]["commit_matches"] is False
    assert result["passed"] is False
