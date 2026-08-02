from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from any2rwkv.errors import ContractError
from any2rwkv.layer_schedule import (
    LayerConvergenceState,
    epoch_permutation,
    update_convergence,
)
from any2rwkv.recipes.qwen35_to_rwkv7.layer_major_runner import (
    _read_progress,
    _sha256_json,
    _validate_resume_permutation,
    _verify_generation_integrity,
    _write_generation_integrity,
    _write_progress,
)
from any2rwkv.artifacts import file_sha256


def test_epoch_permutation_visits_every_row_exactly_once() -> None:
    rows, digest = epoch_permutation(row_count=17, seed=7, layer=2, epoch=3)
    assert sorted(rows) == list(range(17))
    assert len(set(rows)) == 17
    assert (rows, digest) == epoch_permutation(row_count=17, seed=7, layer=2, epoch=3)
    assert rows != epoch_permutation(row_count=17, seed=7, layer=3, epoch=3)[0]


def test_convergence_waits_for_min_epochs_and_patience_then_selects_best() -> None:
    state = LayerConvergenceState()
    decisions = []
    for metric in (1.0, 0.8, 0.805, 0.81):
        decision = update_convergence(
            state,
            metric=metric,
            min_epochs=3,
            max_epochs=6,
            min_delta=0.01,
            patience=2,
        )
        decisions.append(decision)
        state = decision.state
    assert not decisions[2].converged
    assert decisions[3].converged
    assert decisions[3].reason == "validation-plateau"
    assert state.best_epoch == 1
    assert state.best_metric == 0.8


def test_max_epochs_fails_closed_when_metric_keeps_improving() -> None:
    state = LayerConvergenceState()
    for metric in (1.0, 0.9, 0.8):
        decision = update_convergence(
            state,
            metric=metric,
            min_epochs=1,
            max_epochs=3,
            min_delta=0.01,
            patience=1,
        )
        state = decision.state
    assert decision.exhausted
    assert not decision.converged


def test_plateau_tracks_sgd_curve_separately_from_frozen_baseline() -> None:
    state = LayerConvergenceState(best_metric=0.5, best_epoch=-1)
    decisions = []
    for metric in (4.0, 0.7, 0.55, 0.56, 0.57):
        decision = update_convergence(
            state,
            metric=metric,
            min_epochs=3,
            max_epochs=8,
            min_delta=0.0,
            patience=2,
        )
        decisions.append(decision)
        state = decision.state

    assert all(not decision.improved for decision in decisions)
    assert all(decision.training_curve_improved for decision in decisions[:3])
    assert decisions[2].state.bad_epochs == 0
    assert not decisions[2].converged
    assert decisions[4].converged
    assert state.best_metric == 0.5
    assert state.best_epoch == -1
    assert state.best_training_metric == 0.55
    assert state.best_training_epoch == 2


def test_strict_checkpoint_improvement_is_kept_without_resetting_patience() -> None:
    state = LayerConvergenceState(
        completed_epochs=1,
        best_metric=0.5,
        best_epoch=0,
        best_training_metric=0.5,
        best_training_epoch=0,
        bad_epochs=0,
    )

    decision = update_convergence(
        state,
        metric=0.4995,
        min_epochs=1,
        max_epochs=4,
        min_delta=0.001,
        patience=2,
    )

    assert decision.improved
    assert not decision.training_curve_improved
    assert decision.state.best_metric == 0.4995
    assert decision.state.best_epoch == 1
    assert decision.state.best_training_metric == 0.5
    assert decision.state.best_training_epoch == 0
    assert decision.state.bad_epochs == 1


def test_resume_manifest_uses_run_relative_generation_even_outside_run_dir() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        generation = root / "run" / "generations" / "generation-0001"
        generation.mkdir(parents=True)
        (generation / "integrity.json").write_text(
            json.dumps({"schema_version": 1, "files": {"fixture": "0" * 64}}),
            encoding="utf-8",
        )
        resume = root / "resume-copies" / "progress.json"
        resume.parent.mkdir()
        _write_progress(
            resume,
            phase="local-complete",
            active_layer=4,
            epoch_index=0,
            next_train_row=0,
            prefix_fingerprint="prefix",
            generation=generation,
            generation_cursor={
                "active_layer": 3,
                "epoch_index": 2,
                "next_train_row": 17,
                "permutation_sha256": "a" * 64,
                "consumed_row_count": 17,
                "consumed_rows_sha256": "b" * 64,
                "train_cache_manifest_sha256": "c" * 64,
                "validation_cache_manifest_sha256": "d" * 64,
            },
            convergence=LayerConvergenceState(),
            completed_optimizer_steps=7,
            active_optimizer_steps=0,
            history=[],
            base_binding={},
            last_train_metrics=None,
        )
        payload = json.loads(resume.read_text(encoding="utf-8"))
        assert payload["generation"] == "generations/generation-0001"
        assert payload["status"] == "layerwise-local-complete"
        assert _read_progress(resume)["active_layer"] == 4


def test_resume_manifest_rejects_path_escape_and_cursor_mismatch(
    tmp_path: Path,
) -> None:
    progress = {
        "schema_version": 4,
        "schedule": "rolling-cache-layer-major-v1",
        "phase": "train",
        "active_layer": 2,
        "epoch_index": 1,
        "next_train_row": 4,
        "generation": "../outside",
        "generation_manifest_sha256": "a" * 64,
        "generation_cursor": {
            "active_layer": 2,
            "epoch_index": 1,
            "next_train_row": 4,
            "permutation_sha256": "b" * 64,
            "consumed_row_count": 4,
            "consumed_rows_sha256": "c" * 64,
            "train_cache_manifest_sha256": "d" * 64,
            "validation_cache_manifest_sha256": "e" * 64,
        },
        "history": [],
        "convergence": {},
    }
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(progress), encoding="utf-8")
    with pytest.raises(ContractError, match="invalid phase, path, cursor, or state"):
        _read_progress(path)
    progress["generation"] = "layer-generations/good"
    progress["generation_cursor"]["next_train_row"] = 3
    progress["generation_cursor"]["consumed_row_count"] = 3
    path.write_text(json.dumps(progress), encoding="utf-8")
    with pytest.raises(ContractError, match="cursor differ"):
        _read_progress(path)


def test_resume_permutation_must_match_frozen_epoch() -> None:
    permutation = (3, 1, 0, 2, 7, 4, 6, 5)
    progress = {
        "phase": "train",
        "next_train_row": 4,
        "generation_cursor": {
            "permutation_sha256": "a" * 64,
            "consumed_row_count": 4,
            "consumed_rows_sha256": _sha256_json(list(permutation[:4])),
            "train_cache_manifest_sha256": "d" * 64,
            "validation_cache_manifest_sha256": "e" * 64,
        },
    }
    _validate_resume_permutation(
        progress,
        permutation=permutation,
        permutation_sha="a" * 64,
    )
    with pytest.raises(ContractError, match="row permutation differs"):
        _validate_resume_permutation(
            progress,
            permutation=permutation,
            permutation_sha="b" * 64,
        )


def test_generation_integrity_rejects_tampered_training_state(tmp_path: Path) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    state = generation / "training-state.pt"
    state.write_bytes(b"original")
    (generation / "cursor.json").write_text("{}", encoding="utf-8")
    _write_generation_integrity(generation)
    manifest_sha = file_sha256(generation / "integrity.json")
    _verify_generation_integrity(
        generation,
        expected_manifest_sha256=manifest_sha,
    )
    state.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="file SHA-256 mismatch"):
        _verify_generation_integrity(
            generation,
            expected_manifest_sha256=manifest_sha,
        )
