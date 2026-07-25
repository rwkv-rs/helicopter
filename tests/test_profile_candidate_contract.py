from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile_any2rwkv_candidate.py"
SPEC = importlib.util.spec_from_file_location("profile_candidate_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)
LAUNCHER_SCRIPT = ROOT / "scripts" / "run_any2rwkv_profile_candidate.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "profile_candidate_launcher_for_test", LAUNCHER_SCRIPT
)
assert LAUNCHER_SPEC is not None and LAUNCHER_SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(LAUNCHER)


def profile_paths(tmp_path: Path) -> argparse.Namespace:
    source = tmp_path / "source"
    zero_step = tmp_path / "zero-step"
    overlays = tmp_path / "overlays"
    train_cache = tmp_path / "train-cache"
    validation_cache = tmp_path / "validation-cache"
    run_dir = tmp_path / "profile-run"
    for path in (source, zero_step, overlays, train_cache, validation_cache, run_dir):
        path.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    dataset_manifest = tmp_path / "data-splits.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        source=source,
        plan=plan,
        dataset_manifest=dataset_manifest,
        zero_step=zero_step,
        overlays=overlays,
        train_cache=train_cache,
        validation_cache=validation_cache,
        scratch=run_dir / "scratch",
        nsys_rep=run_dir / "profile.nsys-rep",
        nsys_sqlite=run_dir / "profile.sqlite",
        output=run_dir / "candidate.json",
    )


def test_profile_paths_accept_only_isolated_run_directory(tmp_path: Path) -> None:
    args = profile_paths(tmp_path)

    PROFILE._validate_profile_paths(args)

    assert args.scratch == (tmp_path / "profile-run/scratch").resolve()


def test_profile_paths_reject_scratch_aliasing_checkpoint(tmp_path: Path) -> None:
    args = profile_paths(tmp_path)
    args.scratch = args.zero_step

    with pytest.raises(SystemExit, match="dedicated RUN_DIR/scratch"):
        PROFILE._validate_profile_paths(args)


def test_profile_paths_reject_run_directory_inside_checkpoint(tmp_path: Path) -> None:
    args = profile_paths(tmp_path)
    run_dir = args.zero_step / "profile-run"
    run_dir.mkdir()
    args.output = run_dir / "candidate.json"
    args.scratch = run_dir / "scratch"
    args.nsys_rep = run_dir / "profile.nsys-rep"
    args.nsys_sqlite = run_dir / "profile.sqlite"

    with pytest.raises(SystemExit, match="must be isolated"):
        PROFILE._validate_profile_paths(args)


def test_profile_paths_reject_stale_tracking_or_report(tmp_path: Path) -> None:
    args = profile_paths(tmp_path)
    (args.output.parent / "experiment-tracking.json").write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="stale evidence"):
        PROFILE._validate_profile_paths(args)


def test_profile_cache_rejects_different_source_checkpoint(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layer_index": 0,
                "split": "distill_train",
                "binding": {
                    "source_checkpoint_sha256": "a" * 64,
                    "prefix_fingerprint": "b" * 64,
                    "split": "distill_train",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="not bound to the measured source"):
        PROFILE._profile_cache_prefix_fingerprint(
            manifest,
            source_checkpoint_sha256="c" * 64,
            zero_step_checkpoint_sha256="d" * 64,
            training_config_sha256="e" * 64,
            dataset_manifest_sha256="f" * 64,
            layer=0,
            split="distill_train",
        )


def test_profile_cases_cover_embedding_later_gdn_and_full_attention() -> None:
    layer_types = [
        kind
        for _ in range(6)
        for kind in (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
    ]
    config = {"num_hidden_layers": 24, "layer_types": layer_types}

    assert PROFILE._source_profile_case(config, 0) == {
        "profile_case_id": "embedding-output:linear_attention",
        "source_mixer_kind": "linear_attention",
        "input_boundary": "embedding-output",
        "representative_layer": 0,
        "layer_count": 1,
        "transition_count": 1,
    }
    assert PROFILE._source_profile_case(config, 1) == {
        "profile_case_id": "recurrent-prefix:linear_attention",
        "source_mixer_kind": "linear_attention",
        "input_boundary": "recurrent-prefix",
        "representative_layer": 1,
        "layer_count": 17,
        "transition_count": 17,
    }
    assert PROFILE._source_profile_case(config, 3) == {
        "profile_case_id": "recurrent-prefix:full_attention",
        "source_mixer_kind": "full_attention",
        "input_boundary": "recurrent-prefix",
        "representative_layer": 3,
        "layer_count": 6,
        "transition_count": 5,
    }

    with pytest.raises(SystemExit, match="earliest representative"):
        PROFILE._source_profile_case(config, 2)


def test_profile_launcher_requires_candidate_as_direct_python_entrypoint() -> None:
    candidate = str((ROOT / "scripts/profile_any2rwkv_candidate.py").resolve())

    LAUNCHER._validate_torchrun_candidate_command(
        [
            "torchrun",
            "--standalone",
            "--nproc-per-node=8",
            "--no-python",
            sys.executable,
            candidate,
            "--run-id",
            "profile-test",
        ]
    )


def test_profile_launcher_rejects_candidate_script_as_unexecuted_argument() -> None:
    candidate = str((ROOT / "scripts/profile_any2rwkv_candidate.py").resolve())

    with pytest.raises(SystemExit, match="must execute .* directly"):
        LAUNCHER._validate_torchrun_candidate_command(
            [
                "torchrun",
                "--standalone",
                "--nproc-per-node=8",
                "--no-python",
                sys.executable,
                "-c",
                "raise SystemExit(0)",
                candidate,
            ]
        )


def test_profile_launcher_rejects_unbound_torchrun_arguments() -> None:
    candidate = str((ROOT / "scripts/profile_any2rwkv_candidate.py").resolve())

    with pytest.raises(SystemExit, match="unbound torchrun"):
        LAUNCHER._validate_torchrun_candidate_command(
            [
                "torchrun",
                "--standalone",
                "--nproc-per-node=8",
                "--max-restarts=1",
                "--no-python",
                sys.executable,
                candidate,
            ]
        )


def test_profile_launcher_writes_plain_failure_report_without_partial_metrics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "failed-profile"
    output_dir.mkdir()

    LAUNCHER._finalize_failed_profile(
        output_dir=output_dir,
        plan_path=tmp_path / "unused-plan.json",
        launch_token="a" * 32,
        reason="Nsight export failed",
    )

    report = (output_dir / "experiment-report.md").read_text(encoding="utf-8")
    assert "8 卡性能候选失败" in report
    assert "Nsight export failed" in report
    assert "不能进入性能选择" in report


def test_profile_launcher_does_not_invent_missing_nsys_export_source(
    tmp_path: Path,
) -> None:
    sqlite = tmp_path / "profile.sqlite"
    with sqlite3.connect(sqlite) as connection:
        connection.execute("CREATE TABLE metadata (name TEXT, value TEXT)")

    assert LAUNCHER._export_input_file(sqlite) is None


def test_profile_launcher_rewrites_candidate_report_with_nsys_results(
    tmp_path: Path,
) -> None:
    metrics = {
        "kernel_covered_wall_fraction_by_rank": [0.95] * 8,
        "window_seconds_by_rank": [10.0] * 8,
        "process_ids_by_rank": list(range(8)),
        "device_ids_by_rank": list(range(8)),
        "kernel_intersection_count_by_rank": [100] * 8,
        "sm_active_fraction_by_gpu": [0.94] * 8,
        "sm_active_sample_count_by_gpu": [100] * 8,
        "sm_active_metric_source_by_gpu": [],
        "gpu_metric_common_window_seconds": 10.0,
    }
    tracking = {
        "run_id": "profile-b2-l1",
        "attempt_id": "a" * 32,
        "status": "completed",
        "config_sha256": "b" * 64,
        "url": "https://wandb.invalid/profile-b2-l1",
    }
    candidate = {
        "per_rank_micro_batch_size": 2,
        "ephemeral_optimizer_steps": 12,
        "loss_tokens_per_second": 1234.0,
        "checkpoint_wall_fraction": 0.02,
        "peak_reserved_memory_fraction_by_rank": [0.90] * 8,
        "binding": {
            "layer": 1,
            "profile_case": {
                "profile_case_id": "recurrent-prefix:linear_attention"
            },
        },
    }

    LAUNCHER._write_final_candidate_report(
        output_dir=tmp_path,
        candidate=candidate,
        tracking=tracking,
        epoch_metrics=metrics,
        transition_metrics=metrics,
    )

    report = (tmp_path / "experiment-report.md").read_text(encoding="utf-8")
    assert report.startswith("<!-- any2rwkv-report-identity:")
    assert "recurrent-prefix:linear_attention" in report
    assert "后续 GDN 层" in report
    assert "SM 真正工作比例" in report
    assert "生成下一层输入缓存" in report
    assert "单个候选不能放行正式训练" in report
