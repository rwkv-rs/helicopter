#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

from any2rwkv.core.experiment_tracking import (
    ExperimentTracker,
    experiment_report_identity_line,
)


ROOT = Path(__file__).resolve().parents[1]
STRICT_WRAPPER = ROOT / "scripts" / "strict_any2rwkv_train.py"


def _load_strict_wrapper():
    spec = importlib.util.spec_from_file_location(
        "any2rwkv_strict_wrapper_for_profile_launcher", STRICT_WRAPPER
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load strict Any2RWKV wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    if command.count(name) != 1:
        raise SystemExit(f"profile command requires exactly one {name}")
    index = command.index(name)
    if index + 1 >= len(command):
        raise SystemExit(f"profile command has no value for {name}")
    return command[index + 1]


def _validate_torchrun_candidate_command(command: list[str]) -> None:
    """Require the profiler candidate to be the exact torchrun entrypoint."""
    if not command or Path(command[0]).name != "torchrun":
        raise SystemExit("profile launcher command must start with torchrun")
    if command.count("--standalone") != 1:
        raise SystemExit("profile launcher requires exactly one --standalone")
    if command.count("--nproc-per-node=8") != 1:
        raise SystemExit("profile launcher requires exact eight-rank torchrun")
    if command.count("--no-python") != 1:
        raise SystemExit("profile launcher requires exactly one --no-python")
    no_python = command.index("--no-python")
    launcher_arguments = command[1:no_python]
    if sorted(launcher_arguments) != sorted(
        ["--standalone", "--nproc-per-node=8"]
    ):
        raise SystemExit(
            "profile launcher rejects unbound torchrun launcher arguments"
        )
    if len(command) <= no_python + 2:
        raise SystemExit(
            "profile launcher requires a Python executable and candidate script"
        )
    python = Path(command[no_python + 1]).expanduser().resolve()
    if not python.is_file() or not python.name.startswith("python"):
        raise SystemExit(
            "profile launcher --no-python entrypoint must be a Python executable"
        )
    candidate_script = (ROOT / "scripts/profile_any2rwkv_candidate.py").resolve()
    actual_script = Path(command[no_python + 2]).expanduser().resolve()
    if actual_script != candidate_script:
        raise SystemExit(
            "profile launcher must execute profile_any2rwkv_candidate.py directly"
        )


def _export_input_file(sqlite_path: Path) -> str | None:
    with sqlite3.connect(sqlite_path) as connection:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        for table in tables:
            columns = [
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            ]
            if "EXPORT_INPUT_FILE" in columns:
                row = connection.execute(
                    f'SELECT EXPORT_INPUT_FILE FROM "{table}" LIMIT 1'
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
            if {"name", "value"}.issubset(columns):
                row = connection.execute(
                    f'SELECT value FROM "{table}" WHERE name = ? LIMIT 1',
                    ("EXPORT_INPUT_FILE",),
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
    return None


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _directory_hashes(path: Path, strict) -> dict[str, str]:
    return {
        str(file.relative_to(path)): strict.sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _profile_input_binding(command: list[str], strict) -> dict[str, object]:
    """Freeze every read-only input before Nsight launches any GPU process."""
    plan_path = Path(_option(command, "--plan")).expanduser().resolve()
    source = Path(_option(command, "--source")).expanduser().resolve()
    source_config = source if source.name == "config.json" else source / "config.json"
    zero_step = Path(_option(command, "--zero-step")).expanduser().resolve()
    overlays = Path(_option(command, "--overlays")).expanduser().resolve()
    dataset_manifest = Path(
        _option(command, "--dataset-manifest")
    ).expanduser().resolve()
    train_cache = Path(_option(command, "--train-cache")).expanduser().resolve()
    validation_cache = Path(
        _option(command, "--validation-cache")
    ).expanduser().resolve()
    plan = strict.load_strict_json(plan_path, label="profile plan")
    if not isinstance(plan, dict):
        raise SystemExit("profile plan must be a JSON object")
    source_checkpoint = strict.checkpoint_content_binding(source_config)
    zero_step_checkpoint = strict.checkpoint_content_binding(
        zero_step / "config.json"
    )
    overlay_files = _directory_hashes(overlays, strict)
    return {
        "plan_sha256": strict.sha256_file(plan_path),
        "workload_sha256": strict.performance_workload_binding(plan)["sha256"],
        "source_config_sha256": strict.sha256_file(source_config),
        "source_checkpoint_sha256": source_checkpoint["sha256"],
        "source_checkpoint_binding": source_checkpoint,
        "zero_step_checkpoint_sha256": zero_step_checkpoint["sha256"],
        "zero_step_checkpoint_binding": zero_step_checkpoint,
        "dataset_content_binding": strict.dataset_content_binding(
            dataset_manifest
        ),
        "overlay_files_sha256": strict.sha256_json(overlay_files),
        "overlay_files": overlay_files,
        "train_cache_content_binding": strict.cache_content_binding(
            train_cache / "manifest.json"
        ),
        "validation_cache_content_binding": strict.cache_content_binding(
            validation_cache / "manifest.json"
        ),
        "training_source_files": strict.training_source_files(),
        "source_revision": strict.source_revision_binding(),
    }


def _wandb_metric_tree(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "rank": {
            str(rank): {
                "kernel_covered_wall_fraction": metrics[
                    "kernel_covered_wall_fraction_by_rank"
                ][rank],
                "window_seconds": metrics["window_seconds_by_rank"][rank],
                "kernel_intersection_count": metrics[
                    "kernel_intersection_count_by_rank"
                ][rank],
            }
            for rank in range(8)
        },
        "gpu": {
            str(gpu): {
                "sm_active_fraction": metrics["sm_active_fraction_by_gpu"][gpu],
                "sm_active_sample_count": metrics[
                    "sm_active_sample_count_by_gpu"
                ][gpu],
            }
            for gpu in range(8)
        },
        "gpu_metric_common_window_seconds": metrics[
            "gpu_metric_common_window_seconds"
        ],
    }


def _plain_profile_case(profile_case: dict[str, object]) -> str:
    boundary = profile_case.get("input_boundary")
    mixer_kind = profile_case.get("source_mixer_kind")
    profile_case_id = str(profile_case.get("profile_case_id", ""))
    if boundary is None or mixer_kind is None:
        parsed_boundary, separator, parsed_mixer_kind = profile_case_id.partition(":")
        if separator:
            boundary = boundary or parsed_boundary
            mixer_kind = mixer_kind or parsed_mixer_kind
    if boundary == "embedding-output":
        return "第 0 层（输入直接来自 embedding）"
    if mixer_kind == "linear_attention":
        return "后续 GDN 层（输入来自已经转换完成的 RWKV7 前缀）"
    if mixer_kind == "full_attention":
        return "full-attention 层（输入来自已经转换完成的 RWKV7 前缀）"
    return f"后续 {mixer_kind} 层（输入来自已经转换完成的 RWKV7 前缀）"


def _resume_wandb_with_nsys_metrics(
    *,
    output_dir: Path,
    plan_path: Path,
    launch_token: str,
    epoch_metrics: dict[str, object],
    transition_metrics: dict[str, object] | None,
) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tracking = plan.get("tracking") if isinstance(plan, dict) else None
    job_types = tracking.get("job_types") if isinstance(tracking, dict) else None
    if (
        not isinstance(tracking, dict)
        or tracking.get("mode") != "online"
        or not isinstance(job_types, dict)
    ):
        raise SystemExit("profile plan lacks online W&B tracking for Nsight results")
    existing = json.loads(
        (output_dir / "experiment-tracking.json").read_text(encoding="utf-8")
    )
    config = existing.get("config") if isinstance(existing, dict) else None
    if not isinstance(config, dict):
        raise SystemExit("profile candidate tracking config is missing")
    tracker_plan = SimpleNamespace(
        wandb_mode="online",
        wandb_project=tracking.get("project"),
        wandb_entity=tracking.get("entity"),
        wandb_group=tracking.get("group"),
        wandb_tags=tuple(tracking.get("tags", ())),
        wandb_job_types=tuple(sorted(job_types.items())),
        evidence_tier=plan.get("evidence_tier"),
    )
    previous = {
        name: os.environ.get(name)
        for name in ("RANK", "WORLD_SIZE", "ANY2RWKV_ATTEMPT_TOKEN")
    }
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "8"
    os.environ["ANY2RWKV_ATTEMPT_TOKEN"] = launch_token
    tracker: ExperimentTracker | None = None
    try:
        tracker = ExperimentTracker.from_plan(
            run_dir=output_dir,
            plan=tracker_plan,
            action="profile",
        )
        tracker.start(config=config, resume_existing=True)
        tracker.log_metrics(
            {
                "nsys": {
                    "epoch": _wandb_metric_tree(epoch_metrics),
                    "cache_transition": (
                        _wandb_metric_tree(transition_metrics)
                        if transition_metrics is not None
                        else {"profiled": False}
                    ),
                }
            }
        )
        tracker.finish(exit_code=0)
    except BaseException as error:
        if tracker is not None:
            try:
                tracker.abort(reason=f"{type(error).__name__}: {error}")
            except BaseException as finalization_error:
                error.add_note(str(finalization_error))
        raise
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return json.loads(
        (output_dir / "experiment-tracking.json").read_text(encoding="utf-8")
    )


def _finalize_failed_profile(
    *,
    output_dir: Path,
    plan_path: Path,
    launch_token: str,
    reason: str,
) -> None:
    """Resume the candidate run when possible and persist one plain failure report."""
    tracking_path = output_dir / "experiment-tracking.json"
    tracking: dict[str, object] = {}
    if tracking_path.is_file():
        existing = json.loads(tracking_path.read_text(encoding="utf-8"))
        config = existing.get("config") if isinstance(existing, dict) else None
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        tracking_plan = plan.get("tracking") if isinstance(plan, dict) else None
        job_types = (
            tracking_plan.get("job_types")
            if isinstance(tracking_plan, dict)
            else None
        )
        if isinstance(config, dict) and isinstance(job_types, dict):
            tracker_plan = SimpleNamespace(
                wandb_mode="online",
                wandb_project=tracking_plan.get("project"),
                wandb_entity=tracking_plan.get("entity"),
                wandb_group=tracking_plan.get("group"),
                wandb_tags=tuple(tracking_plan.get("tags", ())),
                wandb_job_types=tuple(sorted(job_types.items())),
                evidence_tier=plan.get("evidence_tier"),
            )
            previous = {
                name: os.environ.get(name)
                for name in ("RANK", "WORLD_SIZE", "ANY2RWKV_ATTEMPT_TOKEN")
            }
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "8"
            os.environ["ANY2RWKV_ATTEMPT_TOKEN"] = launch_token
            tracker: ExperimentTracker | None = None
            try:
                tracker = ExperimentTracker.from_plan(
                    run_dir=output_dir,
                    plan=tracker_plan,
                    action="profile",
                )
                tracker.start(config=config, resume_existing=True)
                tracker.abort(reason=reason)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
    if tracking_path.is_file():
        value = json.loads(tracking_path.read_text(encoding="utf-8"))
        tracking = value if isinstance(value, dict) else {}
    lines = [
        experiment_report_identity_line(tracking, status="failed"),
        "",
        "# 8 卡性能候选失败",
        "",
        "## 发生了什么",
        "",
        f"- 失败原因：`{reason}`",
        f"- W&B：{tracking.get('url') or '未能建立在线 run'}",
        "",
        "## 结论",
        "",
        "这次候选不能进入性能选择，也不能放行正式训练。修复原因后必须使用新的 run ID 重跑。",
        "",
    ]
    temporary = output_dir / "experiment-report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output_dir / "experiment-report.md")


def _write_final_candidate_report(
    *,
    output_dir: Path,
    candidate: dict[str, object],
    tracking: dict[str, object],
    epoch_metrics: dict[str, object],
    transition_metrics: dict[str, object] | None,
) -> None:
    binding = candidate.get("binding")
    binding = binding if isinstance(binding, dict) else {}
    profile_case = binding.get("profile_case")
    profile_case = profile_case if isinstance(profile_case, dict) else {}
    epoch_kernel = epoch_metrics["kernel_covered_wall_fraction_by_rank"]
    epoch_sm = epoch_metrics["sm_active_fraction_by_gpu"]
    epoch_walls = epoch_metrics["window_seconds_by_rank"]
    reserved = candidate["peak_reserved_memory_fraction_by_rank"]
    lines = [
        experiment_report_identity_line(tracking, status="completed"),
        "",
        "# 8 卡性能候选结果",
        "",
        "## 这次实际测了什么",
        "",
        f"- 场景：{_plain_profile_case(profile_case)}",
        f"- 机器可读场景 ID：`{profile_case.get('profile_case_id', '未知')}`",
        f"- 代表层：`{binding.get('layer', '未知')}`",
        f"- 每卡 batch：`{candidate.get('per_rank_micro_batch_size')}`",
        f"- 临时 AdamW update：`{candidate.get('ephemeral_optimizer_steps')}` 次；没有写入正式 checkpoint",
        f"- W&B：{tracking.get('url') or '未建立在线 run'}",
        "",
        "## 结果",
        "",
        (
            "- 测量窗口里有 CUDA 工作的时间比例："
            f"`{min(epoch_kernel):.1%}..{max(epoch_kernel):.1%}`"
        ),
        (
            "- 每张卡的 SM 真正工作比例（Nsight `SMs Active`）："
            f"`{min(epoch_sm):.1%}..{max(epoch_sm):.1%}`"
        ),
        (
            "- 每张卡被 PyTorch/CUDA 保留的显存比例："
            f"`{min(reserved):.1%}..{max(reserved):.1%}`"
        ),
        f"- 卡间完整周期墙钟：`{min(epoch_walls):.3f}..{max(epoch_walls):.3f}` 秒",
        f"- checkpoint 占完整周期：`{float(candidate.get('checkpoint_wall_fraction', 0)):.1%}`",
        (
            "- 端到端吞吐：每秒处理 "
            f"`{candidate.get('loss_tokens_per_second')}` 个真正参与误差计算的 token"
        ),
    ]
    if transition_metrics is None:
        lines.append("- 这是模型末层场景，不需要生成下一层输入缓存")
    else:
        transition_kernel = transition_metrics[
            "kernel_covered_wall_fraction_by_rank"
        ]
        transition_sm = transition_metrics["sm_active_fraction_by_gpu"]
        transition_walls = transition_metrics["window_seconds_by_rank"]
        lines.extend(
            [
                "- 生成下一层输入缓存时，有 CUDA 工作的时间比例："
                f"`{min(transition_kernel):.1%}..{max(transition_kernel):.1%}`",
                "- 生成下一层输入缓存时，每张卡的 SM 真正工作比例："
                f"`{min(transition_sm):.1%}..{max(transition_sm):.1%}`",
                "- 生成下一层输入缓存的卡间墙钟："
                f"`{min(transition_walls):.3f}..{max(transition_walls):.3f}` 秒",
            ]
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "这个文件只陈述当前候选的实测结果。是否合格由 selection 对全部场景统一复算；单个候选不能放行正式训练。",
            "",
        ]
    )
    temporary = output_dir / "experiment-report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output_dir / "experiment-report.md")


def _execute_and_finalize_profile(
    *,
    run_id: str,
    output_dir: Path,
    command: list[str],
    strict,
    frozen_input_binding: dict[str, object],
    nsys: str,
    profile_command: list[str],
    child_environment: dict[str, str],
    launch_token: str,
) -> int:
    candidate_path = output_dir / "candidate.json"
    report_path = output_dir / "profile.nsys-rep"
    sqlite_path = output_dir / "profile.sqlite"
    subprocess.run(profile_command, cwd=ROOT, env=child_environment, check=True)
    if not report_path.is_file() or not candidate_path.is_file():
        raise SystemExit("Nsight profile did not publish report and candidate files")
    export_command = [
        nsys,
        "export",
        "--type=sqlite",
        "--force-overwrite=false",
        f"--output={sqlite_path}",
        str(report_path),
    ]
    subprocess.run(export_command, cwd=ROOT, check=True)
    if not sqlite_path.is_file():
        raise SystemExit("Nsight export did not publish SQLite evidence")

    candidate = strict.load_strict_json(candidate_path, label="profile candidate")
    if not isinstance(candidate, dict) or candidate.get("run_id") != run_id:
        raise SystemExit("profile candidate identity differs from launcher run id")
    candidate_binding = candidate.get("binding")
    if (
        not isinstance(candidate_binding, dict)
        or any(
            candidate_binding.get(key) != value
            for key, value in frozen_input_binding.items()
        )
        or _profile_input_binding(command, strict) != frozen_input_binding
    ):
        raise SystemExit(
            "profile input binding changed before, during, or after measurement"
        )
    epoch_metrics = strict.derive_nsys_profile_metrics(
        sqlite_path, candidate.get("nvtx_range_names_by_rank", [])
    )
    transition_names = candidate.get("transition_nvtx_range_names_by_rank", [])
    transition_metrics = (
        strict.derive_nsys_profile_metrics(sqlite_path, transition_names)
        if transition_names
        else None
    )
    export_input = _export_input_file(sqlite_path)
    if export_input is None or Path(export_input).name != report_path.name:
        raise SystemExit(
            "Nsight SQLite metadata does not bind the exact input report name"
        )
    version = subprocess.run(
        [nsys, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    tracking = _resume_wandb_with_nsys_metrics(
        output_dir=output_dir,
        plan_path=Path(_option(command, "--plan")).expanduser().resolve(),
        launch_token=launch_token,
        epoch_metrics=epoch_metrics,
        transition_metrics=transition_metrics,
    )
    _write_final_candidate_report(
        output_dir=output_dir,
        candidate=candidate,
        tracking=tracking,
        epoch_metrics=epoch_metrics,
        transition_metrics=transition_metrics,
    )
    binding = {
        "schema_version": 1,
        "run_id": run_id,
        "profile_nonce": candidate.get("profile_nonce"),
        "launch_token_sha256": hashlib.sha256(
            launch_token.encode("ascii")
        ).hexdigest(),
        "candidate": {
            "path": str(candidate_path),
            "sha256": strict.sha256_file(candidate_path),
        },
        "nsys_report": {
            "path": str(report_path),
            "sha256": strict.sha256_file(report_path),
        },
        "nsys_sqlite": {
            "path": str(sqlite_path),
            "sha256": strict.sha256_file(sqlite_path),
            "export_input_file": export_input,
        },
        "nsys_version": version,
        "profile_command": profile_command,
        "export_command": export_command,
        "frozen_input_binding": frozen_input_binding,
        "derived_metrics": {
            "epoch": epoch_metrics,
            "cache_transition": transition_metrics,
        },
    }
    _write_json(output_dir / "profiler-export-binding.json", binding)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one exact 8-rank Any2RWKV profile, export its Nsight report to "
            "SQLite, and bind both files to the candidate."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if os.environ.get("HELICOPTER_RUN_ID") != args.run_id:
        raise SystemExit("launcher --run-id must equal HELICOPTER_RUN_ID")
    _validate_torchrun_candidate_command(command)
    strict = _load_strict_wrapper()
    frozen_input_binding = _profile_input_binding(command, strict)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit("profile output directory already exists; use a new run id")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate.json"
    report_path = output_dir / "profile.nsys-rep"
    sqlite_path = output_dir / "profile.sqlite"
    scratch_path = output_dir / "scratch"
    expected_options = {
        "--run-id": args.run_id,
        "--output": str(candidate_path),
        "--nsys-rep": str(report_path),
        "--nsys-sqlite": str(sqlite_path),
        "--scratch": str(scratch_path),
    }
    for name, expected in expected_options.items():
        actual = _option(command, name)
        actual = (
            str(Path(actual).expanduser().resolve()) if name != "--run-id" else actual
        )
        if actual != expected:
            raise SystemExit(f"profile command {name} differs from launcher output")

    nsys = shutil.which("nsys")
    if nsys is None:
        raise SystemExit("Nsight Systems CLI 'nsys' is unavailable")
    profile_prefix = output_dir / "profile"
    profile_command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--gpu-metrics-devices=all",
        "--gpu-metrics-frequency=1000",
        "--force-overwrite=false",
        f"--output={profile_prefix}",
        *command,
    ]
    launch_token = secrets.token_hex(16)
    child_environment = os.environ.copy()
    child_environment["ANY2RWKV_ATTEMPT_TOKEN"] = launch_token
    child_environment["ANY2RWKV_PROFILE_NONCE"] = launch_token
    try:
        return _execute_and_finalize_profile(
            run_id=args.run_id,
            output_dir=output_dir,
            command=command,
            strict=strict,
            frozen_input_binding=frozen_input_binding,
            nsys=nsys,
            profile_command=profile_command,
            child_environment=child_environment,
            launch_token=launch_token,
        )
    except BaseException as error:
        try:
            _finalize_failed_profile(
                output_dir=output_dir,
                plan_path=Path(_option(command, "--plan")).expanduser().resolve(),
                launch_token=launch_token,
                reason=f"{type(error).__name__}: {error}",
            )
        except BaseException as finalization_error:
            error.add_note(
                "failed to persist W&B/Markdown failure evidence: "
                + str(finalization_error)
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
