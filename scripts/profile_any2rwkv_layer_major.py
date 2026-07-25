#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from any2rwkv.core.experiment_tracking import ExperimentTracker
from any2rwkv.core.experiment_tracking import experiment_report_identity_line


ROOT = Path(__file__).resolve().parents[1]
STRICT_WRAPPER = ROOT / "scripts" / "strict_any2rwkv_train.py"
PROFILER_CAPTURE_CONTRACT = (
    "nsys-full-process-exact-nvtx-rank-transition-and-input-bindings-v5"
)

PROFILE_INPUT_BINDING_FIELDS = (
    "plan_sha256",
    "workload_sha256",
    "source_config_sha256",
    "source_checkpoint_sha256",
    "source_checkpoint_binding",
    "zero_step_checkpoint_sha256",
    "zero_step_checkpoint_binding",
    "dataset_content_binding",
    "overlay_files_sha256",
    "overlay_files",
    "train_cache_content_binding",
    "validation_cache_content_binding",
    "training_source_files",
    "source_revision",
)


def _load_strict_wrapper():
    spec = importlib.util.spec_from_file_location(
        "any2rwkv_strict_wrapper_for_profile", STRICT_WRAPPER
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load strict Any2RWKV wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(
    path: Path,
    strict,
    *,
    expected_project: str,
    expected_entity: str | None,
    expected_group: str | None,
    expected_tags: list[str],
    expected_job_type: str,
) -> dict[str, object]:
    value = strict.load_strict_json(path, label=f"performance candidate {path}")
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise SystemExit(f"performance candidate has unsupported schema: {path}")
    original_value = copy.deepcopy(value)
    original_sha256 = strict.sha256_file(path)
    profile_nonce = value.get("profile_nonce")
    if (
        not isinstance(profile_nonce, str)
        or len(profile_nonce) != 32
        or any(character not in "0123456789abcdef" for character in profile_nonce)
    ):
        raise SystemExit(f"performance candidate has invalid profile nonce: {path}")
    launch_token_sha256 = hashlib.sha256(profile_nonce.encode("ascii")).hexdigest()
    artifacts = value.get("profiler_artifacts")
    if not isinstance(artifacts, list):
        raise SystemExit(f"performance candidate lacks profiler artifacts: {path}")
    kinds = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") not in {
            "nsys-rep",
            "nsys-sqlite",
        }:
            raise SystemExit(f"candidate profiler artifact is malformed: {path}")
        artifact_path = Path(str(artifact.get("path", ""))).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = (path.parent / artifact_path).resolve()
        if not artifact_path.is_file():
            raise SystemExit(f"candidate profiler artifact is missing: {artifact_path}")
        artifact["path"] = str(artifact_path)
        artifact["sha256"] = strict.sha256_file(artifact_path)
        kinds.add(str(artifact["kind"]))
    if kinds != {"nsys-rep", "nsys-sqlite"}:
        raise SystemExit(f"candidate requires nsys-rep and nsys-sqlite: {path}")
    export_binding_path = path.parent / "profiler-export-binding.json"
    export_binding = strict.load_strict_json(
        export_binding_path, label=f"candidate profiler export binding {path}"
    )
    rep_artifact = next(
        artifact for artifact in artifacts if artifact["kind"] == "nsys-rep"
    )
    sqlite_artifact = next(
        artifact for artifact in artifacts if artifact["kind"] == "nsys-sqlite"
    )
    exported_candidate = (
        export_binding.get("candidate") if isinstance(export_binding, dict) else None
    )
    exported_report = (
        export_binding.get("nsys_report") if isinstance(export_binding, dict) else None
    )
    exported_sqlite = (
        export_binding.get("nsys_sqlite") if isinstance(export_binding, dict) else None
    )
    if (
        not isinstance(export_binding, dict)
        or not isinstance(exported_candidate, dict)
        or not isinstance(exported_report, dict)
        or not isinstance(exported_sqlite, dict)
        or export_binding.get("schema_version") != 1
        or export_binding.get("run_id") != value.get("run_id")
        or export_binding.get("profile_nonce") != value.get("profile_nonce")
        or export_binding.get("launch_token_sha256") != launch_token_sha256
        or exported_candidate.get("path") != str(path.resolve())
        or exported_candidate.get("sha256") != strict.sha256_file(path)
        or exported_report.get("path") != rep_artifact["path"]
        or exported_report.get("sha256") != rep_artifact["sha256"]
        or exported_sqlite.get("path") != sqlite_artifact["path"]
        or exported_sqlite.get("sha256") != sqlite_artifact["sha256"]
        or Path(str(exported_sqlite.get("export_input_file", ""))).name
        != Path(str(rep_artifact["path"])).name
    ):
        raise SystemExit(f"candidate Nsight report/export binding is invalid: {path}")
    frozen_input_binding = export_binding.get("frozen_input_binding")
    binding = value.get("binding")
    if (
        not isinstance(binding, dict)
        or not isinstance(frozen_input_binding, dict)
        or frozen_input_binding
        != {key: binding.get(key) for key in PROFILE_INPUT_BINDING_FIELDS}
    ):
        raise SystemExit(f"candidate frozen input binding is invalid: {path}")
    artifacts.append(
        {
            "kind": "nsys-export-binding",
            "path": str(export_binding_path.resolve()),
            "sha256": strict.sha256_file(export_binding_path),
        }
    )
    cache_artifacts = value.get("cache_artifacts")
    if not isinstance(cache_artifacts, list) or len(cache_artifacts) != 2:
        raise SystemExit(f"candidate requires both frozen cache manifests: {path}")
    expected_cache_rows = {
        "train-cache-manifest": (
            "train_cache_manifest_sha256",
            "measured_train_rows",
            "distill_train",
        ),
        "validation-cache-manifest": (
            "validation_cache_manifest_sha256",
            "measured_validation_rows",
            "validation",
        ),
    }
    dataset_binding = binding.get("dataset_content_binding")
    dataset_files = (
        dataset_binding.get("files") if isinstance(dataset_binding, dict) else None
    )
    seen_cache_kinds: set[str] = set()
    for artifact in cache_artifacts:
        if not isinstance(artifact, dict):
            raise SystemExit(f"candidate cache artifact is malformed: {path}")
        kind = str(artifact.get("kind", ""))
        if kind not in expected_cache_rows or kind in seen_cache_kinds:
            raise SystemExit(f"candidate cache artifact kind is invalid: {path}")
        cache_path = Path(str(artifact.get("path", ""))).expanduser().resolve()
        if not cache_path.is_file():
            raise SystemExit(f"candidate cache manifest is missing: {cache_path}")
        digest = strict.sha256_file(cache_path)
        manifest = strict.load_strict_json(cache_path, label=f"{kind} {path}")
        digest_field, rows_field, expected_split = expected_cache_rows[kind]
        content_field = (
            "train_cache_content_binding"
            if kind == "train-cache-manifest"
            else "validation_cache_content_binding"
        )
        manifest_binding = (
            manifest.get("binding") if isinstance(manifest, dict) else None
        )
        if (
            not isinstance(binding, dict)
            or digest != artifact.get("sha256")
            or digest != binding.get(digest_field)
            or strict.cache_content_binding(cache_path)
            != binding.get(content_field)
            or not isinstance(manifest, dict)
            or manifest.get("row_count") != binding.get(rows_field)
            or manifest.get("layer_index") != binding.get("layer")
            or manifest.get("split") != expected_split
            or manifest.get("has_shared_states")
            is not binding.get("cache_has_shared_states")
            or not isinstance(manifest_binding, dict)
            or manifest_binding.get("source_checkpoint_sha256")
            != binding.get("source_checkpoint_sha256")
            or manifest_binding.get("zero_step_checkpoint_sha256")
            != binding.get("zero_step_checkpoint_sha256")
            or manifest_binding.get("training_config_sha256")
            != binding.get("plan_sha256")
            or manifest_binding.get("dataset_manifest_sha256")
            != (dataset_files.get("manifest") if isinstance(dataset_files, dict) else None)
            or manifest_binding.get("prefix_fingerprint")
            != binding.get("cache_prefix_fingerprint")
            or manifest_binding.get("split") != expected_split
        ):
            raise SystemExit(
                f"candidate did not execute the complete frozen {kind}: {path}"
            )
        artifact["path"] = str(cache_path)
        seen_cache_kinds.add(kind)
    tracking_path = path.parent / "experiment-tracking.json"
    report_path = path.parent / "experiment-report.md"
    if not tracking_path.is_file() or not report_path.is_file():
        raise SystemExit(f"candidate lacks W&B tracking or Chinese report: {path}")
    value["experiment_artifacts"] = [
        {
            "kind": "wandb-tracking",
            "path": str(tracking_path.resolve()),
            "sha256": strict.sha256_file(tracking_path),
        },
        {
            "kind": "experiment-report",
            "path": str(report_path.resolve()),
            "sha256": strict.sha256_file(report_path),
        },
    ]
    tracking = strict.validate_experiment_artifacts(
        value["experiment_artifacts"],
        base_dir=path.parent,
        expected_run_id=str(value.get("run_id")),
        expected_project=expected_project,
        expected_entity=expected_entity,
        expected_group=expected_group,
        expected_tags=expected_tags,
        expected_job_type=expected_job_type,
    )
    if (
        tracking.get("launch_token_sha256") != launch_token_sha256
        or tracking.get("coordinated_world_size") != 8
        or tracking.get("resume_existing") is not True
    ):
        raise SystemExit(
            f"candidate W&B startup/final Nsight resume is not fully bound: {path}"
        )
    config = tracking.get("config")
    if not isinstance(binding, dict) or not isinstance(config, dict):
        raise SystemExit(f"candidate tracking/config binding is malformed: {path}")
    expected_config_fields = {
        "action": "performance-profile-candidate",
        "run_id": value.get("run_id"),
        "plan_sha256": binding.get("plan_sha256"),
        "workload_sha256": binding.get("workload_sha256"),
        "source_config_sha256": binding.get("source_config_sha256"),
        "source_checkpoint_sha256": binding.get("source_checkpoint_sha256"),
        "source_checkpoint_binding": binding.get("source_checkpoint_binding"),
        "zero_step_checkpoint_sha256": binding.get("zero_step_checkpoint_sha256"),
        "zero_step_checkpoint_binding": binding.get("zero_step_checkpoint_binding"),
        "dataset_content_binding": binding.get("dataset_content_binding"),
        "overlay_files_sha256": binding.get("overlay_files_sha256"),
        "overlay_files": binding.get("overlay_files"),
        "train_cache_manifest_sha256": binding.get("train_cache_manifest_sha256"),
        "train_cache_content_binding": binding.get(
            "train_cache_content_binding"
        ),
        "validation_cache_manifest_sha256": binding.get(
            "validation_cache_manifest_sha256"
        ),
        "validation_cache_content_binding": binding.get(
            "validation_cache_content_binding"
        ),
        "training_source_files": binding.get("training_source_files"),
        "source_revision": binding.get("source_revision"),
        "cache_prefix_fingerprint": binding.get("cache_prefix_fingerprint"),
        "optimizer": binding.get("optimizer"),
        "micro_batch_size_per_rank": value.get("per_rank_micro_batch_size"),
        "world_size": binding.get("world_size"),
        "gradient_checkpointing": binding.get("gradient_checkpointing"),
        "measured_train_rows": binding.get("measured_train_rows"),
        "measured_validation_rows": binding.get("measured_validation_rows"),
        "row_permutation_sha256": binding.get("row_permutation_sha256"),
        "profile_case": binding.get("profile_case"),
        "cache_has_shared_states": binding.get("cache_has_shared_states"),
        "profile_nonce": value.get("profile_nonce"),
        "nvtx_range_names_by_rank": value.get("nvtx_range_names_by_rank"),
        "transition_nvtx_range_names_by_rank": value.get(
            "transition_nvtx_range_names_by_rank"
        ),
    }
    if any(
        config.get(key) != expected for key, expected in expected_config_fields.items()
    ):
        raise SystemExit(
            f"candidate W&B config differs from candidate evidence: {path}"
        )
    value["source_candidate"] = {
        "path": str(path.resolve()),
        "sha256": original_sha256,
    }
    if original_value != strict.load_strict_json(
        path, label=f"performance candidate post-validation {path}"
    ):
        raise SystemExit(f"candidate changed while selection read it: {path}")
    return value


def _candidate_artifact_binding(path: Path, strict) -> dict[str, object]:
    path = path.expanduser().resolve()
    files = {
        "candidate": path,
        "nsys_export_binding": path.parent / "profiler-export-binding.json",
        "wandb_tracking": path.parent / "experiment-tracking.json",
        "experiment_report": path.parent / "experiment-report.md",
    }
    if any(not value.is_file() for value in files.values()):
        raise SystemExit(f"candidate artifact set is incomplete: {path}")
    return {
        name: {"path": str(value), "sha256": strict.sha256_file(value)}
        for name, value in files.items()
    }


def _tracking_plan(plan: dict[str, object]) -> SimpleNamespace:
    tracking = plan.get("tracking")
    if not isinstance(tracking, dict):
        raise SystemExit("performance selection requires W&B tracking")
    job_types = tracking.get("job_types")
    if (
        tracking.get("mode") != "online"
        or not isinstance(tracking.get("project"), str)
        or not tracking["project"]
        or not isinstance(job_types, dict)
        or set(job_types) != {"profile", "distill", "corrective"}
    ):
        raise SystemExit(
            "performance selection requires online W&B and all three job types"
        )
    return SimpleNamespace(
        wandb_mode="online",
        wandb_project=tracking["project"],
        wandb_entity=tracking.get("entity"),
        wandb_group=tracking.get("group"),
        wandb_tags=tuple(tracking.get("tags", ())),
        wandb_job_types=tuple(sorted(job_types.items())),
        evidence_tier=plan.get("evidence_tier"),
    )


def _plain_profile_case(profile_case: object) -> str:
    if not isinstance(profile_case, dict):
        return "未知场景"
    boundary = profile_case.get("input_boundary")
    mixer_kind = profile_case.get("source_mixer_kind")
    if boundary == "embedding-output":
        return "第 0 层（输入直接来自 embedding）"
    if mixer_kind == "linear_attention":
        return "后续 GDN 层（输入来自已经转换完成的 RWKV7 前缀）"
    if mixer_kind == "full_attention":
        return "full-attention 层（输入来自已经转换完成的 RWKV7 前缀）"
    return f"后续 {mixer_kind} 层（输入来自已经转换完成的 RWKV7 前缀）"


def _write_selection_report(
    run_dir: Path,
    *,
    status: str,
    plan: dict[str, object],
    candidates: list[dict[str, object]] | None,
    selected: dict[str, object] | None,
    reason: str | None = None,
) -> None:
    tracking_path = run_dir / "experiment-tracking.json"
    tracking = (
        json.loads(tracking_path.read_text(encoding="utf-8"))
        if tracking_path.is_file()
        else {}
    )
    comparison = plan.get("experiment_comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    only_changes = comparison.get("only_changes")
    only_changes = only_changes if isinstance(only_changes, list) else ["未声明"]
    lines = [
        experiment_report_identity_line(tracking, status=status),
        "",
        "# 8 卡性能门禁结果",
        "",
        "## 这次想验证什么",
        "",
        "比较至少三个每卡 batch；每个 batch 都覆盖 embedding 输入、后续 GDN 输入和 full-attention 输入，并单独检查生成下一层输入缓存的 GPU 工作情况。",
        "",
        "## 实际结果",
        "",
        f"- 状态：`{status}`",
        f"- W&B：{tracking.get('url') or '未建立在线 run'}",
        "",
        "## 与上一次相比只改了什么",
        "",
        f"- 对照 run：`{comparison.get('baseline_run_id', '未声明')}`",
        *[f"- {change}" for change in only_changes],
        "",
        "## 候选对比",
        "",
    ]
    for row in candidates or []:
        profile_case = row.get("binding", {}).get("profile_case", {})
        kernel_coverage = row.get("kernel_covered_wall_fraction_by_rank", ())
        sm_active = row.get("sm_active_fraction_by_gpu", ())
        reserved = row.get("peak_reserved_memory_fraction_by_rank", ())
        walls = row.get("epoch_wall_seconds_by_rank", ())
        if not kernel_coverage or not sm_active or not reserved or not walls:
            lines.append(
                "- 每卡 batch "
                f"`{row.get('per_rank_micro_batch_size')}`：指标尚未完整复算，门禁不通过"
            )
            continue
        lines.append(
            "- 每卡 batch "
            f"`{row.get('per_rank_micro_batch_size')}` / 场景 "
            f"{_plain_profile_case(profile_case)}："
            f"有 CUDA 工作的时间比例 `{min(kernel_coverage):.1%}..{max(kernel_coverage):.1%}`，"
            f"计算单元真正工作的比例 `{min(sm_active):.1%}..{max(sm_active):.1%}`，"
            f"保留显存比例 `{min(reserved):.1%}..{max(reserved):.1%}`，"
            f"卡间墙钟 `{min(walls):.3f}..{max(walls):.3f}s`，"
            f"checkpoint `{float(row.get('checkpoint_wall_fraction', 0)):.1%}`，"
            f"每秒处理 `{row.get('loss_tokens_per_second')}` 个参与误差计算的 token，"
            f"门禁 `{'通过' if row.get('eligible') else '不通过'}`"
        )
        if row.get("cache_transition_profiled"):
            transition_kernel = row.get(
                "cache_transition_kernel_covered_wall_fraction_by_rank", ()
            )
            transition_sm = row.get(
                "cache_transition_sm_active_fraction_by_gpu", ()
            )
            transition_walls = row.get(
                "cache_transition_profiler_window_seconds_by_rank", ()
            )
            lines.append(
                "  - 生成下一层输入缓存：有 CUDA 工作的时间比例 "
                f"`{min(transition_kernel):.1%}..{max(transition_kernel):.1%}`，"
                f"计算单元真正工作的比例 `{min(transition_sm):.1%}..{max(transition_sm):.1%}`，"
                f"卡间墙钟 `{min(transition_walls):.3f}..{max(transition_walls):.3f}s`，"
                f"门禁 `{'通过' if row.get('cache_transition_eligible') else '不通过'}`"
            )
    if reason:
        lines.append(f"- 失败原因：{reason}")
    lines.extend(["", "## 结论", ""])
    if selected is None:
        lines.append("没有候选同时通过全部门槛，正式训练保持禁止。")
    else:
        lines.append(
            "选中每卡 batch "
            f"`{selected.get('per_rank_micro_batch_size')}`；它在所有实际层场景和缓存切换都通过后，按源模型各类层数量加权的端到端吞吐最高。"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            (
                "把选中 batch 和本 profile 的 SHA 写入正式 plan，strict wrapper 复算一致后才允许训练。"
                if selected is not None
                else "继续只做 benchmark/profile，先消除 GPU 空泡或增大可用 batch；不要启动训练。"
            ),
            "",
        ]
    )
    temporary = run_dir / "experiment-report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(run_dir / "experiment-report.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate short, ephemeral 8-GPU Nsight candidate runs into the only "
            "performance artifact accepted by strict Any2RWKV training."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--target-config", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--head-size", required=True, type=int)
    parser.add_argument("--candidate", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("HELICOPTER_RUN_ID") != args.run_id:
        raise SystemExit("selection --run-id must equal HELICOPTER_RUN_ID")
    args.output = args.output.expanduser().resolve()
    reserved = (
        args.output,
        args.output.parent / "experiment-tracking.json",
        args.output.parent / "experiment-report.md",
    )
    if any(path.exists() for path in reserved):
        raise SystemExit("performance selection requires a fresh output run directory")
    strict = _load_strict_wrapper()
    plan = strict.load_strict_json(args.plan, label="performance profile plan")
    if not isinstance(plan, dict):
        raise SystemExit("performance profile plan must be a JSON object")
    required_profile_cases = strict.source_profile_cases(args.source_config)
    if len(args.candidate) < 3 * len(required_profile_cases):
        raise SystemExit(
            "at least three batches must cover every required source-layer profile case"
        )
    candidate_artifact_bindings = [
        _candidate_artifact_binding(path, strict) for path in args.candidate
    ]
    tracker = ExperimentTracker.from_plan(
        run_dir=args.output.parent,
        plan=_tracking_plan(plan),
        action="profile",
    )
    candidates: list[dict[str, object]] | None = None
    selected: dict[str, object] | None = None
    try:
        tracker.start(
            config={
                "schema_version": 1,
                "action": "performance-profile-selection",
                "run_id": args.run_id,
                "plan_sha256": strict.sha256_file(args.plan),
                "workload_sha256": strict.performance_workload_binding(plan)["sha256"],
                "source_config_sha256": strict.sha256_file(args.source_config),
                "source_checkpoint_sha256": strict.checkpoint_content_binding(
                    args.source_config
                )["sha256"],
                "source_checkpoint_binding": strict.checkpoint_content_binding(
                    args.source_config
                ),
                "target_checkpoint_sha256": strict.checkpoint_content_binding(
                    args.target_config
                )["sha256"],
                "target_checkpoint_binding": strict.checkpoint_content_binding(
                    args.target_config
                ),
                "dataset_manifest_sha256": strict.sha256_file(args.dataset_manifest),
                "training_source_files": strict.training_source_files(),
                "source_revision": strict.source_revision_binding(),
                "comparison": plan.get("experiment_comparison"),
                "required_profile_cases": required_profile_cases,
                "candidate_artifacts": candidate_artifact_bindings,
            }
        )
        tracking = plan["tracking"]
        candidates = [
            _candidate(
                path,
                strict,
                expected_project=str(tracking["project"]),
                expected_entity=tracking.get("entity"),
                expected_group=tracking.get("group"),
                expected_tags=list(tracking.get("tags", [])),
                expected_job_type=str(tracking["job_types"]["profile"]),
            )
            for path in args.candidate
        ]
        candidate_run_ids = [str(row.get("run_id", "")) for row in candidates]
        if (
            len(set(candidate_run_ids)) != len(candidate_run_ids)
            or args.run_id in candidate_run_ids
        ):
            raise SystemExit("profile candidate and selection run IDs must be unique")
        optimizer = plan.get("optimizer")
        expected_binding = {
            "plan_sha256": strict.sha256_file(args.plan),
            "source_config_sha256": strict.sha256_file(args.source_config),
            "source_checkpoint_sha256": strict.checkpoint_content_binding(
                args.source_config
            )["sha256"],
            "source_checkpoint_binding": strict.checkpoint_content_binding(
                args.source_config
            ),
            "zero_step_checkpoint_sha256": strict.checkpoint_content_binding(
                args.target_config
            )["sha256"],
            "zero_step_checkpoint_binding": strict.checkpoint_content_binding(
                args.target_config
            ),
            "dataset_content_binding": strict.dataset_content_binding(
                args.dataset_manifest
            ),
            "workload_sha256": strict.performance_workload_binding(plan)["sha256"],
            "training_source_files": strict.training_source_files(),
            "source_revision": strict.source_revision_binding(),
            "optimizer": optimizer,
            "learning_rate_schedule": plan.get("learning_rate_schedule"),
            "gradient_clip_norm": plan.get("gradient_clip_norm"),
            "accumulation_steps": 1,
            "gradient_checkpointing": False,
            "burn_in_tokens": plan.get("burn_in_tokens"),
            "supervised_tokens": plan.get("supervised_tokens"),
            "head_size": args.head_size,
            "world_size": 8,
        }
        cache_bindings_by_case: dict[
            str, set[tuple[object, object, object]]
        ] = {case_id: set() for case_id in required_profile_cases}
        candidate_matrix: dict[tuple[int, str], dict[str, object]] = {}
        target_signatures_by_case: dict[str, set[str]] = {
            case_id: set() for case_id in required_profile_cases
        }
        source_signatures_by_case: dict[str, set[str]] = {
            case_id: set() for case_id in required_profile_cases
        }
        overlays_by_case: dict[str, set[str]] = {
            case_id: set() for case_id in required_profile_cases
        }
        budgets = {row.get("loss_token_count") for row in candidates}
        if len(budgets) != 1:
            raise SystemExit(
                "performance candidates must use one equal loss-token budget"
            )
        for row in candidates:
            binding = row.get("binding")
            if (
                row.get("profiler_capture_contract") != PROFILER_CAPTURE_CONTRACT
                or not isinstance(binding, dict)
                or any(
                    binding.get(key) != value for key, value in expected_binding.items()
                )
            ):
                raise SystemExit(
                    "performance candidate does not bind the measured plan/checkpoint"
                )
            train_rows = binding.get("measured_train_rows")
            validation_rows = binding.get("measured_validation_rows")
            batch = row.get("per_rank_micro_batch_size")
            supervised_tokens = plan.get("supervised_tokens")
            profile_case = binding.get("profile_case")
            case_id = (
                profile_case.get("profile_case_id")
                if isinstance(profile_case, dict)
                else None
            )
            expected_case = required_profile_cases.get(str(case_id))
            source_signature = binding.get("source_parameter_signature_sha256")
            target_signature = binding.get("target_parameter_signature_sha256")
            if (
                type(train_rows) is not int
                or train_rows <= 0
                or type(validation_rows) is not int
                or validation_rows <= 0
                or type(batch) is not int
                or batch <= 0
                or type(supervised_tokens) is not int
                or supervised_tokens <= 0
                or train_rows % (8 * batch)
                or row.get("ephemeral_optimizer_steps") != train_rows // (8 * batch)
                or row["ephemeral_optimizer_steps"] <= int(optimizer["warmup_steps"])
                or row.get("loss_token_count") != train_rows * supervised_tokens
                or not isinstance(binding.get("row_permutation_sha256"), str)
                or len(binding["row_permutation_sha256"]) != 64
                or not isinstance(binding.get("cache_prefix_fingerprint"), str)
                or len(binding["cache_prefix_fingerprint"]) != 64
                or expected_case is None
                or profile_case != expected_case
                or binding.get("layer") != expected_case["representative_layer"]
                or binding.get("cache_has_shared_states")
                is not (expected_case["input_boundary"] == "recurrent-prefix")
                or not isinstance(source_signature, str)
                or len(source_signature) != 64
                or not isinstance(target_signature, str)
                or len(target_signature) != 64
            ):
                raise SystemExit(
                    "performance candidate row/step budget is inconsistent"
                )
            matrix_key = (batch, str(case_id))
            if matrix_key in candidate_matrix:
                raise SystemExit("duplicate batch/profile-case candidate")
            candidate_matrix[matrix_key] = row
            source_signatures_by_case[str(case_id)].add(source_signature)
            target_signatures_by_case[str(case_id)].add(target_signature)
            overlay_sha = binding.get("overlay_files_sha256")
            if not isinstance(overlay_sha, str) or len(overlay_sha) != 64:
                raise SystemExit("candidate overlay binding is malformed")
            overlays_by_case[str(case_id)].add(overlay_sha)
            cache_bindings_by_case[str(case_id)].add(
                (
                    binding.get("train_cache_content_binding", {}).get("sha256"),
                    binding.get("validation_cache_content_binding", {}).get(
                        "sha256"
                    ),
                    binding.get("cache_prefix_fingerprint"),
                )
            )
            sqlite_path = next(
                Path(str(artifact["path"]))
                for artifact in row["profiler_artifacts"]
                if artifact["kind"] == "nsys-sqlite"
            )
            nsys_metrics = strict.derive_nsys_profile_metrics(
                sqlite_path, row.get("nvtx_range_names_by_rank", [])
            )
            row["kernel_covered_wall_fraction_by_rank"] = nsys_metrics[
                "kernel_covered_wall_fraction_by_rank"
            ]
            row["sm_active_fraction_by_gpu"] = nsys_metrics[
                "sm_active_fraction_by_gpu"
            ]
            row["sm_active_sample_count_by_gpu"] = nsys_metrics[
                "sm_active_sample_count_by_gpu"
            ]
            row["sm_active_metric_source_by_gpu"] = nsys_metrics[
                "sm_active_metric_source_by_gpu"
            ]
            row["gpu_metric_common_window_seconds"] = nsys_metrics[
                "gpu_metric_common_window_seconds"
            ]
            row["profiler_window_seconds_by_rank"] = nsys_metrics[
                "window_seconds_by_rank"
            ]
            row["nsys_process_ids_by_rank"] = nsys_metrics["process_ids_by_rank"]
            row["nsys_device_ids_by_rank"] = nsys_metrics["device_ids_by_rank"]
            row["nsys_kernel_intersection_count_by_rank"] = nsys_metrics[
                "kernel_intersection_count_by_rank"
            ]
            transition_required = int(expected_case["transition_count"]) > 0
            transition_names = row.get("transition_nvtx_range_names_by_rank")
            if (
                row.get("cache_transition_profiled") is not transition_required
                or (
                    transition_required
                    and (
                        not isinstance(transition_names, list)
                        or len(transition_names) != 8
                    )
                )
                or (not transition_required and transition_names != [])
            ):
                raise SystemExit(
                    "candidate does not profile its required cache transition"
                )
            transition_metrics = (
                strict.derive_nsys_profile_metrics(sqlite_path, transition_names)
                if transition_required
                else None
            )
            if transition_metrics is not None:
                row["cache_transition_kernel_covered_wall_fraction_by_rank"] = (
                    transition_metrics["kernel_covered_wall_fraction_by_rank"]
                )
                row["cache_transition_sm_active_fraction_by_gpu"] = (
                    transition_metrics["sm_active_fraction_by_gpu"]
                )
                row["cache_transition_sm_active_sample_count_by_gpu"] = (
                    transition_metrics["sm_active_sample_count_by_gpu"]
                )
                row["cache_transition_sm_active_metric_source_by_gpu"] = (
                    transition_metrics["sm_active_metric_source_by_gpu"]
                )
                row["cache_transition_gpu_metric_common_window_seconds"] = (
                    transition_metrics["gpu_metric_common_window_seconds"]
                )
                row["cache_transition_profiler_window_seconds_by_rank"] = (
                    transition_metrics["window_seconds_by_rank"]
                )
                row["cache_transition_nsys_process_ids_by_rank"] = (
                    transition_metrics["process_ids_by_rank"]
                )
                row["cache_transition_nsys_device_ids_by_rank"] = (
                    transition_metrics["device_ids_by_rank"]
                )
                row["cache_transition_nsys_kernel_intersection_count_by_rank"] = (
                    transition_metrics["kernel_intersection_count_by_rank"]
                )
            kernel_coverage = row.get("kernel_covered_wall_fraction_by_rank")
            sm_active = row.get("sm_active_fraction_by_gpu")
            reserved = row.get("peak_reserved_memory_fraction_by_rank")
            python_wall = row.get("epoch_wall_seconds_by_rank")
            wall = row.get("profiler_window_seconds_by_rank")
            checkpoint_walls = row.get("checkpoint_wall_seconds_by_rank")
            finite_lists = all(
                isinstance(values, list)
                and len(values) == 8
                and all(strict._finite_number(value) for value in values)
                for values in (
                    kernel_coverage,
                    sm_active,
                    reserved,
                    wall,
                    python_wall,
                    checkpoint_walls,
                )
            )
            if not finite_lists:
                raise SystemExit("candidate rank measurements are malformed")
            if any(
                abs(float(measured) - float(profiled)) / float(profiled) > 0.05
                for measured, profiled in zip(python_wall, wall, strict=True)
            ):
                raise SystemExit(
                    "Python epoch wall differs by more than 5% from bound NVTX window"
                )
            row["loss_tokens_per_second"] = float(row["loss_token_count"]) / max(
                map(float, wall)
            )
            row["checkpoint_wall_fraction"] = max(map(float, checkpoint_walls)) / max(
                map(float, wall)
            )
            checkpoint = row["checkpoint_wall_fraction"]
            epoch_eligible = strict.performance_candidate_passes_gates(
                kernel_coverage=kernel_coverage,
                sm_active=sm_active,
                reserved=reserved,
                wall_seconds=wall,
                checkpoint_fraction=checkpoint,
            )
            transition_eligible = True
            if transition_metrics is not None:
                transition_python_wall = row.get(
                    "cache_transition_wall_seconds_by_rank"
                )
                transition_reserved = row.get(
                    "cache_transition_peak_reserved_memory_fraction_by_rank"
                )
                transition_wall = transition_metrics["window_seconds_by_rank"]
                if not all(
                    isinstance(values, list)
                    and len(values) == 8
                    and all(strict._finite_number(value) for value in values)
                    for values in (
                        transition_python_wall,
                        transition_reserved,
                        transition_wall,
                    )
                ):
                    raise SystemExit("cache-transition measurements are malformed")
                if any(
                    abs(float(measured) - float(profiled)) / float(profiled) > 0.05
                    for measured, profiled in zip(
                        transition_python_wall, transition_wall, strict=True
                    )
                ):
                    raise SystemExit(
                        "cache-transition Python wall differs from its NVTX window"
                    )
                transition_eligible = strict.performance_candidate_passes_gates(
                    kernel_coverage=transition_metrics[
                        "kernel_covered_wall_fraction_by_rank"
                    ],
                    sm_active=transition_metrics["sm_active_fraction_by_gpu"],
                    reserved=transition_reserved,
                    wall_seconds=transition_wall,
                    checkpoint_fraction=0.0,
                )
            row["epoch_eligible"] = epoch_eligible
            row["cache_transition_eligible"] = transition_eligible
            row["eligible"] = epoch_eligible and transition_eligible
        validation_budgets = {
            row["binding"]["measured_validation_rows"] for row in candidates
        }
        batches = {
            int(row["per_rank_micro_batch_size"]) for row in candidates
        }
        expected_matrix = {
            (batch, case_id)
            for batch in batches
            for case_id in required_profile_cases
        }
        if (
            len(batches) < 3
            or set(candidate_matrix) != expected_matrix
            or len(validation_budgets) != 1
            or any(len(values) != 1 for values in cache_bindings_by_case.values())
            or any(
                not isinstance(value, str) or len(value) != 64
                for bindings in cache_bindings_by_case.values()
                for value in next(iter(bindings))
            )
            or any(len(values) != 1 for values in source_signatures_by_case.values())
            or any(len(values) != 1 for values in target_signatures_by_case.values())
            or any(len(values) != 1 for values in overlays_by_case.values())
        ):
            raise SystemExit(
                "each batch must cover every required layer case with one frozen cache/signature"
            )
        layer_count = sum(
            int(case["layer_count"]) for case in required_profile_cases.values()
        )
        equal_loss_token_budget = float(next(iter(budgets)))
        batch_summaries: list[dict[str, object]] = []
        for batch in sorted(batches):
            rows = [
                candidate_matrix[(batch, case_id)]
                for case_id in required_profile_cases
            ]
            epoch_wall = sum(
                int(required_profile_cases[case_id]["layer_count"])
                * max(
                    map(
                        float,
                        candidate_matrix[(batch, case_id)][
                            "profiler_window_seconds_by_rank"
                        ],
                    )
                )
                for case_id in required_profile_cases
            )
            transition_wall = sum(
                int(required_profile_cases[case_id]["transition_count"])
                * (
                    max(
                        map(
                            float,
                            candidate_matrix[(batch, case_id)][
                                "cache_transition_profiler_window_seconds_by_rank"
                            ],
                        )
                    )
                    if int(required_profile_cases[case_id]["transition_count"]) > 0
                    else 0.0
                )
                for case_id in required_profile_cases
            )
            total_wall = epoch_wall + transition_wall
            batch_summaries.append(
                {
                    "per_rank_micro_batch_size": batch,
                    "eligible": all(bool(row["eligible"]) for row in rows),
                    "weighted_epoch_wall_seconds": epoch_wall,
                    "weighted_cache_transition_wall_seconds": transition_wall,
                    "weighted_end_to_end_wall_seconds": total_wall,
                    "weighted_end_to_end_loss_tokens_per_second": (
                        equal_loss_token_budget * layer_count / total_wall
                    ),
                }
            )
        eligible_batches = [row for row in batch_summaries if row["eligible"]]
        if not eligible_batches:
            raise SystemExit(
                "no batch passes every required layer and cache-transition gate"
            )
        selected = max(
            eligible_batches,
            key=lambda row: float(
                row["weighted_end_to_end_loss_tokens_per_second"]
            ),
        )
        payload = {
            "schema_version": 3,
            "status": "accepted",
            "run_id": args.run_id,
            "selection_rule": (
                "highest-weighted-all-layer-case-end-to-end-loss-token-throughput-with-all-gates"
            ),
            "binding": {
                "action": "distill",
                "selection_plan_sha256": strict.sha256_file(args.plan),
                "world_size": 8,
                "source_config_sha256": strict.sha256_file(args.source_config),
                "source_checkpoint_sha256": strict.checkpoint_content_binding(
                    args.source_config
                )["sha256"],
                "source_checkpoint_binding": strict.checkpoint_content_binding(
                    args.source_config
                ),
                "target_checkpoint_sha256": strict.checkpoint_content_binding(
                    args.target_config
                )["sha256"],
                "target_checkpoint_binding": strict.checkpoint_content_binding(
                    args.target_config
                ),
                "dataset_manifest_sha256": strict.sha256_file(args.dataset_manifest),
                "dataset_content_binding": strict.dataset_content_binding(
                    args.dataset_manifest
                ),
                "dataset_content_binding": strict.dataset_content_binding(
                    args.dataset_manifest
                ),
                "training_source_files": strict.training_source_files(),
                "source_revision": strict.source_revision_binding(),
                "candidate_artifacts": candidate_artifact_bindings,
                "overlay_files_sha256_by_case": {
                    case_id: next(iter(values))
                    for case_id, values in overlays_by_case.items()
                },
                "workload_sha256": strict.performance_workload_binding(plan)["sha256"],
                "head_size": args.head_size,
                "burn_in_tokens": plan.get("burn_in_tokens"),
                "supervised_tokens": plan.get("supervised_tokens"),
                "accumulation_steps": plan.get("accumulation_steps"),
                "gradient_checkpointing": plan.get("gradient_checkpointing"),
                "checkpoint_interval_micro_batches": plan.get(
                    "checkpoint_interval_micro_batches"
                ),
            },
            "equal_loss_token_budget": next(iter(budgets)),
            "required_profile_cases": required_profile_cases,
            "candidates": candidates,
            "batch_summaries": batch_summaries,
            "selected_per_rank_micro_batch_size": selected["per_rank_micro_batch_size"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
        tracker.log_metrics(
            {
                "selection": {
                    "selected_per_rank_micro_batch_size": selected[
                        "per_rank_micro_batch_size"
                    ],
                    "selected_weighted_end_to_end_loss_tokens_per_second": selected[
                        "weighted_end_to_end_loss_tokens_per_second"
                    ],
                    "candidate": {
                        (
                            f"b{row['per_rank_micro_batch_size']}/"
                            f"{row['binding']['profile_case']['profile_case_id']}"
                        ): {
                            "eligible": row["eligible"],
                            "epoch_eligible": row["epoch_eligible"],
                            "cache_transition_eligible": row[
                                "cache_transition_eligible"
                            ],
                            "loss_tokens_per_second": row["loss_tokens_per_second"],
                            "checkpoint_wall_fraction": row["checkpoint_wall_fraction"],
                            "rank": {
                                str(rank): {
                                    "kernel_covered_wall_fraction": row[
                                        "kernel_covered_wall_fraction_by_rank"
                                    ][rank],
                                    "sm_active_fraction": row[
                                        "sm_active_fraction_by_gpu"
                                    ][rank],
                                    "peak_reserved_memory_fraction": row[
                                        "peak_reserved_memory_fraction_by_rank"
                                    ][rank],
                                    "epoch_wall_seconds": row[
                                        "epoch_wall_seconds_by_rank"
                                    ][rank],
                                    "cache_transition_kernel_covered_wall_fraction": (
                                        row[
                                            "cache_transition_kernel_covered_wall_fraction_by_rank"
                                        ][rank]
                                        if row["cache_transition_profiled"]
                                        else None
                                    ),
                                    "cache_transition_sm_active_fraction": (
                                        row[
                                            "cache_transition_sm_active_fraction_by_gpu"
                                        ][rank]
                                        if row["cache_transition_profiled"]
                                        else None
                                    ),
                                    "cache_transition_peak_reserved_memory_fraction": (
                                        row[
                                            "cache_transition_peak_reserved_memory_fraction_by_rank"
                                        ][rank]
                                        if row["cache_transition_profiled"]
                                        else None
                                    ),
                                }
                                for rank in range(8)
                            },
                        }
                        for row in candidates
                    },
                }
            }
        )
        tracker.finish(exit_code=0)
        _write_selection_report(
            args.output.parent,
            status="accepted",
            plan=plan,
            candidates=candidates,
            selected=selected,
        )
        tracking_path = args.output.parent / "experiment-tracking.json"
        report_path = args.output.parent / "experiment-report.md"
        payload["experiment_artifacts"] = [
            {
                "kind": "wandb-tracking",
                "path": str(tracking_path.resolve()),
                "sha256": strict.sha256_file(tracking_path),
            },
            {
                "kind": "experiment-report",
                "path": str(report_path.resolve()),
                "sha256": strict.sha256_file(report_path),
            },
        ]
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
        print(args.output.read_text(encoding="utf-8"))
    except BaseException as error:
        try:
            tracker.abort(reason=f"{type(error).__name__}: {error}")
            _write_selection_report(
                args.output.parent,
                status="failed",
                plan=plan,
                candidates=candidates,
                selected=None,
                reason=f"{type(error).__name__}: {error}",
            )
        except BaseException as finalization_error:
            error.add_note(str(finalization_error))
        raise


if __name__ == "__main__":
    main()
