#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch

for name, value in {
    "RWKV_JIT_ON": "0",
    "RWKV_MY_TESTING": "x070",
    "RWKV_KERNEL": "",
    "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
    "RWKV_TRAIN_TYPE": "infctx",
    "RWKV_FLOAT_MODE": "bf16",
    "WKV_MODE": "fp32io16",
}.items():
    os.environ.setdefault(name, value)

from any2rwkv.artifacts import file_sha256, write_json
from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.core import LayerInputCacheReader
from any2rwkv.core.experiment_tracking import (
    ExperimentTracker,
    experiment_report_identity_line,
)
from any2rwkv.distributed import DistributedContext
from any2rwkv.distill_runner import read_distillation_plan
from any2rwkv.layer_schedule import epoch_permutation
from any2rwkv.mixer_store import RWKV7MixerLayerStore
from any2rwkv.recipes.qwen35_to_rwkv7.layer_major_runner import (
    _commit_distributed_generation,
    _cursor,
    _ensure_next_layer_caches,
    _local_loss,
    _validate,
)
from any2rwkv.streamed_teacher import (
    StreamedQwen35HybridExecutor,
    StreamedQwen35Teacher,
)
from any2rwkv.streaming_training import ActiveLayerOptimizer


ROOT = Path(__file__).resolve().parents[1]
STRICT_WRAPPER = ROOT / "scripts" / "strict_any2rwkv_train.py"
PROFILER_CAPTURE_CONTRACT = (
    "nsys-full-process-exact-nvtx-rank-transition-and-input-bindings-v5"
)


def _load_strict_wrapper():
    spec = importlib.util.spec_from_file_location(
        "any2rwkv_strict_wrapper_for_profile_candidate", STRICT_WRAPPER
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load strict Any2RWKV wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _validate_profile_paths(args: argparse.Namespace) -> None:
    args.source = args.source.expanduser().resolve()
    args.plan = args.plan.expanduser().resolve()
    args.dataset_manifest = args.dataset_manifest.expanduser().resolve()
    args.zero_step = args.zero_step.expanduser().resolve()
    args.overlays = args.overlays.expanduser().resolve()
    args.train_cache = args.train_cache.expanduser().resolve()
    args.validation_cache = args.validation_cache.expanduser().resolve()
    args.scratch = args.scratch.expanduser().resolve()
    args.nsys_rep = args.nsys_rep.expanduser().resolve()
    args.nsys_sqlite = args.nsys_sqlite.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    run_dir = args.output.parent
    if args.output != run_dir / "candidate.json":
        raise SystemExit("profile output must be RUN_DIR/candidate.json")
    if args.scratch != run_dir / "scratch":
        raise SystemExit("profile scratch must be the dedicated RUN_DIR/scratch")
    if args.nsys_rep != run_dir / "profile.nsys-rep":
        raise SystemExit("profile nsys report must be RUN_DIR/profile.nsys-rep")
    if args.nsys_sqlite != run_dir / "profile.sqlite":
        raise SystemExit("profile SQLite must be RUN_DIR/profile.sqlite")
    readonly_roots = {
        args.plan,
        args.dataset_manifest,
        _source_config(args.source).parent,
        args.zero_step,
        args.overlays,
        args.train_cache,
        args.validation_cache,
    }
    for readonly in readonly_roots:
        if _paths_overlap(run_dir, readonly):
            raise SystemExit(
                "profile RUN_DIR must be isolated from every source/checkpoint/cache: "
                f"{readonly}"
            )
    reserved = (
        args.output,
        args.nsys_rep,
        args.nsys_sqlite,
        run_dir / "experiment-tracking.json",
        run_dir / "experiment-report.md",
    )
    existing = [str(path) for path in reserved if path.exists()]
    if existing:
        raise SystemExit(
            "profile candidate run directory contains stale evidence; use a new run id: "
            + ", ".join(existing)
        )


def _directory_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        str(file.relative_to(path)): file_sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _module_parameter_signature(module: torch.nn.Module) -> str:
    return _sha256_json(
        [
            {
                "name": name,
                "shape": list(parameter.shape),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in module.named_parameters()
        ]
    )


def _source_profile_case(
    source_config: dict[str, object], layer: int
) -> dict[str, object]:
    text = source_config.get("text_config", source_config)
    if not isinstance(text, dict):
        raise SystemExit("source config text_config must be an object")
    layer_types = text.get("layer_types")
    layer_count = int(text.get("num_hidden_layers", 0))
    if (
        not isinstance(layer_types, list)
        or layer_count <= 0
        or len(layer_types) != layer_count
        or not 0 <= layer < layer_count
        or not all(isinstance(value, str) and value for value in layer_types)
    ):
        raise SystemExit(
            "profile source requires one mixer kind for every source layer"
        )
    input_boundary = "embedding-output" if layer == 0 else "recurrent-prefix"
    mixer_kind = str(layer_types[layer])
    matching_layers = [
        index
        for index, value in enumerate(layer_types)
        if str(value) == mixer_kind
        and ("embedding-output" if index == 0 else "recurrent-prefix")
        == input_boundary
    ]
    representative = min(matching_layers)
    if layer != representative:
        raise SystemExit(
            "profile candidate must use the earliest representative layer for its case: "
            f"expected={representative} actual={layer}"
        )
    return {
        "profile_case_id": f"{input_boundary}:{mixer_kind}",
        "source_mixer_kind": mixer_kind,
        "input_boundary": input_boundary,
        "representative_layer": representative,
        "layer_count": len(matching_layers),
        "transition_count": sum(index + 1 < layer_count for index in matching_layers),
    }


def _cache_has_shared_states(path: Path) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid cache manifest: {error}") from error
    value = manifest.get("has_shared_states")
    if type(value) is not bool:
        raise SystemExit("cache manifest must declare has_shared_states")
    return value


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


def _load_plan(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid performance profile plan: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 3:
        raise SystemExit("performance profile plan must use schema_version=3")
    return value


def _workload_sha256(plan: dict[str, object]) -> str:
    normalized = {
        key: value
        for key, value in plan.items()
        if key
        not in {
            "micro_batch_size",
            "performance_evidence",
            "throughput_evidence",
            "notes",
        }
    }
    return _sha256_json(normalized)


def _source_config(path: Path) -> Path:
    config = path if path.name == "config.json" else path / "config.json"
    if not config.is_file():
        raise SystemExit(f"source config is missing: {config}")
    return config


def _checkpoint_content_sha256(config_path: Path) -> str:
    root = config_path.parent
    index_path = root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid zero-step shard index: {error}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("zero-step shard index is empty")
    names = {
        "config.json",
        "model.safetensors.index.json",
        *map(str, weight_map.values()),
    }
    for tokenizer_name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ):
        if (root / tokenizer_name).is_file():
            names.add(tokenizer_name)
    for optional in (
        "mapping.json",
        "mapping-coverage.json",
        "warm-start-plan.json",
    ):
        if (root / optional).is_file():
            names.add(optional)
    files = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise SystemExit(f"zero-step checkpoint file is missing: {path}")
        files[name] = file_sha256(path)
    return _sha256_json(files)


def _profile_cache_prefix_fingerprint(
    manifest_path: Path,
    *,
    source_checkpoint_sha256: str,
    zero_step_checkpoint_sha256: str,
    training_config_sha256: str,
    dataset_manifest_sha256: str,
    layer: int,
    split: str,
) -> str:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {split} cache manifest: {error}") from error
    binding = manifest.get("binding") if isinstance(manifest, dict) else None
    prefix = binding.get("prefix_fingerprint") if isinstance(binding, dict) else None
    if (
        manifest.get("schema_version") != 1
        or manifest.get("layer_index") != layer
        or manifest.get("split") != split
        or not isinstance(binding, dict)
        or binding.get("source_checkpoint_sha256") != source_checkpoint_sha256
        or binding.get("zero_step_checkpoint_sha256")
        != zero_step_checkpoint_sha256
        or binding.get("training_config_sha256") != training_config_sha256
        or binding.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or binding.get("split") != split
        or not isinstance(prefix, str)
        or len(prefix) != 64
    ):
        raise SystemExit(
            f"{split} cache is not bound to the measured source/layer/prefix"
        )
    return prefix


def _tracking_plan(plan: dict[str, object]) -> SimpleNamespace:
    tracking = plan.get("tracking")
    if not isinstance(tracking, dict):
        raise SystemExit("performance profile plan requires W&B tracking")
    job_types = tracking.get("job_types")
    if (
        tracking.get("mode") != "online"
        or not isinstance(tracking.get("project"), str)
        or not tracking["project"]
        or not isinstance(job_types, dict)
        or set(job_types) != {"profile", "distill", "corrective"}
    ):
        raise SystemExit(
            "performance profile requires online W&B and all three job types"
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


def _write_profile_report(
    run_dir: Path,
    *,
    status: str,
    plan: dict[str, object],
    candidate: dict[str, object] | None,
    reason: str | None,
) -> None:
    tracking_path = run_dir / "experiment-tracking.json"
    tracking = (
        json.loads(tracking_path.read_text(encoding="utf-8"))
        if tracking_path.is_file()
        else {}
    )
    optimizer = plan.get("optimizer", {})
    comparison = plan.get("experiment_comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    only_changes = comparison.get("only_changes")
    only_changes = only_changes if isinstance(only_changes, list) else ["未声明"]
    binding = candidate.get("binding", {}) if candidate is not None else {}
    reserved = (
        candidate.get("peak_reserved_memory_fraction_by_rank", ())
        if candidate is not None
        else ()
    )
    walls = (
        candidate.get("epoch_wall_seconds_by_rank", ()) if candidate is not None else ()
    )
    lines = [
        experiment_report_identity_line(tracking, status=status),
        "",
        "# 性能扫描结果",
        "",
        "## 这次想验证什么",
        "",
        "验证这个每卡 batch 是否能让 8 张卡持续工作，同时把显存用到安全上限内。",
        "",
        "## 实际跑了什么",
        "",
        f"- 运行状态：`{status}`",
        f"- 每卡 batch：`{candidate.get('per_rank_micro_batch_size') if candidate else '未知'}`",
        f"- 临时 AdamW update：`{candidate.get('ephemeral_optimizer_steps') if candidate else '未知'}` 次；正式 checkpoint 没有被修改",
        f"- 优化器：`{optimizer}`",
        f"- 测量场景：{_plain_profile_case(binding.get('profile_case'))}",
        f"- 代表层：`{binding.get('layer', '未知')}`",
        f"- W&B：{tracking.get('url') or '未建立在线 run'}",
        "",
        "## 与上一次相比只改了什么",
        "",
        f"- 对照 run：`{comparison.get('baseline_run_id', '未声明')}`",
        *[f"- {change}" for change in only_changes],
        "",
        "## 看到的结果",
        "",
    ]
    if candidate is not None:
        lines.extend(
            [
                (
                    "- 端到端吞吐：每秒处理 "
                    f"`{candidate.get('loss_tokens_per_second')}` 个真正参与误差计算的 token"
                ),
                (
                    "- 每卡 peak reserved memory 比例："
                    f"`{min(reserved):.1%}..{max(reserved):.1%}`"
                    if reserved
                    else "- 没有可用的显存记录",
                ),
                (
                    f"- 每卡完整周期墙钟：`{min(walls):.3f}..{max(walls):.3f}` 秒"
                    if walls
                    else "- 没有可用的墙钟记录",
                ),
                f"- checkpoint 占完整周期：`{float(candidate.get('checkpoint_wall_fraction', 0)):.1%}`",
                (
                    "- 生成下一层输入缓存的每卡墙钟："
                    f"`{min(candidate['cache_transition_wall_seconds_by_rank']):.3f}.."
                    f"{max(candidate['cache_transition_wall_seconds_by_rank']):.3f}` 秒"
                    if candidate.get("cache_transition_profiled")
                    else "- 这个场景在模型末层，不需要生成下一层输入缓存",
                ),
            ]
        )
    if reason:
        lines.append(f"- 失败原因：{reason}")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "这个候选已经完成，但还要等 Nsight SQLite 分别复算“测量窗口里有 CUDA 工作的时间比例”和“每张卡的计算单元真正工作的比例”；单个候选不能决定正式 batch。"
                if status == "completed"
                else "这个候选没有形成可接受的性能证据，不能启动训练。"
            ),
            "",
            "## 下一步",
            "",
            "每个 batch 都要覆盖 embedding 输入、后续 GDN 输入和 full-attention 输入等全部实际场景；聚合器复算全部门槛后才能选择正式 batch。",
            "",
        ]
    )
    temporary = run_dir / "experiment-report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(run_dir / "experiment-report.md")


def _finalize_profile_tracking(
    tracker: ExperimentTracker,
    distributed: DistributedContext,
    *,
    plan: dict[str, object],
    candidate: dict[str, object] | None,
    status: str,
    reason: str | None = None,
) -> None:
    error_message = None
    if distributed.is_primary:
        try:
            if candidate is not None:
                tracker.log_metrics(
                    {
                        "profile": {
                            "per_rank_micro_batch_size": candidate.get(
                                "per_rank_micro_batch_size"
                            ),
                            "loss_token_count": candidate.get("loss_token_count"),
                            "loss_tokens_per_second": candidate.get(
                                "loss_tokens_per_second"
                            ),
                            "ephemeral_optimizer_steps": candidate.get(
                                "ephemeral_optimizer_steps"
                            ),
                            "checkpoint_wall_fraction": candidate.get(
                                "checkpoint_wall_fraction"
                            ),
                            "profile_case": candidate.get("binding", {}).get(
                                "profile_case"
                            ),
                            "rank": {
                                str(rank): {
                                    "epoch_wall_seconds": candidate[
                                        "epoch_wall_seconds_by_rank"
                                    ][rank],
                                    "peak_reserved_memory_fraction": candidate[
                                        "peak_reserved_memory_fraction_by_rank"
                                    ][rank],
                                    "cache_transition_wall_seconds": (
                                        candidate[
                                            "cache_transition_wall_seconds_by_rank"
                                        ][rank]
                                        if candidate.get(
                                            "cache_transition_profiled"
                                        )
                                        else None
                                    ),
                                }
                                for rank in range(8)
                            },
                        }
                    }
                )
            tracker.finish(exit_code=0 if status == "completed" else 1)
            _write_profile_report(
                tracker.run_dir,
                status=status,
                plan=plan,
                candidate=candidate,
                reason=reason,
            )
        except BaseException as error:
            error_message = f"{type(error).__name__}: {error}"
    error_message = distributed.broadcast_object(error_message)
    if error_message is not None:
        raise RuntimeError("distributed profile tracking failed: " + error_message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile one ephemeral, end-to-end Any2RWKV batch candidate."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--zero-step", required=True, type=Path)
    parser.add_argument("--overlays", required=True, type=Path)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--validation-cache", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--micro-batch", required=True, type=int)
    parser.add_argument("--measured-train-rows", required=True, type=int)
    parser.add_argument("--measured-validation-rows", required=True, type=int)
    parser.add_argument("--nsys-rep", required=True, type=Path)
    parser.add_argument("--nsys-sqlite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _validate_profile_paths(args)
    strict = _load_strict_wrapper()
    if os.environ.get("HELICOPTER_RUN_ID") != args.run_id:
        raise SystemExit("profile --run-id must equal HELICOPTER_RUN_ID")
    if (
        args.layer < 0
        or args.micro_batch <= 0
        or args.measured_train_rows <= 0
        or args.measured_validation_rows <= 0
    ):
        raise SystemExit("profile candidate sizes are invalid")
    plan = _load_plan(args.plan)
    plan_contract = read_distillation_plan(args.plan)
    optimizer_contract = plan.get("optimizer")
    if (
        not isinstance(optimizer_contract, dict)
        or optimizer_contract.get("name") != "adamw"
        or plan.get("accumulation_steps") != 1
        or plan.get("gradient_checkpointing") is not False
        or plan.get("distributed_world_size") != 8
        or plan.get("learning_rate_schedule") != "warmup-constant"
        or plan.get("checkpoint_interval_micro_batches") != 0
        or not isinstance(plan.get("gradient_clip_norm"), (int, float))
        or not isinstance(plan.get("max_cached_layer_input_bytes_per_rank"), int)
        or int(plan["max_cached_layer_input_bytes_per_rank"]) <= 0
        or not isinstance(plan.get("cache_shard_rows"), int)
        or int(plan["cache_shard_rows"]) <= 0
    ):
        raise SystemExit(
            "profile candidate requires the explicit 8-rank AdamW plan contract"
        )
    try:
        learning_rate = float(plan_contract.learning_rate)
        final_learning_rate = float(plan_contract.final_learning_rate)
        warmup_steps = int(plan_contract.learning_rate_warmup_steps or 0)
        beta1, beta2 = plan_contract.optimizer_betas
        epsilon = float(plan_contract.optimizer_epsilon)
        weight_decay = float(plan_contract.optimizer_weight_decay)
        gradient_clip = float(plan["gradient_clip_norm"])
        burn_in_tokens = int(plan["burn_in_tokens"])
        supervised_tokens = int(plan["supervised_tokens"])
        max_cached_bytes = int(plan["max_cached_layer_input_bytes_per_rank"])
        cache_shard_rows = int(plan["cache_shard_rows"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"performance profile plan has an incomplete optimizer/workload: {error}"
        ) from error
    zero_step_config = json.loads(
        (args.zero_step / "config.json").read_text(encoding="utf-8")
    )
    head_size = int(zero_step_config.get("head_size", 0))
    if head_size <= 0 or os.environ.get("RWKV_HEAD_SIZE") != str(head_size):
        raise SystemExit("RWKV_HEAD_SIZE must match the zero-step checkpoint")

    # Freeze every read-only input before creating scratch, W&B, or report files.
    source_config_path = _source_config(args.source)
    source_config_payload = json.loads(source_config_path.read_text(encoding="utf-8"))
    profile_case = _source_profile_case(source_config_payload, args.layer)
    source_config_sha256 = file_sha256(source_config_path)
    source_checkpoint_binding = strict.checkpoint_content_binding(source_config_path)
    source_checkpoint_sha256 = source_checkpoint_binding["sha256"]
    zero_step_binding = strict.checkpoint_content_binding(
        args.zero_step / "config.json"
    )
    zero_step_before = zero_step_binding["sha256"]
    overlays_before = _directory_hashes(args.overlays)
    overlay_files_sha256 = _sha256_json(overlays_before)
    plan_sha256 = file_sha256(args.plan)
    dataset_binding = strict.dataset_content_binding(args.dataset_manifest)
    train_cache_binding = strict.cache_content_binding(
        args.train_cache / "manifest.json"
    )
    validation_cache_binding = strict.cache_content_binding(
        args.validation_cache / "manifest.json"
    )
    training_sources = strict.training_source_files()
    source_revision = strict.source_revision_binding()
    train_cache_manifest_sha256 = file_sha256(args.train_cache / "manifest.json")
    validation_cache_manifest_sha256 = file_sha256(
        args.validation_cache / "manifest.json"
    )
    train_prefix_fingerprint = _profile_cache_prefix_fingerprint(
        args.train_cache / "manifest.json",
        source_checkpoint_sha256=source_checkpoint_sha256,
        zero_step_checkpoint_sha256=zero_step_before,
        training_config_sha256=plan_sha256,
        dataset_manifest_sha256=dataset_binding["files"]["manifest"],
        layer=args.layer,
        split="distill_train",
    )
    validation_prefix_fingerprint = _profile_cache_prefix_fingerprint(
        args.validation_cache / "manifest.json",
        source_checkpoint_sha256=source_checkpoint_sha256,
        zero_step_checkpoint_sha256=zero_step_before,
        training_config_sha256=plan_sha256,
        dataset_manifest_sha256=dataset_binding["files"]["manifest"],
        layer=args.layer,
        split="validation",
    )
    if train_prefix_fingerprint != validation_prefix_fingerprint:
        raise SystemExit("train and validation caches use different recurrent prefixes")
    train_has_shared_states = _cache_has_shared_states(
        args.train_cache / "manifest.json"
    )
    validation_has_shared_states = _cache_has_shared_states(
        args.validation_cache / "manifest.json"
    )
    expected_shared_states = profile_case["input_boundary"] == "recurrent-prefix"
    if (
        train_has_shared_states is not validation_has_shared_states
        or train_has_shared_states is not expected_shared_states
    ):
        raise SystemExit(
            "profile caches do not match the embedding/recurrent input boundary"
        )
    workload_sha256 = _workload_sha256(plan)
    measured_permutation, measured_permutation_sha256 = epoch_permutation(
        row_count=args.measured_train_rows,
        seed=plan_contract.seed,
        layer=args.layer,
        epoch=0,
    )
    profile_nonce = os.environ.get("ANY2RWKV_PROFILE_NONCE")
    if (
        not isinstance(profile_nonce, str)
        or len(profile_nonce) != 32
        or any(character not in "0123456789abcdef" for character in profile_nonce)
    ):
        raise SystemExit(
            "profile launcher must provide ANY2RWKV_PROFILE_NONCE as 32 lowercase hex digits"
        )
    nvtx_range_names = [
        f"any2rwkv-profile:{args.run_id}:{profile_nonce}:rank={rank}"
        for rank in range(8)
    ]
    transition_nvtx_range_names = [
        f"any2rwkv-transition:{args.run_id}:{profile_nonce}:rank={rank}"
        for rank in range(8)
    ]
    tracker = ExperimentTracker.from_plan(
        run_dir=args.output.parent,
        plan=_tracking_plan(plan),
        action="profile",
    )
    distributed: DistributedContext | None = None
    tracking_finalized = False
    nvtx_pushed = False
    try:
        tracker.start(
            config={
                "schema_version": 1,
                "action": "performance-profile-candidate",
                "run_id": args.run_id,
                "plan_sha256": plan_sha256,
                "workload_sha256": workload_sha256,
                "source_config_sha256": source_config_sha256,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "source_checkpoint_binding": source_checkpoint_binding,
                "zero_step_checkpoint_sha256": zero_step_before,
                "zero_step_checkpoint_binding": zero_step_binding,
                "dataset_content_binding": dataset_binding,
                "overlay_files_sha256": overlay_files_sha256,
                "overlay_files": overlays_before,
                "train_cache_manifest_sha256": train_cache_manifest_sha256,
                "train_cache_content_binding": train_cache_binding,
                "validation_cache_manifest_sha256": (
                    validation_cache_manifest_sha256
                ),
                "validation_cache_content_binding": validation_cache_binding,
                "training_source_files": training_sources,
                "source_revision": source_revision,
                "cache_prefix_fingerprint": train_prefix_fingerprint,
                "optimizer": optimizer_contract,
                "comparison": plan.get("experiment_comparison"),
                "micro_batch_size_per_rank": args.micro_batch,
                "world_size": 8,
                "gradient_checkpointing": False,
                "measured_train_rows": args.measured_train_rows,
                "measured_validation_rows": args.measured_validation_rows,
                "row_permutation_sha256": measured_permutation_sha256,
                "profile_case": profile_case,
                "cache_has_shared_states": train_has_shared_states,
                "profile_nonce": profile_nonce,
                "nvtx_range_names_by_rank": nvtx_range_names,
                "transition_nvtx_range_names_by_rank": (
                    transition_nvtx_range_names
                    if int(profile_case["transition_count"]) > 0
                    else []
                ),
            }
        )
        distributed = DistributedContext.initialize()
        if distributed.world_size != 8:
            raise SystemExit("performance candidates require exactly eight ranks")
        global_batch = args.micro_batch * distributed.world_size
        if args.measured_train_rows % global_batch:
            raise SystemExit("measured train rows must divide every global micro-batch")
        prepare_status = None
        if distributed.is_primary:
            try:
                if args.scratch.exists():
                    shutil.rmtree(args.scratch)
                args.scratch.mkdir(parents=True)
                prepare_status = {"status": "ok"}
            except BaseException as error:
                prepare_status = {"status": "error", "error": repr(error)}
        prepare_status = distributed.broadcast_object(prepare_status)
        if prepare_status["status"] != "ok":
            raise RuntimeError(prepare_status["error"])
        source = read_checkpoint(args.source, require_final_layers=False)
        train_reader = LayerInputCacheReader(
            args.train_cache,
            verification="full",
            max_cached_bytes=max_cached_bytes,
        )
        validation_reader = LayerInputCacheReader(
            args.validation_cache,
            verification="full",
            max_cached_bytes=max_cached_bytes,
        )
        if args.measured_train_rows != train_reader.row_count:
            raise SystemExit(
                "profile must execute one complete frozen train-cache epoch"
            )
        if args.measured_validation_rows != validation_reader.row_count:
            raise SystemExit(
                "profile must execute the complete frozen validation cache"
            )
        teacher = StreamedQwen35Teacher(
            source,
            device=distributed.device,
            dtype=torch.bfloat16,
            cache_layers=False,
            load_output_head=False,
        )
        executor = StreamedQwen35HybridExecutor(teacher)
        loaded_layer = teacher.loader.load_layer(
            args.layer, device=distributed.device, dtype=torch.bfloat16
        )
        store = RWKV7MixerLayerStore(args.zero_step, args.overlays)
        mixer = store.load_mixer(
            args.layer, device=distributed.device, dtype=torch.bfloat16
        )
        source_parameter_signature_sha256 = _module_parameter_signature(
            loaded_layer.module
        )
        target_parameter_signature_sha256 = _module_parameter_signature(mixer)
        source_text_config = source.config.get("text_config", source.config)
        source_layer_types = tuple(source_text_config.get("layer_types", ()))
        if args.layer >= len(source_layer_types):
            raise SystemExit("profile layer is absent from source layer_types")
        learning_rate_profiles = dict(plan_contract.local_learning_rate_by_mixer_kind)
        unknown_profiles = set(learning_rate_profiles) - set(source_layer_types)
        if unknown_profiles:
            raise SystemExit(
                "profile plan has learning-rate overrides for absent layer types: "
                f"{sorted(unknown_profiles)}"
            )
        configured_learning_rate = learning_rate_profiles.get(
            str(profile_case["source_mixer_kind"]), learning_rate
        )
        final_ratio = final_learning_rate / learning_rate
        configured_final_learning_rate = (
            configured_learning_rate
            if math.isclose(final_ratio, 1.0, rel_tol=1e-12, abs_tol=0.0)
            else configured_learning_rate * final_ratio
        )
        optimizer_steps_per_epoch = math.ceil(
            math.ceil(train_reader.row_count / global_batch)
            / plan_contract.accumulation_steps
        )
        scheduled_epochs = (
            plan_contract.layer_learning_rate_schedule_epochs
            or plan_contract.layer_fixed_epochs
            or plan_contract.layer_max_epochs
        )
        total_steps = optimizer_steps_per_epoch * scheduled_epochs
        optimizer = ActiveLayerOptimizer(
            optimizer_name="adamw",
            learning_rate=configured_learning_rate,
            final_learning_rate=configured_final_learning_rate,
            adam_betas=(beta1, beta2),
            adam_epsilon=epsilon,
            weight_decay=weight_decay,
            learning_rate_schedule="warmup-constant",
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            gradient_clip_norm=gradient_clip,
            max_parameter_update_relative_l2=(
                plan_contract.max_parameter_update_relative_l2
            ),
            gradient_sync=distributed.synchronize_gradients,
            detailed_telemetry_interval_steps=(
                plan_contract.optimizer_telemetry_interval_steps
            ),
        )
        optimizer.activate(
            args.layer,
            mixer,
            trainable_names={name for name, _ in mixer.named_parameters()},
        )
        loss_weights = plan_contract.local_loss_weights

        distributed.barrier()
        torch.cuda.synchronize(distributed.device)
        torch.cuda.reset_peak_memory_stats(distributed.device)
        torch.cuda.nvtx.range_push(nvtx_range_names[distributed.rank])
        nvtx_pushed = True
        torch.cuda.synchronize(distributed.device)
        epoch_started = time.perf_counter()
        train_started = epoch_started
        for start in range(0, args.measured_train_rows, global_batch):
            global_rows = measured_permutation[start : start + global_batch]
            local_rows = distributed.shard_rows(global_rows)
            cached = train_reader.read_rows(local_rows)
            output = executor.forward_cached_layer_local(
                cached.hidden_states,
                shared_states=cached.shared_states,
                active_layer_index=args.layer,
                active_mixer=mixer,
                loaded_layer=loaded_layer,
            )
            loss, _ = _local_loss(
                output, burn_in_tokens, loss_weights, materialize_metrics=False
            )
            optimizer.backward(
                loss,
                accumulation_steps=1,
                sample_weight=len(global_rows),
            )
        torch.cuda.synchronize(distributed.device)
        train_wall = time.perf_counter() - train_started

        validation_started = time.perf_counter()
        validation_metrics = _validate(
            executor=executor,
            reader=validation_reader,
            mixer=mixer,
            loaded_layer=loaded_layer,
            layer_index=args.layer,
            burn_in_tokens=burn_in_tokens,
            micro_batch_size=args.micro_batch,
            loss_weights=loss_weights,
            distributed=distributed,
        )
        torch.cuda.synchronize(distributed.device)
        validation_wall = time.perf_counter() - validation_started

        checkpoint_started = time.perf_counter()
        cursor = _cursor(
            args.layer,
            0,
            args.measured_train_rows,
            measured_permutation_sha256,
            consumed_rows=measured_permutation,
            train_cache_manifest_sha256=train_cache_manifest_sha256,
            validation_cache_manifest_sha256=validation_cache_manifest_sha256,
        )
        _commit_distributed_generation(
            distributed,
            args.scratch,
            store,
            mixer,
            optimizer,
            cursor,
        )
        torch.cuda.synchronize(distributed.device)
        checkpoint_wall = time.perf_counter() - checkpoint_started
        torch.cuda.nvtx.range_pop()
        nvtx_pushed = False
        epoch_wall = time.perf_counter() - epoch_started
        gathered = distributed.gather_scalar_metrics(
            {
                "epoch_wall_seconds": epoch_wall,
                "train_wall_seconds": train_wall,
                "validation_wall_seconds": validation_wall,
                "checkpoint_wall_seconds": checkpoint_wall,
                "peak_reserved_memory_fraction": (
                    torch.cuda.max_memory_reserved(distributed.device)
                    / torch.cuda.get_device_properties(distributed.device).total_memory
                ),
            }
        )
        transition_wall_seconds_by_rank: list[float] = []
        transition_peak_reserved_memory_fraction_by_rank: list[float] = []
        if int(profile_case["transition_count"]) > 0:
            # Formal layer transition releases the active optimizer before it
            # forwards the selected mixer into the next rolling cache.
            optimizer.release()
            distributed.barrier()
            torch.cuda.synchronize(distributed.device)
            torch.cuda.reset_peak_memory_stats(distributed.device)
            torch.cuda.nvtx.range_push(
                transition_nvtx_range_names[distributed.rank]
            )
            nvtx_pushed = True
            torch.cuda.synchronize(distributed.device)
            transition_started = time.perf_counter()
            next_prefix_fingerprint = _sha256_json(
                {
                    "profile_case_id": profile_case["profile_case_id"],
                    "source_prefix_fingerprint": train_prefix_fingerprint,
                    "target_parameter_signature_sha256": (
                        target_parameter_signature_sha256
                    ),
                    "boundary": f"after-layer-{args.layer}",
                }
            )
            _ensure_next_layer_caches(
                executor=executor,
                cache_root=args.scratch / "transition-cache",
                current_layer=args.layer,
                train_reader=train_reader,
                validation_reader=validation_reader,
                mixer=mixer,
                loaded_layer=loaded_layer,
                shard_rows=cache_shard_rows,
                hidden_size=int(source_text_config["hidden_size"]),
                base_binding=dict(train_reader.manifest.get("binding", {})),
                next_prefix_fingerprint=next_prefix_fingerprint,
                distributed=distributed,
            )
            torch.cuda.synchronize(distributed.device)
            transition_wall = time.perf_counter() - transition_started
            torch.cuda.nvtx.range_pop()
            nvtx_pushed = False
            transition_gathered = distributed.gather_scalar_metrics(
                {
                    "wall_seconds": transition_wall,
                    "peak_reserved_memory_fraction": (
                        torch.cuda.max_memory_reserved(distributed.device)
                        / torch.cuda.get_device_properties(
                            distributed.device
                        ).total_memory
                    ),
                }
            )
            transition_wall_seconds_by_rank = list(
                transition_gathered["wall_seconds"]
            )
            transition_peak_reserved_memory_fraction_by_rank = list(
                transition_gathered["peak_reserved_memory_fraction"]
            )
        zero_step_after = _checkpoint_content_sha256(args.zero_step / "config.json")
        overlays_after = _directory_hashes(args.overlays)
        if (
            zero_step_after != zero_step_before
            or overlays_after != overlays_before
            or file_sha256(_source_config(args.source)) != source_config_sha256
            or _checkpoint_content_sha256(_source_config(args.source))
            != source_checkpoint_sha256
            or file_sha256(args.plan) != plan_sha256
            or strict.dataset_content_binding(args.dataset_manifest)
            != dataset_binding
            or strict.cache_content_binding(args.train_cache / "manifest.json")
            != train_cache_binding
            or strict.cache_content_binding(args.validation_cache / "manifest.json")
            != validation_cache_binding
            or strict.training_source_files() != training_sources
            or strict.source_revision_binding() != source_revision
        ):
            raise RuntimeError(
                "profile input changed during measurement; evidence is invalid"
            )

        candidate_payload = None
        if distributed.is_primary:
            max_epoch_wall = max(gathered["epoch_wall_seconds"])
            max_checkpoint_wall = max(gathered["checkpoint_wall_seconds"])
            loss_tokens = args.measured_train_rows * supervised_tokens
            candidate_payload = {
                "schema_version": 1,
                "run_id": args.run_id,
                "per_rank_micro_batch_size": args.micro_batch,
                "loss_token_count": loss_tokens,
                "loss_tokens_per_second": loss_tokens / max_epoch_wall,
                "includes_validation": True,
                "includes_checkpoint": True,
                "ephemeral_optimizer_steps": optimizer.optimizer_step,
                "persistent_weight_update": False,
                "epoch_wall_seconds_by_rank": list(gathered["epoch_wall_seconds"]),
                "train_wall_seconds_by_rank": list(gathered["train_wall_seconds"]),
                "validation_wall_seconds_by_rank": list(
                    gathered["validation_wall_seconds"]
                ),
                "checkpoint_wall_seconds_by_rank": list(
                    gathered["checkpoint_wall_seconds"]
                ),
                "checkpoint_wall_fraction": (max_checkpoint_wall / max_epoch_wall),
                "peak_reserved_memory_fraction_by_rank": list(
                    gathered["peak_reserved_memory_fraction"]
                ),
                "cache_transition_profiled": bool(
                    int(profile_case["transition_count"]) > 0
                ),
                "cache_transition_wall_seconds_by_rank": (
                    transition_wall_seconds_by_rank
                ),
                "cache_transition_peak_reserved_memory_fraction_by_rank": (
                    transition_peak_reserved_memory_fraction_by_rank
                ),
                "profiler_capture_contract": PROFILER_CAPTURE_CONTRACT,
                "profile_nonce": profile_nonce,
                "nvtx_range_names_by_rank": nvtx_range_names,
                "transition_nvtx_range_names_by_rank": (
                    transition_nvtx_range_names
                    if int(profile_case["transition_count"]) > 0
                    else []
                ),
                "validation_metrics": validation_metrics,
                "profiler_artifacts": [
                    {"kind": "nsys-rep", "path": str(args.nsys_rep)},
                    {"kind": "nsys-sqlite", "path": str(args.nsys_sqlite)},
                ],
                "cache_artifacts": [
                    {
                        "kind": "train-cache-manifest",
                        "path": str(args.train_cache / "manifest.json"),
                        "sha256": train_cache_manifest_sha256,
                    },
                    {
                        "kind": "validation-cache-manifest",
                        "path": str(args.validation_cache / "manifest.json"),
                        "sha256": validation_cache_manifest_sha256,
                    },
                ],
                "binding": {
                    "plan_sha256": plan_sha256,
                    "source_config_sha256": source_config_sha256,
                    "source_checkpoint_sha256": source_checkpoint_sha256,
                    "source_checkpoint_binding": source_checkpoint_binding,
                    "zero_step_checkpoint_sha256": zero_step_before,
                    "zero_step_checkpoint_binding": zero_step_binding,
                    "dataset_content_binding": dataset_binding,
                    "overlay_files_sha256": overlay_files_sha256,
                    "overlay_files": overlays_before,
                    "workload_sha256": workload_sha256,
                    "optimizer": optimizer_contract,
                    "learning_rate_schedule": plan.get("learning_rate_schedule"),
                    "gradient_clip_norm": gradient_clip,
                    "accumulation_steps": 1,
                    "gradient_checkpointing": False,
                    "burn_in_tokens": burn_in_tokens,
                    "supervised_tokens": supervised_tokens,
                    "measured_train_rows": args.measured_train_rows,
                    "measured_validation_rows": (args.measured_validation_rows),
                    "row_permutation_sha256": measured_permutation_sha256,
                    "train_cache_manifest_sha256": train_cache_manifest_sha256,
                    "train_cache_content_binding": train_cache_binding,
                    "validation_cache_manifest_sha256": (
                        validation_cache_manifest_sha256
                    ),
                    "validation_cache_content_binding": validation_cache_binding,
                    "training_source_files": training_sources,
                    "source_revision": source_revision,
                    "cache_prefix_fingerprint": train_prefix_fingerprint,
                    "layer": args.layer,
                    "profile_case": profile_case,
                    "cache_has_shared_states": train_has_shared_states,
                    "source_parameter_signature_sha256": (
                        source_parameter_signature_sha256
                    ),
                    "target_parameter_signature_sha256": (
                        target_parameter_signature_sha256
                    ),
                    "head_size": head_size,
                    "world_size": distributed.world_size,
                },
            }
            write_json(args.output, candidate_payload)
        candidate_payload = distributed.broadcast_object(candidate_payload)
        distributed.barrier()
        cleanup_status = None
        if distributed.is_primary:
            try:
                shutil.rmtree(args.scratch)
                cleanup_status = {"status": "ok"}
            except BaseException as error:
                cleanup_status = {"status": "error", "error": repr(error)}
        cleanup_status = distributed.broadcast_object(cleanup_status)
        if cleanup_status["status"] != "ok":
            raise RuntimeError(cleanup_status["error"])
        assert tracker is not None
        _finalize_profile_tracking(
            tracker,
            distributed,
            plan=plan,
            candidate=candidate_payload,
            status="completed",
        )
        tracking_finalized = True
    except BaseException as error:
        if nvtx_pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except BaseException:
                pass
            nvtx_pushed = False
        if not tracking_finalized:
            try:
                if tracker.is_primary:
                    reason = f"{type(error).__name__}: {error}"
                    tracker.abort(reason=reason)
                    _write_profile_report(
                        tracker.run_dir,
                        status="failed",
                        plan=plan,
                        candidate=None,
                        reason=reason,
                    )
            except BaseException as finalization_error:
                error.add_note(str(finalization_error))
        raise
    finally:
        if distributed is not None:
            distributed.close()


if __name__ == "__main__":
    main()
