from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "strict_any2rwkv_train.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "strict_any2rwkv_train_for_test", SCRIPT
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
STRICT_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(STRICT_SCRIPT)
AGGREGATOR_SCRIPT = ROOT / "scripts" / "profile_any2rwkv_layer_major.py"
AGGREGATOR_SPEC = importlib.util.spec_from_file_location(
    "profile_any2rwkv_layer_major_for_test", AGGREGATOR_SCRIPT
)
assert AGGREGATOR_SPEC is not None and AGGREGATOR_SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(AGGREGATOR_SPEC)
AGGREGATOR_SPEC.loader.exec_module(AGGREGATOR)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_completed_experiment(
    directory: Path,
    *,
    run_id: str,
    project: str,
    job_type: str,
    config: dict[str, object],
    report_status: str,
    launch_token: str | None = None,
    coordinated_world_size: int = 1,
    resume_existing: bool = False,
) -> tuple[Path, Path]:
    attempt_id = hashlib.md5(run_id.encode(), usedforsecurity=False).hexdigest()
    launch_token = launch_token or attempt_id
    config_sha = STRICT_SCRIPT.sha256_json(config)
    tracking = directory / "experiment-tracking.json"
    tracking.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "wandb",
                "mode": "online",
                "status": "completed",
                "resume_existing": resume_existing,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "launch_token_sha256": hashlib.sha256(
                    launch_token.encode("ascii")
                ).hexdigest(),
                "coordinated_world_size": coordinated_world_size,
                "project": project,
                "entity": None,
                "group": "strict-wrapper",
                "tags": ["test"],
                "job_type": job_type,
                "url": f"https://wandb.invalid/{run_id}",
                "config": config,
                "config_sha256": config_sha,
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    report = directory / "experiment-report.md"
    report.write_text(
        "<!-- any2rwkv-report-identity: "
        + json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "status": report_status,
                "tracking_status": "completed",
                "config_sha256": config_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + " -->\n\n# 实验结果\n",
        encoding="utf-8",
    )
    return tracking, report


def test_experiment_artifacts_reject_wrong_wandb_group_or_tags(tmp_path: Path) -> None:
    tracking, report = write_completed_experiment(
        tmp_path,
        run_id="identity-test",
        project="any2rwkv-tests",
        job_type="performance-profile",
        config={"schema_version": 1},
        report_status="complete",
    )
    artifacts = [
        {"kind": "wandb-tracking", "path": str(tracking), "sha256": sha256(tracking)},
        {
            "kind": "experiment-report",
            "path": str(report),
            "sha256": sha256(report),
        },
    ]

    with pytest.raises(SystemExit, match="W&B identity"):
        STRICT_SCRIPT.validate_experiment_artifacts(
            artifacts,
            base_dir=tmp_path,
            expected_run_id="identity-test",
            expected_project="any2rwkv-tests",
            expected_entity=None,
            expected_group="different-group",
            expected_tags=["test"],
            expected_job_type="performance-profile",
        )

    with pytest.raises(SystemExit, match="W&B identity"):
        STRICT_SCRIPT.validate_experiment_artifacts(
            artifacts,
            base_dir=tmp_path,
            expected_run_id="identity-test",
            expected_project="any2rwkv-tests",
            expected_entity=None,
            expected_group="strict-wrapper",
            expected_tags=["different-tag"],
            expected_job_type="performance-profile",
        )


def strict_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    checkpoint = tmp_path / "config.json"
    checkpoint.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "hidden_size": 2048,
                "num_hidden_layers": 1,
                "layer_types": ["linear_attention"],
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "num_attention_heads": 8,
                "head_dim": 256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_shard = tmp_path / "model-00001-of-00001.safetensors"
    source_shard.write_bytes(b"source-weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.self_attn.q_proj.weight": source_shard.name}}),
        encoding="utf-8",
    )
    dataset = tmp_path / "data-splits.json"
    dataset.write_text('{"schema_version":1}\n', encoding="utf-8")
    target_config = tmp_path / "output" / "checkpoint-zero-step" / "config.json"
    target_config.parent.mkdir(parents=True)
    target_config.write_text('{"head_size":128}\n', encoding="utf-8")
    target_shard = target_config.parent / "model-00001-of-00001.safetensors"
    target_shard.write_bytes(b"zero-step")
    (target_config.parent / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.attn.r_k": target_shard.name}}),
        encoding="utf-8",
    )
    (target_config.parent / "mapping.json").write_text("{}\n", encoding="utf-8")
    (target_config.parent / "mapping-coverage.json").write_text(
        "{}\n", encoding="utf-8"
    )
    plan_payload = {
        "schema_version": 3,
        "classification": "strict-wrapper-test",
        "evidence_tier": "exploratory",
        "execution_mode": "streamed_layer_store",
        "distributed_world_size": 8,
        "seed": 20260714,
        "experiment_comparison": {
            "baseline_run_id": "fixture-baseline",
            "only_changes": ["strict wrapper fixture"],
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-6,
            "final_learning_rate": 1e-6,
            "warmup_steps": 10,
            "betas": [0.9, 0.99],
            "epsilon": 1e-8,
            "weight_decay": 0.1,
        },
        "learning_rate": 1e-6,
        "learning_rate_schedule": "warmup-constant",
        "gradient_clip_norm": 1.0,
        "burn_in_tokens": 128,
        "supervised_tokens": 512,
        "micro_batch_size": 2,
        "accumulation_steps": 1,
        "gradient_checkpointing": False,
        "cache_shard_rows": 4,
        "max_layer_input_cache_bytes": 1_000_000,
        "max_cached_layer_input_bytes_per_rank": 262_144,
        "checkpoint_interval_micro_batches": 0,
        "tracking": {
            "mode": "online",
            "project": "any2rwkv-tests",
            "entity": None,
            "group": "strict-wrapper",
            "tags": ["test"],
            "job_types": {
                "profile": "performance-profile",
                "distill": "layerwise-distillation",
                "corrective": "fully-recurrent-corrective",
            },
        },
    }
    candidates = []
    target_checkpoint_sha = STRICT_SCRIPT.checkpoint_content_binding(target_config)[
        "sha256"
    ]
    source_checkpoint_sha = STRICT_SCRIPT.checkpoint_content_binding(checkpoint)[
        "sha256"
    ]
    workload_sha = STRICT_SCRIPT.performance_workload_binding(plan_payload)["sha256"]
    required_profile_cases = STRICT_SCRIPT.source_profile_cases(checkpoint)
    profile_case = required_profile_cases["embedding-output:linear_attention"]
    profile_plan = tmp_path / "profile-plan.json"
    profile_plan.write_text(json.dumps(plan_payload), encoding="utf-8")
    profile_plan_sha = sha256(profile_plan)
    measured_train_rows = 768
    measured_validation_rows = 64
    cache_prefix_fingerprint = "d" * 64
    source_checkpoint_binding = STRICT_SCRIPT.checkpoint_content_binding(checkpoint)
    target_checkpoint_binding = STRICT_SCRIPT.checkpoint_content_binding(target_config)
    dataset_content_binding = STRICT_SCRIPT.dataset_content_binding(dataset)
    overlay_files: dict[str, str] = {}
    overlay_files_sha = STRICT_SCRIPT.sha256_json(overlay_files)
    train_cache_dir = tmp_path / "train-cache"
    validation_cache_dir = tmp_path / "validation-cache"
    train_cache_dir.mkdir()
    validation_cache_dir.mkdir()
    train_shard = train_cache_dir / "shard-000000.safetensors"
    validation_shard = validation_cache_dir / "shard-000000.safetensors"
    train_shard.write_bytes(b"train-cache")
    validation_shard.write_bytes(b"validation-cache")
    train_cache_manifest = train_cache_dir / "manifest.json"
    validation_cache_manifest = validation_cache_dir / "manifest.json"
    train_cache_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layer_index": 0,
                "split": "distill_train",
                "row_count": measured_train_rows,
                "has_shared_states": False,
                "binding": {
                    "source_checkpoint_sha256": source_checkpoint_sha,
                    "zero_step_checkpoint_sha256": target_checkpoint_sha,
                    "training_config_sha256": profile_plan_sha,
                    "dataset_manifest_sha256": sha256(dataset),
                    "prefix_fingerprint": cache_prefix_fingerprint,
                    "split": "distill_train",
                },
                "shards": [
                    {
                        "path": train_shard.name,
                        "sha256": sha256(train_shard),
                        "row_indices": list(range(measured_train_rows)),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    validation_cache_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layer_index": 0,
                "split": "validation",
                "row_count": measured_validation_rows,
                "has_shared_states": False,
                "binding": {
                    "source_checkpoint_sha256": source_checkpoint_sha,
                    "zero_step_checkpoint_sha256": target_checkpoint_sha,
                    "training_config_sha256": profile_plan_sha,
                    "dataset_manifest_sha256": sha256(dataset),
                    "prefix_fingerprint": cache_prefix_fingerprint,
                    "split": "validation",
                },
                "shards": [
                    {
                        "path": validation_shard.name,
                        "sha256": sha256(validation_shard),
                        "row_indices": list(range(measured_validation_rows)),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    train_cache_sha = sha256(train_cache_manifest)
    validation_cache_sha = sha256(validation_cache_manifest)
    train_cache_content_binding = STRICT_SCRIPT.cache_content_binding(
        train_cache_manifest
    )
    validation_cache_content_binding = STRICT_SCRIPT.cache_content_binding(
        validation_cache_manifest
    )
    for batch_size, throughput in ((2, 300.0), (4, 200.0), (8, 100.0)):
        candidate_dir = tmp_path / f"candidate-{batch_size}"
        candidate_dir.mkdir()
        nsys_report = candidate_dir / "profile.nsys-rep"
        sqlite = candidate_dir / "profile.sqlite"
        nsys_report.write_bytes(f"report-{batch_size}".encode())
        loss_tokens = measured_train_rows * 512
        window_seconds = loss_tokens / throughput
        window_ns = int(window_seconds * 1e9)
        active_ns = int(window_ns * 0.95)
        candidate_run_id = f"test-profile-b{batch_size}"
        profile_nonce = f"{batch_size:032x}"
        nvtx_names = [
            f"any2rwkv-profile:{candidate_run_id}:{profile_nonce}:rank={rank}"
            for rank in range(8)
        ]
        with sqlite3.connect(sqlite) as connection:
            connection.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                "(globalPid INTEGER, deviceId INTEGER, start INTEGER, end INTEGER)"
            )
            connection.execute(
                "CREATE TABLE NVTX_EVENTS "
                "(start INTEGER, end INTEGER, globalTid INTEGER, textId INTEGER)"
            )
            connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
            connection.execute(
                "CREATE TABLE TARGET_INFO_GPU_METRICS "
                "(typeId INTEGER, sourceId INTEGER, typeName TEXT, "
                "metricId INTEGER, metricName TEXT)"
            )
            connection.execute(
                "CREATE TABLE GPU_METRICS "
                "(rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, "
                "metricId INTEGER, value REAL)"
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
                [((1000 + rank) << 24, rank, 0, active_ns) for rank in range(8)],
            )
            connection.executemany(
                "INSERT INTO StringIds VALUES (?, ?)",
                list(enumerate(nvtx_names, start=1)),
            )
            connection.executemany(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
                [
                    (0, window_ns, ((1000 + rank) << 24) + rank, rank + 1)
                    for rank in range(8)
                ],
            )
            connection.executemany(
                "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?, ?, ?)",
                [
                    (rank + 1, rank, f"GPU {rank}", 1, "SMs Active")
                    for rank in range(8)
                ],
            )
            connection.executemany(
                "INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?, ?)",
                [
                    (timestamp, timestamp, rank + 1, 1, 95.0)
                    for rank in range(8)
                    for timestamp in (
                        int(sample * window_ns / 100) for sample in range(100)
                    )
                ],
            )
        binding = {
            "plan_sha256": profile_plan_sha,
            "source_config_sha256": sha256(checkpoint),
            "source_checkpoint_sha256": source_checkpoint_sha,
            "source_checkpoint_binding": source_checkpoint_binding,
            "zero_step_checkpoint_sha256": target_checkpoint_sha,
            "zero_step_checkpoint_binding": target_checkpoint_binding,
            "dataset_content_binding": dataset_content_binding,
            "overlay_files_sha256": overlay_files_sha,
            "overlay_files": overlay_files,
            "workload_sha256": workload_sha,
            "optimizer": plan_payload["optimizer"],
            "learning_rate_schedule": "warmup-constant",
            "gradient_clip_norm": 1.0,
            "accumulation_steps": 1,
            "gradient_checkpointing": False,
            "burn_in_tokens": 128,
            "supervised_tokens": 512,
            "measured_train_rows": measured_train_rows,
            "measured_validation_rows": measured_validation_rows,
            "row_permutation_sha256": "c" * 64,
            "train_cache_manifest_sha256": train_cache_sha,
            "train_cache_content_binding": train_cache_content_binding,
            "validation_cache_manifest_sha256": validation_cache_sha,
            "validation_cache_content_binding": validation_cache_content_binding,
            "training_source_files": STRICT_SCRIPT.training_source_files(),
            "source_revision": STRICT_SCRIPT.source_revision_binding(),
            "cache_prefix_fingerprint": cache_prefix_fingerprint,
            "layer": 0,
            "profile_case": profile_case,
            "cache_has_shared_states": False,
            "source_parameter_signature_sha256": "e" * 64,
            "target_parameter_signature_sha256": "f" * 64,
            "head_size": 128,
            "world_size": 8,
        }
        candidate_tracking_config = {
            "schema_version": 1,
            "action": "performance-profile-candidate",
            "run_id": candidate_run_id,
            "plan_sha256": profile_plan_sha,
            "workload_sha256": workload_sha,
            "source_config_sha256": sha256(checkpoint),
            "source_checkpoint_sha256": source_checkpoint_sha,
            "source_checkpoint_binding": source_checkpoint_binding,
            "zero_step_checkpoint_sha256": target_checkpoint_sha,
            "zero_step_checkpoint_binding": target_checkpoint_binding,
            "dataset_content_binding": dataset_content_binding,
            "overlay_files_sha256": overlay_files_sha,
            "overlay_files": overlay_files,
            "train_cache_manifest_sha256": train_cache_sha,
            "train_cache_content_binding": train_cache_content_binding,
            "validation_cache_manifest_sha256": validation_cache_sha,
            "validation_cache_content_binding": validation_cache_content_binding,
            "training_source_files": STRICT_SCRIPT.training_source_files(),
            "source_revision": STRICT_SCRIPT.source_revision_binding(),
            "cache_prefix_fingerprint": cache_prefix_fingerprint,
            "optimizer": plan_payload["optimizer"],
            "comparison": plan_payload["experiment_comparison"],
            "micro_batch_size_per_rank": batch_size,
            "world_size": 8,
            "gradient_checkpointing": False,
            "measured_train_rows": measured_train_rows,
            "measured_validation_rows": measured_validation_rows,
            "row_permutation_sha256": "c" * 64,
            "profile_case": profile_case,
            "cache_has_shared_states": False,
            "profile_nonce": profile_nonce,
            "nvtx_range_names_by_rank": nvtx_names,
            "transition_nvtx_range_names_by_rank": [],
        }
        candidate_tracking, candidate_report = write_completed_experiment(
            candidate_dir,
            run_id=candidate_run_id,
            project="any2rwkv-tests",
            job_type="performance-profile",
            config=candidate_tracking_config,
            report_status="completed",
            launch_token=profile_nonce,
            coordinated_world_size=8,
            resume_existing=True,
        )
        source_candidate = {
            "schema_version": 1,
            "run_id": candidate_run_id,
            "per_rank_micro_batch_size": batch_size,
            "loss_token_count": loss_tokens,
            "loss_tokens_per_second": throughput,
            "includes_validation": True,
            "includes_checkpoint": True,
            "ephemeral_optimizer_steps": measured_train_rows // (8 * batch_size),
            "persistent_weight_update": False,
            "epoch_wall_seconds_by_rank": [window_seconds] * 8,
            "train_wall_seconds_by_rank": [window_seconds * 0.9] * 8,
            "validation_wall_seconds_by_rank": [window_seconds * 0.05] * 8,
            "checkpoint_wall_seconds_by_rank": [window_seconds * 0.04] * 8,
            "checkpoint_wall_fraction": 0.04,
            "peak_reserved_memory_fraction_by_rank": [0.90] * 8,
            "profiler_capture_contract": (
                "nsys-full-process-exact-nvtx-rank-transition-and-input-bindings-v5"
            ),
            "profile_nonce": profile_nonce,
            "nvtx_range_names_by_rank": nvtx_names,
            "transition_nvtx_range_names_by_rank": [],
            "cache_transition_profiled": False,
            "cache_transition_wall_seconds_by_rank": [],
            "cache_transition_peak_reserved_memory_fraction_by_rank": [],
            "validation_metrics": {"loss": 0.1},
            "profiler_artifacts": [
                {"kind": "nsys-rep", "path": str(nsys_report)},
                {"kind": "nsys-sqlite", "path": str(sqlite)},
            ],
            "cache_artifacts": [
                {
                    "kind": "train-cache-manifest",
                    "path": str(train_cache_manifest),
                    "sha256": train_cache_sha,
                },
                {
                    "kind": "validation-cache-manifest",
                    "path": str(validation_cache_manifest),
                    "sha256": validation_cache_sha,
                },
            ],
            "binding": binding,
        }
        source_candidate_path = candidate_dir / "candidate.json"
        source_candidate_path.write_text(json.dumps(source_candidate), encoding="utf-8")
        derived = STRICT_SCRIPT.derive_nsys_profile_metrics(sqlite, nvtx_names)
        export_binding_path = candidate_dir / "profiler-export-binding.json"
        export_binding_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": candidate_run_id,
                    "profile_nonce": profile_nonce,
                    "launch_token_sha256": hashlib.sha256(
                        profile_nonce.encode("ascii")
                    ).hexdigest(),
                    "candidate": {
                        "path": str(source_candidate_path),
                        "sha256": sha256(source_candidate_path),
                    },
                    "nsys_report": {
                        "path": str(nsys_report),
                        "sha256": sha256(nsys_report),
                    },
                    "nsys_sqlite": {
                        "path": str(sqlite),
                        "sha256": sha256(sqlite),
                        "export_input_file": nsys_report.name,
                    },
                    "nsys_version": "NVIDIA Nsight Systems version test",
                    "profile_command": [
                        "/usr/bin/nsys",
                        "profile",
                        str(source_candidate_path),
                    ],
                    "export_command": [
                        "/usr/bin/nsys",
                        "export",
                        f"--output={sqlite}",
                        str(nsys_report),
                    ],
                    "frozen_input_binding": {
                        key: binding[key]
                        for key in STRICT_SCRIPT.PROFILE_INPUT_BINDING_FIELDS
                    },
                    "derived_metrics": {
                        "epoch": derived,
                        "cache_transition": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        aggregated = dict(source_candidate)
        aggregated["source_candidate"] = {
            "path": str(source_candidate_path),
            "sha256": sha256(source_candidate_path),
        }
        aggregated["profiler_artifacts"] = [
            {
                "kind": "nsys-rep",
                "path": str(nsys_report),
                "sha256": sha256(nsys_report),
            },
            {
                "kind": "nsys-sqlite",
                "path": str(sqlite),
                "sha256": sha256(sqlite),
            },
            {
                "kind": "nsys-export-binding",
                "path": str(export_binding_path),
                "sha256": sha256(export_binding_path),
            },
        ]
        aggregated.update(
            {
                "eligible": True,
                "epoch_eligible": True,
                "cache_transition_eligible": True,
                "kernel_covered_wall_fraction_by_rank": derived[
                    "kernel_covered_wall_fraction_by_rank"
                ],
                "sm_active_fraction_by_gpu": derived["sm_active_fraction_by_gpu"],
                "sm_active_sample_count_by_gpu": derived[
                    "sm_active_sample_count_by_gpu"
                ],
                "sm_active_metric_source_by_gpu": derived[
                    "sm_active_metric_source_by_gpu"
                ],
                "gpu_metric_common_window_seconds": derived[
                    "gpu_metric_common_window_seconds"
                ],
                "profiler_window_seconds_by_rank": derived["window_seconds_by_rank"],
                "nsys_process_ids_by_rank": derived["process_ids_by_rank"],
                "nsys_device_ids_by_rank": derived["device_ids_by_rank"],
                "nsys_kernel_intersection_count_by_rank": derived[
                    "kernel_intersection_count_by_rank"
                ],
                "experiment_artifacts": [
                    {
                        "kind": "wandb-tracking",
                        "path": str(candidate_tracking),
                        "sha256": sha256(candidate_tracking),
                    },
                    {
                        "kind": "experiment-report",
                        "path": str(candidate_report),
                        "sha256": sha256(candidate_report),
                    },
                ],
            }
        )
        candidates.append(aggregated)
    selection_dir = tmp_path / "performance-selection"
    selection_dir.mkdir()
    current_sources = STRICT_SCRIPT.training_source_files()
    current_revision = STRICT_SCRIPT.source_revision_binding()
    candidate_artifact_bindings = []
    for row in candidates:
        candidate_path = Path(row["source_candidate"]["path"])
        files = {
            "candidate": candidate_path,
            "nsys_export_binding": candidate_path.parent
            / "profiler-export-binding.json",
            "wandb_tracking": candidate_path.parent / "experiment-tracking.json",
            "experiment_report": candidate_path.parent / "experiment-report.md",
        }
        candidate_artifact_bindings.append(
            {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in files.items()
            }
        )
    selection_config = {
        "schema_version": 1,
        "action": "performance-profile-selection",
        "run_id": "test-profile-selection",
        "plan_sha256": profile_plan_sha,
        "workload_sha256": workload_sha,
        "source_config_sha256": sha256(checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha,
        "source_checkpoint_binding": source_checkpoint_binding,
        "target_checkpoint_sha256": target_checkpoint_sha,
        "target_checkpoint_binding": target_checkpoint_binding,
        "dataset_manifest_sha256": sha256(dataset),
        "dataset_content_binding": dataset_content_binding,
        "training_source_files": current_sources,
        "source_revision": current_revision,
        "comparison": plan_payload["experiment_comparison"],
        "required_profile_cases": required_profile_cases,
        "candidate_artifacts": candidate_artifact_bindings,
    }
    selection_tracking, selection_report = write_completed_experiment(
        selection_dir,
        run_id="test-profile-selection",
        project="any2rwkv-tests",
        job_type="performance-profile",
        config=selection_config,
        report_status="accepted",
    )
    performance_profile = selection_dir / "performance-profile.json"
    performance_profile.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "accepted",
                "run_id": "test-profile-selection",
                "selection_rule": "highest-weighted-all-layer-case-end-to-end-loss-token-throughput-with-all-gates",
                "binding": {
                    "action": "distill",
                    "selection_plan_sha256": profile_plan_sha,
                    "world_size": 8,
                    "source_config_sha256": sha256(checkpoint),
                    "source_checkpoint_sha256": source_checkpoint_sha,
                    "source_checkpoint_binding": source_checkpoint_binding,
                    "target_checkpoint_sha256": STRICT_SCRIPT.checkpoint_content_binding(
                        target_config
                    )["sha256"],
                    "target_checkpoint_binding": target_checkpoint_binding,
                    "dataset_manifest_sha256": sha256(dataset),
                    "dataset_content_binding": dataset_content_binding,
                    "training_source_files": current_sources,
                    "source_revision": current_revision,
                    "candidate_artifacts": candidate_artifact_bindings,
                    "overlay_files_sha256_by_case": {
                        "embedding-output:linear_attention": overlay_files_sha
                    },
                    "workload_sha256": STRICT_SCRIPT.performance_workload_binding(
                        plan_payload
                    )["sha256"],
                    "head_size": 128,
                    "burn_in_tokens": 128,
                    "supervised_tokens": 512,
                    "accumulation_steps": 1,
                    "gradient_checkpointing": False,
                    "checkpoint_interval_micro_batches": 0,
                },
                "equal_loss_token_budget": measured_train_rows * 512,
                "required_profile_cases": required_profile_cases,
                "candidates": candidates,
                "batch_summaries": [
                    {
                        "per_rank_micro_batch_size": batch,
                        "eligible": True,
                        "weighted_epoch_wall_seconds": (
                            measured_train_rows * 512 / throughput
                        ),
                        "weighted_cache_transition_wall_seconds": 0.0,
                        "weighted_end_to_end_wall_seconds": (
                            measured_train_rows * 512 / throughput
                        ),
                        "weighted_end_to_end_loss_tokens_per_second": throughput,
                    }
                    for batch, throughput in ((2, 300.0), (4, 200.0), (8, 100.0))
                ],
                "selected_per_rank_micro_batch_size": 2,
                "experiment_artifacts": [
                    {
                        "kind": "wandb-tracking",
                        "path": str(selection_tracking),
                        "sha256": sha256(selection_tracking),
                    },
                    {
                        "kind": "experiment-report",
                        "path": str(selection_report),
                        "sha256": sha256(selection_report),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plan_payload["performance_evidence"] = {
        "status": "accepted",
        "artifact": str(performance_profile),
        "artifact_sha256": sha256(performance_profile),
        "selected_per_rank_micro_batch_size": 2,
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_payload), encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "for index in {0..7}; do printf '%s, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 12.0, 97887\\n' \"$index\"; done\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IXUSR)
    torchrun = fake_bin / "torchrun"
    torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in --nproc-per-node=8|--no-python) shift ;; *) break ;; esac\n'
        "done\n"
        'exec "$@"\n',
        encoding="utf-8",
    )
    torchrun.chmod(torchrun.stat().st_mode | stat.S_IXUSR)
    log_dir = tmp_path / "run-log"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REMOTE_RUN_LOG_DIR": str(log_dir),
        "HELICOPTER_RUN_ID": "test-run",
        "HELICOPTER_CHECKPOINT_PATH": str(checkpoint),
        "HELICOPTER_CHECKPOINT_SHA256": sha256(checkpoint),
        "HELICOPTER_DATASET_MANIFEST": str(dataset),
        "HELICOPTER_CONFIG_PATH": str(plan),
        "HELICOPTER_SEED": "20260714",
        "HELICOPTER_BATCH_JSON": '{"micro_batch_size":2,"accumulation_steps":1}',
        "HELICOPTER_PRECISION": "bf16",
        "HELICOPTER_WKV_MODE": "fp32io16",
        "HELICOPTER_RUN_PHASE": "any2rwkv-layerwise",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "RWKV_JIT_ON": "0",
        "RWKV_MY_TESTING": "x070",
        "RWKV_KERNEL": "",
        "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
        "RWKV_TRAIN_TYPE": "infctx",
        "RWKV_FLOAT_MODE": "bf16",
        "WKV_MODE": "fp32io16",
    }
    return env, log_dir


def test_source_profile_cases_require_all_qwen35_2b_layer_classes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "num_hidden_layers": 24,
                "layer_types": [
                    kind
                    for _ in range(6)
                    for kind in (
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    assert STRICT_SCRIPT.source_profile_cases(config) == {
        "embedding-output:linear_attention": {
            "profile_case_id": "embedding-output:linear_attention",
            "source_mixer_kind": "linear_attention",
            "input_boundary": "embedding-output",
            "representative_layer": 0,
            "layer_count": 1,
            "transition_count": 1,
        },
        "recurrent-prefix:linear_attention": {
            "profile_case_id": "recurrent-prefix:linear_attention",
            "source_mixer_kind": "linear_attention",
            "input_boundary": "recurrent-prefix",
            "representative_layer": 1,
            "layer_count": 17,
            "transition_count": 17,
        },
        "recurrent-prefix:full_attention": {
            "profile_case_id": "recurrent-prefix:full_attention",
            "source_mixer_kind": "full_attention",
            "input_boundary": "recurrent-prefix",
            "representative_layer": 3,
            "layer_count": 6,
            "transition_count": 5,
        },
    }


def valid_distill_command(tmp_path: Path) -> list[str]:
    return [
        "torchrun",
        "--nproc-per-node=8",
        "--no-python",
        sys.executable,
        "-m",
        "any2rwkv.cli",
        "distill",
        "--source",
        str(tmp_path / "config.json"),
        "--recipe",
        "qwen35_to_rwkv7",
        "--output",
        str(tmp_path / "output"),
        "--precision",
        "bf16",
        "--rwkv-hf-sha",
        "a" * 40,
        "--rwkv-lm-sha",
        "b" * 40,
        "--dataset-manifest",
        str(tmp_path / "data-splits.json"),
        "--training-config",
        str(tmp_path / "plan.json"),
        "--run-id",
        "test-run",
        "--allow-proxy-layers",
    ]


def test_nsys_metrics_use_only_exact_rank_nvtx_windows(tmp_path: Path) -> None:
    sqlite = tmp_path / "profile.sqlite"
    names = [f"profile-nonce-rank-{rank}" for rank in range(8)]
    with sqlite3.connect(sqlite) as connection:
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(globalPid INTEGER, deviceId INTEGER, start INTEGER, end INTEGER)"
        )
        connection.execute(
            "CREATE TABLE NVTX_EVENTS "
            "(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT)"
        )
        connection.execute(
            "CREATE TABLE TARGET_INFO_GPU_METRICS "
            "(typeId INTEGER, sourceId INTEGER, typeName TEXT, "
            "metricId INTEGER, metricName TEXT)"
        )
        connection.execute(
            "CREATE TABLE GPU_METRICS "
            "(rawTimestamp INTEGER, timestamp INTEGER, typeId INTEGER, "
            "metricId INTEGER, value REAL)"
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (1_000, 11_000, ((2000 + rank) << 24) + rank, names[rank])
                for rank in range(8)
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
            [((2000 + rank) << 24, rank, 1_000, 10_500) for rank in range(8)]
            + [((9999 << 24), 0, 0, 1_000_000_000)],
        )
        connection.executemany(
            "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?, ?, ?)",
            [
                (rank + 1, rank, f"GPU {rank}", 1, "SMs Active")
                for rank in range(8)
            ],
        )
        connection.executemany(
            "INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?, ?)",
            [
                (timestamp, timestamp, rank + 1, 1, 95.0)
                for rank in range(8)
                for timestamp in range(1_000, 11_000, 100)
            ],
        )

    metrics = STRICT_SCRIPT.derive_nsys_profile_metrics(sqlite, names)

    assert metrics["kernel_covered_wall_fraction_by_rank"] == [0.95] * 8
    assert metrics["sm_active_fraction_by_gpu"] == [0.95] * 8

    with sqlite3.connect(sqlite) as connection:
        connection.execute("UPDATE TARGET_INFO_GPU_METRICS SET sourceId = 0")
    with pytest.raises(SystemExit, match="one SMs Active source"):
        STRICT_SCRIPT.derive_nsys_profile_metrics(sqlite, names)
    assert metrics["sm_active_sample_count_by_gpu"] == [100] * 8
    assert metrics["window_seconds_by_rank"] == [0.00001] * 8
    assert metrics["device_ids_by_rank"] == list(range(8))


def test_nsys_busy_rejects_missing_bound_rank_range(tmp_path: Path) -> None:
    sqlite = tmp_path / "profile.sqlite"
    names = [f"profile-nonce-rank-{rank}" for rank in range(8)]
    with sqlite3.connect(sqlite) as connection:
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(globalPid INTEGER, deviceId INTEGER, start INTEGER, end INTEGER)"
        )
        connection.execute(
            "CREATE TABLE NVTX_EVENTS "
            "(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT)"
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (0, 10_000, ((3000 + rank) << 24) + rank, names[rank])
                for rank in range(7)
            ],
        )

    with pytest.raises(SystemExit, match="exactly one bound NVTX range"):
        STRICT_SCRIPT.derive_nsys_profile_metrics(sqlite, names)


def test_low_sm_activity_fails_even_when_kernel_coverage_is_high() -> None:
    assert not STRICT_SCRIPT.performance_candidate_passes_gates(
        kernel_coverage=[0.95] * 8,
        sm_active=[0.949, *([0.95] * 7)],
        reserved=[0.90] * 8,
        wall_seconds=[10.0] * 8,
        checkpoint_fraction=0.04,
    )


def test_kernel_bubbles_fail_the_near_full_admission_policy() -> None:
    assert not STRICT_SCRIPT.performance_candidate_passes_gates(
        kernel_coverage=[0.949, *([0.95] * 7)],
        sm_active=[0.95] * 8,
        reserved=[0.90] * 8,
        wall_seconds=[10.0] * 8,
        checkpoint_fraction=0.04,
    )


def test_profile_aggregator_recomputes_and_selects_bound_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strict_environment(tmp_path)
    selection = tmp_path / "fresh-selection"

    class FakeRun:
        id = "fresh-selection"
        project = "any2rwkv-tests"
        group = "strict-wrapper"
        job_type = "performance-profile"
        url = "https://wandb.invalid/fresh-selection"

        def log(self, _metrics):
            pass

        def finish(self, *, exit_code):
            assert exit_code == 0

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **_kwargs: FakeRun(),
            Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setenv("HELICOPTER_RUN_ID", "fresh-selection")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(AGGREGATOR_SCRIPT),
            "--plan",
            str(tmp_path / "profile-plan.json"),
            "--run-id",
            "fresh-selection",
            "--source-config",
            str(tmp_path / "config.json"),
            "--target-config",
            str(tmp_path / "output/checkpoint-zero-step/config.json"),
            "--dataset-manifest",
            str(tmp_path / "data-splits.json"),
            "--head-size",
            "128",
            *[
                value
                for batch in (2, 4, 8)
                for value in (
                    "--candidate",
                    str(tmp_path / f"candidate-{batch}/candidate.json"),
                )
            ],
            "--output",
            str(selection / "performance-profile.json"),
        ],
    )

    AGGREGATOR.main()

    result = json.loads(
        (selection / "performance-profile.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "accepted"
    assert result["selected_per_rank_micro_batch_size"] == 2
    assert min(
        result["candidates"][0]["kernel_covered_wall_fraction_by_rank"]
    ) == pytest.approx(0.95)
    assert min(result["candidates"][0]["sm_active_fraction_by_gpu"]) == pytest.approx(
        0.95
    )


def test_wrapper_records_successful_eight_gpu_layer_major_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, log_dir = strict_environment(tmp_path)
    marker = tmp_path / "completed.txt"
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(ROOT)
    original_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command[0] == "git":
            return original_run(command, *args, **kwargs)
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="".join(
                    f"{index}, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 12.0, 97887\n"
                    for index in range(8)
                ),
                stderr="",
            )
        output = tmp_path / "output"
        plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
        write_completed_experiment(
            output,
            run_id="test-run",
            project="any2rwkv-tests",
            job_type="layerwise-distillation",
            config={
                "schema_version": 1,
                "classification": plan["classification"],
                "evidence_tier": plan["evidence_tier"],
                "seed": plan["seed"],
                "optimizer": {
                    **plan["optimizer"],
                    "gradient_clip_norm": plan["gradient_clip_norm"],
                },
                "batch": {
                    "per_rank": plan["micro_batch_size"],
                    "accumulation_steps": plan["accumulation_steps"],
                    "world_size": plan["distributed_world_size"],
                    "gradient_checkpointing": plan["gradient_checkpointing"],
                },
                "comparison": plan["experiment_comparison"],
                "training_config": str((tmp_path / "plan.json").resolve()),
                "dataset_manifest": str((tmp_path / "data-splits.json").resolve()),
            },
            report_status="layerwise-local-complete",
            launch_token=kwargs["env"]["ANY2RWKV_ATTEMPT_TOKEN"],
            coordinated_world_size=8,
        )
        marker.write_text("ok", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(STRICT_SCRIPT.subprocess, "run", fake_run)
    command = [
        "torchrun",
        "--nproc-per-node=8",
        "--no-python",
        sys.executable,
        "-m",
        "any2rwkv.cli",
        "distill",
        "--source",
        str(tmp_path / "config.json"),
        "--recipe",
        "qwen35_to_rwkv7",
        "--output",
        str(tmp_path / "output"),
        "--precision",
        "bf16",
        "--rwkv-hf-sha",
        "a" * 40,
        "--rwkv-lm-sha",
        "b" * 40,
        "--dataset-manifest",
        str(tmp_path / "data-splits.json"),
        "--training-config",
        str(tmp_path / "plan.json"),
        "--run-id",
        "test-run",
        "--allow-proxy-layers",
    ]
    assert STRICT_SCRIPT.main(["--", *command]) == 0
    assert marker.read_text(encoding="utf-8") == "ok"
    metadata = json.loads((log_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["cuda_visible_devices"] == "0,1,2,3,4,5,6,7"
    assert metadata["distributed"]["world_size"] == 8
    assert metadata["precision"] == "bf16"
    assert metadata["wkv_mode"] == "fp32io16"
    assert metadata["native_training_environment"]["RWKV_HEAD_SIZE"] == "128"
    assert metadata["layer_major"]["execution_mode"] == "streamed_layer_store"
    assert metadata["layer_major"]["cache_shard_rows"] == 4
    assert set(STRICT_SCRIPT.TRAINING_SOURCE_FILES).issubset(
        metadata["training_source_files"]
    )
    assert all(len(value) == 64 for value in metadata["training_source_files"].values())


def test_wrapper_rejects_checkpoint_digest_mismatch_before_training(
    tmp_path: Path,
) -> None:
    env, log_dir = strict_environment(tmp_path)
    env["HELICOPTER_CHECKPOINT_SHA256"] = "0" * 64

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--",
            "torchrun",
            "--nproc-per-node=8",
            "--no-python",
            "python",
            "-m",
            "any2rwkv.cli",
            "distill",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "checkpoint SHA-256 mismatch" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_real_training_without_wandb_and_performance_gate(
    tmp_path: Path,
) -> None:
    env, log_dir = strict_environment(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["evidence_tier"] = "exploratory"
    plan.pop("tracking")
    plan.pop("performance_evidence")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--",
            "torchrun",
            "--nproc-per-node=8",
            "--no-python",
            "python",
            "-m",
            "any2rwkv.cli",
            "distill",
            "--source",
            str(tmp_path / "config.json"),
            "--output",
            str(tmp_path / "output"),
            "--dataset-manifest",
            str(tmp_path / "data-splits.json"),
            "--training-config",
            str(plan_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires online W&B tracking" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_missing_performance_evidence_independently(
    tmp_path: Path,
) -> None:
    env, log_dir = strict_environment(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.pop("performance_evidence")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", *valid_distill_command(tmp_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires accepted performance_evidence" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_duplicate_training_config_before_execution(
    tmp_path: Path,
) -> None:
    env, log_dir = strict_environment(tmp_path)
    duplicate = tmp_path / "duplicate-plan.json"
    duplicate.write_text((tmp_path / "plan.json").read_text(), encoding="utf-8")
    command = [
        *valid_distill_command(tmp_path),
        "--training-config",
        str(duplicate),
    ]

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", *command],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "rejects duplicate option: --training-config" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_profile_without_candidates(tmp_path: Path) -> None:
    env, log_dir = strict_environment(tmp_path)
    profile_path = tmp_path / "performance-selection" / "performance-profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidates"] = []
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["performance_evidence"]["artifact_sha256"] = sha256(profile_path)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", *valid_distill_command(tmp_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "candidate-set binding" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_nonfinite_profile_measurement(tmp_path: Path) -> None:
    env, log_dir = strict_environment(tmp_path)
    profile_path = tmp_path / "performance-selection" / "performance-profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidates"][0]["sm_active_fraction_by_gpu"][0] = float("nan")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["performance_evidence"]["artifact_sha256"] = sha256(profile_path)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", *valid_distill_command(tmp_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "non-finite JSON constant is forbidden" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_selection_row_that_changes_original_candidate(
    tmp_path: Path,
) -> None:
    env, log_dir = strict_environment(tmp_path)
    profile_path = tmp_path / "performance-selection" / "performance-profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["candidates"][0]["peak_reserved_memory_fraction_by_rank"] = [0.91] * 8
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["performance_evidence"]["artifact_sha256"] = sha256(profile_path)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", *valid_distill_command(tmp_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "changed original measured field" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_fixture_plan_on_eight_gpu_entrypoint(tmp_path: Path) -> None:
    env, log_dir = strict_environment(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["evidence_tier"] = "fixture"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", *valid_distill_command(tmp_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refuses fixture evidence" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_non_native_training_precision(tmp_path: Path) -> None:
    env, _ = strict_environment(tmp_path)
    env["HELICOPTER_WKV_MODE"] = "fp16"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", "torchrun", "--nproc-per-node=8", "true"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "HELICOPTER_WKV_MODE=fp32io16" in result.stderr


def test_wrapper_rejects_exploratory_layer_limit_for_p1_before_training(
    tmp_path: Path,
) -> None:
    env, log_dir = strict_environment(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["evidence_tier"] = "p1"
    plan["exploratory_layer_limit"] = 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--",
            "torchrun",
            "--nproc-per-node=8",
            "--no-python",
            "true",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "only allowed for exploratory evidence" in result.stderr
    assert not (log_dir / "metadata.json").exists()


@pytest.mark.parametrize(
    ("evidence_tier", "limit"),
    (("p1", None), ("scale", True), ("fixture", 1), ("exploratory", 24)),
)
def test_wrapper_rejects_invalid_exploratory_layer_limit_by_key_presence(
    tmp_path: Path, evidence_tier: str, limit: object
) -> None:
    env, log_dir = strict_environment(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["evidence_tier"] = evidence_tier
    plan["exploratory_layer_limit"] = limit
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--",
            "torchrun",
            "--nproc-per-node=8",
            "--no-python",
            "true",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "only allowed for exploratory evidence" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_exploratory_limit_on_corrective_before_parent_lookup(
    tmp_path: Path,
) -> None:
    env, log_dir = strict_environment(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["evidence_tier"] = "exploratory"
    plan["exploratory_layer_limit"] = 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    command = valid_distill_command(tmp_path)
    command[command.index("distill")] = "corrective"
    command.extend(
        [
            "--parent-run",
            str(tmp_path / "missing-parent"),
            "--parent-checkpoint-sha256",
            "0" * 64,
        ]
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", *command],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "corrective command forbids exploratory_layer_limit" in result.stderr
    assert not (log_dir / "metadata.json").exists()


def test_wrapper_rejects_head_size_that_conflicts_with_source(tmp_path: Path) -> None:
    env, _ = strict_environment(tmp_path)
    env["RWKV_HEAD_SIZE"] = "64"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--", "torchrun", "--nproc-per-node=8", "true"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "RWKV_HEAD_SIZE conflicts with source-compatible geometry" in result.stderr


def test_wrapper_rejects_non_distill_payload(tmp_path: Path) -> None:
    env, _ = strict_environment(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--",
            "torchrun",
            "--nproc-per-node=8",
            "--no-python",
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "any2rwkv.cli distill" in result.stderr


def test_throughput_evidence_rejects_a_different_zero_step_geometry(
    tmp_path: Path,
) -> None:
    source_config = tmp_path / "source-config.json"
    source_config.write_text('{"head_dim":128}\n', encoding="utf-8")
    zero_step_config = tmp_path / "zero-step-config.json"
    zero_step_config.write_text('{"head_size":128}\n', encoding="utf-8")
    profile = tmp_path / "profile.json"
    candidates = [
        {
            "per_rank_micro_batch_size": batch,
            "loss_token_count": 1024,
            "loss_tokens_per_second": throughput,
            "eligible": True,
        }
        for batch, throughput in ((2, 30.0), (4, 20.0), (8, 10.0))
    ]
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "selection_rule": (
                    "highest-slowest-rank-loss-token-throughput-within-memory-limit"
                ),
                "binding": {
                    "world_size": 8,
                    "head_size": 64,
                    "source_config_sha256": sha256(source_config),
                    "zero_step_config_sha256": "0" * 64,
                },
                "candidates": candidates,
                "equal_loss_token_budget": 1024,
                "selected_per_rank_micro_batch_size": 2,
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "micro_batch_size": 2,
        "throughput_evidence": {
            "status": "measured-exploratory",
            "artifact": str(profile),
            "artifact_sha256": sha256(profile),
            "selected_per_rank_micro_batch_size": 2,
        },
    }

    with pytest.raises(SystemExit, match="does not bind"):
        STRICT_SCRIPT.validate_throughput_evidence(
            plan,
            source_config=source_config,
            target_config=zero_step_config,
            head_size=128,
        )


def test_performance_workload_hash_excludes_only_batch_selection_fields() -> None:
    base = {
        "schema_version": 3,
        "micro_batch_size": 2,
        "optimizer": {"name": "adamw", "learning_rate": 1e-6},
        "notes": ["candidate"],
        "performance_evidence": {"status": "pending"},
    }
    selected = {
        **base,
        "micro_batch_size": 16,
        "notes": ["selected"],
        "performance_evidence": {"status": "accepted"},
    }
    changed_optimizer = {
        **selected,
        "optimizer": {"name": "adamw", "learning_rate": 2e-6},
    }

    assert (
        STRICT_SCRIPT.performance_workload_binding(base)["sha256"]
        == STRICT_SCRIPT.performance_workload_binding(selected)["sha256"]
    )
    assert (
        STRICT_SCRIPT.performance_workload_binding(selected)["sha256"]
        != STRICT_SCRIPT.performance_workload_binding(changed_optimizer)["sha256"]
    )


def test_wrapper_hash_binds_corrective_parent_checkpoint(tmp_path: Path) -> None:
    strict_environment(tmp_path)
    parent = tmp_path / "parent"
    checkpoint = parent / "checkpoint-global-corrective"
    checkpoint.mkdir(parents=True)
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"frozen-parent-weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.attn.r_k": shard.name}}),
        encoding="utf-8",
    )
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "any2rwkv": {
                    "recurrence": "native_rwkv7",
                    "fully_recurrent_proxy": True,
                    "mixer_overlay_fingerprint": "a" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    source_sha = sha256(tmp_path / "config.json")
    (parent / "metadata.json").write_text(
        json.dumps(
            {
                "source": {"files": {"config.json": source_sha}},
                "recipe": {"id": "qwen35_to_rwkv7"},
                "precision": "bf16",
            }
        ),
        encoding="utf-8",
    )
    files = {
        name: sha256(checkpoint / name)
        for name in (
            "config.json",
            "model.safetensors.index.json",
            shard.name,
        )
    }
    parent_sha = hashlib.sha256(
        json.dumps(
            {"files": files, "mixer_fingerprint": "a" * 64},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    parsed = STRICT_SCRIPT.ParsedTrainingCommand(
        action="corrective",
        source=(tmp_path / "config.json").resolve(),
        output=(tmp_path / "output").resolve(),
        dataset_manifest=(tmp_path / "data-splits.json").resolve(),
        training_config=(tmp_path / "plan.json").resolve(),
        recipe="qwen35_to_rwkv7",
        precision="bf16",
        run_id="test-run",
        parent_run=parent.resolve(),
        parent_checkpoint_sha256=parent_sha,
    )
    binding = STRICT_SCRIPT.validate_distill_command(
        parsed,
        checkpoint=(tmp_path / "config.json").resolve(),
        dataset_manifest=(tmp_path / "data-splits.json").resolve(),
        config=(tmp_path / "plan.json").resolve(),
        expected_run_id="test-run",
    )
    assert binding is not None
    assert binding["path"] == str(checkpoint)
    assert binding["files"][shard.name] == sha256(shard)

    shard.write_bytes(b"mutated-parent-weights")
    with pytest.raises(SystemExit, match="parent checkpoint SHA-256 mismatch"):
        STRICT_SCRIPT.validate_distill_command(
            parsed,
            checkpoint=(tmp_path / "config.json").resolve(),
            dataset_manifest=(tmp_path / "data-splits.json").resolve(),
            config=(tmp_path / "plan.json").resolve(),
            expected_run_id="test-run",
        )
