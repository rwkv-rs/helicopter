import hashlib
import importlib.util
import io
import json
import signal
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "strict_train_run.py"
SPEC = importlib.util.spec_from_file_location("strict_train_run", SCRIPT)
strict_train_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(strict_train_run)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_metadata_uses_remote_revision_manifest_without_git(tmp_path):
    revisions = {
        "product_commit": "a" * 40,
        "submodules": {"src/train/verl-rwkv": "b" * 40},
    }

    metadata = strict_train_run.source_metadata(tmp_path, revisions)

    assert metadata == {
        "commit": "a" * 40,
        "branch": None,
        "dirty": None,
        "status": [],
        "patch_sha256": None,
        "submodules": [f"{'b' * 40} src/train/verl-rwkv"],
        "git_metadata_available": False,
        "source": "synchronized revision manifest",
    }
    with pytest.raises(RuntimeError, match="revision manifest is missing"):
        strict_train_run.source_metadata(tmp_path, None)


def test_vllm_capacity_is_extracted_per_replica(tmp_path):
    command_log = tmp_path / "command.log"
    command_log.write_text(
        "GPU KV cache size: 12,345 tokens\n"
        "Maximum concurrency for 10,240 tokens per request: 1.20x\n"
        "GPU KV cache size: 12,345 tokens\n"
        "Maximum concurrency for 10,240 tokens per request: 1.20x\n"
    )

    assert strict_train_run.extract_vllm_capacity(command_log) == {
        "replica_observation_count": 2,
        "capacity_mode": "kv-cache",
        "kv_cache_applicable": True,
        "per_replica": [],
        "structured_errors": [],
        "gpu_kv_cache_tokens": [12345, 12345],
        "maximum_concurrency": [1.2, 1.2],
    }


def test_vllm_capacity_requires_structured_observation_for_each_recurrent_replica(tmp_path):
    command_log = tmp_path / "command.log"
    command_log.write_text(
        "Keeping chunked prefill enabled for no-KVCache causal recurrent model.\n",
        encoding="utf-8",
    )
    topology_path = tmp_path / "rollout_topology.json"
    capacity = {
        "capacity_mode": "recurrent-state-no-kv-cache",
        "kv_cache_applicable": False,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.6,
    }
    topology_path.write_text(
        json.dumps(
            {
                "deployments": [
                    {"replica_rank": replica_rank, "capacity": capacity}
                    for replica_rank in range(8)
                ]
            }
        ),
        encoding="utf-8",
    )

    assert strict_train_run.extract_vllm_capacity(command_log, topology_path) == {
        "replica_observation_count": 8,
        "capacity_mode": "recurrent-state-no-kv-cache",
        "kv_cache_applicable": False,
        "per_replica": [
            {"replica_rank": replica_rank, **capacity}
            for replica_rank in range(8)
        ],
        "structured_errors": [],
        "gpu_kv_cache_tokens": [],
        "maximum_concurrency": [],
    }

    topology_path.write_text(json.dumps({"deployments": [{"replica_rank": 0, "capacity": capacity}]}))
    assert strict_train_run.extract_vllm_capacity(command_log, topology_path)["replica_observation_count"] == 0


def test_formal_rollout_capacity_must_match_resolved_rwkv_config():
    capacity = {
        "capacity_mode": "recurrent-state-no-kv-cache",
        "kv_cache_applicable": False,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.6,
    }
    observations = {
        "per_replica": [{"replica_rank": rank, **capacity} for rank in range(8)],
        "structured_errors": [],
    }
    config = {
        "takeoff": {
            "grpo": {
                "rollout_max_num_seqs": 16,
                "rollout_max_num_batched_tokens": 2048,
                "rollout_gpu_memory_utilization": 0.6,
            }
        }
    }

    strict_train_run.verify_rollout_capacity_observations(observations, config)

    observations["per_replica"][7]["capacity_mode"] = "kv-cache"
    with pytest.raises(RuntimeError, match="does not match"):
        strict_train_run.verify_rollout_capacity_observations(observations, config)


def test_tee_keeps_persisting_after_live_stdout_breaks(monkeypatch):
    class BrokenBuffer:
        def write(self, _chunk):
            raise BrokenPipeError("closed")

        def flush(self):
            raise AssertionError("flush must not run after write fails")

    class BrokenStdout:
        buffer = BrokenBuffer()

    command_log = io.BytesIO()
    output_errors = []
    monkeypatch.setattr(strict_train_run.sys, "stdout", BrokenStdout())

    strict_train_run.tee_child_output(io.BytesIO(b"complete child output"), command_log, output_errors)

    assert command_log.getvalue() == b"complete child output"
    assert output_errors == []


def test_verify_dataset_manifest_checks_every_file(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps({"files": [{"path": str(first), "sha256": _sha(first)}, {"path": str(second), "sha256": _sha(second)}]})
    )

    verified = strict_train_run.verify_dataset_manifest(manifest)
    assert [item["sha256"] for item in verified["files"]] == [_sha(first), _sha(second)]

    second.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        strict_train_run.verify_dataset_manifest(manifest)


def test_visible_devices_requires_all_eight_in_stable_order():
    assert strict_train_run.parse_visible_devices("0,1,2,3,4,5,6,7") == tuple(range(8))
    for invalid in ("0", "0,1,2,3,4,5,6", "1,0,2,3,4,5,6,7", "all"):
        with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
            strict_train_run.parse_visible_devices(invalid)


def test_run_phase_fails_closed_and_run_dir_is_absolute_within_workspace(tmp_path):
    assert strict_train_run.validate_run_phase("correctness") == "correctness"
    assert strict_train_run.validate_run_phase("nsys") == "nsys"
    with pytest.raises(RuntimeError, match="basline"):
        strict_train_run.validate_run_phase("basline")

    assert strict_train_run.resolve_run_dir(tmp_path, ".helicopter-dev/runs/1") == (
        tmp_path / ".helicopter-dev/runs/1"
    ).resolve()
    with pytest.raises(RuntimeError, match="within the workspace root"):
        strict_train_run.resolve_run_dir(tmp_path, "../outside")

    run_dir = (tmp_path / ".helicopter-dev/runs/1").resolve()
    child_env = strict_train_run.strict_child_environment({}, run_dir)
    assert child_env == {
        "REMOTE_RUN_LOG_DIR": str(run_dir),
        "VERL_FILE_LOGGER_PATH": str(run_dir / "metrics.jsonl"),
        "VERL_POLICY_IDENTITY_LOG_PATH": str(run_dir / "policy_identity.jsonl"),
    }


def test_stop_process_escalates_from_term_to_kill():
    class StubbornProcess:
        returncode = None

        def __init__(self):
            self.signals = []
            self.waits = 0

        def poll(self):
            return self.returncode

        def send_signal(self, value):
            self.signals.append(value)

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise strict_train_run.subprocess.TimeoutExpired("child", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

        def kill(self):
            self.signals.append(signal.SIGKILL)

    process = StubbornProcess()
    strict_train_run.stop_process(process)
    assert process.signals == [signal.SIGTERM, signal.SIGKILL]


def test_topology_contract_and_observation_must_match(tmp_path):
    expected = strict_train_run.verify_topology_contract(
        {"trainer_gpus": 8, "rollout_replicas": 8, "rollout_tp": 1, "rollout_pp": 1, "rollout_internal_dp": 1}
    )
    observed_path = tmp_path / "rollout_topology.json"
    observed_path.write_text(
        json.dumps(
            {
                "replicas": 8,
                "gpus_per_replica": 1,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "endpoints": [f"http://127.0.0.1:{30000 + index}" for index in range(8)],
                "deployments": [
                    {
                        "node_id": "node-0",
                        "actor_id": f"actor-{index}",
                        "http_port": 30000 + index,
                        "master_port": 31000 + index,
                        "dp_rpc_port": 32000 + index,
                        "dp_master_port": 33000 + index,
                        "cuda_visible_devices": [str(index)],
                    }
                    for index in range(8)
                ],
            }
        )
    )
    assert strict_train_run.verify_observed_topology(observed_path, expected)["replicas"] == 8

    observed = json.loads(observed_path.read_text())
    observed["endpoints"][-1] = observed["endpoints"][0]
    observed_path.write_text(json.dumps(observed))
    with pytest.raises(RuntimeError, match="duplicate"):
        strict_train_run.verify_observed_topology(observed_path, expected)

    observed["endpoints"][-1] = "http://127.0.0.1:39999"
    observed["deployments"][-1]["master_port"] = observed["deployments"][0][
        "dp_rpc_port"
    ]
    observed_path.write_text(json.dumps(observed))
    with pytest.raises(RuntimeError, match="duplicate runtime ports"):
        strict_train_run.verify_observed_topology(observed_path, expected)

    with pytest.raises(RuntimeError, match="8 independent TP1"):
        strict_train_run.verify_topology_contract(
            {"trainer_gpus": 8, "rollout_replicas": 4, "rollout_tp": 2, "rollout_pp": 1, "rollout_internal_dp": 1}
        )


def test_declared_contract_must_match_resolved_training_config():
    config = {
        "takeoff": {
            "grpo": {
                "seed": 42,
                "wkv_mode": "fp32io16",
                "train_batch_size": 8,
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size": 1,
                "ppo_max_token_len_per_gpu": 8192,
                "rollout_n": 2,
                "max_prompt_length": 256,
                "max_response_length": 128,
                "infctx": True,
                "chunk_ctx": 2048,
                "trainer_n_gpus_per_node": 8,
                "rollout_tensor_parallel_size": 1,
                "rollout_pipeline_parallel_size": 1,
                "rollout_data_parallel_size": 1,
            }
        }
    }
    batch = {
        "train_batch_size": 8,
        "ppo_mini_batch_size": 8,
        "ppo_micro_batch_size": 1,
        "ppo_max_token_len_per_gpu": 8192,
        "rollout_n": 2,
        "max_prompt_length": 256,
        "max_response_length": 128,
    }
    topology = {
        "trainer_gpus": 8,
        "rollout_replicas": 8,
        "rollout_tp": 1,
        "rollout_pp": 1,
        "rollout_internal_dp": 1,
    }

    strict_train_run.verify_declared_contract(
        config,
        seed="42",
        batch=batch,
        topology=topology,
        precision="fp32io16",
        wkv_mode="fp32io16",
    )
    with pytest.raises(RuntimeError, match="declared batch"):
        strict_train_run.verify_declared_contract(
            config,
            seed="42",
            batch={**batch, "train_batch_size": 56},
            topology=topology,
            precision="fp32io16",
            wkv_mode="fp32io16",
        )

    with pytest.raises(RuntimeError, match="actor token budget 8192"):
        strict_train_run.verify_declared_contract(
            {
                "takeoff": {
                    "grpo": {
                        **config["takeoff"]["grpo"],
                        "ppo_max_token_len_per_gpu": 16384,
                    }
                }
            },
            seed="42",
            batch={**batch, "ppo_max_token_len_per_gpu": 16384},
            topology=topology,
            precision="fp32io16",
            wkv_mode="fp32io16",
        )


def test_correctness_metrics_require_on_policy_same_version_correction_and_update(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    aligned = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": 0.001,
        "training/rollout_actor_probs_pearson_corr": 0.999,
        "training/off_policy/trajectory_spans/max": 1,
        "training/off_policy/trajectory_staleness/max": 0,
        "training/off_policy/trajectory_staleness_worst/max": 0,
        "training/on_policy/version_count": 1,
        "training/on_policy/weight_digest_count": 1,
        "training/on_policy/sampling_config_count": 1,
        "training/on_policy/runtime_identity_count": 1,
        "rollout_corr/rollout_is_mean": 0.9,
        "rollout_corr/rollout_is_min": 0.1,
        "rollout_corr/rollout_is_max": 2.0,
        "rollout_corr/rollout_is_std": 0.4,
        "rollout_corr/rollout_is_eff_sample_size": 0.7,
        "rollout_corr/rollout_is_ratio_fraction_high": 0.1,
        "rollout_corr/rollout_is_ratio_fraction_low": 0.1,
        "actor/loss": 0.25,
        "actor/grad_norm": 1.5,
        "actor/optimizer_steps": 1,
    }
    metrics.write_text("".join(json.dumps({"step": step, "data": aligned}) + "\n" for step in (1, 2)))
    assert strict_train_run.verify_correctness_metrics(metrics, expected_rounds=2)["rounds"] == 2

    cross_runtime_mismatch = {**aligned, "training/rollout_probs_diff_max": 0.5}
    metrics.write_text(json.dumps({"step": 1, "data": cross_runtime_mismatch}) + "\n")
    assert strict_train_run.verify_correctness_metrics(metrics, expected_rounds=1)[
        "cross_runtime_diagnostics"
    ][0]["max_probability_diff"] == 0.5

    unhealthy_correction = {**aligned, "rollout_corr/rollout_is_eff_sample_size": 0.2}
    metrics.write_text(json.dumps({"step": 1, "data": unhealthy_correction}) + "\n")
    with pytest.raises(RuntimeError, match="effective sample size"):
        strict_train_run.verify_correctness_metrics(metrics, expected_rounds=1)


def test_performance_metrics_exclude_warmup_and_recompute_throughput(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    records = []
    for step in range(1, 8):
        records.append(
            {
                "step": step,
                "data": {
                    "training/global_step": step,
                    "training/rollout_probs_diff_valid": 1,
                    "training/rollout_probs_diff_max": 0.001,
                    "training/rollout_actor_probs_pearson_corr": 0.999,
                    "training/off_policy/trajectory_spans/max": 1,
                    "training/off_policy/trajectory_staleness/max": 0,
                    "training/off_policy/trajectory_staleness_worst/max": 0,
                    "training/on_policy/version_count": 1,
                    "training/on_policy/weight_digest_count": 1,
                    "training/on_policy/sampling_config_count": 1,
                    "training/on_policy/runtime_identity_count": 1,
                    "rollout_corr/rollout_is_mean": 0.9,
                    "rollout_corr/rollout_is_min": 0.1,
                    "rollout_corr/rollout_is_max": 2.0,
                    "rollout_corr/rollout_is_std": 0.4,
                    "rollout_corr/rollout_is_eff_sample_size": 0.7,
                    "rollout_corr/rollout_is_ratio_fraction_high": 0.1,
                    "rollout_corr/rollout_is_ratio_fraction_low": 0.1,
                    "actor/loss": 0.25,
                    "actor/grad_norm": 1.5,
                    "actor/optimizer_steps": 1,
                    "training/actual_samples": 8,
                    "training/actual_prompt_tokens": 80,
                    "training/actual_response_tokens": 160,
                    "training/actual_total_tokens": 240,
                    "training/actual_policy_loss_tokens": 40,
                    "timing/rollout_seconds": float(step),
                    "timing/train_seconds": float(step * 2),
                    "timing/full_step_seconds": float(step * 4),
                    "timing_s/weight_publish": float(step),
                    "critic/rewards/mean": float(step),
                    "actor/entropy": 0.5,
                    "actor/actual_micro_batches": 4,
                    "training/rollout_group_completion_seconds/p95": 3.0,
                    "training/rollout_group_completion_seconds/max": 4.0,
                    "training/rollout_group_tail_seconds": 2.0,
                    "training/rollout_preemptions": 0,
                    "training/rollout_effective_concurrency": 8.0,
                    "training/rollout_train_overlap_seconds": 0.0,
                },
            }
        )
    metrics.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = strict_train_run.verify_performance_metrics(metrics, expected_rounds=7)

    assert summary["warmup_steps"] == [1, 2]
    assert summary["timed_steps"] == [3, 4, 5, 6, 7]
    assert summary["actual_total_tokens"] == 1200
    assert summary["actual_policy_loss_tokens"] == 200
    assert summary["throughput"]["train_tokens_per_second"] == pytest.approx(4)
    assert summary["throughput"]["full_step_tokens_per_second"] == pytest.approx(12)
    assert summary["quality"]["critic/rewards/mean"] == pytest.approx(5)
    assert summary["detailed_stage_timing"]["weight_publish"]["mean_seconds"] == 5
    assert summary["capacity_observations"]["rollout_preemptions_available"] is True


def test_performance_metrics_reject_insufficient_timed_steps(tmp_path):
    with pytest.raises(RuntimeError, match="2 warmup and 5 timed"):
        strict_train_run.verify_performance_metrics(tmp_path / "missing.jsonl", expected_rounds=6)


def test_validation_curve_tracks_equal_sample_and_wall_clock_axis(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [
        {
            "step": 0,
            "elapsed_seconds": 3.0,
            "data": {
                "val-core/dapo/reward/acc/mean@1": 0.10,
                "timing_s/testing": 3.0,
            },
        },
        {
            "step": 1,
            "elapsed_seconds": 9.0,
            "data": {
                "training/global_step": 1,
                "training/actual_samples": 8,
                "training/actual_prompt_tokens": 16,
                "training/actual_response_tokens": 24,
                "training/actual_total_tokens": 40,
                "critic/rewards/mean": 0.2,
                "actor/entropy": 0.3,
                "actor/grad_norm": 1.0,
                "actor/optimizer_steps": 1.0,
                "timing_s/step": 2.0,
                "val-core/dapo/reward/acc/mean@1": 0.20,
                "timing_s/testing": 4.0,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    summary = strict_train_run.verify_validation_curve(path, expected_rounds=1)

    assert summary["accuracy_metrics"] == ["val-core/dapo/reward/acc/mean@1"]
    assert summary["points"][0]["cumulative_samples"] == 0
    assert summary["points"][0]["cumulative_wall_seconds"] == 3.0
    assert summary["points"][1]["cumulative_samples"] == 8
    assert summary["points"][1]["cumulative_wall_seconds"] == 9.0


def test_validation_curve_requires_initial_timing(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 0,
                "elapsed_seconds": 1.0,
                "data": {"val-core/dapo/reward/acc/mean@1": 0.1},
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="timing_s/testing"):
        strict_train_run.verify_validation_curve(path, expected_rounds=0)


@pytest.mark.parametrize("testing_seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_validation_curve_rejects_invalid_testing_time(tmp_path, testing_seconds):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 0,
                "elapsed_seconds": 1.0,
                "data": {
                    "val-core/dapo/reward/acc/mean@1": 0.1,
                    "timing_s/testing": testing_seconds,
                },
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="positive finite testing time"):
        strict_train_run.verify_validation_curve(path, expected_rounds=0)


@pytest.mark.parametrize(("batch_size", "rounds", "test_freq"), [(56, 14, 2), (112, 7, 1)])
def test_global_batch_quality_schedule_is_equal_sampled(batch_size, rounds, test_freq):
    config = {
        "takeoff": {
            "grpo": {
                "train_batch_size": batch_size,
                "ppo_mini_batch_size": batch_size,
                "rollout_n": 8,
                "test_freq": test_freq,
            }
        }
    }
    validation = {
        "points": [
            {"cumulative_samples": cumulative_samples}
            for cumulative_samples in range(0, 6272 + 1, 896)
        ]
    }

    strict_train_run.verify_global_batch_quality_schedule(
        config, validation, expected_rounds=rounds
    )


@pytest.mark.parametrize(
    ("config_name", "batch_size", "rounds", "test_freq"),
    [
        ("strict-quality-b56.toml", 56, 14, 2),
        ("strict-quality-b112.toml", 112, 7, 1),
    ],
)
def test_checked_in_quality_configs_share_equal_sample_contract(
    config_name, batch_size, rounds, test_freq
):
    config_path = Path(__file__).parents[1] / "configs" / config_name
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    takeoff = config["takeoff"]["grpo"]

    assert takeoff["train_batch_size"] == batch_size
    assert takeoff["ppo_mini_batch_size"] == batch_size
    assert takeoff["total_training_steps"] == rounds
    assert takeoff["test_freq"] == test_freq
    assert takeoff["val_before_train"] is True
    assert takeoff["rollout_n"] == 8
    assert takeoff["ppo_max_token_len_per_gpu"] == 8192
    assert takeoff["rollout_max_num_seqs"] == 64
    assert takeoff["rollout_max_num_batched_tokens"] == 8192


def test_all_checked_in_strict_configs_fix_state_passing_actor_capacity():
    config_root = Path(__file__).parents[1] / "configs"
    strict_configs = sorted(config_root.glob("strict-*.toml"))

    assert strict_configs
    for config_path in strict_configs:
        takeoff = tomllib.loads(config_path.read_text(encoding="utf-8"))["takeoff"][
            "grpo"
        ]
        assert takeoff["ppo_max_token_len_per_gpu"] == 8192, config_path.name
        assert takeoff["infctx"] is True, config_path.name
        assert takeoff["chunk_ctx"] == 2048, config_path.name


@pytest.mark.parametrize("config_name", ["strict-quality-b56.toml", "strict-quality-b112.toml"])
def test_quality_configs_satisfy_exact_remote_declared_contract(config_name):
    config_path = Path(__file__).parents[1] / "configs" / config_name
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    takeoff = config["takeoff"]["grpo"]
    batch_fields = (
        "train_batch_size",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_max_token_len_per_gpu",
        "rollout_n",
        "max_prompt_length",
        "max_response_length",
    )

    strict_train_run.verify_declared_contract(
        config,
        seed="42",
        batch={field: takeoff[field] for field in batch_fields},
        topology={
            "trainer_gpus": 8,
            "rollout_replicas": 8,
            "rollout_tp": 1,
            "rollout_pp": 1,
            "rollout_internal_dp": 1,
        },
        precision="fp32io16",
        wkv_mode="fp32io16",
    )


def test_quality_configs_differ_only_by_equal_sample_batch_schedule():
    root = Path(__file__).parents[1]
    baseline = tomllib.loads((root / "configs" / "strict-quality-b56.toml").read_text())[
        "takeoff"
    ]["grpo"]
    candidate = tomllib.loads(
        (root / "configs" / "strict-quality-b112.toml").read_text()
    )["takeoff"]["grpo"]

    allowed_changes = {
        "experiment_name",
        "train_batch_size",
        "ppo_mini_batch_size",
        "total_training_steps",
        "test_freq",
    }
    assert {
        key for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)
    } == allowed_changes
    assert baseline["train_batch_size"] * baseline["total_training_steps"] == 56 * 14
    assert candidate["train_batch_size"] * candidate["total_training_steps"] == 112 * 7
    assert baseline["test_freq"] * baseline["train_batch_size"] == 112
    assert candidate["test_freq"] * candidate["train_batch_size"] == 112


def test_checked_in_global_batch_probe_changes_only_algorithm_batch_capacity():
    root = Path(__file__).parents[1]
    baseline = tomllib.loads((root / "configs" / "strict-baseline.toml").read_text())[
        "takeoff"
    ]["grpo"]
    candidate = tomllib.loads(
        (root / "configs" / "strict-global-batch-112.toml").read_text()
    )["takeoff"]["grpo"]

    allowed_changes = {"experiment_name", "train_batch_size", "ppo_mini_batch_size"}
    assert {
        key for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)
    } == allowed_changes
    assert candidate["train_batch_size"] == candidate["ppo_mini_batch_size"] == 112


def test_global_batch_quality_schedule_rejects_sparse_validation():
    config = {
        "takeoff": {
            "grpo": {
                "train_batch_size": 112,
                "ppo_mini_batch_size": 112,
                "rollout_n": 8,
                "test_freq": 1,
            }
        }
    }
    validation = {"points": [{"cumulative_samples": 0}, {"cumulative_samples": 6272}]}

    with pytest.raises(RuntimeError, match="every 896"):
        strict_train_run.verify_global_batch_quality_schedule(
            config, validation, expected_rounds=7
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records[3].__setitem__("step", records[2]["step"]), "unique"),
        (lambda records: records[3]["data"].__setitem__("timing/train_seconds", float("nan")), "finite"),
        (lambda records: records[3]["data"].__setitem__("training/actual_total_tokens", 999), "total=prompt"),
        (lambda records: records[3]["data"].pop("actor/entropy"), "missing"),
    ],
)
def test_performance_metrics_fail_closed_on_malformed_evidence(tmp_path, mutation, message):
    records = [
        {
            "step": step,
            "data": {
                "training/global_step": step,
                "training/rollout_probs_diff_valid": 1,
                "training/rollout_probs_diff_max": 0.001,
                "training/rollout_actor_probs_pearson_corr": 0.999,
                "training/off_policy/trajectory_spans/max": 1,
                "training/off_policy/trajectory_staleness/max": 0,
                "training/off_policy/trajectory_staleness_worst/max": 0,
                "training/on_policy/version_count": 1,
                "training/on_policy/weight_digest_count": 1,
                "training/on_policy/sampling_config_count": 1,
                "training/on_policy/runtime_identity_count": 1,
                "rollout_corr/rollout_is_mean": 0.9,
                "rollout_corr/rollout_is_min": 0.1,
                "rollout_corr/rollout_is_max": 2.0,
                "rollout_corr/rollout_is_std": 0.4,
                "rollout_corr/rollout_is_eff_sample_size": 0.7,
                "rollout_corr/rollout_is_ratio_fraction_high": 0.1,
                "rollout_corr/rollout_is_ratio_fraction_low": 0.1,
                "actor/loss": 0.25,
                "actor/grad_norm": 1.5,
                "actor/optimizer_steps": 1,
                "training/actual_samples": 1,
                "training/actual_prompt_tokens": 4,
                "training/actual_response_tokens": 2,
                "training/actual_total_tokens": 6,
                "training/actual_policy_loss_tokens": 1,
                "timing/rollout_seconds": 1.0,
                "timing/train_seconds": 1.0,
                "timing/full_step_seconds": 2.0,
                "critic/rewards/mean": 1.0,
                "actor/entropy": 1.0,
                "actor/actual_micro_batches": 2,
                "training/rollout_group_completion_seconds/p95": 1.0,
                "training/rollout_group_completion_seconds/max": 1.0,
                "training/rollout_group_tail_seconds": 0.0,
                "training/rollout_preemptions": 0,
                "training/rollout_effective_concurrency": 1.0,
                "training/rollout_train_overlap_seconds": 0.0,
            },
        }
        for step in range(1, 8)
    ]
    mutation(records)
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(RuntimeError, match=message):
        strict_train_run.verify_performance_metrics(path, expected_rounds=7)


def test_gpu_sample_summary_reports_idle_peak_and_missing_intervals(tmp_path):
    samples = tmp_path / "gpu.csv"
    rows = [
        "sequence,wall_time_utc,monotonic_ns,gpu_index,utilization_gpu_percent,power_watts,memory_used_mib"
    ]
    for gpu in range(8):
        rows.extend(
            [
                f"0,2026-07-15T00:00:01+00:00,0,{gpu},0,100,{1000 + gpu}",
                f"1,2026-07-15T00:00:00+00:00,100000000,{gpu},50,200,{2000 + gpu}",
                f"2,2026-07-14T23:59:59+00:00,500000000,{gpu},90,300,{1500 + gpu}",
            ]
        )
    samples.write_text("\n".join(rows) + "\n")

    summary = strict_train_run.parse_gpu_samples(samples)
    assert summary["missing_gpus"] == []
    assert summary["per_gpu"]["0"]["idle_fraction"] == pytest.approx(1 / 3)
    assert summary["per_gpu"]["0"]["peak_memory_used_mib"] == 2000
    assert summary["per_gpu"]["0"]["missing_interval_count"] == 1
    assert summary["per_gpu"]["0"]["max_sample_gap_ms"] == 400
    assert summary["per_gpu"]["0"]["missing_intervals"] == [
        {"after_sequence": 1, "before_sequence": 2, "gap_ms": 400}
    ]
    assert summary["per_gpu"]["0"]["sample_coverage"] == pytest.approx(0.5)
    assert summary["incomplete_sequences"] == {}
    with pytest.raises(RuntimeError, match="coverage fell below"):
        strict_train_run.verify_gpu_telemetry(summary)


def test_gpu_sample_summary_rejects_incomplete_sequence(tmp_path):
    samples = tmp_path / "gpu.csv"
    rows = [
        "sequence,wall_time_utc,monotonic_ns,gpu_index,utilization_gpu_percent,power_watts,memory_used_mib"
    ]
    rows.extend(f"0,now,0,{gpu},1,2,3" for gpu in range(8))
    rows.extend(f"1,now,100000000,{gpu},1,2,3" for gpu in range(7))
    samples.write_text("\n".join(rows) + "\n")

    summary = strict_train_run.parse_gpu_samples(samples)

    assert summary["incomplete_sequences"] == {"1": [7]}
    with pytest.raises(RuntimeError, match="incomplete 8-GPU sequences"):
        strict_train_run.verify_gpu_telemetry(summary)


def test_nsys_phase_requires_nonempty_trace_and_stage_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="non-empty"):
        strict_train_run.verify_nsys_trace(tmp_path)

    report = tmp_path / "profile.nsys-rep"
    report.write_bytes(b"trace")
    (tmp_path / "nsys_trace_manifest.json").write_text(
        json.dumps(
            {
                "profiled_steps": [2],
                "nvtx_stage_markers": ["gen", "reward", "update_actor", "step"],
            }
        )
    )
    assert strict_train_run.verify_nsys_trace(tmp_path)["formal_performance"] is False


def test_nvml_sampler_writes_one_complete_sequence_and_shuts_down(tmp_path, monkeypatch):
    state = {"shutdown": False}

    class OneRoundEvent:
        def __init__(self):
            self.stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, _timeout=None):
            self.stopped = True
            return True

    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: state.__setitem__("shutdown", True),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=10 + handle),
        nvmlDeviceGetPowerUsage=lambda handle: 100000 + handle,
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=(1024 * 1024) * (1000 + handle)),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setattr(strict_train_run.threading, "Event", OneRoundEvent)
    samples = tmp_path / "gpu.csv"
    ready = tmp_path / "ready"

    assert strict_train_run.sample_gpus(samples, ready) == 0

    summary = strict_train_run.parse_gpu_samples(samples)
    assert ready.read_text() == "ready\n"
    assert summary["missing_gpus"] == []
    assert summary["incomplete_sequences"] == {}
    assert summary["per_gpu"]["7"]["peak_memory_used_mib"] == 1007
    assert state["shutdown"] is True


def test_telemetry_affinity_reserves_one_allowed_cpu(monkeypatch):
    monkeypatch.setattr(strict_train_run.os, "sched_getaffinity", lambda _pid: {9, 3, 7})

    sampler_cpu, child_cpus = strict_train_run.telemetry_cpu_affinity_plan()

    assert sampler_cpu == 9
    assert child_cpus == (3, 7)


def test_telemetry_affinity_rejects_single_cpu(monkeypatch):
    monkeypatch.setattr(strict_train_run.os, "sched_getaffinity", lambda _pid: {3})

    with pytest.raises(RuntimeError, match="at least two"):
        strict_train_run.telemetry_cpu_affinity_plan()


def test_sampler_heartbeat_uses_monotonic_gap(tmp_path):
    heartbeat = tmp_path / "heartbeat.csv"
    heartbeat.write_text("1000000000\n1100000000\n1450000000\n")

    assert strict_train_run.parse_sampler_heartbeat(heartbeat) == {
        "samples": 3,
        "max_gap_ms": 350.0,
    }


def test_policy_identity_log_requires_monotonic_publication_and_matching_training(tmp_path):
    identity_log = tmp_path / "policy_identity.jsonl"
    common = {
        "sampling_config_digest": "sampling",
        "runtime_identity": "runtime",
    }
    records = [
        {"event": "publish_initial", "global_steps": 0, "policy_version": 0, "weight_digest": "digest-0", **common},
        {"event": "train_begin", "global_steps": 1, "policy_version": 0, "weight_digest": "digest-0", "effective_sampling_digest": "effective", **common},
        {"event": "publish", "global_steps": 1, "policy_version": 1, "weight_digest": "digest-1", **common},
        {"event": "train_begin", "global_steps": 2, "policy_version": 1, "weight_digest": "digest-1", "effective_sampling_digest": "effective", **common},
        {"event": "publish", "global_steps": 2, "policy_version": 2, "weight_digest": "digest-2", **common},
    ]
    identity_log.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = strict_train_run.verify_policy_identity_log(identity_log, expected_rounds=2)
    assert summary["train_versions"] == [0, 1]
    assert summary["last_published_version"] == 2

    records[-1]["policy_version"] = 3
    identity_log.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(RuntimeError, match="advance"):
        strict_train_run.verify_policy_identity_log(identity_log, expected_rounds=2)

    records[0]["runtime_identity"] = ""
    identity_log.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(RuntimeError, match="non-empty"):
        strict_train_run.verify_policy_identity_log(identity_log, expected_rounds=2)


@pytest.mark.parametrize(
    "records",
    [
        [{"event": "publish_initial", "global_steps": 0, "policy_version": 0}],
        [
            {"event": "publish_initial", "global_steps": 0, "policy_version": 0},
            {"event": "train_begin", "global_steps": 1, "policy_version": 0},
            {"event": "train_begin", "global_steps": 1, "policy_version": 0},
        ],
    ],
)
def test_policy_identity_log_rejects_incomplete_or_repeated_rounds(tmp_path, records):
    identity_log = tmp_path / "policy_identity.jsonl"
    common = {
        "weight_digest": "digest",
        "sampling_config_digest": "sampling",
        "runtime_identity": "runtime",
    }
    identity_log.write_text("".join(json.dumps({**common, **record}) + "\n" for record in records))
    with pytest.raises(RuntimeError):
        strict_train_run.verify_policy_identity_log(identity_log, expected_rounds=1)
