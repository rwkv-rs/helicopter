import copy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_global_batch_quality.py"
SPEC = importlib.util.spec_from_file_location("compare_global_batch_quality", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare)

STRICT_SCRIPT = Path(__file__).parents[1] / "scripts" / "strict_train_run.py"
STRICT_SPEC = importlib.util.spec_from_file_location("strict_train_run_for_quality", STRICT_SCRIPT)
strict_train_run = importlib.util.module_from_spec(STRICT_SPEC)
assert STRICT_SPEC.loader is not None
STRICT_SPEC.loader.exec_module(strict_train_run)


def _metadata(run_id, batch_size, values, times):
    metric = "val-core/dapo/reward/acc/mean@1"
    if len(values) == len(times) == 3:
        values = [values[0], *([values[1]] * 6), values[2]]
        times = [
            times[0],
            *(times[1] + (times[2] - times[1]) * offset / 6 for offset in range(6)),
            times[2],
        ]
    return {
        "run_id": run_id,
        "source": {"commit": "a" * 40},
        "source_revisions": {"product_commit": "a" * 40, "submodules": {}},
        "checkpoint": {"sha256": "b" * 64},
        "dataset_manifest": {"sha256": "c" * 64},
        "seed": 42,
        "precision": "fp32io16",
        "wkv_mode": "fp32io16",
        "gemm_policy": "fp32-state-fp16-io_no-fp16-accumulation",
        "topology": {"trainer_gpus": 8, "rollout_replicas": 8, "rollout_tp": 1},
        "rollout_capacity": {"rollout_max_num_seqs": 64},
        "measurement_contract": {"minimum_sample_coverage": 0.99},
        "environment": {"host": "rwkv-pro6000x8", "torch": "2.11.0+cu130"},
        "batch": {
            "train_batch_size": batch_size,
            "ppo_mini_batch_size": batch_size,
            "ppo_max_token_len_per_gpu": 8192,
            "rollout_n": 8,
        },
        "validation": {
            "accuracy_metrics": [metric],
            "common_metrics": [metric],
            "points": [
                {
                    "metrics": {metric: value},
                    "cumulative_wall_seconds": wall_seconds,
                    "cumulative_samples": index * 896,
                }
                for index, (value, wall_seconds) in enumerate(
                    zip(values, times, strict=True)
                )
            ],
        },
    }


def _successful_artifact(batch_size):
    artifact = _metadata("run", batch_size, [0.10, 0.20, 0.30], [10.0, 30.0, 60.0])
    artifact.update(
        status="done",
        exit_code=0,
        run_phase="global-batch-quality",
        config={"path": f"/remote/configs/strict-quality-b{batch_size}.toml"},
    )
    artifact["validation"]["final_cumulative_samples"] = 6272
    return artifact


def test_load_metadata_accepts_only_canonical_quality_contract(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(_successful_artifact(56)))

    loaded = compare.load_metadata(path, expected_batch_size=56)

    assert loaded["batch"]["train_batch_size"] == 56


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(status="failed"), "not successful"),
        (lambda payload: payload.update(run_phase="performance"), "wrong run phase"),
        (lambda payload: payload["config"].update(path="strict-baseline.toml"), "must use"),
        (lambda payload: payload.update(precision="fp16"), "fp32io16"),
        (lambda payload: payload.update(gemm_policy="fp16"), "GEMM policy"),
    ],
)
def test_load_metadata_rejects_noncanonical_quality_contract(tmp_path, mutation, message):
    path = tmp_path / "metadata.json"
    artifact = _successful_artifact(56)
    mutation(artifact)
    path.write_text(json.dumps(artifact))

    with pytest.raises(RuntimeError, match=message):
        compare.load_metadata(path, expected_batch_size=56)


def test_comparison_accepts_noninferior_faster_candidate():
    baseline = _metadata("b", 56, [0.10, 0.20, 0.30], [10.0, 30.0, 60.0])
    candidate = _metadata("c", 112, [0.10, 0.28, 0.29], [10.0, 20.0, 40.0])

    result = compare.compare_quality(baseline, candidate)

    assert result["accepted"] is True
    assert result["decision"] == "accept-batch-112"
    assert result["quality_gate_passed"] is True
    assert result["wall_clock_gate_passed"] is True


def test_comparison_retains_baseline_when_candidate_is_slower():
    baseline = _metadata("b", 56, [0.10, 0.20, 0.30], [10.0, 30.0, 60.0])
    candidate = _metadata("c", 112, [0.10, 0.28, 0.29], [10.0, 70.0, 80.0])

    result = compare.compare_quality(baseline, candidate)

    assert result["accepted"] is False
    assert result["decision"] == "retain-batch-56"
    assert result["quality_gate_passed"] is True
    assert result["wall_clock_gate_passed"] is False


def test_comparison_retains_baseline_when_quality_regresses():
    baseline = _metadata("b", 56, [0.10, 0.20, 0.30], [10.0, 30.0, 60.0])
    candidate = _metadata("c", 112, [0.10, 0.20, 0.26], [10.0, 20.0, 40.0])

    result = compare.compare_quality(baseline, candidate)

    assert result["accepted"] is False
    assert result["quality_gate_passed"] is False


def test_wall_clock_gate_ignores_initial_validation_crossing():
    baseline = _metadata("b", 56, [0.30, 0.31, 0.31], [10.0, 30.0, 100.0])
    candidate = _metadata("c", 112, [0.30, 0.29, 0.29], [10.0, 70.0, 80.0])

    result = compare.compare_quality(baseline, candidate)

    assert result["quality_gate_passed"] is True
    assert result["wall_clock_gate_passed"] is False
    assert result["baseline"]["time_to_target_seconds"] == 20.0
    assert result["candidate"]["time_to_target_seconds"] == 60.0
    assert result["accepted"] is False


def test_comparison_rejects_incomparable_artifacts():
    baseline = _metadata("b", 56, [0.10, 0.20, 0.30], [10.0, 30.0, 60.0])
    mutations = (
        ("checkpoint.sha256", lambda payload: payload["checkpoint"].update(sha256="d" * 64)),
        ("seed", lambda payload: payload.update(seed=7)),
        ("precision", lambda payload: payload.update(precision="fp16")),
        ("topology", lambda payload: payload["topology"].update(rollout_tp=2)),
        (
            "source_revisions",
            lambda payload: payload.update(
                source_revisions={"product_commit": "e" * 40, "submodules": {}}
            ),
        ),
    )

    for field, mutate in mutations:
        candidate = _metadata("c", 112, [0.10, 0.28, 0.29], [10.0, 20.0, 40.0])
        candidate = copy.deepcopy(candidate)
        mutate(candidate)
        with pytest.raises(RuntimeError, match=field.replace(".", r"\.")):
            compare.compare_quality(baseline, candidate)


def _write_quality_metrics(path, *, batch_size, rounds, test_freq, final_accuracy):
    metric = "val-core/dapo/reward/acc/mean@1"
    records = [
        {
            "step": 0,
            "elapsed_seconds": 5.0,
            "data": {metric: 0.10, "timing_s/testing": 5.0},
        }
    ]
    samples_per_round = batch_size * 8
    for step in range(1, rounds + 1):
        elapsed_seconds = 5.0 + step * (20.0 if batch_size == 56 else 30.0)
        data = {
            "training/global_step": step,
            "training/actual_samples": samples_per_round,
            "training/actual_prompt_tokens": samples_per_round * 64,
            "training/actual_response_tokens": samples_per_round * 128,
            "training/actual_total_tokens": samples_per_round * 192,
            "critic/rewards/mean": 0.1 + step / 100,
            "actor/entropy": 1.0,
            "actor/grad_norm": 2.0,
            "actor/optimizer_steps": 1.0,
        }
        if step % test_freq == 0 or step == rounds:
            data[metric] = 0.10 + (final_accuracy - 0.10) * step / rounds
            data["timing_s/testing"] = 4.0
        records.append({"step": step, "elapsed_seconds": elapsed_seconds, "data": data})
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_equal_sample_artifacts_flow_from_metrics_to_quality_decision(tmp_path):
    baseline_path = tmp_path / "b56.jsonl"
    candidate_path = tmp_path / "b112.jsonl"
    _write_quality_metrics(
        baseline_path, batch_size=56, rounds=14, test_freq=2, final_accuracy=0.30
    )
    _write_quality_metrics(
        candidate_path, batch_size=112, rounds=7, test_freq=1, final_accuracy=0.29
    )

    baseline_validation = strict_train_run.verify_validation_curve(
        baseline_path, expected_rounds=14
    )
    candidate_validation = strict_train_run.verify_validation_curve(
        candidate_path, expected_rounds=7
    )
    strict_train_run.verify_global_batch_quality_schedule(
        {
            "takeoff": {
                "grpo": {
                    "train_batch_size": 56,
                    "ppo_mini_batch_size": 56,
                    "rollout_n": 8,
                    "test_freq": 2,
                }
            }
        },
        baseline_validation,
        expected_rounds=14,
    )
    strict_train_run.verify_global_batch_quality_schedule(
        {
            "takeoff": {
                "grpo": {
                    "train_batch_size": 112,
                    "ppo_mini_batch_size": 112,
                    "rollout_n": 8,
                    "test_freq": 1,
                }
            }
        },
        candidate_validation,
        expected_rounds=7,
    )

    baseline_metadata = _metadata("b56", 56, [0.1], [1.0])
    candidate_metadata = _metadata("b112", 112, [0.1], [1.0])
    baseline_metadata["validation"] = baseline_validation
    candidate_metadata["validation"] = candidate_validation
    result = compare.compare_quality(baseline_metadata, candidate_metadata)

    assert baseline_validation["final_cumulative_samples"] == 6272
    assert candidate_validation["final_cumulative_samples"] == 6272
    assert len(baseline_validation["training_trajectory"]) == 14
    assert len(candidate_validation["training_trajectory"]) == 7
    assert result["accepted"] is True
