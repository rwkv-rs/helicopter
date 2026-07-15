import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_global_batch_quality.py"
SPEC = importlib.util.spec_from_file_location("compare_global_batch_quality", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare)


def _metadata(run_id, batch_size, values, times):
    metric = "val-core/dapo/reward/acc/mean@1"
    return {
        "run_id": run_id,
        "batch": {"train_batch_size": batch_size},
        "validation": {
            "accuracy_metrics": [metric],
            "points": [
                {
                    "metrics": {metric: value},
                    "cumulative_wall_seconds": wall_seconds,
                }
                for value, wall_seconds in zip(values, times, strict=True)
            ],
        },
    }


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
