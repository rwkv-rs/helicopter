from types import SimpleNamespace

import pytest

from lighteval.metrics.metrics_sample import ExactMatches

from helicopter_lighteval.evaluation import (
    EvaluationRequest,
    _accounting,
    _eligibility,
    _scorer_revision,
    _terminal_payload,
)
from helicopter_lighteval.vllm_rwkv import AttestationDecision


def test_max_samples_are_selection_not_dataset_rejection() -> None:
    task = SimpleNamespace(
        dataset={"test": ["row-1", "row-2", "row-3"]},
        config=SimpleNamespace(evaluation_splits=("test",)),
    )
    accounting = _accounting(task, sample_count=1)
    assert accounting["source_rows"] == 3
    assert accounting["dataset_accepted"] == 3
    assert accounting["dataset_rejected"] == 0
    assert accounting["selected"] == 1


def test_callable_object_scorer_revision_is_process_stable() -> None:
    def make_task():
        return SimpleNamespace(
            config=SimpleNamespace(scorer=ExactMatches(strip_strings=True), version=0),
            metrics=[
                SimpleNamespace(
                    metric_name="extractive_match",
                    sample_level_fn=ExactMatches(strip_strings=True),
                )
            ],
        )

    assert _scorer_revision(make_task()) == _scorer_revision(make_task())


def _actual_metric_a(doc, response):
    return 1.0


def _actual_metric_b(doc, response):
    return 0.0


def _inspect_scorer_a(doc, response):
    return 1.0


def _inspect_scorer_b(doc, response):
    return 0.0


def _metric_task(*, sample_level_fn, config_scorer):
    return SimpleNamespace(
        config=SimpleNamespace(scorer=config_scorer, version=0),
        metrics=[
            SimpleNamespace(
                metric_name="extractive_match",
                sample_level_fn=sample_level_fn,
                higher_is_better=True,
                category="GENERATIVE",
                batched_compute=False,
                corpus_level_fn=None,
            )
        ],
    )


def test_scorer_revision_tracks_executed_primary_metric() -> None:
    unchanged_inspect_scorer = _scorer_revision(
        _metric_task(sample_level_fn=_actual_metric_a, config_scorer=_inspect_scorer_a)
    )
    changed_metric = _scorer_revision(
        _metric_task(sample_level_fn=_actual_metric_b, config_scorer=_inspect_scorer_a)
    )
    assert unchanged_inspect_scorer != changed_metric


def test_scorer_revision_ignores_unexecuted_inspect_scorer() -> None:
    first = _scorer_revision(
        _metric_task(sample_level_fn=_actual_metric_a, config_scorer=_inspect_scorer_a)
    )
    second = _scorer_revision(
        _metric_task(sample_level_fn=_actual_metric_a, config_scorer=_inspect_scorer_b)
    )
    assert first == second


def test_provider_mismatch_is_proxy_while_partial_verified_runs_are_sanity() -> None:
    request = EvaluationRequest(
        model="model",
        task="lighteval/math/gsm8k@0",
        endpoint_url="http://server/v1",
        max_samples=1,
    )
    assert (
        _eligibility(
            request,
            AttestationDecision(official=False, mismatches=("missing_attestation",)),
        )
        == "proxy"
    )
    assert (
        _eligibility(request, AttestationDecision(official=True, mismatches=()))
        == "sanity"
    )


def test_evaluation_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_concurrent_requests must be positive"):
        EvaluationRequest(
            model="model",
            task="lighteval/math/gsm8k@0",
            endpoint_url="http://server/v1",
            max_concurrent_requests=0,
        )


def test_multi_metric_task_projects_only_signed_primary() -> None:
    task = SimpleNamespace(
        metrics=(
            SimpleNamespace(metric_name="pass@k:k=1"),
            SimpleNamespace(metric_name="avg@n:n=1"),
        )
    )
    payload = _terminal_payload(
        task=task,
        samples=[{"generation": {"truncated": False}}],
        result_dict={
            "results": {
                "aime24|0": {
                    "pass@k:k=1": 1.0,
                    "avg@n:n=1": 1.0,
                    "pass@k:k=1_stderr": 0.0,
                }
            }
        },
    )
    assert payload["metrics"] == {"pass@k:k=1": 1.0}
    assert payload["native_metrics"] == {"pass@k:k=1": 1.0, "avg@n:n=1": 1.0}


def test_terminal_payload_ignores_lighteval_all_summary() -> None:
    task = SimpleNamespace(
        full_name="gsm8k|0",
        metrics=(SimpleNamespace(metric_name="extractive_match"),),
    )
    payload = _terminal_payload(
        task=task,
        samples=[{"generation": {"truncated": False}}],
        result_dict={
            "results": {
                "gsm8k|0": {"extractive_match": 1.0},
                "all": {"extractive_match": 0.5},
            }
        },
    )
    assert payload["metrics"] == {"extractive_match": 1.0}


def test_grouped_instruction_metric_projects_prompt_strict_primary() -> None:
    task = SimpleNamespace(
        metrics=(
            SimpleNamespace(
                metric_name=[
                    "prompt_level_strict_acc",
                    "inst_level_strict_acc",
                    "prompt_level_loose_acc",
                    "inst_level_loose_acc",
                ]
            ),
        )
    )
    payload = _terminal_payload(
        task=task,
        samples=[{"generation": {"truncated": True}}],
        result_dict={
            "results": {
                "ifeval|0": {
                    "prompt_level_strict_acc": 0.5,
                    "inst_level_strict_acc": 0.75,
                    "prompt_level_loose_acc": 0.6,
                    "inst_level_loose_acc": 0.8,
                }
            }
        },
    )
    assert payload["metrics"] == {"prompt_level_strict_acc": 0.5}
    assert payload["truncated_samples"] == 1


def test_multi_metric_task_without_signed_primary_fails_closed() -> None:
    task = SimpleNamespace(metrics=(SimpleNamespace(metric_name="avg@n:n=1"),))
    with pytest.raises(ValueError, match="exactly one signed primary metric"):
        _terminal_payload(
            task=task,
            samples=[{"generation": {"truncated": False}}],
            result_dict={"results": {"task|0": {"avg@n:n=1": 1.0}}},
        )
