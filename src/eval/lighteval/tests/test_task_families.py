from __future__ import annotations

import pytest

from helicopter_lighteval.evaluation import (
    DEFAULT_GENERATION_LIMIT,
    _SIGNED_BINARY_METRICS,
    _SUPPORTED_GENERATIVE_METRICS,
    _ensure_generation_size,
    _light_eval_task_configs,
    _override_generation_size,
    _primary_metric_name,
    UnsupportedTaskError,
    _select_benchmark,
)


@pytest.mark.parametrize(
    ("canonical", "lighteval_task"),
    [
        ("lighteval/knowledge/mmlu-abstract-algebra@0", "mmlu:abstract_algebra|0"),
        ("lighteval/knowledge/mmlu-pro@0", "mmlu_pro|0"),
        ("lighteval/knowledge/gpqa-diamond@1", "gpqa:diamond|0"),
        ("lighteval/knowledge/gpqa-main@0", "gpqa:main|0"),
        ("lighteval/knowledge/gpqa-extended@0", "gpqa:extended|0"),
    ],
)
def test_generation_only_knowledge_names_select_pinned_lighteval_tasks(
    canonical: str, lighteval_task: str
) -> None:
    selection = _select_benchmark(canonical)
    assert selection.lighteval_task == lighteval_task
    assert selection.info.field.value == "knowledge"


def test_all_pinned_mmlu_subjects_resolve_through_knowledge_family() -> None:
    configs = _light_eval_task_configs()
    subjects = sorted(
        name.removeprefix("mmlu:") for name in configs if name.startswith("mmlu:")
    )
    assert len(subjects) == 57
    for subject in subjects:
        canonical = f"lighteval/knowledge/mmlu-{subject.replace('_', '-')}@0"
        selection = _select_benchmark(canonical)
        assert selection.lighteval_task == f"mmlu:{subject}|0"


@pytest.mark.parametrize(
    "canonical",
    [
        "lighteval/knowledge/mmlu_redux_2:abstract_algebra@0",
        "lighteval/knowledge/gpqa-diamond@0",
        "lighteval/knowledge/not-a-real-task@0",
    ],
)
def test_registered_but_unsupported_or_unknown_knowledge_tasks_fail_closed(
    canonical: str,
) -> None:
    with pytest.raises(UnsupportedTaskError):
        _select_benchmark(canonical)


@pytest.mark.parametrize(
    "canonical",
    [
        "lighteval/knowledge/hle@0",
        "lighteval/knowledge/simpleqa@0",
        "lighteval/knowledge/mt-bench@0",
        "lighteval/knowledge/long-horizon-execution@0",
        "lighteval/knowledge/lcb:codegeneration@0",
        "lighteval/math/lcb:codegeneration@0",
    ],
)
def test_benchmark_names_cannot_cross_coding_or_judge_boundaries(
    canonical: str,
) -> None:
    with pytest.raises(UnsupportedTaskError):
        _select_benchmark(canonical)


def test_supported_knowledge_metrics_are_signed_binary_metrics() -> None:
    configs = _light_eval_task_configs()
    for upstream in (
        "mmlu:abstract_algebra",
        "mmlu_pro",
        "gpqa:diamond",
        "gpqa:main",
        "gpqa:extended",
    ):
        assert {
            str(metric.metric_name) for metric in configs[upstream].metrics
        } <= _SIGNED_BINARY_METRICS


@pytest.mark.parametrize(
    ("canonical", "lighteval_task"),
    [
        ("lighteval/math/aime24@2", "aime24|0"),
        ("lighteval/math/aime25@2", "aime25|0"),
        ("lighteval/math/asdiv@0", "asdiv|0"),
        ("lighteval/math/gsm-plus@0", "gsm_plus|0"),
        (
            "lighteval/math/olympiadbench@1",
            "olympiad_bench:OE_TO_maths_en_COMP|0",
        ),
    ],
)
def test_low_cost_maths_names_select_pinned_lighteval_tasks(
    canonical: str, lighteval_task: str
) -> None:
    selection = _select_benchmark(canonical)
    assert selection.lighteval_task == lighteval_task
    assert selection.info.field.value == "maths"


@pytest.mark.parametrize(
    ("canonical", "lighteval_task"),
    [
        ("lighteval/instruction-following/ifeval@0.1", "ifeval|0"),
        ("lighteval/instruction-following/ifbench-test@0.1", "ifbench_test|0"),
    ],
)
def test_single_turn_instruction_names_select_pinned_lighteval_tasks(
    canonical: str, lighteval_task: str
) -> None:
    selection = _select_benchmark(canonical)
    assert selection.lighteval_task == lighteval_task
    assert selection.task_version == "0.1"
    assert selection.info.field.value == "instruction_following"


def test_multiturn_instruction_benchmark_is_not_fabricated() -> None:
    with pytest.raises(UnsupportedTaskError):
        _select_benchmark("lighteval/instruction-following/ifbench-multiturn@0.1")


def test_low_cost_tasks_have_one_signed_primary_metric() -> None:
    configs = _light_eval_task_configs()
    for upstream in (
        "aime24",
        "aime25",
        "asdiv",
        "gsm_plus",
        "olympiad_bench:OE_TO_maths_en_COMP",
        "ifeval",
        "ifbench_test",
    ):
        config = configs[upstream]
        assert _primary_metric_name(config) in _SIGNED_BINARY_METRICS
        for metric in config.metrics:
            names = (
                metric.metric_name
                if isinstance(metric.metric_name, list)
                else [metric.metric_name]
            )
            assert set(map(str, names)) <= _SUPPORTED_GENERATIVE_METRICS


def test_missing_generation_size_gets_a_stable_default() -> None:
    task = type("Task", (), {})()
    task.generation_size = None
    task.config = type("Config", (), {"generation_size": None})()
    task._docs = [type("Doc", (), {"generation_size": None})()]
    _ensure_generation_size(task)
    assert task.generation_size == DEFAULT_GENERATION_LIMIT
    assert task.config.generation_size == DEFAULT_GENERATION_LIMIT
    assert task._docs[0].generation_size == DEFAULT_GENERATION_LIMIT
    _override_generation_size(task, 128)
    assert task.generation_size == 128
    assert task.config.generation_size == 128
    assert task._docs[0].generation_size == 128


@pytest.mark.parametrize(
    ("canonical", "lighteval_task"),
    [
        ("lighteval/coding/livecodebench@0", "lcb:codegeneration|0"),
        ("lighteval/coding/livecodebench-v6@0", "lcb:codegeneration_v6|0"),
        (
            "lighteval/coding/livecodebench-release-latest@0",
            "lcb:codegeneration_release_latest|0",
        ),
    ],
)
def test_livecodebench_names_select_tasks_without_enabling_execution(
    canonical: str, lighteval_task: str
) -> None:
    selection = _select_benchmark(canonical)
    assert selection.lighteval_task == lighteval_task
    assert selection.info.field.value == "coding"


@pytest.mark.parametrize("family", ["function-calling", "agent"])
def test_absent_function_and_agent_families_are_not_fabricated(family: str) -> None:
    with pytest.raises(UnsupportedTaskError, match="has no"):
        _select_benchmark(f"lighteval/{family}/default@0")
