from __future__ import annotations

from dataclasses import replace

import pytest

from lighteval_runner.context import (
    ContextBudgetPolicy,
    ContextDocument,
    ContextSection,
    apply_context_budget,
)
from lighteval_runner.execution import Eligibility, ModelIdentity, ProviderIdentity
from lighteval_runner.provider.attestation import (
    Comparability,
    ProviderAttestation,
    validate_attestation,
)
from lighteval_runner.registry import (
    CANONICAL_TASKS,
    MetricContract,
    MetricDirection,
    TaskContract,
    get_task_definition,
)


def test_context_budget_drops_only_whole_task_owned_sections():
    calls = []
    document = ContextDocument(
        (
            ContextSection("question", "user", "12345", 100),
            ContextSection("fewshot", "assistant", "67890", 0, can_drop=True),
        )
    )
    budgeted = apply_context_budget(
        document,
        ContextBudgetPolicy(5),
        lambda sections: (
            calls.append(sections) or sum(len(section.content) for section in sections)
        ),
    )

    assert budgeted.dropped_sections == ("fewshot",)
    assert budgeted.prompt_token_count == 5
    assert calls[-1] == (ContextSection("question", "user", "12345", 100),)


def attestation():
    return ProviderAttestation(
        ModelIdentity("model", "checkpoint", "tokenizer", "chat"),
        ProviderIdentity("server", "fp32io16", "fp16", "fp32", "launch"),
        ("token_ids", "stop_reason"),
    )


def test_official_attestation_must_match_exactly():
    expected = attestation()
    assert (
        validate_attestation(
            expected, expected, official=True, allow_non_comparable=False
        ).comparability
        is Comparability.OFFICIAL
    )
    with pytest.raises(ValueError):
        validate_attestation(
            expected,
            replace(expected, capabilities=("token_ids",)),
            official=True,
            allow_non_comparable=True,
        )
    with pytest.raises(ValueError):
        validate_attestation(expected, None, official=True, allow_non_comparable=True)


def test_proxy_may_explicitly_degrade_to_non_comparable():
    decision = validate_attestation(
        attestation(), None, official=False, allow_non_comparable=True
    )
    assert decision.comparability is Comparability.NON_COMPARABLE
    assert decision.mismatches == ("missing_attestation",)


def test_metric_contract_never_infers_correctness_from_metric_name_or_positive_value():
    metric = MetricContract("exact_match", MetricDirection.HIGHER_IS_BETTER, 0.0, 1.0)
    assert metric.correctness({"exact_match": 1.0}) is None
    binary = replace(metric, binary_correctness_metric="is_correct")
    assert binary.correctness({"exact_match": 0.2, "is_correct": 0.0}) is False
    with pytest.raises(ValueError):
        binary.correctness({"exact_match": 1.0})


def test_metric_aggregation_uses_only_the_signed_rule():
    mean = MetricContract("score", MetricDirection.HIGHER_IS_BETTER, 0.0, 10.0)
    total = replace(mean, aggregation_rule="sum")
    assert mean.aggregate([1.0, 3.0]) == 2.0
    assert total.aggregate([1.0, 3.0]) == 4.0
    with pytest.raises(ValueError, match="unsupported aggregation"):
        replace(mean, aggregation_rule="metric-name-heuristic")


def test_only_official_task_contract_can_publish_officially():
    metric = MetricContract("score", MetricDirection.HIGHER_IS_BETTER, 0.0, 1.0)
    official = TaskContract(
        identity="task",
        dataset_revision="dataset",
        prompt_revision="prompt",
        scorer_revision="scorer",
        generation_contract="generation",
        dataset_owner="upstream",
        prompt_owner="upstream",
        scorer_owner="upstream",
        harness_revision=None,
        eligibility=Eligibility.OFFICIAL,
        metric=metric,
    )
    proxy = replace(official, eligibility=Eligibility.PROXY)
    assert official.can_publish_official is True
    assert proxy.can_publish_official is False


@pytest.mark.parametrize("identity", sorted(CANONICAL_TASKS))
def test_canonical_registry_loads_signed_runtime(identity):
    registered = CANONICAL_TASKS[identity]
    definition = get_task_definition(
        identity, allow_proxy=registered.contract.eligibility is Eligibility.PROXY
    )
    assert definition.load_runtime() is not None


@pytest.mark.parametrize(
    "identity",
    [
        "helicopter-proxy/function-calling/exact-json@1",
        "helicopter-proxy/coding/python-stdio@1",
    ],
)
def test_proxy_registry_requires_explicit_opt_in(identity):
    with pytest.raises(ValueError, match="explicit opt-in"):
        get_task_definition(identity)
    assert (
        get_task_definition(identity, allow_proxy=True).contract.eligibility
        is Eligibility.PROXY
    )


def test_unknown_or_unproven_task_is_not_silently_registered():
    with pytest.raises(ValueError, match="unknown canonical"):
        get_task_definition("official-looking/swe-bench")


def test_mmlu_context_exactly_matches_lighteval_api_message_semantics():
    definition = get_task_definition("lighteval/knowledge/mmlu-abstract-algebra@0")
    prepared = definition.load_runtime().prepare(
        {
            "question": "Which option?",
            "subject": "abstract_algebra",
            "choices": ["zero", "one", "two", "three"],
            "answer": 1,
        }
    )
    doc = prepared.scoring_state
    assert [
        (section.role, section.content) for section in prepared.context.sections
    ] == [("user", doc.query)]
