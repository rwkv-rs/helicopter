from __future__ import annotations

from dataclasses import replace

import pytest

from lighteval_runner.execution import (
    Eligibility,
    GenerationLimitResolution,
    ModelIdentity,
    ProviderIdentity,
    RunStatus,
    SampleAccounting,
    TaskIdentity,
    identity_digest,
    validate_run_transition,
)


def identities():
    task = TaskIdentity(
        "math",
        "gsm8k",
        "1",
        "d1",
        "test",
        0,
        "p1",
        "s1",
        "g1",
        "cot",
        "A",
        Eligibility.OFFICIAL,
    )
    model = ModelIdentity("model", "c1", "t1", "chat1")
    provider = ProviderIdentity("server1", "fp32io16", "fp16", "fp32", "launch1")
    return task, model, provider


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("task", "dataset_digest", "d2"),
        ("task", "prompt_revision", "p2"),
        ("task", "scorer_revision", "s2"),
        ("task", "cot_mode", "none"),
        ("task", "repair_strategy", "C"),
        ("task", "fewshot", 5),
        ("task", "generation_contract", "g2"),
        ("task", "eligibility", Eligibility.PROXY),
        ("model", "served_name", "other-model"),
        ("model", "checkpoint_sha256", "c2"),
        ("model", "tokenizer_revision", "t2"),
        ("model", "chat_template_revision", "chat2"),
        ("provider", "server_revision", "server2"),
        ("provider", "precision", "bf16"),
        ("provider", "wkv_mode", "fp16"),
        ("provider", "gemm_policy", "fp16"),
        ("provider", "launch_contract", "launch2"),
    ],
)
def test_every_signed_identity_dimension_invalidates_cache(target, field, value):
    task, model, provider = identities()
    baseline = identity_digest(task, model, provider, config_digest="cfg")
    values = {"task": task, "model": model, "provider": provider}
    values[target] = replace(values[target], **{field: value})

    assert (
        identity_digest(
            values["task"], values["model"], values["provider"], config_digest="cfg"
        )
        != baseline
    )


def test_config_digest_invalidates_cache():
    task, model, provider = identities()
    assert identity_digest(task, model, provider, config_digest="a") != identity_digest(
        task, model, provider, config_digest="b"
    )


def test_generation_limit_resolution_is_validated_and_part_of_identity():
    baseline = GenerationLimitResolution(2048, None, None, 2048)
    override = GenerationLimitResolution(2048, 512, "cli", 512)
    assert identity_digest(baseline, config_digest="cfg") != identity_digest(
        override, config_digest="cfg"
    )
    with pytest.raises(ValueError):
        GenerationLimitResolution(2048, 512, None, 512)


def test_sample_accounting_partitions_close():
    accounting = SampleAccounting(
        10,
        9,
        1,
        8,
        7,
        1,
        2,
        model_invalid=1,
        provider_error=1,
        cache_error=1,
        scorer_error=1,
        harness_error=1,
    )
    accounting.validate()


@pytest.mark.parametrize(
    "accounting",
    [
        SampleAccounting(
            10,
            9,
            0,
            8,
            7,
            1,
            2,
            model_invalid=1,
            provider_error=1,
            scorer_error=1,
            harness_error=1,
        ),
        SampleAccounting(
            10,
            9,
            1,
            8,
            7,
            0,
            2,
            model_invalid=1,
            provider_error=1,
            cache_error=1,
            scorer_error=1,
            harness_error=1,
        ),
        SampleAccounting(10, 9, 1, 8, 7, 1, 2),
    ],
)
def test_sample_accounting_rejects_open_partitions(accounting):
    with pytest.raises(ValueError):
        accounting.validate()


def test_run_state_machine_accepts_only_signed_transitions():
    validate_run_transition(RunStatus.PLANNED, RunStatus.RUNNING)
    validate_run_transition(RunStatus.RUNNING, RunStatus.FINALIZING)
    validate_run_transition(RunStatus.FINALIZING, RunStatus.COMPLETED)
    with pytest.raises(ValueError):
        validate_run_transition(RunStatus.RUNNING, RunStatus.COMPLETED)
    with pytest.raises(ValueError):
        validate_run_transition(RunStatus.COMPLETED, RunStatus.RUNNING)
