from __future__ import annotations

import pytest

from lighteval_runner.generation import (
    ProviderTerminalError,
    StopReason,
    normalize_generation_outcome,
)


@pytest.mark.parametrize(
    ("completion", "tokens", "finish", "provider_stop"),
    [
        ("answer", (12, 0), "stop", None),
        ("answer", (12,), "stop", 0),
        ("answer", (12,), "stop", "0"),
        ("answer", (12,), "stop", "\nUser:"),
        ("answer\nUser:", (12,), "stop", "\nUser:"),
    ],
)
def test_signed_stop_evidence_is_not_truncation(
    completion, tokens, finish, provider_stop
):
    outcome = normalize_generation_outcome(
        completion=completion,
        output_token_ids=tokens,
        provider_finish_reason=finish,
        provider_stop_reason=provider_stop,
        generation_limit=8,
    )

    assert outcome.stop_reason is StopReason.STOP
    assert outcome.truncated is False
    assert outcome.raw_completion == "answer"


@pytest.mark.parametrize(
    ("finish", "tokens"),
    [("length", (1, 2, 3, 4))],
)
def test_generation_limit_is_truncation(finish, tokens):
    outcome = normalize_generation_outcome(
        completion="partial",
        output_token_ids=tokens,
        provider_finish_reason=finish,
        provider_stop_reason=None,
        generation_limit=4,
    )

    assert outcome.stop_reason is StopReason.LENGTH
    assert outcome.truncated is True


@pytest.mark.parametrize(
    "finish", [None, "content_filter", "tool_calls", "function_call", "error"]
)
def test_unsigned_terminal_reasons_fail_explicitly(finish):
    with pytest.raises(ProviderTerminalError):
        normalize_generation_outcome(
            completion="partial",
            output_token_ids=(1,),
            provider_finish_reason=finish,
            provider_stop_reason=None,
            generation_limit=4,
        )


def test_unknown_terminal_without_stop_or_limit_fails_explicitly():
    with pytest.raises(ProviderTerminalError, match="did not terminate"):
        normalize_generation_outcome(
            completion="partial",
            output_token_ids=(1,),
            provider_finish_reason="mystery",
            provider_stop_reason=None,
            generation_limit=4,
        )


def test_stop_without_signed_stop_evidence_is_not_reclassified_as_length():
    with pytest.raises(ProviderTerminalError, match="did not terminate"):
        normalize_generation_outcome(
            completion="partial",
            output_token_ids=(1, 2, 3, 4),
            provider_finish_reason="stop",
            provider_stop_reason=None,
            generation_limit=4,
        )


def test_token_zero_must_be_the_terminal_output_token():
    with pytest.raises(ProviderTerminalError, match="final"):
        normalize_generation_outcome(
            completion="answer",
            output_token_ids=(1, 0, 2),
            provider_finish_reason="stop",
            provider_stop_reason=0,
            generation_limit=4,
        )


def test_text_after_stop_delimiter_is_rejected():
    with pytest.raises(ProviderTerminalError, match="after the signed stop"):
        normalize_generation_outcome(
            completion="answer\nUser:ignored",
            output_token_ids=(1,),
            provider_finish_reason="stop",
            provider_stop_reason="\nUser:",
            generation_limit=4,
        )
