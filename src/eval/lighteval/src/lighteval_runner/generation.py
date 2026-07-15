from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


DEFAULT_STOP_TOKEN_ID = 0
DEFAULT_STOP_TEXT = "\nUser:"


class StopReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"


class ProviderTerminalError(ValueError):
    """The provider response did not satisfy the signed generation contract."""


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    raw_completion: str
    output_token_ids: tuple[int, ...]
    output_token_count: int
    provider_finish_reason: str
    provider_stop_reason: str | int | None
    stop_reason: StopReason
    truncated: bool
    generation_limit: int
    prompt_text: str | None = None
    prompt_token_ids: tuple[int, ...] = ()
    request_id: str | None = None
    usage: ProviderUsage | None = None


def normalize_generation_outcome(
    *,
    completion: str,
    output_token_ids: tuple[int, ...],
    provider_finish_reason: str | None,
    provider_stop_reason: str | int | None,
    generation_limit: int,
    prompt_text: str | None = None,
    prompt_token_ids: tuple[int, ...] = (),
    request_id: str | None = None,
    usage: ProviderUsage | None = None,
) -> GenerationOutcome:
    if generation_limit <= 0:
        raise ValueError("generation_limit must be positive")
    if provider_finish_reason is None:
        raise ProviderTerminalError("provider response is missing finish_reason")
    if provider_finish_reason in {
        "content_filter",
        "tool_calls",
        "function_call",
        "error",
    }:
        raise ProviderTerminalError(
            f"unsupported provider terminal reason: {provider_finish_reason}"
        )

    token_count = len(output_token_ids)
    stop_index = completion.find(DEFAULT_STOP_TEXT)
    if stop_index >= 0 and completion[stop_index + len(DEFAULT_STOP_TEXT) :]:
        raise ProviderTerminalError(
            "provider returned text after the signed stop delimiter"
        )
    stopped_by_text = stop_index >= 0 or provider_stop_reason == DEFAULT_STOP_TEXT
    token_zero_positions = tuple(
        index
        for index, token_id in enumerate(output_token_ids)
        if token_id == DEFAULT_STOP_TOKEN_ID
    )
    if token_zero_positions and token_zero_positions != (len(output_token_ids) - 1,):
        raise ProviderTerminalError("token 0 must be the final generated token")
    stopped_by_token = (
        bool(token_zero_positions)
        or provider_stop_reason == DEFAULT_STOP_TOKEN_ID
        or provider_stop_reason == str(DEFAULT_STOP_TOKEN_ID)
    )
    if provider_finish_reason == "length" and (stopped_by_text or stopped_by_token):
        raise ProviderTerminalError(
            "provider terminal metadata conflicts: length with stop evidence"
        )
    if stopped_by_text or stopped_by_token:
        raw_completion = completion[:stop_index] if stop_index >= 0 else completion
        reason = StopReason.STOP
        truncated = False
    elif provider_finish_reason == "length":
        if token_count != generation_limit:
            raise ProviderTerminalError(
                "length termination must contain exactly generation_limit output tokens"
            )
        raw_completion = completion
        reason = StopReason.LENGTH
        truncated = True
    elif token_count > generation_limit:
        raise ProviderTerminalError(
            "provider returned more tokens than the generation limit"
        )
    else:
        raise ProviderTerminalError(
            "provider response did not terminate with token 0, \\nUser:, or the generation limit"
        )

    return GenerationOutcome(
        raw_completion=raw_completion,
        output_token_ids=output_token_ids,
        output_token_count=token_count,
        provider_finish_reason=provider_finish_reason,
        provider_stop_reason=provider_stop_reason,
        stop_reason=reason,
        truncated=truncated,
        generation_limit=generation_limit,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
        request_id=request_id,
        usage=usage,
    )
