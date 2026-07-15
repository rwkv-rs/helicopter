from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx

from ..generation import (
    DEFAULT_STOP_TEXT,
    DEFAULT_STOP_TOKEN_ID,
    GenerationOutcome,
    ProviderUsage,
    ProviderTerminalError,
    normalize_generation_outcome,
)
from ..context import BudgetedContext, ContextSection


class ProviderTransportError(RuntimeError):
    """The signed endpoint request could not be completed."""


class ProviderResponseSchemaError(ProviderTerminalError):
    """The endpoint returned an incomplete or invalid response schema."""


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    completion: str
    output_token_ids: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]
    prompt_text: str | None
    finish_reason: str | None
    stop_reason: str | int | None
    request_id: str
    usage: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TokenizedContext:
    token_ids: tuple[int, ...]
    max_model_len: int


@dataclass(frozen=True, slots=True)
class EndpointGenerationRequest:
    model: str
    context: BudgetedContext
    generation_limit: int
    generation_prompt_mode: str
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("endpoint model must not be empty")
        if not self.context.sections:
            raise ValueError("endpoint context must not be empty")
        if self.generation_limit <= 0:
            raise ValueError("generation_limit must be positive")
        if self.generation_prompt_mode not in {"open_think", "fake_think"}:
            raise ValueError("generation_prompt_mode must be open_think or fake_think")


class OpenAIEndpoint:
    """Signed vLLM-RWKV chat-completion adapter preserving terminal metadata."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._api_base_url = base_url.rstrip("/")
        self._server_base_url = (
            self._api_base_url[:-3]
            if self._api_base_url.endswith("/v1")
            else self._api_base_url
        )
        self._api_key = api_key
        self._tokenized_contexts: dict[
            tuple[str, tuple[ContextSection, ...]], TokenizedContext
        ] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def generate(self, request: EndpointGenerationRequest) -> GenerationOutcome:
        expected = self.tokenize_context(
            request.model, request.context.sections, request.generation_prompt_mode
        )
        response = self._post_completion(request)
        if response.prompt_token_ids != expected.token_ids:
            raise ProviderResponseSchemaError(
                "generation prompt token ids differ from the budgeted context"
            )
        return normalize_generation_outcome(
            completion=response.completion,
            output_token_ids=response.output_token_ids,
            provider_finish_reason=response.finish_reason,
            provider_stop_reason=response.stop_reason,
            generation_limit=request.generation_limit,
            prompt_text=response.prompt_text,
            prompt_token_ids=response.prompt_token_ids,
            request_id=response.request_id,
            usage=ProviderUsage(**response.usage),
        )

    def context_token_count(
        self,
        model: str,
        sections: tuple[ContextSection, ...],
        generation_prompt_mode: str,
    ) -> int:
        return len(
            self.tokenize_context(model, sections, generation_prompt_mode).token_ids
        )

    def tokenize_context(
        self,
        model: str,
        sections: tuple[ContextSection, ...],
        generation_prompt_mode: str,
    ) -> TokenizedContext:
        key = (f"{model}:{generation_prompt_mode}", sections)
        cached = self._tokenized_contexts.get(key)
        if cached is not None:
            return cached
        payload = {
            "model": model,
            "messages": _messages(sections),
            "chat_template_kwargs": {"rwkv_generation_prompt": generation_prompt_mode},
        }
        body = self._request_json(f"{self._server_base_url}/tokenize", payload)
        if not isinstance(body, dict):
            raise ProviderResponseSchemaError("tokenize response must be an object")
        token_ids = _integer_sequence(body.get("tokens"), "tokens")
        count = body.get("count")
        max_model_len = body.get("max_model_len")
        if count != len(token_ids):
            raise ProviderResponseSchemaError("tokenize count does not match token ids")
        if (
            not isinstance(max_model_len, int)
            or isinstance(max_model_len, bool)
            or max_model_len <= 0
        ):
            raise ProviderResponseSchemaError("tokenize max_model_len must be positive")
        tokenized = TokenizedContext(token_ids, max_model_len)
        self._tokenized_contexts[key] = tokenized
        return tokenized

    def _post_completion(self, request: EndpointGenerationRequest) -> ProviderResponse:
        payload = {
            "model": request.model,
            "messages": _messages(request.context.sections),
            "chat_template_kwargs": {
                "rwkv_generation_prompt": request.generation_prompt_mode
            },
            "max_tokens": request.generation_limit,
            "temperature": request.temperature,
            "stop": [DEFAULT_STOP_TEXT],
            "stop_token_ids": [DEFAULT_STOP_TOKEN_ID],
            "return_token_ids": True,
            "return_prompt_text": True,
        }
        return parse_provider_response(
            self._request_json(f"{self._api_base_url}/chat/completions", payload)
        )

    def _request_json(self, url: str, payload: Mapping[str, Any]) -> Any:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ProviderTransportError(f"provider request failed: {error}") from error
        try:
            body = response.json()
        except ValueError as error:
            raise ProviderResponseSchemaError(
                "provider response is not JSON"
            ) from error
        return body


def parse_provider_response(body: Any) -> ProviderResponse:
    if not isinstance(body, dict):
        raise ProviderResponseSchemaError("provider response must be an object")
    choices = body.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise ProviderResponseSchemaError(
            "provider response must contain exactly one choice"
        )
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ProviderResponseSchemaError(
            "provider choice must contain string message content"
        )
    if message.get("tool_calls") or message.get("function_call"):
        raise ProviderTerminalError(
            "unsigned tool call response is not a text generation"
        )
    token_ids = _integer_sequence(choice.get("token_ids"), "choice.token_ids")
    prompt_token_ids = _integer_sequence(
        body.get("prompt_token_ids"), "prompt_token_ids"
    )
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ProviderResponseSchemaError("finish_reason must be a string or null")
    stop_reason = choice.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, (str, int)):
        raise ProviderResponseSchemaError(
            "stop_reason must be a string, integer, or null"
        )
    usage_raw = body.get("usage")
    if not isinstance(usage_raw, dict):
        raise ProviderResponseSchemaError("provider response must contain usage")
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage_raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProviderResponseSchemaError(
                f"usage.{key} must be a non-negative integer"
            )
        usage[key] = value
    request_id = body.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ProviderResponseSchemaError(
            "provider response id must be a non-empty string"
        )
    prompt_text = body.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ProviderResponseSchemaError("prompt_text must be a non-empty string")
    if usage["prompt_tokens"] != len(prompt_token_ids):
        raise ProviderResponseSchemaError(
            "usage.prompt_tokens does not match prompt_token_ids"
        )
    if usage["completion_tokens"] != len(token_ids):
        raise ProviderResponseSchemaError(
            "usage.completion_tokens does not match choice.token_ids"
        )
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ProviderResponseSchemaError("usage.total_tokens is inconsistent")
    return ProviderResponse(
        completion=message["content"],
        output_token_ids=token_ids,
        prompt_token_ids=prompt_token_ids,
        prompt_text=prompt_text,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        request_id=request_id,
        usage=usage,
    )


def _integer_sequence(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProviderResponseSchemaError(f"{field} must be an integer array")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value
    ):
        raise ProviderResponseSchemaError(f"{field} must be an integer array")
    return tuple(value)


def _messages(sections: tuple[ContextSection, ...]) -> list[dict[str, str]]:
    return [{"role": section.role, "content": section.content} for section in sections]
