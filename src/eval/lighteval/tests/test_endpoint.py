from __future__ import annotations

import json

import httpx
import pytest

from lighteval_runner.generation import ProviderTerminalError, StopReason
from lighteval_runner.context import BudgetedContext, ContextSection
from lighteval_runner.provider.endpoint import (
    EndpointGenerationRequest,
    OpenAIEndpoint,
    ProviderResponseSchemaError,
    ProviderTransportError,
    parse_provider_response,
)


def signed_body(**choice_overrides):
    choice = {
        "message": {"content": "answer"},
        "finish_reason": "stop",
        "stop_reason": 0,
        "token_ids": [12, 0],
        **choice_overrides,
    }
    return {
        "id": "request-1",
        "choices": [choice],
        "prompt_token_ids": [1, 2],
        "prompt_text": "rendered",
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


def test_endpoint_uses_only_signed_rwkv_stops_and_preserves_metadata():
    captured = []

    def handler(request):
        payload = json.loads(request.content)
        captured.append((request.url.path, payload))
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(
                200, json={"count": 2, "max_model_len": 128, "tokens": [1, 2]}
            )
        return httpx.Response(200, json=signed_body())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    endpoint = OpenAIEndpoint(base_url="http://server/v1/", client=client)
    outcome = endpoint.generate(
        EndpointGenerationRequest(
            model="model",
            context=BudgetedContext(
                (ContextSection("question", "user", "question", 100),), (), 2
            ),
            generation_limit=8,
            generation_prompt_mode="open_think",
        )
    )

    completion = captured[-1][1]
    tokenize = captured[0][1]
    assert captured[0][0] == "/tokenize"
    assert captured[-1][0] == "/v1/chat/completions"
    expected_template = {"rwkv_generation_prompt": "open_think"}
    assert tokenize["chat_template_kwargs"] == expected_template
    assert completion["chat_template_kwargs"] == expected_template
    assert completion["stop"] == ["\nUser:"]
    assert completion["stop_token_ids"] == [0]
    assert completion["return_token_ids"] is True
    assert completion["return_prompt_text"] is True
    assert outcome.stop_reason is StopReason.STOP
    assert outcome.output_token_ids == (12, 0)
    assert outcome.prompt_text == "rendered"
    assert outcome.prompt_token_ids == (1, 2)
    assert outcome.request_id == "request-1"
    assert outcome.usage is not None
    assert outcome.usage.total_tokens == 4


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        signed_body(token_ids=None),
        signed_body(message={"content": None}),
        {**signed_body(), "usage": {}},
        {**signed_body(), "prompt_token_ids": [True]},
    ],
)
def test_incomplete_provider_schema_fails_explicitly(body):
    with pytest.raises(ProviderResponseSchemaError):
        parse_provider_response(body)


def test_unsigned_tool_call_is_not_scored_as_text():
    body = signed_body(message={"content": "", "tool_calls": [{"id": "call"}]})
    with pytest.raises(ProviderTerminalError, match="tool call"):
        parse_provider_response(body)


def test_transport_error_has_typed_boundary():
    def handler(_request):
        return httpx.Response(503, text="unavailable")

    endpoint = OpenAIEndpoint(
        base_url="http://server/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderTransportError):
        endpoint.generate(
            EndpointGenerationRequest(
                model="model",
                context=BudgetedContext(
                    (ContextSection("question", "user", "question", 100),), (), 2
                ),
                generation_limit=8,
                generation_prompt_mode="fake_think",
            )
        )
