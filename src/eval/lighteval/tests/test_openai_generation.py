import asyncio
from types import SimpleNamespace

import pytest

from helicopter_lighteval import evaluation


class _Client:
    requests = []
    finish_reason = "stop"
    content = "raw answer"

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


@pytest.mark.parametrize(
    ("cot_mode", "generation_prompt"),
    [("none", "fake_think"), ("cot", "open_think")],
)
def test_generate_uses_one_raw_user_message_and_server_chat_template(
    monkeypatch, cot_mode, generation_prompt
) -> None:
    _Client.requests = []
    _Client.finish_reason = "stop"
    monkeypatch.setattr(evaluation, "AsyncOpenAI", _Client)
    request = evaluation.EvaluationRequest(
        model="model",
        task="lighteval/math/gsm8k@0",
        endpoint_url="http://server/v1",
        cot_mode=cot_mode,
    )
    responses, finish_reasons = asyncio.run(
        evaluation._generate(
            request=request,
            documents=[SimpleNamespace(query="What is 1 + 1?", generation_size=32)],
        )
    )

    assert responses[0].text == ["raw answer"]
    assert finish_reasons == ["stop"]
    assert _Client.requests == [
        {
            "model": "model",
            "messages": [{"role": "user", "content": "What is 1 + 1?"}],
            "max_tokens": 32,
            "temperature": 0.0,
            "stop": ["\nUser:"],
            "extra_body": {
                "stop_token_ids": [0],
                "chat_template_kwargs": {"rwkv_generation_prompt": generation_prompt},
            },
        }
    ]


def test_generate_accepts_length_without_repair(monkeypatch) -> None:
    _Client.requests = []
    _Client.finish_reason = "length"
    _Client.content = "<think>unfinished"
    monkeypatch.setattr(evaluation, "AsyncOpenAI", _Client)
    request = evaluation.EvaluationRequest(
        model="model",
        task="lighteval/math/gsm8k@0",
        endpoint_url="http://server/v1",
        cot_mode="cot",
    )

    responses, finish_reasons = asyncio.run(
        evaluation._generate(
            request=request,
            documents=[SimpleNamespace(query="question", generation_size=8)],
        )
    )

    assert responses[0].text == ["<think>unfinished"]
    assert finish_reasons == ["length"]


def test_generate_rejects_unknown_finish_reason(monkeypatch) -> None:
    _Client.finish_reason = "content_filter"
    monkeypatch.setattr(evaluation, "AsyncOpenAI", _Client)
    request = evaluation.EvaluationRequest(
        model="model",
        task="lighteval/math/gsm8k@0",
        endpoint_url="http://server/v1",
    )

    with pytest.raises(ValueError, match="unsupported finish_reason"):
        asyncio.run(
            evaluation._generate(
                request=request,
                documents=[SimpleNamespace(query="question", generation_size=8)],
            )
        )
