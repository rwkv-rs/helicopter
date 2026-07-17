import asyncio
from types import SimpleNamespace

import pytest
import httpx

from lighteval.tasks.requests import Doc

from helicopter_lighteval.vllm_rwkv import (
    ProviderResponseError,
    TokenUsage,
    TerminalReason,
    VllmRwkvModel,
    normalize_generation,
)
from helicopter_lighteval.vllm_rwkv import fetch_attestation


def _kwargs(**overrides):
    values = {
        "completion": "answer",
        "output_token_ids": [11, 12],
        "finish_reason": "stop",
        "stop_reason": 0,
        "generation_limit": 4,
        "prompt_token_ids": [1, 2, 3],
        "prompt_text": "prompt",
        "request_id": "req-1",
        "usage": TokenUsage(3, 2, 5),
    }
    values.update(overrides)
    return values


def test_token_zero_is_the_stop_terminal() -> None:
    evidence = normalize_generation(**_kwargs(output_token_ids=[11, 0]))
    assert evidence.terminal_reason is TerminalReason.STOP
    assert evidence.truncated is False


def test_user_delimiter_is_removed_and_recorded_as_stop() -> None:
    evidence = normalize_generation(
        **_kwargs(
            completion="answer\nUser:",
            output_token_ids=[11],
            stop_reason="\nUser:",
            usage=TokenUsage(3, 1, 4),
        )
    )
    assert evidence.raw_completion == "answer"
    assert evidence.terminal_reason is TerminalReason.STOP


def test_length_is_truncation_only_at_the_requested_limit() -> None:
    evidence = normalize_generation(
        **_kwargs(
            output_token_ids=[1, 2, 3, 4],
            finish_reason="length",
            stop_reason=None,
            generation_limit=4,
            usage=TokenUsage(3, 4, 7),
        )
    )
    assert evidence.terminal_reason is TerminalReason.LENGTH
    assert evidence.truncated is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"finish_reason": "unknown"},
        {
            "finish_reason": "length",
            "output_token_ids": [1, 0],
            "stop_reason": 0,
            "usage": TokenUsage(3, 2, 5),
        },
        {
            "finish_reason": "length",
            "stop_reason": None,
            "output_token_ids": [1],
            "usage": TokenUsage(3, 1, 4),
        },
        {"output_token_ids": [0, 2]},
        {
            "completion": "answer\nUser: trailing",
            "output_token_ids": [1],
            "stop_reason": "\nUser:",
            "usage": TokenUsage(3, 1, 4),
        },
    ],
)
def test_invalid_terminal_evidence_fails_closed(overrides) -> None:
    with pytest.raises(ProviderResponseError):
        normalize_generation(**_kwargs(**overrides))


def test_math_repair_is_applied_before_light_eval_scores() -> None:
    evidence = normalize_generation(
        **_kwargs(
            completion="reasoning",
            output_token_ids=[1, 2, 3, 4],
            finish_reason="length",
            stop_reason=None,
            generation_limit=4,
            usage=TokenUsage(3, 4, 7),
            math_task=True,
            repair_strategy="C",
            prompt_for_repair="<think>",
        )
    )
    assert evidence.raw_completion == "reasoning"
    assert evidence.scored_completion == "reasoning</think>\nTherefore..."


def test_generation_limit_reaches_provider_request_and_evidence() -> None:
    model = VllmRwkvModel(
        model="model",
        base_url="http://server/v1",
        api_key=None,
        checkpoint_sha256="c" * 64,
        tokenizer_revision="tokenizer",
        chat_template_revision="chat",
        server_revision="server",
        wkv_mode="fp32io16",
        precision="fp16",
        gemm_policy="fp32",
        launch_contract="launch",
        cot_mode="none",
        math_repair_strategy="A",
        math_task=False,
        max_concurrent_requests=1,
    )
    requested_limits: list[int] = []
    model.prompt_manager.prepare_prompt_api = lambda doc: [
        {"role": "user", "content": "prompt"}
    ]

    async def fake_complete(client, messages, generation_limit):
        requested_limits.append(generation_limit)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", reasoning_content=None),
                    token_ids=[11, 0],
                    finish_reason="stop",
                    stop_reason=0,
                )
            ],
            prompt_token_ids=[1],
            prompt_text="prompt",
            id="request-1",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    model._complete = fake_complete
    doc = Doc(
        query="question",
        choices=["answer"],
        gold_index=0,
        id="0",
        task_name="gsm8k|0",
        generation_size=128,
    )
    try:
        asyncio.run(model.greedy_until([doc]))
    finally:
        model.cleanup()
    evidence = model.evidence[("gsm8k|0", "0")][0]
    assert requested_limits == [128]
    assert evidence.generation_limit == 128


def test_greedy_until_uses_bounded_concurrency_and_preserves_doc_order() -> None:
    model = VllmRwkvModel(
        model="model",
        base_url="http://server/v1",
        api_key=None,
        checkpoint_sha256="c" * 64,
        tokenizer_revision="tokenizer",
        chat_template_revision="chat",
        server_revision="server",
        wkv_mode="fp32io16",
        precision="fp16",
        gemm_policy="fp32",
        launch_contract="launch",
        cot_mode="none",
        math_repair_strategy="A",
        math_task=False,
        max_concurrent_requests=2,
    )
    all_started = asyncio.Event()
    active = 0
    peak_active = 0
    model.prompt_manager.prepare_prompt_api = lambda doc: [
        {"role": "user", "content": doc.query}
    ]

    async def fake_complete(client, messages, generation_limit):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        if active == 2:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        query = messages[0]["content"]
        active -= 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"answer-{query}", reasoning_content=None
                    ),
                    token_ids=[11, 0],
                    finish_reason="stop",
                    stop_reason=0,
                )
            ],
            prompt_token_ids=[1],
            prompt_text=f"prompt-{query}",
            id=f"request-{query}",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    model._complete = fake_complete
    docs = [
        Doc(
            query=str(index),
            choices=[f"answer-{index}"],
            gold_index=0,
            id=str(index),
            task_name="gsm8k|0",
            generation_size=128,
        )
        for index in range(2)
    ]
    try:
        responses = asyncio.run(model.greedy_until(docs))
    finally:
        model.cleanup()

    assert peak_active == 2
    assert [response.text for response in responses] == [["answer-0"], ["answer-1"]]
    assert model.evidence[("gsm8k|0", "0")][0].request_id == "request-0"
    assert model.evidence[("gsm8k|0", "1")][0].request_id == "request-1"


def test_concurrent_failure_cancels_peers_without_partial_evidence() -> None:
    model = VllmRwkvModel(
        model="model",
        base_url="http://server/v1",
        api_key=None,
        checkpoint_sha256="c" * 64,
        tokenizer_revision="tokenizer",
        chat_template_revision="chat",
        server_revision="server",
        wkv_mode="fp32io16",
        precision="fp16",
        gemm_policy="fp32",
        launch_contract="launch",
        cot_mode="none",
        math_repair_strategy="A",
        math_task=False,
        max_concurrent_requests=2,
    )
    peer_started = asyncio.Event()
    peer_cancelled = asyncio.Event()
    model.prompt_manager.prepare_prompt_api = lambda doc: [
        {"role": "user", "content": doc.query}
    ]

    async def fake_complete(client, messages, generation_limit):
        if messages[0]["content"] == "fail":
            await asyncio.wait_for(peer_started.wait(), timeout=2)
            raise RuntimeError("provider failed")
        peer_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            peer_cancelled.set()
            raise

    model._complete = fake_complete
    docs = [
        Doc(
            query=query,
            choices=["answer"],
            gold_index=0,
            id=str(index),
            task_name="gsm8k|0",
            generation_size=128,
        )
        for index, query in enumerate(("fail", "wait"))
    ]

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(model.greedy_until(docs))

    assert peer_cancelled.is_set()
    assert model.evidence == {}


def test_attestation_uses_the_vllm_v1_route() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "model": {
                    "served_name": "model",
                    "checkpoint_sha256": "c" * 64,
                    "tokenizer_revision": "tok",
                    "chat_template_revision": "chat",
                },
                "provider": {
                    "server_revision": "server",
                    "wkv_mode": "fp32io16",
                    "precision": "fp16",
                    "gemm_policy": "fp32",
                    "launch_contract": "launch",
                },
                "capabilities": [
                    "openai-chat",
                    "output-token-ids",
                    "terminal-reason",
                    "prompt-evidence",
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert fetch_attestation(base_url="http://server/v1", client=client) is not None
    assert requested == ["http://server/v1/helicopter/attestation"]
