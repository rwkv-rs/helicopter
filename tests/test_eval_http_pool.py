from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from helicopter_lighteval import http_pool
from helicopter_lighteval.http_pool import (
    CapacityScheduler,
    PoolError,
    PoolManifest,
    VLLMHttpPool,
)


def _manifest(tmp_path: Path, replicas: list[dict[str, object]]) -> PoolManifest:
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "global_step": 7,
                "wkv_mode": "fp32io16",
                "vllm_version": "0.23.1.dev0",
                "max_model_len": 10240,
                "replicas": replicas,
            }
        ),
        encoding="utf-8",
    )
    return PoolManifest.read(path)


def test_manifest_rejects_duplicate_or_non_origin_endpoints(tmp_path: Path) -> None:
    with pytest.raises(PoolError, match="duplicate"):
        _manifest(
            tmp_path,
            [
                {"base_url": "http://10.0.0.1:8000", "max_concurrency": 1},
                {"base_url": "http://10.0.0.1:8000", "max_concurrency": 2},
            ],
        )

    with pytest.raises(PoolError, match="HTTP\\(S\\) origin"):
        _manifest(
            tmp_path,
            [
                {
                    "base_url": "http://10.0.0.1:8000/v1?token=secret",
                    "max_concurrency": 1,
                }
            ],
        )


def test_scheduler_respects_capacity_and_unblocks_on_release() -> None:
    scheduler = CapacityScheduler([2, 1])

    assert [scheduler.acquire() for _ in range(3)] == [0, 1, 0]
    acquired: list[int] = []
    ready = threading.Event()

    def acquire() -> None:
        acquired.append(scheduler.acquire())
        ready.set()

    thread = threading.Thread(target=acquire)
    thread.start()
    assert not ready.wait(0.05)
    scheduler.release(0)
    assert ready.wait(1)
    thread.join()
    assert acquired == [0]

    scheduler.release(0)
    scheduler.release(0)
    scheduler.release(1)
    assert scheduler.inflight == (0, 0)


def test_pool_preflights_every_replica_and_uses_all_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    class Response:
        def __init__(self, payload: dict[str, object], status_code: int = 200):
            self._payload = payload
            self.status_code = status_code
            self.text = json.dumps(payload)
            self.request = httpx.Request("GET", "http://test")

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "failure",
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

    class Client:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        def get(self, path):
            if path == "/health":
                return Response({})
            assert path == "/v1/models"
            return Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            if path == "/detokenize":
                assert json == {"model": "rwkv-current", "tokens": [1, 2]}
                return Response({"prompt": "User✿question✿\nBot✿<think"})
            assert path == "/v1/chat/completions"
            with lock:
                calls.append((self.base_url, json))
                token = len(calls)
            barrier.wait(timeout=1)
            return Response(
                {
                    "prompt_token_ids": [1, 2],
                    "choices": [
                        {
                            "message": {"content": f"answer-{token}"},
                            "token_ids": [token],
                        }
                    ],
                }
            )

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    manifest = _manifest(
        tmp_path,
        [
            {"base_url": "http://10.0.0.1:8000", "max_concurrency": 1},
            {"base_url": "http://10.0.0.2:8000", "max_concurrency": 1},
        ],
    )
    pool = VLLMHttpPool(manifest)

    assert pool.preflight() == "rwkv-current"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: pool.complete(
                    [{"role": "user", "content": "question"}],
                    {
                        "temperature": 0.96,
                        "return_token_ids": True,
                    },
                ),
                range(2),
            )
        )

    assert {base_url for base_url, _ in calls} == {
        "http://10.0.0.1:8000",
        "http://10.0.0.2:8000",
    }
    assert all(payload["model"] == "rwkv-current" for _, payload in calls)
    assert all(payload["n"] == 1 for _, payload in calls)
    assert {result.prompt_text for result in results} == {
        "User✿question✿\nBot✿<think"
    }
    assert sorted(result.output_token_ids for result in results) == [(1,), (2,)]
    pool.close()


def test_pool_tokenizes_and_returns_aligned_prompt_logprobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    class Response:
        status_code = 200
        text = ""
        request = httpx.Request("POST", "http://test")

        def __init__(self, payload: dict[str, object]):
            self.payload = payload

        def json(self):
            return self.payload

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            if path == "/health":
                return Response({})
            return Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            requests.append((path, json))
            if path == "/tokenize":
                return Response({"tokens": [11, 12]})
            assert path == "/v1/completions"
            return Response(
                {
                    "choices": [
                        {
                            "token_ids": [99],
                            "prompt_token_ids": [0, 11, 12],
                            "logprobs": {
                                "token_logprobs": [None, -0.25, -0.5],
                                "top_logprobs": [
                                    None,
                                    {"token_id:11": -0.25},
                                    {"token_id:12": -0.5},
                                ],
                            },
                        }
                    ]
                }
            )

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = VLLMHttpPool(
        _manifest(
            tmp_path,
            [{"base_url": "http://10.0.0.1:8000", "max_concurrency": 2}],
        )
    )
    pool.preflight()

    assert pool.tokenize("ab") == (11, 12)
    scored = pool.score_tokens([0, 11, 12])

    assert scored.token_ids == (0, 11, 12)
    assert scored.token_logprobs == (None, -0.25, -0.5)
    normalized = pool.score_tokens(
        [11, 12],
        implicit_prefix_token_id=0,
    )
    assert normalized.token_ids == (11, 12)
    assert normalized.token_logprobs == (None, -0.5)
    tokenize_payload = requests[0][1]
    assert tokenize_payload["add_special_tokens"] is False
    scoring_payload = requests[1][1]
    assert scoring_payload == {
        "model": "rwkv-current",
        "prompt": [0, 11, 12],
        "echo": True,
        "max_tokens": 0,
        "logprobs": 1,
        "prompt_logprobs": 1,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "add_special_tokens": False,
        "stream": False,
        "n": 1,
    }
    pool.close()


def test_prompt_scoring_normalizes_implicit_prefix_without_top_logprobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status_code = 200
        text = ""
        request = httpx.Request("POST", "http://test")

        def __init__(self, payload: dict[str, object]):
            self.payload = payload

        def json(self):
            return self.payload

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            if path == "/health":
                return Response({})
            return Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            assert path == "/v1/completions"
            return Response(
                {
                    "choices": [
                        {
                            "prompt_token_ids": [0, 11, 12],
                            "logprobs": {
                                "token_logprobs": [None, -0.25, -0.5],
                            },
                        }
                    ]
                }
            )

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = VLLMHttpPool(
        _manifest(
            tmp_path,
            [{"base_url": "http://10.0.0.1:8000", "max_concurrency": 1}],
        )
    )
    pool.preflight()

    scored = pool.score_tokens([11, 12], implicit_prefix_token_id=0)

    assert scored.token_ids == (11, 12)
    assert scored.token_logprobs == (None, -0.5)
    assert scored.top_logprobs == (None, None)
    pool.close()


def test_pool_generates_text_from_pretokenized_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    class Response:
        status_code = 200
        text = ""
        request = httpx.Request("POST", "http://test")

        def __init__(self, payload: dict[str, object]):
            self.payload = payload

        def json(self):
            return self.payload

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            if path == "/health":
                return Response({})
            return Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            requests.append((path, json))
            assert path == "/v1/completions"
            return Response(
                {
                    "choices": [
                        {
                            "text": "generated",
                            "prompt_token_ids": [0, 11, 12],
                            "token_ids": [21, 22],
                            "finish_reason": "stop",
                            "stop_reason": "END",
                        }
                    ]
                }
            )

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = VLLMHttpPool(
        _manifest(
            tmp_path,
            [{"base_url": "http://10.0.0.1:8000", "max_concurrency": 1}],
        )
    )
    pool.preflight()

    generated = pool.generate_text(
        [11, 12],
        {"max_tokens": 32, "stop": ["END"], "temperature": 0.0},
    )

    assert generated.text == "generated"
    assert generated.prompt_token_ids == (0, 11, 12)
    assert generated.output_token_ids == (21, 22)
    assert generated.stop_reason == "END"
    assert requests == [
        (
            "/v1/completions",
            {
                "max_tokens": 32,
                "stop": ["END"],
                "temperature": 0.0,
                "model": "rwkv-current",
                "prompt": [11, 12],
                "add_special_tokens": False,
                "return_token_ids": True,
                "n": 1,
                "stream": False,
            },
        )
    ]
    pool.close()


def test_prompt_scoring_retries_429_on_another_replica(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion_hosts: list[str] = []

    class Response:
        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code
            self.text = "busy" if status_code == 429 else ""
            self.request = httpx.Request("POST", "http://test")

        def json(self):
            return self.payload

        def raise_for_status(self):
            if self.status_code >= 400:
                response = httpx.Response(
                    self.status_code,
                    request=self.request,
                    text=self.text,
                )
                raise httpx.HTTPStatusError(
                    "failure",
                    request=self.request,
                    response=response,
                )

    class Client:
        def __init__(self, *, base_url, **_kwargs):
            self.base_url = str(base_url)

        def get(self, path):
            if path == "/health":
                return Response({})
            return Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            assert path == "/v1/completions"
            completion_hosts.append(self.base_url)
            if self.base_url.endswith("1:8000"):
                return Response({}, status_code=429)
            return Response(
                {
                    "choices": [
                        {
                            "token_ids": [99],
                            "prompt_token_ids": json["prompt"],
                            "logprobs": {
                                "token_logprobs": [None, -0.5],
                                "top_logprobs": [None, {"token_id:2": -0.5}],
                            },
                        }
                    ]
                }
            )

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = VLLMHttpPool(
        _manifest(
            tmp_path,
            [
                {"base_url": "http://10.0.0.1:8000", "max_concurrency": 1},
                {"base_url": "http://10.0.0.2:8000", "max_concurrency": 1},
            ],
        )
    )
    pool.preflight()

    assert pool.score_tokens([1, 2]).token_logprobs == (None, -0.5)
    assert completion_hosts == [
        "http://10.0.0.1:8000",
        "http://10.0.0.2:8000",
    ]
    pool.close()


def test_prompt_scoring_does_not_retry_non_retryable_4xx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion_calls = 0

    class Response:
        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code
            self.text = "invalid request"
            self.request = httpx.Request("POST", "http://test")

        def json(self):
            return self.payload

        def raise_for_status(self):
            if self.status_code >= 400:
                response = httpx.Response(
                    self.status_code,
                    request=self.request,
                    text=self.text,
                )
                raise httpx.HTTPStatusError(
                    "failure",
                    request=self.request,
                    response=response,
                )

    class Client:
        def __init__(self, **_kwargs):
            pass

        def get(self, path):
            if path == "/health":
                return Response({})
            return Response({"data": [{"id": "rwkv-current"}]})

        def post(self, path, *, json):
            nonlocal completion_calls
            del json
            assert path == "/v1/completions"
            completion_calls += 1
            return Response({}, status_code=400)

        def close(self):
            pass

    monkeypatch.setattr(http_pool.httpx, "Client", Client)
    pool = VLLMHttpPool(
        _manifest(
            tmp_path,
            [
                {"base_url": "http://10.0.0.1:8000", "max_concurrency": 1},
                {"base_url": "http://10.0.0.2:8000", "max_concurrency": 1},
            ],
        )
    )
    pool.preflight()

    with pytest.raises(PoolError, match="rejected prompt scoring with HTTP 400"):
        pool.score_tokens([1, 2])
    assert completion_calls == 1
    pool.close()
