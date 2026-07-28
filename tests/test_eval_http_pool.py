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
