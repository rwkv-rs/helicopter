from __future__ import annotations

import httpx
import pytest
import hashlib
import gzip
import json

from lighteval_runner.execution import RunStatus, SampleAccounting
from lighteval_runner.results.artifacts import RunArtifacts
from lighteval_runner.results.scoreboard import (
    PublicationResult,
    ScoreboardClient,
    ScoreboardPublicationError,
    retry_publication,
)


def test_http_publication_uses_one_authenticated_idempotent_put() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201,
            json={
                "run_id": "run-1",
                "status": "completed",
                "task_id": 7,
                "disposition": "created",
            },
        )

    transport = httpx.MockTransport(handler)
    client = ScoreboardClient(
        base_url="https://scoreboard.example",
        bearer_token="secret",
        client=httpx.Client(transport=transport),
    )
    result = client.publish(
        run_id="run-1",
        payload={"terminal_status": "completed"},
        idempotency_key="publish:digest",
    )
    assert result.status == "published"
    assert result.task_id == 7
    assert [request.url.path for request in requests] == [
        "/api/v1/evaluation-publications/run-1"
    ]
    assert requests[0].method == "PUT"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert requests[0].headers["content-encoding"] == "gzip"
    assert json.loads(gzip.decompress(requests[0].content)) == {
        "terminal_status": "completed"
    }
    assert requests[0].headers["idempotency-key"] == "publish:digest"
    assert "if-match" not in requests[0].headers


def test_publication_retry_repeats_the_same_put() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "run_id": "run-1",
                "status": "completed",
                "task_id": 7,
                "disposition": "unchanged",
            },
        )

    client = ScoreboardClient(
        base_url="https://scoreboard.example",
        bearer_token="secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert (
        client.publish(
            run_id="run-1",
            payload={"terminal_status": "completed"},
            idempotency_key="publish:digest",
        ).status
        == "published"
    )
    assert paths == ["/api/v1/evaluation-publications/run-1"]


def test_publication_never_falls_back_when_http_backend_fails() -> None:
    client = ScoreboardClient(
        base_url="https://scoreboard.example",
        bearer_token="secret",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(503, json={"error": {"code": "down"}})
            )
        ),
    )
    with pytest.raises(ScoreboardPublicationError, match=r"503 \(down\)"):
        client.publish(
            run_id="run-1",
            payload={"terminal_status": "completed"},
            idempotency_key="publish:digest",
        )


def test_retry_publication_reconstructs_only_the_committed_run(
    tmp_path, monkeypatch
) -> None:
    identity = {
        "task": {"primary_metric": "score"},
        "model": {"name": "rwkv"},
        "evaluator": {"product_revision": "a" * 40, "dirty": False},
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    accounting = SampleAccounting(1, 1, 0, 1, 1, 0, 1)
    artifacts = RunArtifacts(tmp_path, run_id="run-1")
    artifacts.write_json(
        "samples.json",
        [
            {
                "sample_index": 0,
                "sample_id": "sample-1",
                "attempt": 1,
                "status": "scored",
                "prompt": "question",
                "raw_completion": "answer",
                "scored_completion": "answer",
                "generation": {
                    "raw_completion": "answer",
                    "output_token_ids": [1, 0],
                    "output_token_count": 2,
                    "finish_reason": "stop",
                    "stop_reason": 0,
                    "terminal_reason": "stop",
                    "truncated": False,
                    "generation_limit": 16,
                    "prompt_text": "question",
                    "prompt_token_ids": [1],
                    "request_id": "request-1",
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                },
                "scoring": {
                    "raw_completion": "answer",
                    "scored_completion": "answer",
                    "scorer_revision": "scorer-v1",
                    "repair_strategy": "A",
                    "repair_action": "none",
                },
                "metrics": {"score": 1.0},
                "error_code": None,
                "error_message": None,
                "reference_answer": "answer",
                "provenance": {},
            }
        ],
    )
    artifacts.write_json(
        "performance.json", {"token_usage_attribution": "not_attributable"}
    )
    artifacts.write_json(
        "summary.json",
        {
            "metrics": {"score": 1.0},
            "rejections": [],
            "truncated_samples": 0,
            "generated_samples": 1,
        },
    )
    manifest = artifacts.finalize(
        status=RunStatus.COMPLETED,
        identity_digest=digest,
        identities={"run": identity},
        accounting=accounting,
    )
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def publish(self, **kwargs):
            captured["publish"] = kwargs
            return PublicationResult("published", kwargs["idempotency_key"])

        def close(self):
            pass

    monkeypatch.setattr("lighteval_runner.results.scoreboard.ScoreboardClient", Client)
    result = retry_publication(
        manifest_path=manifest,
        base_url="https://scoreboard.example",
        bearer_token="secret",
    )
    assert result.status == "published"
    assert captured["publish"]["run_id"] == "run-1"
    assert captured["publish"]["payload"]["metrics"] == {"score": 1.0}
    assert captured["publish"]["payload"]["performance"] == {
        "token_usage_attribution": "not_attributable"
    }
