from __future__ import annotations

import httpx
import pytest
import hashlib
import json

from lighteval_runner.execution import RunStatus, SampleAccounting
from lighteval_runner.results.artifacts import RunArtifacts
from lighteval_runner.results.scoreboard import (
    PublicationResult,
    ScoreboardClient,
    ScoreboardPublicationError,
    retry_publication,
)


def test_http_publication_uses_versioned_authenticated_idempotent_state_machine() -> (
    None
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/runs":
            return httpx.Response(
                201,
                json={
                    "run_id": "run-1",
                    "status": "planned",
                    "revision": 1,
                    "disposition": "created",
                },
            )
        if request.url.path.endswith("/resume"):
            return httpx.Response(
                200,
                json={
                    "run_id": "run-1",
                    "status": "running",
                    "revision": 2,
                    "disposition": "created",
                },
            )
        return httpx.Response(
            200,
            json={
                "run_id": "run-1",
                "status": "completed",
                "revision": 4,
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
        create_payload={"run_id": "run-1"},
        ingest_payload={"terminal_status": "completed"},
        ingest_identity="ingest:digest",
    )
    assert result.status == "published"
    assert [request.url.path for request in requests] == [
        "/api/v1/runs",
        "/api/v1/runs/run-1/resume",
        "/api/v1/runs/run-1/ingest",
    ]
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert requests[0].headers["idempotency-key"] == "create:run-1"
    assert requests[2].headers["idempotency-key"] == "ingest:digest"
    assert requests[2].headers["if-match"] == "2"


def test_publication_retry_skips_resume_for_existing_terminal_run() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "run_id": "run-1",
                "status": "completed",
                "revision": 4,
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
            create_payload={"run_id": "run-1"},
            ingest_payload={"terminal_status": "completed"},
            ingest_identity="ingest:digest",
        ).status
        == "published"
    )
    assert paths == ["/api/v1/runs", "/api/v1/runs/run-1/ingest"]


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
            create_payload={"run_id": "run-1"},
            ingest_payload={"terminal_status": "completed"},
            ingest_identity="ingest:digest",
        )


def test_retry_publication_reconstructs_only_the_committed_run(
    tmp_path, monkeypatch
) -> None:
    identity = {"task": {"primary_metric": "score"}, "model": {"name": "rwkv"}}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    accounting = SampleAccounting(1, 1, 0, 1, 1, 0, 1)
    artifacts = RunArtifacts(tmp_path, run_id="run-1")
    artifacts.write_json(
        "samples.json", [{"sample_id": "sample-1", "status": "scored"}]
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
            return PublicationResult("published", kwargs["ingest_identity"])

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
    assert captured["publish"]["ingest_payload"]["metrics"] == {"score": 1.0}
