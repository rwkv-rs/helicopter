from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from scoreboard_server.application import create_app
from scoreboard_server.settings import AuthPrincipal, ScoreboardSettings


PUBLISHER = {"Authorization": "Bearer publisher-token"}
PUBLISHER_B = {"Authorization": "Bearer publisher-b-token"}
READER = {"Authorization": "Bearer reader-token"}
ADMIN = {"Authorization": "Bearer admin-token"}


def _settings() -> ScoreboardSettings:
    return ScoreboardSettings(
        cors_origins=("https://scoreboard.example",),
        tokens=(
            ("publisher-token", AuthPrincipal("publisher-a", frozenset({"publisher"}))),
            (
                "publisher-b-token",
                AuthPrincipal("publisher-b", frozenset({"publisher"})),
            ),
            ("reader-token", AuthPrincipal("reader-a", frozenset({"evidence_reader"}))),
            ("admin-token", AuthPrincipal("admin-a", frozenset({"admin"}))),
        ),
    )


def _create_payload(run_id: str = "run-1") -> dict:
    digest = "a" * 64
    return {
        "run_id": run_id,
        "identity": {
            "task": {
                "suite": "lighteval",
                "task": "gsm8k",
                "version": "1",
                "split": "test",
                "fewshot": 0,
                "prompt_revision": "upstream-64f4f5a",
                "scorer_revision": "extractive-match-v1",
                "generation_contract": "stop-v1",
                "cot_mode": "cot",
                "repair_strategy": "A",
                "dataset_digest": digest,
                "primary_metric": "extractive_match",
                "metrics": [
                    {
                        "name": "extractive_match",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "aggregation": "mean",
                        "binary_correctness": True,
                    }
                ],
            },
            "model": {
                "served_name": "rwkv-test",
                "checkpoint_sha256": "b" * 64,
                "tokenizer_revision": "tok-v1",
                "chat_template_revision": "chat-v1",
            },
            "provider": {
                "server_revision": "server-v1",
                "wkv_mode": "fp32io16",
                "precision": "fp16-io-fp32-state",
                "gemm_policy": "fp32-accumulation",
                "launch_contract": "launch-v1",
                "attestation_digest": "e" * 64,
                "attestation_verified": True,
                "attestation_present": True,
                "attestation_mismatches": [],
            },
            "config_digest": "c" * 64,
            "eligibility": "official",
            "comparable": True,
        },
        "expected_sample_ids": ["sample-1"],
    }


def _ingest_payload(*, terminal_status: str = "completed", score: float = 1.0) -> dict:
    accounting = {
        "source_rows": 1,
        "dataset_accepted": 1,
        "dataset_rejected": 0,
        "selected": 1,
        "formatter_accepted": 1,
        "formatter_rejected": 0,
        "scored": 1,
    }
    identity = _create_payload()["identity"]

    def canonical(value: object) -> str:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    return {
        "samples": [
            {
                "sample_id": "sample-1",
                "attempt": 1,
                "status": "scored",
                "prompt": "1 + 1 = ?",
                "raw_completion": "<think>2</think>2",
                "scored_completion": "<think>2</think>2",
                "generation": {
                    "raw_completion": "<think>2</think>2",
                    "output_token_ids": [1, 2, 0],
                    "output_token_count": 3,
                    "finish_reason": "stop",
                    "stop_reason": 0,
                    "terminal_reason": "stop",
                    "truncated": False,
                    "generation_limit": 256,
                    "prompt_text": "1 + 1 = ?",
                    "prompt_token_ids": [10, 11],
                    "request_id": "request-1",
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                    },
                },
                "scoring": {
                    "raw_completion": "<think>2</think>2",
                    "scored_completion": "<think>2</think>2",
                    "scorer_revision": "extractive-match-v1",
                    "repair_strategy": "A",
                    "repair_action": "none",
                },
                "metrics": {"extractive_match": score},
                "provenance": {"cache": {"hit": False, "key": "cache-1"}},
            }
        ],
        "accounting": accounting,
        "rejections": [],
        "metrics": {"extractive_match": score},
        "primary_metric": "extractive_match",
        "truncated_samples": 0,
        "generated_samples": 1,
        "manifest": {
            "digest": "d" * 64,
            "identity_digest": hashlib.sha256(canonical(identity).encode()).hexdigest(),
            "accounting_digest": hashlib.sha256(
                canonical(
                    {
                        **accounting,
                        "model_invalid": 0,
                        "provider_error": 0,
                        "cache_error": 0,
                        "scorer_error": 0,
                        "harness_error": 0,
                        "cancelled": 0,
                    }
                ).encode()
            ).hexdigest(),
            "terminal_status": terminal_status,
            "checksums_verified": True,
        },
        "terminal_status": terminal_status,
    }


def _reseal_accounting(payload: dict) -> None:
    for name in (
        "model_invalid",
        "provider_error",
        "cache_error",
        "scorer_error",
        "harness_error",
        "cancelled",
    ):
        payload["accounting"].setdefault(name, 0)
    canonical = json.dumps(
        payload["accounting"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    payload["manifest"]["accounting_digest"] = hashlib.sha256(
        canonical.encode()
    ).hexdigest()


@pytest.fixture(params=("sqlite", "postgres"))
async def client(request, tmp_path: Path):
    admin_connection = None
    database_name = None
    if request.param == "sqlite":
        database_url = f"sqlite:///{tmp_path / 'scoreboard.db'}"
    else:
        admin_url = os.environ.get("SCOREBOARD_TEST_POSTGRES_URL")
        if not admin_url:
            pytest.skip("SCOREBOARD_TEST_POSTGRES_URL is not configured")
        database_name = f"scoreboard_test_{uuid4().hex}"
        admin_connection = await asyncpg.connect(admin_url)
        await admin_connection.execute(f'CREATE DATABASE "{database_name}"')
        parsed = urlsplit(admin_url)
        database_url = urlunsplit(parsed._replace(path=f"/{database_name}"))
    app = create_app(_settings(), database_url=database_url)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as value:
            yield value
    finally:
        await app.state.database.close()
        if admin_connection is not None and database_name is not None:
            await admin_connection.execute(
                f'DROP DATABASE "{database_name}" WITH (FORCE)'
            )
            await admin_connection.close()


async def _migrate(client: AsyncClient) -> None:
    response = await client.post("/api/v1/admin/migrations", headers=ADMIN)
    assert response.status_code == 200, response.text


async def _create_and_resume(client: AsyncClient, run_id: str = "run-1") -> int:
    created = await client.post(
        "/api/v1/runs",
        headers={**PUBLISHER, "Idempotency-Key": f"create-{run_id}"},
        json=_create_payload(run_id),
    )
    assert created.status_code == 201, created.text
    resumed = await client.post(
        f"/api/v1/runs/{run_id}/resume",
        headers={**PUBLISHER, "If-Match": created.headers["etag"]},
    )
    assert resumed.status_code == 200, resumed.text
    return resumed.json()["revision"]


async def test_schema_is_explicitly_migrated_and_old_routes_are_absent(
    client: AsyncClient,
) -> None:
    health = await client.get("/api/v1/health")
    assert health.json() == {"status": "degraded", "schema_state": "missing"}
    assert (await client.get("/api/v1/meta")).status_code == 503
    assert (await client.post("/api/v1/admin/migrations")).status_code == 401
    await _migrate(client)
    assert (await client.get("/api/v1/health")).json()["schema_state"] == "ready"
    assert (await client.get("/api/leaderboard")).status_code == 404
    assert (await client.post("/api/capture-page")).status_code == 404


async def test_auth_cors_and_raw_evidence_boundary(client: AsyncClient) -> None:
    await _migrate(client)
    assert (
        await client.post("/api/v1/runs", json=_create_payload())
    ).status_code == 401
    assert (await client.get("/api/v1/history")).status_code == 401
    preflight = await client.options(
        "/api/v1/leaderboard",
        headers={
            "Origin": "https://scoreboard.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert (
        preflight.headers["access-control-allow-origin"] == "https://scoreboard.example"
    )
    denied = await client.options(
        "/api/v1/leaderboard",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in denied.headers


async def test_create_idempotency_and_revision_cas(client: AsyncClient) -> None:
    await _migrate(client)
    headers = {**PUBLISHER, "Idempotency-Key": "same-key"}
    first, second = await asyncio.gather(
        client.post("/api/v1/runs", headers=headers, json=_create_payload()),
        client.post("/api/v1/runs", headers=headers, json=_create_payload()),
    )
    assert sorted((first.status_code, second.status_code)) == [200, 201]
    conflict = _create_payload("run-other")
    assert (
        await client.post("/api/v1/runs", headers=headers, json=conflict)
    ).status_code == 409
    revision = first.json()["revision"]
    resumed = await client.post(
        "/api/v1/runs/run-1/resume",
        headers={**PUBLISHER, "If-Match": str(revision)},
    )
    assert resumed.status_code == 200
    stale = await client.post(
        "/api/v1/runs/run-1/resume",
        headers={**PUBLISHER, "If-Match": str(revision)},
    )
    assert stale.status_code == 412


async def test_ingest_is_atomic_idempotent_and_projects_only_official_completed(
    client: AsyncClient,
) -> None:
    await _migrate(client)
    revision = await _create_and_resume(client)
    headers = {
        **PUBLISHER,
        "If-Match": str(revision),
        "Idempotency-Key": "ingest-1",
    }
    payload = _ingest_payload()
    created = await client.put(
        "/api/v1/runs/run-1/ingest", headers=headers, json=payload
    )
    assert created.status_code == 200, created.text
    assert created.json()["status"] == "completed"
    unchanged = await client.put(
        "/api/v1/runs/run-1/ingest", headers=headers, json=payload
    )
    assert unchanged.json()["disposition"] == "unchanged"
    modified = _ingest_payload(score=0.0)
    assert (
        await client.put("/api/v1/runs/run-1/ingest", headers=headers, json=modified)
    ).status_code == 409
    board = (await client.get("/api/v1/leaderboard")).json()
    assert board["items"][0]["run_id"] == "run-1"
    samples = await client.get("/api/v1/runs/run-1/samples", headers=READER)
    assert samples.status_code == 200
    assert "<think>" in samples.json()["items"][0]["raw_completion"]
    assert samples.json()["items"][0]["provenance"]["cache"]["key"] == "cache-1"


async def test_invalid_ingest_rolls_back_all_evidence(client: AsyncClient) -> None:
    await _migrate(client)
    revision = await _create_and_resume(client)
    invalid = _ingest_payload()
    invalid["accounting"]["scored"] = 0
    response = await client.put(
        "/api/v1/runs/run-1/ingest",
        headers={**PUBLISHER, "If-Match": str(revision), "Idempotency-Key": "bad"},
        json=invalid,
    )
    assert response.status_code == 422
    run = await client.get("/api/v1/runs/run-1", headers=READER)
    assert run.json()["status"] == "running"
    samples = await client.get("/api/v1/runs/run-1/samples", headers=READER)
    assert samples.json()["items"] == []


async def test_migration_checksum_drift_is_visible(client: AsyncClient) -> None:
    await _migrate(client)
    app = client._transport.app  # test-only access to the in-process backend
    async with app.state.database.transaction(immediate=True) as session:
        await session.execute(
            "UPDATE schema_migrations SET checksum=? WHERE version=1", ("0" * 64,)
        )
    assert (await client.get("/api/v1/health")).json()["schema_state"] == "drift"
    retry = await client.post("/api/v1/admin/migrations", headers=ADMIN)
    assert retry.status_code == 409


async def test_partial_resume_preserves_attempts_and_replaces_projection(
    client: AsyncClient,
) -> None:
    await _migrate(client)
    revision = await _create_and_resume(client)
    partial = _ingest_payload(terminal_status="partial")
    partial["samples"][0].update(
        {
            "status": "scorer_error",
            "scoring": None,
            "metrics": {},
            "error_code": "scorer_failed",
            "error_message": "first attempt failed",
        }
    )
    partial["accounting"]["scored"] = 0
    partial["accounting"]["scorer_error"] = 1
    partial["metrics"] = {}
    partial["primary_metric"] = None
    _reseal_accounting(partial)
    first = await client.put(
        "/api/v1/runs/run-1/ingest",
        headers={
            **PUBLISHER,
            "If-Match": str(revision),
            "Idempotency-Key": "partial-1",
        },
        json=partial,
    )
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "partial"
    resumed = await client.post(
        "/api/v1/runs/run-1/resume",
        headers={**PUBLISHER, "If-Match": str(first.json()["revision"])},
    )
    completed = _ingest_payload()
    completed["samples"][0]["attempt"] = 2
    second = await client.put(
        "/api/v1/runs/run-1/ingest",
        headers={
            **PUBLISHER,
            "If-Match": str(resumed.json()["revision"]),
            "Idempotency-Key": "complete-2",
        },
        json=completed,
    )
    assert second.status_code == 200, second.text
    assert second.json()["status"] == "completed"
    page1 = (
        await client.get(
            "/api/v1/runs/run-1/samples", headers=READER, params={"limit": 1}
        )
    ).json()
    page2 = (
        await client.get(
            "/api/v1/runs/run-1/samples",
            headers=READER,
            params={"limit": 1, "cursor": page1["next_cursor"]},
        )
    ).json()
    assert [page1["items"][0]["attempt"], page2["items"][0]["attempt"]] == [1, 2]


async def test_transaction_rolls_back_after_sample_write_fault(
    client: AsyncClient, monkeypatch
) -> None:
    await _migrate(client)
    revision = await _create_and_resume(client)
    commands = client._transport.app.state.runs.commands
    original = commands.insert_sample

    async def fail_after_insert(session, values):
        await original(session, values)
        raise RuntimeError("fault after sample insert")

    monkeypatch.setattr(commands, "insert_sample", fail_after_insert)
    with pytest.raises(RuntimeError, match="fault after sample insert"):
        await client.put(
            "/api/v1/runs/run-1/ingest",
            headers={
                **PUBLISHER,
                "If-Match": str(revision),
                "Idempotency-Key": "fault-injected",
            },
            json=_ingest_payload(),
        )
    run = await client.get("/api/v1/runs/run-1", headers=READER)
    assert run.json()["status"] == "running"
    samples = await client.get("/api/v1/runs/run-1/samples", headers=READER)
    assert samples.json()["items"] == []


async def test_performance_patch_is_separate_and_revision_cas_is_atomic(
    client: AsyncClient,
) -> None:
    await _migrate(client)
    revision = await _create_and_resume(client)
    await client.put(
        "/api/v1/runs/run-1/ingest",
        headers={
            **PUBLISHER,
            "If-Match": str(revision),
            "Idempotency-Key": "complete",
        },
        json=_ingest_payload(),
    )
    headers = {**PUBLISHER, "If-Match": "0"}
    first, second = await asyncio.gather(
        client.put(
            "/api/v1/runs/run-1/performance",
            headers=headers,
            json={"metrics": {"tokens_per_second": 12.3456789012345}},
        ),
        client.put(
            "/api/v1/runs/run-1/performance",
            headers=headers,
            json={"metrics": {"tokens_per_second": 99.0}},
        ),
    )
    assert sorted((first.status_code, second.status_code)) == [200, 412]
    detail = (await client.get("/api/v1/runs/run-1", headers=READER)).json()
    assert detail["performance"]["revision"] == 1
    assert detail["aggregate"]["metrics"] == {"extractive_match": 1.0}


async def test_first_performance_patch_requires_zero_revision(
    client: AsyncClient,
) -> None:
    await _migrate(client)
    revision = await _create_and_resume(client)
    await client.put(
        "/api/v1/runs/run-1/ingest",
        headers={
            **PUBLISHER,
            "If-Match": str(revision),
            "Idempotency-Key": "complete",
        },
        json=_ingest_payload(),
    )
    rejected = await client.put(
        "/api/v1/runs/run-1/performance",
        headers={**PUBLISHER, "If-Match": "99"},
        json={"metrics": {"tokens_per_second": 12.0}},
    )
    assert rejected.status_code == 412
    accepted = await client.put(
        "/api/v1/runs/run-1/performance",
        headers={**PUBLISHER, "If-Match": "0"},
        json={"metrics": {"tokens_per_second": 12.0}},
    )
    assert accepted.status_code == 200
    assert accepted.json()["revision"] == 1


async def test_schema_fingerprint_rejects_missing_product_table(
    client: AsyncClient,
) -> None:
    await _migrate(client)
    app = client._transport.app
    async with app.state.database.transaction(immediate=True) as session:
        await session.execute("DROP TABLE sample_evidence")
    assert (await client.get("/api/v1/health")).json()["schema_state"] == "drift"
    assert (
        await client.post("/api/v1/admin/migrations", headers=ADMIN)
    ).status_code == 409


async def test_schema_fingerprint_rejects_wrong_metric_type(
    client: AsyncClient,
) -> None:
    await _migrate(client)
    app = client._transport.app
    async with app.state.database.transaction(immediate=True) as session:
        await session.execute("DROP TABLE run_metrics")
        await session.execute(
            """CREATE TABLE run_metrics (
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE RESTRICT,
                metric_name TEXT NOT NULL, metric_value TEXT NOT NULL,
                is_primary INTEGER NOT NULL, PRIMARY KEY (run_id, metric_name)
            )"""
        )
        await session.execute(
            "CREATE INDEX idx_metrics_query ON run_metrics(metric_name, metric_value, run_id)"
        )
    assert (await client.get("/api/v1/health")).json()["schema_state"] == "drift"


async def test_postgres_fingerprint_requires_exact_composite_unique(
    client: AsyncClient,
) -> None:
    await _migrate(client)
    app = client._transport.app
    if app.state.database.target.backend != "postgres":
        return
    async with app.state.database.transaction(immediate=True) as session:
        row = await session.fetchone(
            """SELECT tc.constraint_name FROM information_schema.table_constraints tc
            WHERE tc.table_schema='public' AND tc.table_name='evaluation_runs'
            AND tc.constraint_type='UNIQUE'"""
        )
        assert row is not None
        name = row["constraint_name"]
        assert name.replace("_", "").isalnum()
        await session.execute(f'ALTER TABLE evaluation_runs DROP CONSTRAINT "{name}"')
        await session.execute(
            "ALTER TABLE evaluation_runs ADD UNIQUE (publisher_subject)"
        )
    assert (await client.get("/api/v1/health")).json()["schema_state"] == "drift"


async def test_concurrent_migration_is_locked_and_idempotent(
    client: AsyncClient,
) -> None:
    first, second = await asyncio.gather(
        client.post("/api/v1/admin/migrations", headers=ADMIN),
        client.post("/api/v1/admin/migrations", headers=ADMIN),
    )
    assert {first.json()["disposition"], second.json()["disposition"]} == {
        "applied",
        "unchanged",
    }


async def test_openapi_has_required_headers_typed_reads_and_error_envelopes(
    client: AsyncClient,
) -> None:
    schema = (await client.get("/openapi.json")).json()
    create = schema["paths"]["/api/v1/runs"]["post"]
    header = next(
        item for item in create["parameters"] if item["name"] == "Idempotency-Key"
    )
    assert header["required"] is True
    read_schema = schema["paths"]["/api/v1/runs/{run_id}"]["get"]["responses"]["200"]
    assert read_schema["content"]["application/json"]["schema"]["$ref"].endswith(
        "/RunDetail"
    )
    assert create["responses"]["409"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("/ErrorEnvelope")


async def test_publisher_cannot_mutate_another_publishers_run(
    client: AsyncClient,
) -> None:
    await _migrate(client)
    created = await client.post(
        "/api/v1/runs",
        headers={**PUBLISHER, "Idempotency-Key": "owned"},
        json=_create_payload(),
    )
    denied = await client.post(
        "/api/v1/runs/run-1/resume",
        headers={**PUBLISHER_B, "If-Match": created.headers["etag"]},
    )
    assert denied.status_code == 403
