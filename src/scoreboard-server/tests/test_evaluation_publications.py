from __future__ import annotations

import getpass
import hashlib
import json
import os
import uuid
import asyncio

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from scoreboard_server.application import create_app
from scoreboard_server.db.connection import close_db, init_db
from scoreboard_server.db.models import (
    Checker,
    Completion,
    EvalRecord,
    EvaluationPublication,
    Score,
    Task,
)
from scoreboard_server.db.settings import DatabaseSettings
from scoreboard_server.services.api.evaluation_publications import (
    MAX_PUBLICATION_TRANSFER_BYTES,
    TokenGrant,
)


def _maintenance_connection_kwargs() -> dict[str, str]:
    return {
        "user": os.environ.get("PGUSER") or getpass.getuser(),
        "host": os.environ.get("PGHOST") or "/var/run/postgresql",
        "database": os.environ.get("PGDATABASE") or "postgres",
    }


@pytest.fixture()
async def database_settings() -> DatabaseSettings:
    db_name = f"helicopter_publication_test_{uuid.uuid4().hex[:12]}"
    kwargs = _maintenance_connection_kwargs()
    connection = await asyncpg.connect(**kwargs)
    try:
        await connection.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await connection.close()
    settings = DatabaseSettings(
        host=kwargs["host"],
        port=int(os.environ.get("PGPORT") or 5432),
        user=kwargs["user"],
        password=os.environ.get("PGPASSWORD") or None,
        database=db_name,
    )
    await init_db(settings, generate_schemas=True)
    try:
        yield settings
    finally:
        await close_db()
        connection = await asyncpg.connect(**kwargs)
        try:
            await connection.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = $1 AND pid <> pg_backend_pid()
                """,
                db_name,
            )
            await connection.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        finally:
            await connection.close()


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _publication_payload(*, binary: bool = True, official: bool = True) -> dict:
    identity = {
        "task": {
            "suite": "lighteval",
            "task": "math/gsm8k",
            "version": "0",
            "split": "test",
            "fewshot": 0,
            "prompt_revision": "prompt-v1",
            "scorer_revision": "scorer-v1",
            "generation_contract": "rwkv-stop-v1",
            "cot_mode": "none",
            "repair_strategy": "A",
            "dataset_digest": "a" * 64,
            "primary_metric": "exact_match",
            "metrics": [
                {
                    "name": "exact_match",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "aggregation": "mean",
                    "binary_correctness": binary,
                }
            ],
        },
        "model": {
            "served_name": "rwkv7-g1g-1.5b",
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
            "attestation_digest": "c" * 64,
            "attestation_verified": official,
            "attestation_present": True,
            "attestation_mismatches": [],
        },
        "evaluator": {"product_revision": "d" * 40, "dirty": False},
        "config_digest": "e" * 64,
        "eligibility": "official" if official else "proxy",
        "comparable": official,
    }
    accounting = {
        "source_rows": 1,
        "dataset_accepted": 1,
        "dataset_rejected": 0,
        "selected": 1,
        "formatter_accepted": 1,
        "formatter_rejected": 0,
        "scored": 1,
        "model_invalid": 0,
        "provider_error": 0,
        "cache_error": 0,
        "scorer_error": 0,
        "harness_error": 0,
        "cancelled": 0,
    }
    sample = {
        "sample_index": 0,
        "sample_id": "gsm8k-0",
        "attempt": 1,
        "status": "scored",
        "prompt": "What is 1 + 1?",
        "raw_completion": "2",
        "scored_completion": "2",
        "generation": {
            "output_token_count": 2,
            "finish_reason": "stop",
            "stop_reason": 0,
            "terminal_reason": "stop",
            "truncated": False,
            "generation_limit": 256,
            "request_id": "request-1",
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        },
        "scoring": {
            "scorer_revision": "scorer-v1",
            "repair_strategy": "A",
            "repair_action": "none",
        },
        "metrics": {"exact_match": 1.0 if binary else 0.75},
        "error_code": None,
        "error_message": None,
        "reference_answer": "2" if binary else None,
        "provenance": {"config_digest": "e" * 64},
    }
    manifest_digest = "f" * 64
    return {
        "identity": identity,
        "samples": [sample],
        "accounting": accounting,
        "rejections": [],
        "metrics": {"exact_match": sample["metrics"]["exact_match"]},
        "native_metrics": {
            "exact_match": sample["metrics"]["exact_match"],
            "secondary_metric": 0.5,
        },
        "primary_metric": "exact_match",
        "truncated_samples": 0,
        "generated_samples": 1,
        "performance": {
            "token_usage_attribution": "per_request_usage",
            "total_tokens": 4,
        },
        "manifest": {
            "digest": manifest_digest,
            "identity_digest": _digest(identity),
            "accounting_digest": _digest(accounting),
            "terminal_status": "completed",
            "checksums_verified": True,
            "completed_at": "2026-07-15T12:00:00+00:00",
        },
        "terminal_status": "completed",
    }


def _app(settings: DatabaseSettings):
    return create_app(
        settings=settings,
        publication_grants={
            "publisher-token": TokenGrant("publisher", frozenset({"publisher"})),
            "reader-token": TokenGrant("reader", frozenset({"reader"})),
        },
    )


async def test_publication_auth_projection_retry_conflict_and_old_queries(
    database_settings: DatabaseSettings,
) -> None:
    app = _app(database_settings)
    payload = _publication_payload()
    path = "/api/v1/evaluation-publications/run-1"
    headers = {
        "Authorization": "Bearer publisher-token",
        "Idempotency-Key": f"publish:{payload['manifest']['digest']}",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        assert (
            await client.put(path, headers={"Idempotency-Key": "x"}, json=payload)
        ).status_code == 401
        assert (
            await client.put(path, headers={"Idempotency-Key": "x"}, content=b"{")
        ).status_code == 401
        assert (
            await client.put(
                path,
                headers={
                    "Authorization": "Bearer reader-token",
                    "Idempotency-Key": headers["Idempotency-Key"],
                },
                json=payload,
            )
        ).status_code == 403
        created = await client.put(path, headers=headers, json=payload)
        assert created.status_code == 201
        receipt = created.json()
        assert receipt["disposition"] == "created"
        task_id = receipt["task_id"]

        replay = await client.put(path, headers=headers, json=payload)
        assert replay.status_code == 200
        assert replay.json() == {**receipt, "disposition": "unchanged"}

        changed = json.loads(json.dumps(payload))
        changed["performance"]["total_tokens"] = 5
        assert (
            await client.put(path, headers=headers, json=changed)
        ).status_code == 409

        leaderboard = (
            await client.get(
                "/api/leaderboard",
                params={
                    "model": "rwkv7-g1g-1.5b",
                    "view": "benchmark_detail_latest",
                },
            )
        ).json()
        math = next(
            domain for domain in leaderboard["domains"] if domain["key"] == "math"
        )
        assert math["rows"][0]["cells"][0]["meta"]["task_id"] == task_id
        records = (
            await client.get("/api/eval-records", params={"task_id": task_id})
        ).json()
        assert records["records"][0]["answer"] == "2"
        context = (
            await client.get(
                "/api/eval-context",
                params={
                    "task_id": task_id,
                    "sample_index": 0,
                    "repeat_index": 0,
                    "pass_index": 0,
                },
            )
        ).json()
        assert context["context"]["stages"][0]["completion"] == "2"
        history = (
            await client.get(
                "/api/score-history",
                params={
                    "model": "rwkv7-g1g-1.5b",
                    "benchmark": "lighteval/math/gsm8k@0_test",
                },
            )
        ).json()
        assert history["total"] == 1

    assert await Task.all().count() == 1
    assert await Completion.all().count() == 1
    assert await EvalRecord.all().count() == 1
    assert await Checker.all().count() == 0
    assert await Score.all().count() == 1
    assert await EvaluationPublication.all().count() == 1
    score = await Score.get(task_id=task_id)
    assert score.metrics == {"exact_match": 1.0, "secondary_metric": 0.5}


async def test_publication_rejects_inconsistent_artifact_evidence(
    database_settings: DatabaseSettings,
) -> None:
    app = _app(database_settings)
    payload = _publication_payload()
    headers = {
        "Authorization": "Bearer publisher-token",
        "Idempotency-Key": f"publish:{payload['manifest']['digest']}",
    }
    cases = []

    wrong_digest = json.loads(json.dumps(payload))
    wrong_digest["manifest"]["identity_digest"] = "0" * 64
    cases.append(wrong_digest)

    wrong_aggregate = json.loads(json.dumps(payload))
    wrong_aggregate["metrics"]["exact_match"] = 0.0
    cases.append(wrong_aggregate)

    wrong_native_aggregate = json.loads(json.dumps(payload))
    wrong_native_aggregate["native_metrics"]["exact_match"] = 0.0
    cases.append(wrong_native_aggregate)

    wrong_accounting = json.loads(json.dumps(payload))
    wrong_accounting["accounting"]["selected"] = 2
    cases.append(wrong_accounting)

    naive_completion_time = json.loads(json.dumps(payload))
    naive_completion_time["manifest"]["completed_at"] = "2026-07-15T12:00:00"
    cases.append(naive_completion_time)

    dirty_official = json.loads(json.dumps(payload))
    dirty_official["identity"]["evaluator"]["dirty"] = True
    cases.append(dirty_official)

    unverified_official = json.loads(json.dumps(payload))
    unverified_official["identity"]["provider"]["attestation_verified"] = False
    cases.append(unverified_official)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        responses = await asyncio.gather(
            *(
                client.put(
                    "/api/v1/evaluation-publications/invalid-run",
                    headers=headers,
                    json=case,
                )
                for case in cases
            )
        )
        oversized_response = await client.put(
            "/api/v1/evaluation-publications/oversized-run",
            headers={
                **headers,
                "Content-Length": str(MAX_PUBLICATION_TRANSFER_BYTES + 1),
            },
            content=b"{}",
        )
    assert [response.status_code for response in responses] == [
        422,
        422,
        422,
        422,
        422,
        422,
        422,
    ]
    assert oversized_response.status_code == 413
    assert await Task.all().count() == 0
    assert await EvaluationPublication.all().count() == 0


async def test_nonbinary_proxy_stays_out_of_official_and_enters_non_official_scope(
    database_settings: DatabaseSettings,
) -> None:
    app = _app(database_settings)
    payload = _publication_payload(binary=False, official=False)
    sample = payload["samples"][0]
    sample["raw_completion"] = "unfinished reasoning"
    sample["scored_completion"] = "unfinished reasoning\nTherefore..."
    sample["generation"].update(
        {
            "output_token_count": 256,
            "finish_reason": "length",
            "stop_reason": None,
            "terminal_reason": "length",
            "truncated": True,
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 256,
                "total_tokens": 258,
            },
        }
    )
    sample["scoring"]["repair_action"] = "append-think-and-therefore"
    payload["truncated_samples"] = 1
    headers = {
        "Authorization": "Bearer publisher-token",
        "Idempotency-Key": f"publish:{payload['manifest']['digest']}",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.put(
            "/api/v1/evaluation-publications/proxy-run", headers=headers, json=payload
        )
        assert response.status_code == 201
        official_meta = (await client.get("/api/meta")).json()
        official_leaderboard = (
            await client.get(
                "/api/leaderboard",
                params={"model": "rwkv7-g1g-1.5b", "view": "benchmark_detail_latest"},
            )
        ).json()
        non_official_meta = (
            await client.get("/api/meta", params={"scope": "non_official"})
        ).json()
        non_official_leaderboard = (
            await client.get(
                "/api/leaderboard",
                params={
                    "model": "rwkv7-g1g-1.5b",
                    "view": "benchmark_detail_latest",
                    "scope": "non_official",
                },
            )
        ).json()
        non_official_refresh = (
            await client.post("/api/refresh", params={"scope": "non_official"})
        ).json()
        task_id = response.json()["task_id"]
        context = (
            await client.get(
                "/api/eval-context",
                params={
                    "task_id": task_id,
                    "sample_index": 0,
                    "repeat_index": 0,
                    "pass_index": 0,
                },
            )
        ).json()
        records = (
            await client.get("/api/eval-records", params={"task_id": task_id})
        ).json()
        official_options = (await client.get("/api/score-history/options")).json()
        non_official_options = (
            await client.get(
                "/api/score-history/options", params={"scope": "non_official"}
            )
        ).json()
        official_history = (
            await client.get(
                "/api/score-history",
                params={
                    "model": "rwkv7-g1g-1.5b",
                    "benchmark": "lighteval/math/gsm8k@0_test",
                },
            )
        ).json()
        non_official_history = (
            await client.get(
                "/api/score-history",
                params={
                    "model": "rwkv7-g1g-1.5b",
                    "benchmark": "lighteval/math/gsm8k@0_test",
                    "scope": "non_official",
                },
            )
        ).json()
        detail = (
            await client.get("/api/score-history/detail", params={"task_id": task_id})
        ).json()
        invalid_scope = await client.get(
            "/api/score-history/options", params={"scope": "everything"}
        )
        invalid_meta_scope = await client.get(
            "/api/meta", params={"scope": "everything"}
        )
        invalid_leaderboard_scope = await client.get(
            "/api/leaderboard", params={"scope": "everything"}
        )
        invalid_refresh_scope = await client.post(
            "/api/refresh", params={"scope": "everything"}
        )
    assert await EvalRecord.all().count() == 0
    assert records["records"][0]["is_passed"] is None
    assert records["records"][0]["answer"] == "unfinished reasoning\nTherefore..."
    assert context["context"]["evidence"]["metrics"] == {"exact_match": 0.75}
    assert context["context"]["evidence"]["generation"]["truncated"] is True
    assert (
        context["context"]["evidence"]["scoring"]["repair_action"]
        == "append-think-and-therefore"
    )
    assert official_meta["scope"] == "official"
    assert official_meta["entry_count"] == 0
    assert official_leaderboard["scope"] == "official"
    assert all(not domain["rows"] for domain in official_leaderboard["domains"])
    assert non_official_meta["scope"] == "non_official"
    assert non_official_meta["entry_count"] == 1
    assert non_official_meta["models"] == ["rwkv7-g1g-1.5b"]
    assert non_official_leaderboard["scope"] == "non_official"
    math_domain = next(
        domain
        for domain in non_official_leaderboard["domains"]
        if domain["key"] == "math"
    )
    cell = math_domain["rows"][0]["cells"][0]
    assert cell["percent"] == 75.0
    assert cell["meta"]["task_id"] == response.json()["task_id"]
    assert cell["meta"]["visibility"] == "non_official"
    assert cell["meta"]["eligibility"] == "proxy"
    assert cell["meta"]["comparable"] is False
    assert cell["meta"]["dirty"] is False
    assert non_official_refresh == {
        "scope": "non_official",
        "entry_count": 1,
        "errors": [],
    }
    assert official_options["pairs"] == []
    assert non_official_options["scope"] == "non_official"
    assert non_official_options["pairs"] == [
        {
            "model": "rwkv7-g1g-1.5b",
            "dataset": "lighteval/math/gsm8k@0_test",
        }
    ]
    assert official_history["total"] == 0
    point = non_official_history["groups"][0]["points"][0]
    assert point["visibility"] == "non_official"
    assert point["eligibility"] == "proxy"
    assert detail["visibility"] == "non_official"
    assert detail["eligibility"] == "proxy"
    assert detail["generated_samples"] == 1
    assert detail["truncated_samples"] == 1
    assert detail["truncation_rate"] == 1.0
    assert detail["accounting"]["generated_samples"] == 1
    assert detail["accounting"]["truncated_samples"] == 1
    assert detail["metrics"] == {"exact_match": 0.75, "secondary_metric": 0.5}
    assert invalid_scope.status_code == 422
    assert invalid_meta_scope.status_code == 422
    assert invalid_leaderboard_scope.status_code == 422
    assert invalid_refresh_scope.status_code == 422


async def test_concurrent_replay_creates_one_projection(
    database_settings: DatabaseSettings,
) -> None:
    app = _app(database_settings)
    payload = _publication_payload()
    headers = {
        "Authorization": "Bearer publisher-token",
        "Idempotency-Key": f"publish:{payload['manifest']['digest']}",
    }
    async with (
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as first_client,
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as second_client,
    ):
        first, second = await asyncio.gather(
            first_client.put(
                "/api/v1/evaluation-publications/concurrent-run",
                headers=headers,
                json=payload,
            ),
            second_client.put(
                "/api/v1/evaluation-publications/concurrent-run",
                headers=headers,
                json=payload,
            ),
        )
    assert sorted((first.status_code, second.status_code)) == [200, 201]
    assert first.json()["task_id"] == second.json()["task_id"]
    assert await Task.all().count() == 1
    assert await Completion.all().count() == 1
    assert await Score.all().count() == 1
    assert await EvaluationPublication.all().count() == 1


async def test_publication_rolls_back_all_rows_when_score_write_fails(
    database_settings: DatabaseSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_score_create(cls, **kwargs):
        del cls, kwargs
        raise RuntimeError("injected score failure")

    monkeypatch.setattr(Score, "create", classmethod(fail_score_create))
    app = _app(database_settings)
    payload = _publication_payload()
    headers = {
        "Authorization": "Bearer publisher-token",
        "Idempotency-Key": f"publish:{payload['manifest']['digest']}",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        response = await client.put(
            "/api/v1/evaluation-publications/rollback-run",
            headers=headers,
            json=payload,
        )
    assert response.status_code == 500
    assert await Task.all().count() == 0
    assert await Completion.all().count() == 0
    assert await EvalRecord.all().count() == 0
    assert await Score.all().count() == 0
    assert await EvaluationPublication.all().count() == 0
