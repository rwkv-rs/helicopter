from __future__ import annotations

import getpass
import gzip
import json
import os
from pathlib import Path
import uuid

import asyncpg
from httpx import ASGITransport, AsyncClient
import pytest

from scoreboard_server.application import create_app
from scoreboard_server.db.settings import DatabaseSettings
from scoreboard_server.dtos.api.evaluation_results import (
    EvaluationPublication,
    StandardDetail,
    content_digest,
    sample_outcome,
)


def _maintenance_kwargs() -> dict[str, str]:
    return {
        "user": os.environ.get("PGUSER") or getpass.getuser(),
        "host": os.environ.get("PGHOST") or "/var/run/postgresql",
        "database": os.environ.get("PGDATABASE") or "postgres",
    }


@pytest.fixture()
async def database_settings() -> DatabaseSettings:
    database = f"helicopter_scoreboard_test_{uuid.uuid4().hex[:12]}"
    kwargs = _maintenance_kwargs()
    connection = await asyncpg.connect(**kwargs)
    try:
        await connection.execute(f'CREATE DATABASE "{database}"')
    finally:
        await connection.close()
    settings = DatabaseSettings(
        host=kwargs["host"],
        port=int(os.environ.get("PGPORT") or 5432),
        user=kwargs["user"],
        password=os.environ.get("PGPASSWORD"),
        database=database,
    )
    try:
        yield settings
    finally:
        connection = await asyncpg.connect(**kwargs)
        try:
            await connection.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = $1 AND pid <> pg_backend_pid()
                """,
                database,
            )
            await connection.execute(f'DROP DATABASE IF EXISTS "{database}"')
        finally:
            await connection.close()


def _payload() -> dict:
    rwkv = {
        "label": "RWKV G1H 1.5B",
        "architecture": "RWKV",
        "generation": "G1H",
        "parameters": "1.5B",
    }
    g1g = {**rwkv, "label": "RWKV G1G 1.5B", "generation": "G1G"}
    return {
        "schema_version": "lighteval-standard-v1",
        "source_run_id": "run-20260725",
        "artifact": {
            "lighteval_version": "0.13.0",
            "results_path": "results/model/results_stamp.json",
            "details_paths": ["details/model/stamp/details_gsm8k_stamp.parquet"],
        },
        "task_name": "gsm8k|0",
        "task_config": {"generation_size": 8192, "num_few_shot_seeds": 1},
        "model": rwkv,
        "benchmark": {
            "label": "GSM8K",
            "domain": "math",
            "evaluation_method": "cot",
            "score_multiplier": 100,
        },
        "evaluation": {
            "prompt_profile": "unified",
            "prompt_template": "User: {task.problem}\\n\\nAssistant: <think",
            "precision": "fp32io16",
        },
        "comparisons": [
            {
                "comparison": {
                    "id": "generation",
                    "label": "G1G vs G1H",
                    "short_label": "代际",
                    "a_label": "G1G",
                    "b_label": "G1H",
                    "contract": "相同 prompt、precision、sampling 与输出边界。",
                },
                "parameter_group": {
                    "id": "1.5b",
                    "label": "1.5B",
                    "a_model": g1g,
                    "b_model": rwkv,
                    "parameter_delta_percent": 0,
                    "comparable": True,
                },
                "arm": "b",
            }
        ],
        "sampling_config": {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "max_tokens": 8192,
            "seed": 42,
        },
        "primary_metric": "exact_match",
        "aggregates": {"exact_match": 0.5, "exact_match_stderr": 0.01},
        "diagnostics": {
            "samples": 2,
            "completions": 3,
            "truncated": 1,
            "non_truncated": 2,
            "truncation_rate": 1 / 3,
            "turn_boundary_violations": 0,
            "turn_boundary_violation_rate": 0,
        },
        "details": [
            {
                "doc": {
                    "id": "gsm8k-0",
                    "task_name": "gsm8k|0",
                    "query": "What is 1 + 1?",
                    "choices": ["2"],
                    "gold_index": 0,
                },
                "metric": {"exact_match": 1},
                "model_response": {
                    "input": "What is 1 + 1?",
                    "text": ["<think>x</think>2", "<think>y</think>2"],
                    "text_post_processed": ["2", "2"],
                    "output_tokens": [[1, 2], [3, 4]],
                },
            },
            {
                "doc": {
                    "id": "gsm8k-1",
                    "task_name": "gsm8k|0",
                    "query": "What is 2 + 2?",
                    "choices": ["4"],
                    "gold_index": 0,
                },
                "metric": {"exact_match": 0},
                "model_response": {
                    "input": "What is 2 + 2?",
                    "text": ["<think>x</think>5"],
                    "text_post_processed": ["5"],
                    "output_tokens": [[5] * 8192],
                },
            },
        ],
    }


async def test_publication_is_atomic_idempotent_and_queryable(
    database_settings: DatabaseSettings,
) -> None:
    app = create_app(
        database_settings, publication_tokens={"publisher-token": "lighteval-ci"}
    )
    await app.state.database.start()
    payload = _payload()
    digest = content_digest(payload)
    headers = {
        "Authorization": "Bearer publisher-token",
        "Content-Encoding": "gzip",
        "Content-Type": "application/json",
        "Idempotency-Key": f"publish:{digest}",
    }
    body = gzip.compress(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            assert (
                await client.put(
                    "/api/v1/evaluation-publications/run%3Agsm8k",
                    content=body,
                    headers={key: value for key, value in headers.items() if key != "Authorization"},
                )
            ).status_code == 401

            invalid = _payload()
            invalid["diagnostics"]["samples"] = 1
            invalid_digest = content_digest(invalid)
            rejected = await client.put(
                "/api/v1/evaluation-publications/run%3Agsm8k",
                content=gzip.compress(
                    json.dumps(invalid, separators=(",", ":")).encode()
                ),
                headers={
                    **headers,
                    "Idempotency-Key": f"publish:{invalid_digest}",
                },
            )
            assert rejected.status_code == 422
            assert (
                await app.state.database.require_pool().fetchval(
                    "SELECT count(*) FROM evaluation_result"
                )
                == 0
            )

            created = await client.put(
                "/api/v1/evaluation-publications/run%3Agsm8k",
                content=body,
                headers=headers,
            )
            assert created.status_code == 201
            receipt = created.json()
            assert receipt["disposition"] == "created"

            replay = await client.put(
                "/api/v1/evaluation-publications/run%3Agsm8k",
                content=body,
                headers=headers,
            )
            assert replay.status_code == 200
            assert replay.json() == {**receipt, "disposition": "unchanged"}

            changed = _payload()
            changed["aggregates"]["exact_match"] = 0.75
            changed_digest = content_digest(changed)
            conflict = await client.put(
                "/api/v1/evaluation-publications/run%3Agsm8k",
                content=gzip.compress(
                    json.dumps(changed, separators=(",", ":")).encode()
                ),
                headers={
                    **headers,
                    "Idempotency-Key": f"publish:{changed_digest}",
                },
            )
            assert conflict.status_code == 409

            evaluations = (await client.get("/api/evaluations")).json()
            assert len(evaluations["evaluations"]) == 1
            summary = evaluations["evaluations"][0]
            assert summary["source"] == "lighteval-ci"
            assert summary["visibility"] == "non_official"
            assert summary["aggregates"]["exact_match"] == 0.5
            assert summary["comparisons"][0]["arm"] == "b"

            samples = (
                await client.get(
                    f"/api/evaluations/{receipt['evaluation_id']}/samples",
                    params={"limit": 10},
                )
            ).json()
            assert samples["groups"]["correct"]["total"] == 1
            assert samples["groups"]["incorrect"]["total"] == 1
            assert len(
                samples["groups"]["correct"]["items"][0]["model_response"]["text"]
            ) == 2

            assert (await client.get("/api/admin/health")).status_code == 404
            assert (await client.post("/api/refresh")).status_code == 404

        pool = app.state.database.require_pool()
        assert await pool.fetchval("SELECT count(*) FROM evaluation_result") == 1
        assert await pool.fetchval("SELECT count(*) FROM evaluation_sample") == 2
    finally:
        await app.state.database.stop()


def test_contract_rejects_unknown_trust_and_classifies_generic_rows() -> None:
    payload = _payload()
    payload["official"] = True
    with pytest.raises(ValueError):
        EvaluationPublication.model_validate(payload)

    unanswered = StandardDetail(doc={}, metric={}, model_response={"text": []})
    judged = StandardDetail(
        doc={},
        metric={"judge_score": 0.6},
        model_response={"text": ["plausible answer"]},
    )
    logprob = StandardDetail(
        doc={},
        metric={},
        model_response={"text": [], "logprobs": [-0.1]},
    )
    assert sample_outcome(unanswered, "missing") == "unanswered"
    assert sample_outcome(judged, "judge_score") == "undetermined"
    assert sample_outcome(logprob, "missing") == "undetermined"


def test_server_preserves_layered_layout_without_legacy_features() -> None:
    package = Path(__file__).parents[1] / "scoreboard_server"
    assert {path.name for path in package.iterdir() if path.is_file()} == {
        "__init__.py",
        "application.py",
    }
    assert {
        path.name for path in package.iterdir() if path.is_dir() and path.name != "__pycache__"
    } == {"adapters", "cores", "db", "dtos", "routes", "services"}
    paths = {str(path.relative_to(package)) for path in package.rglob("*.py")}
    assert {
        "db/connection.py",
        "db/repository.py",
        "db/schema.py",
        "db/settings.py",
        "dtos/api/evaluation_results.py",
        "routes/api/evaluation_publications.py",
        "routes/api/evaluation_results.py",
        "routes/api/health.py",
        "services/api/evaluation_publications.py",
    } <= paths
    assert not any(
        fragment in path
        for path in paths
        for fragment in ("admin", "scheduler", "lease", "checker", "resume", "capture")
    )
