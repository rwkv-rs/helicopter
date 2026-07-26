from __future__ import annotations

import getpass
import gzip
import json
import os
from pathlib import Path
from urllib.parse import quote
import uuid

import asyncpg
from httpx import ASGITransport, AsyncClient
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import pytest_asyncio

from helicopter_eval import artifacts, campaign
from helicopter_eval.config import EvaluationConfig, WeightIdentity
from helicopter_eval.plan import build_plan
from helicopter_eval.registry import RegistrySnapshot, RegistryTask
from scoreboard_server.application import create_app
from scoreboard_server.db.settings import DatabaseSettings


ROOT = Path(__file__).parents[1]
TOKEN = "shared-fixture-publisher-token"
AUTHORIZATION = {"Authorization": f"Bearer {TOKEN}"}


def _maintenance_kwargs() -> dict[str, str]:
    return {
        "user": os.environ.get("PGUSER") or getpass.getuser(),
        "host": os.environ.get("PGHOST") or "/var/run/postgresql",
        "database": os.environ.get("PGDATABASE") or "postgres",
    }


@pytest_asyncio.fixture()
async def e2e_database() -> DatabaseSettings:
    database = f"helicopter_eval_e2e_{uuid.uuid4().hex[:12]}"
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


def _gzip(value: object) -> bytes:
    return gzip.compress(artifacts.canonical_json(value))


def _publication_headers(digest: str) -> dict[str, str]:
    return {
        **AUTHORIZATION,
        "Content-Encoding": "gzip",
        "Content-Type": "application/json",
        "Idempotency-Key": f"publish:{digest}",
    }


@pytest.mark.asyncio
async def test_standard_artifact_to_database_query_contract(
    tmp_path: Path,
    e2e_database: DatabaseSettings,
) -> None:
    fixture = json.loads(
        (ROOT / "fixtures/lighteval_e2e.json").read_text(encoding="utf-8")
    )
    task = RegistryTask(**fixture["registry_task"])
    registry = RegistrySnapshot(
        lighteval_version="0.13.0",
        tasks=(task,),
        module_count=1,
        digest="b" * 64,
        domain_rules_version="shared-fixture",
        domain_rules_digest="c" * 64,
        unknown_domain_modules=(),
    )
    weight_path = tmp_path / fixture["model_execution"]["weight_display_name"]
    weight_path.write_bytes(b"shared fixture does not load a model")
    weight = WeightIdentity(
        configured_path=weight_path.name,
        path=weight_path,
        display_name=fixture["model_execution"]["weight_display_name"],
        sha256=fixture["model_execution"]["weight_sha256"],
    )
    plan = build_plan(
        EvaluationConfig(schema_version=1, weights=(weight_path.name,)),
        (weight,),
        registry,
    )
    resume_key = campaign._resume_key(plan)
    create_payload = campaign._campaign_payload(plan, resume_key)
    app = create_app(e2e_database, publication_tokens={TOKEN: "shared-fixture"})
    await app.state.database.start()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            created = await client.post(
                "/api/v1/evaluation-campaigns",
                content=_gzip(create_payload),
                headers={
                    **AUTHORIZATION,
                    "Content-Encoding": "gzip",
                    "Content-Type": "application/json",
                    "Idempotency-Key": f"campaign:{resume_key}",
                },
            )
            assert created.status_code == 201
            campaign_id = created.json()["campaign_id"]

            evaluation_ids: list[str] = []
            for unit in plan.units:
                shard = unit.shards[0]
                shard_dir = tmp_path / unit.wkv_mode
                result_path = shard_dir / "results/model/results_shared-fixture.json"
                detail_path = (
                    shard_dir
                    / "details/model/shared-fixture"
                    / "details_gsm8k|0_shared-fixture.parquet"
                )
                result_path.parent.mkdir(parents=True)
                detail_path.parent.mkdir(parents=True)
                result_path.write_text(
                    json.dumps(fixture["standard_results"]),
                    encoding="utf-8",
                )
                parquet.write_table(
                    pa.Table.from_pylist(fixture["standard_rows"]),
                    detail_path,
                )
                model_execution = {
                    **fixture["model_execution"],
                    "wkv_mode": unit.wkv_mode,
                    "gemm_policy": (
                        "fp16-accumulation"
                        if unit.wkv_mode == "fp16"
                        else "fp32-accumulation"
                    ),
                }
                publications = artifacts.publications_from_shard(
                    shard_dir=shard_dir,
                    campaign_id=campaign_id,
                    unit=unit,
                    shard=shard,
                    model_execution=model_execution,
                    registry_tasks=plan.registry.tasks,
                )
                assert len(publications) == 1
                identity, publication, digest = publications[0]
                response = await client.put(
                    (
                        f"/api/v1/evaluation-campaigns/{campaign_id}/tasks/"
                        f"{quote(identity, safe='')}"
                    ),
                    content=_gzip(publication),
                    headers=_publication_headers(digest),
                )
                assert response.status_code == 201
                evaluation_ids.append(response.json()["evaluation_id"])

            finalized = await client.post(
                f"/api/v1/evaluation-campaigns/{campaign_id}/finalize",
                headers={
                    **AUTHORIZATION,
                    "Idempotency-Key": f"finalize:{campaign_id}",
                },
            )
            assert finalized.status_code == 200
            summaries = (await client.get("/api/evaluations")).json()
            assert summaries["total"] == 2
            assert {item["model"]["wkv_mode"] for item in summaries["evaluations"]} == {
                "fp16",
                "fp32io16",
            }
            assert all(
                item["aggregates"]
                == {
                    "exact_match": 1.0,
                    "exact_match_stderr": 0.0,
                }
                for item in summaries["evaluations"]
            )
            sample_page = (
                await client.get(f"/api/evaluations/{evaluation_ids[0]}/samples")
            ).json()
            assert sample_page["total"] == 2
            model_response = sample_page["items"][0]["model_response"]
            for key, value in fixture["standard_rows"][0]["model_response"].items():
                assert model_response[key] == value
            assert model_response["logprobs"] is None
            assert model_response["argmax_logits_eq_gold"] is None
            logprob_response = sample_page["items"][1]["model_response"]
            for key, value in fixture["standard_rows"][1]["model_response"].items():
                assert logprob_response[key] == value
            assert logprob_response["reasonings"] is None
            assert logprob_response["text"] == []
            assert logprob_response["logprobs"] == [-0.1, -0.2]
    finally:
        await app.state.database.stop()
