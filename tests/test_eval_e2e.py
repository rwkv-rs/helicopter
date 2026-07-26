from __future__ import annotations

import getpass
import gzip
import json
import os
import uuid
from pathlib import Path
from urllib.parse import quote

import asyncpg
from httpx import ASGITransport, AsyncClient
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import pytest_asyncio

from helicopter_lighteval import evaluate, publish
from helicopter_lighteval.config import LightEvalConfig
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
    database = f"helicopter_lighteval_e2e_{uuid.uuid4().hex[:12]}"
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
    return gzip.compress(publish.canonical_json(value))


def _headers(key: str) -> dict[str, str]:
    return {
        **AUTHORIZATION,
        "Content-Encoding": "gzip",
        "Content-Type": "application/json",
        "Idempotency-Key": key,
    }


def _task(fixture: dict[str, object]) -> dict[str, object]:
    source = fixture["registry_task"]
    assert isinstance(source, dict)
    return {
        "selector": source["selector"],
        "task_name": source["identity"],
        "task_version": source["version"],
        "module_family": source["module_family"],
        "module": source["module"],
        "dataset": source["dataset"],
        "subset": source["subset"],
        "evaluation_splits": source["evaluation_splits"],
        "languages": source["languages"],
        "upstream_tags": source["upstream_tags"],
    }


def _write_standard(
    root: Path,
    fixture: dict[str, object],
) -> None:
    result_path = root / "results/model/results_shared-fixture.json"
    detail_path = (
        root / "details/model/shared-fixture" / "details_gsm8k|0_shared-fixture.parquet"
    )
    result_path.parent.mkdir(parents=True)
    detail_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(fixture["standard_results"]),
        encoding="utf-8",
    )
    rows = fixture["standard_rows"]
    assert isinstance(rows, list)
    parquet.write_table(pa.Table.from_pylist(rows), detail_path)


class RecordingClient:
    def __init__(self) -> None:
        self.publications: list[tuple[str, str, dict[str, object]]] = []

    def publish_task(
        self,
        campaign_id: str,
        identity: str,
        payload: dict[str, object],
    ) -> None:
        self.publications.append((campaign_id, identity, payload))


@pytest.mark.asyncio
async def test_standard_artifact_to_database_query_contract(
    tmp_path: Path,
    e2e_database: DatabaseSettings,
) -> None:
    fixture = json.loads(
        (ROOT / "fixtures/lighteval_e2e.json").read_text(encoding="utf-8")
    )
    task = _task(fixture)
    weight = tmp_path / "shared-fixture.pth"
    weight.write_bytes(b"fixture")
    config = LightEvalConfig(
        prompt_template="assistant",
        weights=(weight,),
        weight_hashes=("a" * 64,),
        benchmarks=("gsm8k",),
        scoreboard_url="http://testserver",
        scoreboard_token=TOKEN,
        staging_root=tmp_path / "staging",
    )
    expected = evaluate._expected_tasks(config, [task])
    campaign = evaluate._campaign_payload(
        config,
        [task],
        [],
        expected,
        "0.13.0",
    )
    run_key = str(campaign["run_key"])
    app = create_app(e2e_database, publication_tokens={TOKEN: "shared-fixture"})
    await app.state.database.start()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            created = await client.post(
                "/api/v1/evaluation-campaigns",
                content=_gzip(campaign),
                headers=_headers(f"campaign:{run_key}"),
            )
            assert created.status_code == 201
            campaign_id = created.json()["campaign_id"]
            evaluation_ids: list[str] = []

            for mode in evaluate.WKV_MODES:
                output_dir = tmp_path / mode
                _write_standard(output_dir, fixture)
                recorder = RecordingClient()
                model = {
                    **fixture["model_execution"],
                    "wkv_mode": mode,
                    "gemm_policy": (
                        "fp16-accumulation" if mode == "fp16" else "fp32-accumulation"
                    ),
                }
                publish.publish_results(
                    output_dir=output_dir,
                    campaign_id=campaign_id,
                    expected_tasks=[
                        item for item in expected if item["wkv_mode"] == mode
                    ],
                    model=model,
                    sampling_config={
                        "temperature": 0.96,
                        "top_p": 0.76,
                        "top_k": 32,
                        "presence_penalty": 1.0,
                        "frequency_penalty": 0.1,
                        "repetition_penalty": 1.0,
                        "penalty_decay": 0.988,
                        "max_new_tokens": 8192,
                        "stop": ["\nUser:"],
                        "ignore_eos": False,
                    },
                    client=recorder,
                )
                _, identity, payload = recorder.publications[0]
                digest = publish.content_digest(payload)
                response = await client.put(
                    (
                        f"/api/v1/evaluation-campaigns/{campaign_id}/tasks/"
                        f"{quote(identity, safe='')}"
                    ),
                    content=_gzip(payload),
                    headers=_headers(f"publish:{digest}"),
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
            assert finalized.json()["status"] == "complete"

            summaries = (await client.get("/api/evaluations")).json()
            assert summaries["total"] == 2
            assert {item["model"]["wkv_mode"] for item in summaries["evaluations"]} == {
                "fp16",
                "fp32io16",
            }
            sample_page = (
                await client.get(f"/api/evaluations/{evaluation_ids[0]}/samples")
            ).json()
            assert sample_page["total"] == 2
            assert sample_page["items"][0]["model_response"]["text"]
            assert sample_page["items"][1]["model_response"]["logprobs"] == [
                -0.1,
                -0.2,
            ]
    finally:
        await app.state.database.stop()
