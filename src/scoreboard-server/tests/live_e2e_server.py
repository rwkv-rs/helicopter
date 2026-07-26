from __future__ import annotations

import asyncio
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote
import uuid

import asyncpg
from httpx import ASGITransport, AsyncClient
import pyarrow as pa
import pyarrow.parquet as parquet
import uvicorn

from helicopter_eval import artifacts, campaign
from helicopter_eval.config import EvaluationConfig, WeightIdentity
from helicopter_eval.plan import build_plan
from helicopter_eval.registry import RegistrySnapshot, RegistryTask
from scoreboard_server.application import create_app
from scoreboard_server.db.settings import DatabaseSettings


REPOSITORY = Path(__file__).resolve().parents[3]
TOKEN = "playwright-live-e2e-token"
AUTHORIZATION = {"Authorization": f"Bearer {TOKEN}"}


def _maintenance_kwargs() -> dict[str, str]:
    return {
        "user": os.environ.get("PGUSER") or getpass.getuser(),
        "host": os.environ.get("PGHOST") or "/var/run/postgresql",
        "database": os.environ.get("PGDATABASE") or "postgres",
    }


def _gzip(value: object) -> bytes:
    return gzip.compress(artifacts.canonical_json(value))


def _headers(idempotency_key: str) -> dict[str, str]:
    return {
        **AUTHORIZATION,
        "Content-Encoding": "gzip",
        "Content-Type": "application/json",
        "Idempotency-Key": idempotency_key,
    }


def _response_json(response, expected_status: int) -> dict:
    if response.status_code != expected_status:
        raise RuntimeError(
            f"live E2E seed failed ({response.status_code}): {response.text}"
        )
    return response.json()


async def _seed(app, temporary_root: Path) -> None:
    fixture = json.loads(
        (REPOSITORY / "fixtures" / "lighteval_e2e.json").read_text(encoding="utf-8")
    )
    task = RegistryTask(**fixture["registry_task"])
    registry = RegistrySnapshot(
        lighteval_version="0.13.0",
        configured_selectors=(task.selector,),
        resolved_selectors=(task.selector,),
        skipped_selectors=(),
        tasks=(task,),
        module_count=1,
        digest="b" * 64,
    )
    identities: list[WeightIdentity] = []
    for name, content in (
        ("small.pth", b"live-e2e-small"),
        ("large.pth", b"live-e2e-large"),
    ):
        path = temporary_root / name
        path.write_bytes(content)
        identities.append(
            WeightIdentity(
                configured_path=name,
                path=path,
                display_name=name,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    plan = build_plan(
        EvaluationConfig(
            schema_version=1,
            weights=tuple(identity.configured_path for identity in identities),
            benchmarks=(task.selector,),
            prompt_template="assistant",
        ),
        tuple(identities),
        registry,
    )
    resume_key = campaign._resume_key(plan)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://live-e2e",
    ) as client:
        created = await client.post(
            "/api/v1/evaluation-campaigns",
            content=_gzip(campaign._campaign_payload(plan, resume_key)),
            headers=_headers(f"campaign:{resume_key}"),
        )
        campaign_id = _response_json(created, 201)["campaign_id"]
        for unit in plan.units:
            shard = unit.shards[0]
            shard_dir = (
                temporary_root / "artifacts" / unit.weight.sha256 / unit.wkv_mode
            )
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
                "weight_sha256": unit.weight.sha256,
                "weight_display_name": unit.weight.display_name,
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
            if len(publications) != 1:
                raise RuntimeError("live E2E shard did not produce one task")
            identity, publication, digest = publications[0]
            published = await client.put(
                (
                    f"/api/v1/evaluation-campaigns/{campaign_id}/tasks/"
                    f"{quote(identity, safe='')}"
                ),
                content=_gzip(publication),
                headers=_headers(f"publish:{digest}"),
            )
            _response_json(published, 201)
        finalized = await client.post(
            f"/api/v1/evaluation-campaigns/{campaign_id}/finalize",
            headers={
                **AUTHORIZATION,
                "Idempotency-Key": f"finalize:{campaign_id}",
            },
        )
        _response_json(finalized, 200)


async def _drop_database(database: str, maintenance: dict[str, str]) -> None:
    connection = await asyncpg.connect(**maintenance)
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


async def main() -> None:
    database = f"helicopter_live_e2e_{uuid.uuid4().hex[:12]}"
    maintenance = _maintenance_kwargs()
    connection = await asyncpg.connect(**maintenance)
    try:
        await connection.execute(f'CREATE DATABASE "{database}"')
    finally:
        await connection.close()
    settings = DatabaseSettings(
        host=maintenance["host"],
        port=int(os.environ.get("PGPORT") or 5432),
        user=maintenance["user"],
        password=os.environ.get("PGPASSWORD"),
        database=database,
    )
    app = create_app(settings, publication_tokens={TOKEN: "playwright-live"})
    await app.state.database.start()
    try:
        with tempfile.TemporaryDirectory(prefix="helicopter-live-e2e-") as raw:
            await _seed(app, Path(raw))
            configuration = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=int(os.environ.get("SCOREBOARD_E2E_PORT") or 7862),
                lifespan="off",
                log_level="warning",
            )
            await uvicorn.Server(configuration).serve()
    finally:
        await app.state.database.stop()
        await _drop_database(database, maintenance)


if __name__ == "__main__":
    asyncio.run(main())
