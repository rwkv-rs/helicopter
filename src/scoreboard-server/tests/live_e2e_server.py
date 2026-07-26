from __future__ import annotations

import asyncio
import getpass
import gzip
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

import asyncpg
from httpx import ASGITransport, AsyncClient
import pyarrow as pa
import pyarrow.parquet as parquet
import uvicorn

from helicopter_lighteval import evaluate, publish
from helicopter_lighteval.config import LightEvalConfig
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
    return gzip.compress(publish.canonical_json(value))


def _headers(key: str) -> dict[str, str]:
    return {
        **AUTHORIZATION,
        "Content-Encoding": "gzip",
        "Content-Type": "application/json",
        "Idempotency-Key": key,
    }


def _checked(response, expected: int) -> dict[str, object]:
    if response.status_code != expected:
        raise RuntimeError(
            f"live E2E seed failed ({response.status_code}): {response.text}"
        )
    return response.json()


def _task(fixture: dict[str, object]) -> dict[str, object]:
    source = fixture["registry_task"]
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
    result = root / "results/model/results_shared-fixture.json"
    details = (
        root / "details/model/shared-fixture" / "details_gsm8k|0_shared-fixture.parquet"
    )
    result.parent.mkdir(parents=True)
    details.parent.mkdir(parents=True)
    result.write_text(json.dumps(fixture["standard_results"]), encoding="utf-8")
    parquet.write_table(pa.Table.from_pylist(fixture["standard_rows"]), details)


class Recorder:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, dict[str, object]]] = []

    def publish_task(
        self,
        campaign_id: str,
        identity: str,
        payload: dict[str, object],
    ) -> None:
        self.items.append((campaign_id, identity, payload))


async def _seed(app, temporary_root: Path) -> None:
    fixture = json.loads(
        (REPOSITORY / "fixtures/lighteval_e2e.json").read_text(encoding="utf-8")
    )
    task = _task(fixture)
    weights: list[Path] = []
    hashes: list[str] = []
    for name, content in (
        ("small.pth", b"live-e2e-small"),
        ("large.pth", b"live-e2e-large"),
    ):
        path = temporary_root / name
        path.write_bytes(content)
        weights.append(path)
        hashes.append(hashlib.sha256(content).hexdigest())
    config = LightEvalConfig(
        prompt_template="assistant",
        weights=tuple(weights),
        weight_hashes=tuple(hashes),
        benchmarks=("gsm8k",),
        scoreboard_url="http://live-e2e",
        scoreboard_token=TOKEN,
        staging_root=temporary_root / "staging",
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

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://live-e2e",
    ) as client:
        created = await client.post(
            "/api/v1/evaluation-campaigns",
            content=_gzip(campaign),
            headers=_headers(f"campaign:{run_key}"),
        )
        campaign_id = str(_checked(created, 201)["campaign_id"])
        for weight, weight_hash in zip(weights, hashes, strict=True):
            for mode in evaluate.WKV_MODES:
                output = temporary_root / "artifacts" / weight_hash / mode
                _write_standard(output, fixture)
                recorder = Recorder()
                model = {
                    **fixture["model_execution"],
                    "weight_sha256": weight_hash,
                    "weight_display_name": weight.name,
                    "wkv_mode": mode,
                    "gemm_policy": (
                        "fp16-accumulation" if mode == "fp16" else "fp32-accumulation"
                    ),
                }
                publish.publish_results(
                    output_dir=output,
                    campaign_id=campaign_id,
                    expected_tasks=[
                        item
                        for item in expected
                        if item["weight_sha256"] == weight_hash
                        and item["wkv_mode"] == mode
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
                _, identity, payload = recorder.items[0]
                response = await client.put(
                    (
                        f"/api/v1/evaluation-campaigns/{campaign_id}/tasks/"
                        f"{quote(identity, safe='')}"
                    ),
                    content=_gzip(payload),
                    headers=_headers(f"publish:{publish.content_digest(payload)}"),
                )
                _checked(response, 201)
        finalized = await client.post(
            f"/api/v1/evaluation-campaigns/{campaign_id}/finalize",
            headers={
                **AUTHORIZATION,
                "Idempotency-Key": f"finalize:{campaign_id}",
            },
        )
        _checked(finalized, 200)


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
