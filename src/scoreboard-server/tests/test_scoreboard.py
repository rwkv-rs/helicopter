from __future__ import annotations

import getpass
import gzip
import json
import os
from urllib.parse import quote
import uuid

import asyncpg
from httpx import ASGITransport, AsyncClient
import pytest
from pydantic import ValidationError

from scoreboard_server.application import create_app
from scoreboard_server.db.settings import DatabaseSettings
from scoreboard_server.dtos.api.evaluation_results import (
    CampaignCreate,
    TaskPublication,
    content_digest,
)
from scoreboard_server.routes.api import evaluation_publications as publication_routes


TOKEN = "publisher-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
OTHER_AUTH = {"Authorization": "Bearer other-publisher-token"}


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


def _expected(
    *,
    task_name: str,
    weight: str = "a" * 64,
    mode: str = "fp16",
) -> dict:
    return {
        "identity": f"{weight}:{mode}:{task_name}",
        "weight_sha256": weight,
        "weight_display_name": f"{weight[:8]}.pth",
        "wkv_mode": mode,
        "selector": task_name.split("|", 1)[0].split(":", 1)[0],
        "task_name": task_name,
        "task_version": "0",
        "module_family": task_name.split(":", 1)[0],
        "module": f"lighteval.tasks.tasks.{task_name.split(':', 1)[0]}",
        "dataset": f"dataset/{task_name}",
        "subset": "default",
        "evaluation_splits": ["test"],
        "languages": ["english"],
        "upstream_tags": ["math"],
    }


def _campaign(*, task_names: tuple[str, ...] = ("gsm8k|0",)) -> dict:
    selectors = [task_name.split("|", 1)[0] for task_name in task_names]
    return {
        "schema_version": "lighteval-campaign-v3",
        "run_key": "1" * 64,
        "config_digest": "2" * 64,
        "registry_digest": "3" * 64,
        "eval_contract_digest": "5" * 64,
        "lighteval_version": "0.13.0",
        "configured_selectors": list(selectors),
        "resolved_selectors": list(selectors),
        "skipped_selectors": [],
        "expected_tasks": [
            _expected(task_name=task, weight=weight, mode=mode)
            for weight in ("a" * 64, "b" * 64)
            for task in task_names
            for mode in ("fp16", "fp32io16")
        ],
    }


def _publication(campaign_id: str, task: dict) -> dict:
    return {
        "schema_version": "lighteval-task-v2",
        "campaign_id": campaign_id,
        "task": task,
        "artifact": {
            "lighteval_version": "0.13.0",
            "results_path": "results/model/results_stamp.json",
            "details_paths": ["details/model/stamp/details_task_stamp.parquet"],
        },
        "task_config": {
            "generation_size": 8192,
            "original_num_docs": 2,
            "effective_num_docs": 2,
            "skipped_multiselect_docs": 0,
        },
        "model": {
            "weight_sha256": task["weight_sha256"],
            "weight_display_name": task["weight_display_name"],
            "wkv_mode": task["wkv_mode"],
            "prompt_template": "assistant",
            "gemm_policy": (
                "fp16-accumulation"
                if task["wkv_mode"] == "fp16"
                else "fp32-accumulation"
            ),
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "max_num_seqs": 2560,
            "max_num_batched_tokens": 2560,
            "dependency_versions": {
                "lighteval": "0.13.0",
                "vllm": "0.13.0",
                "torch": "2.10.0",
            },
        },
        "sampling_config": {
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
        "primary_metric": "exact_match",
        "aggregates": {"exact_match": 0.5, "exact_match_stderr": 0.01},
        "diagnostics": {
            "samples": 2,
            "completions": 3,
            "truncated": 0,
            "non_truncated": 3,
            "truncation_rate": 0,
            "turn_boundary_violations": 1,
            "turn_boundary_violation_rate": 1 / 3,
        },
        "details": [
            {
                "sample_index": 0,
                "document_index": 0,
                "doc": {
                    "id": "sample-0",
                    "task_name": task["task_name"],
                    "query": "What is 1 + 1?",
                    "choices": ["2"],
                    "gold_index": 0,
                    "specific": {"helicopter_document_index": 0},
                },
                "metric": {"exact_match": 1},
                "model_response": {
                    "input": "What is 1 + 1?",
                    "input_tokens": [10, 11],
                    "text": [
                        "<think>x</think>2",
                        "<think>y</think>2\nUser: bad",
                    ],
                    "text_post_processed": ["2", "2"],
                    "output_tokens": [[1, 2], [3, 4]],
                },
            },
            {
                "sample_index": 1,
                "document_index": 1,
                "doc": {
                    "id": "sample-1",
                    "task_name": task["task_name"],
                    "query": "What is 2 + 2?",
                    "choices": ["4"],
                    "gold_index": 0,
                    "specific": {"helicopter_document_index": 1},
                },
                "metric": {"exact_match": 0},
                "model_response": {
                    "input": "What is 2 + 2?",
                    "input_tokens": [20, 21],
                    "text": ["<think>x</think>5"],
                    "text_post_processed": ["5"],
                    "output_tokens": [[5, 6, 7, 8]],
                },
            },
        ],
    }


def _body(payload: dict) -> bytes:
    return gzip.compress(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    )


def _campaign_headers(campaign: dict) -> dict[str, str]:
    return {
        **AUTH,
        "Content-Encoding": "gzip",
        "Content-Type": "application/json",
        "Idempotency-Key": f"campaign:{campaign['run_key']}",
    }


def _task_headers(payload: dict) -> dict[str, str]:
    return {
        **AUTH,
        "Content-Encoding": "gzip",
        "Content-Type": "application/json",
        "Idempotency-Key": f"publish:{content_digest(payload)}",
    }


async def test_publication_rejects_non_standard_json_constants() -> None:
    app = create_app(publication_tokens={TOKEN: "test-publisher"})
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/evaluation-campaigns",
            headers={
                **AUTH,
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
                "Idempotency-Key": "campaign:" + "1" * 64,
            },
            content=gzip.compress(
                '{"schema_version":"lighteval-campaign-v3","unexpected":NaN}'.encode()
            ),
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid JSON body"


def test_campaign_contract_requires_same_tasks_and_both_modes_per_weight() -> None:
    valid = _campaign(task_names=("gsm8k|0", "aime24|0"))
    assert len(CampaignCreate.model_validate(valid).expected_tasks) == 8

    missing_mode = _campaign(task_names=("gsm8k|0", "aime24|0"))
    missing_mode["expected_tasks"].pop()
    with pytest.raises(ValidationError, match="both WKV modes"):
        CampaignCreate.model_validate(missing_mode)

    missing_task = _campaign(task_names=("gsm8k|0", "aime24|0"))
    missing_task["expected_tasks"] = [
        task
        for task in missing_task["expected_tasks"]
        if not (task["weight_sha256"] == "b" * 64 and task["task_name"] == "aime24|0")
    ]
    with pytest.raises(ValidationError, match="same task set"):
        CampaignCreate.model_validate(missing_task)

    invalid_selector_status = _campaign(task_names=("gsm8k|0", "aime24|0"))
    invalid_selector_status["skipped_selectors"] = ["gsm8k"]
    with pytest.raises(ValidationError, match="disjoint"):
        CampaignCreate.model_validate(invalid_selector_status)

    missing_resolved_tasks = _campaign(task_names=("gsm8k|0", "aime24|0"))
    missing_resolved_tasks["configured_selectors"].append("empty")
    missing_resolved_tasks["resolved_selectors"].append("empty")
    with pytest.raises(ValidationError, match="resolved selector"):
        CampaignCreate.model_validate(missing_resolved_tasks)


def test_campaign_contract_requires_stable_task_and_weight_metadata() -> None:
    inconsistent_task = _campaign()
    inconsistent_task["expected_tasks"][-1]["dataset"] = "dataset/forged"
    with pytest.raises(ValidationError, match="task metadata differs"):
        CampaignCreate.model_validate(inconsistent_task)

    inconsistent_weight = _campaign()
    inconsistent_weight["expected_tasks"][0]["weight_display_name"] = "forged.pth"
    with pytest.raises(ValidationError, match="weight display name differs"):
        CampaignCreate.model_validate(inconsistent_weight)

    untrimmed = _campaign()
    untrimmed["expected_tasks"][0]["module_family"] = " gsm8k"
    with pytest.raises(ValidationError, match="trimmed string"):
        CampaignCreate.model_validate(untrimmed)


def test_contract_rejects_inconsistent_evaluation_evidence() -> None:
    task = _expected(task_name="gsm8k|0")
    payload = _publication(str(uuid.uuid4()), task)
    payload["diagnostics"]["truncated"] = 1
    with pytest.raises(ValidationError, match="diagnostics do not match"):
        TaskPublication.model_validate(payload)

    payload = _publication(str(uuid.uuid4()), task)
    payload["sampling_config"]["temperature"] = 0.7
    with pytest.raises(ValidationError, match="sampling config"):
        TaskPublication.model_validate(payload)

    fp32_task = _expected(task_name="gsm8k|0", mode="fp32io16")
    payload = _publication(str(uuid.uuid4()), fp32_task)
    payload["model"]["gemm_policy"] = "fp16-accumulation"
    with pytest.raises(ValidationError, match="GEMM policy"):
        TaskPublication.model_validate(payload)

    payload = _publication(str(uuid.uuid4()), task)
    payload["artifact"]["results_path"] = "../../results.json"
    with pytest.raises(ValidationError, match="normalized relative paths"):
        TaskPublication.model_validate(payload)

    payload = _publication(str(uuid.uuid4()), task)
    payload["task_config"]["original_num_docs"] = 0
    payload["task_config"]["effective_num_docs"] = 0
    payload["details"] = []
    payload["diagnostics"] = {
        "samples": 0,
        "completions": 0,
        "truncated": 0,
        "non_truncated": 0,
        "truncation_rate": 0,
        "turn_boundary_violations": 0,
        "turn_boundary_violation_rate": 0,
    }
    with pytest.raises(ValidationError, match="full evaluation split"):
        TaskPublication.model_validate(payload)


@pytest.mark.parametrize(
    ("prompt_template", "stop"),
    [
        ("bot", "✿"),
        ("assistant", "\nUser:"),
        ("function_calling", "\n### User"),
    ],
)
def test_contract_validates_each_prompt_template_stop(
    prompt_template: str,
    stop: str,
) -> None:
    payload = _publication(str(uuid.uuid4()), _expected(task_name="gsm8k|0"))
    payload["model"]["prompt_template"] = prompt_template
    payload["sampling_config"]["stop"] = [stop]
    payload["details"][0]["model_response"]["text"][1] = f"<think>y</think>2{stop} bad"

    publication = TaskPublication.model_validate(payload)

    assert publication.model.prompt_template == prompt_template
    assert publication.diagnostics.turn_boundary_violations == 1


def test_contract_accepts_logprob_rows_with_output_token_evidence() -> None:
    task = _expected(task_name="gsm8k|0")
    payload = _publication(str(uuid.uuid4()), task)
    payload["details"][0]["model_response"] = {
        "input": "What is 1 + 1?",
        "input_tokens": [10, 11],
        "text": [],
        "text_post_processed": None,
        "output_tokens": [[1], [2]],
        "logprobs": [-0.1, -0.2],
        "argmax_logits_eq_gold": [True, False],
    }
    payload["details"][1]["model_response"] = {
        "input": "What is 2 + 2?",
        "input_tokens": [20, 21],
        "text": [],
        "output_tokens": [[3]],
        "logprobs": [-0.3],
    }
    payload["diagnostics"] = {
        "samples": 2,
        "completions": 0,
        "truncated": 0,
        "non_truncated": 0,
        "truncation_rate": 0,
        "turn_boundary_violations": 0,
        "turn_boundary_violation_rate": 0,
    }

    publication = TaskPublication.model_validate(payload)

    assert publication.diagnostics.completions == 0
    assert publication.details[0].model_response["output_tokens"] == [
        [1],
        [2],
    ]


def test_contract_rejects_misaligned_completion_and_logprob_evidence() -> None:
    task = _expected(task_name="gsm8k|0")
    payload = _publication(str(uuid.uuid4()), task)
    payload["details"][0]["model_response"]["text_post_processed"] = ["2"]
    with pytest.raises(
        ValidationError,
        match="text_post_processed must align",
    ):
        TaskPublication.model_validate(payload)

    payload = _publication(str(uuid.uuid4()), task)
    payload["details"][0]["model_response"] = {
        "input": "What is 1 + 1?",
        "input_tokens": [10, 11],
        "text": [],
        "output_tokens": [[1]],
        "logprobs": [-0.1, -0.2],
    }
    with pytest.raises(
        ValidationError,
        match="log-likelihood evidence and output-token counts differ",
    ):
        TaskPublication.model_validate(payload)

    payload = _publication(str(uuid.uuid4()), task)
    payload["details"][0]["model_response"] = {
        "input": "What is 1 + 1?",
        "input_tokens": [10, 11],
        "text": [],
        "logprobs": [-0.1, -0.2],
    }
    with pytest.raises(
        ValidationError,
        match="output_tokens must be a non-empty array",
    ):
        TaskPublication.model_validate(payload)

    payload = _publication(str(uuid.uuid4()), task)
    payload["details"][0]["model_response"] = {
        "input": "What is 1 + 1?",
        "input_tokens": [10, 11],
        "text": [],
        "output_tokens": [[]],
        "logprobs": [-0.1],
    }
    with pytest.raises(
        ValidationError,
        match="output token groups must be non-empty",
    ):
        TaskPublication.model_validate(payload)


async def test_publication_transport_enforces_gzip_size_and_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(publication_tokens={TOKEN: "ci"})
    monkeypatch.setattr(publication_routes, "MAX_COMPRESSED_BYTES", 8)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        oversized = await client.post(
            "/api/v1/evaluation-campaigns",
            content=gzip.compress(b'{"value":"long enough"}'),
            headers={**AUTH, "Content-Encoding": "gzip"},
        )
        assert oversized.status_code == 413

    monkeypatch.setattr(publication_routes, "MAX_COMPRESSED_BYTES", 1024)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        invalid = await client.post(
            "/api/v1/evaluation-campaigns",
            content=b"not-gzip",
            headers={
                **AUTH,
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
        )
        assert invalid.status_code == 400
        identity = await client.post(
            "/api/v1/evaluation-campaigns",
            content=b"{}",
            headers={**AUTH, "Content-Type": "application/json"},
        )
        assert identity.status_code == 415
        trailing = await client.post(
            "/api/v1/evaluation-campaigns",
            content=gzip.compress(b"{}") + b"trailing",
            headers={
                **AUTH,
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
        )
        assert trailing.status_code == 413


async def test_unauthorized_publication_is_rejected_before_body_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = False

    async def unexpected_decode(_request):
        nonlocal decoded
        decoded = True
        raise AssertionError("unauthorized request body was decoded")

    monkeypatch.setattr(
        publication_routes,
        "_publication_json",
        unexpected_decode,
    )
    app = create_app(publication_tokens={TOKEN: "ci"})
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/evaluation-campaigns",
            content=b"not-gzip",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 401
    assert decoded is False


async def test_campaign_publication_finalize_and_complete_queries(
    database_settings: DatabaseSettings,
) -> None:
    app = create_app(
        database_settings,
        publication_tokens={
            TOKEN: "lighteval-production",
            "other-publisher-token": "other-worker",
        },
    )
    await app.state.database.start()
    campaign = _campaign()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            assert (
                await client.get("/api/v1/evaluation-publication-preflight")
            ).status_code == 401
            preflight = await client.get(
                "/api/v1/evaluation-publication-preflight",
                headers=AUTH,
            )
            assert preflight.json() == {
                "status": "ready",
                "publisher_principal": "lighteval-production",
                "schema_version": "lighteval-campaign-v3",
                "lighteval_version": "0.13.0",
            }
            unauthorized = await client.post(
                "/api/v1/evaluation-campaigns",
                content=_body(campaign),
                headers={
                    key: value
                    for key, value in _campaign_headers(campaign).items()
                    if key != "Authorization"
                },
            )
            assert unauthorized.status_code == 401

            created = await client.post(
                "/api/v1/evaluation-campaigns",
                content=_body(campaign),
                headers=_campaign_headers(campaign),
            )
            assert created.status_code == 201
            campaign_id = created.json()["campaign_id"]
            assert created.json()["acknowledged_task_digests"] == {}
            hidden_from_other = await client.get(
                f"/api/v1/evaluation-campaigns/{campaign_id}",
                headers=OTHER_AUTH,
            )
            assert hidden_from_other.status_code == 404

            unchanged = await client.post(
                "/api/v1/evaluation-campaigns",
                content=_body(campaign),
                headers=_campaign_headers(campaign),
            )
            assert unchanged.status_code == 200
            assert unchanged.json()["disposition"] == "unchanged"
            assert unchanged.json()["campaign_id"] == campaign_id

            assert (await client.get("/api/evaluations")).json()["evaluations"] == []
            incomplete = await client.post(
                f"/api/v1/evaluation-campaigns/{campaign_id}/finalize",
                headers={
                    **AUTH,
                    "Idempotency-Key": f"finalize:{campaign_id}",
                },
            )
            assert incomplete.status_code == 409
            assert len(incomplete.json()["detail"]["missing"]) == 4

            receipts = []
            for task in campaign["expected_tasks"]:
                payload = _publication(campaign_id, task)
                path = (
                    f"/api/v1/evaluation-campaigns/{campaign_id}/tasks/"
                    f"{quote(task['identity'], safe='')}"
                )
                published = await client.put(
                    path,
                    content=_body(payload),
                    headers=_task_headers(payload),
                )
                assert published.status_code == 201
                receipts.append(published.json())
                replay = await client.put(
                    path,
                    content=_body(payload),
                    headers=_task_headers(payload),
                )
                assert replay.status_code == 200
                assert replay.json()["disposition"] == "unchanged"

            changed = _publication(campaign_id, campaign["expected_tasks"][0])
            changed["aggregates"]["exact_match"] = 0.25
            conflict = await client.put(
                (
                    f"/api/v1/evaluation-campaigns/{campaign_id}/tasks/"
                    f"{quote(changed['task']['identity'], safe='')}"
                ),
                content=_body(changed),
                headers=_task_headers(changed),
            )
            assert conflict.status_code == 409

            status = (
                await client.get(
                    f"/api/v1/evaluation-campaigns/{campaign_id}",
                    headers=AUTH,
                )
            ).json()
            assert status["missing_task_identities"] == []
            assert len(status["acknowledged_task_digests"]) == 4
            assert (await client.get("/api/evaluations")).json()["evaluations"] == []

            finalized = await client.post(
                f"/api/v1/evaluation-campaigns/{campaign_id}/finalize",
                headers={
                    **AUTH,
                    "Idempotency-Key": f"finalize:{campaign_id}",
                },
            )
            assert finalized.status_code == 200
            assert finalized.json()["task_count"] == 4

            evaluations = (await client.get("/api/evaluations")).json()
            assert len(evaluations["evaluations"]) == 4
            assert evaluations["total"] == 4
            assert evaluations["next_offset"] is None
            first_page = (
                await client.get(
                    "/api/evaluations",
                    params={"offset": 0, "limit": 2},
                )
            ).json()
            assert len(first_page["evaluations"]) == 2
            assert first_page["next_offset"] == 2
            second_page = (
                await client.get(
                    "/api/evaluations",
                    params={
                        "offset": first_page["next_offset"],
                        "limit": 2,
                        "completed_before": first_page["generated_at"],
                    },
                )
            ).json()
            assert len(second_page["evaluations"]) == 2
            assert second_page["next_offset"] is None
            assert {
                item["evaluation_id"]
                for item in first_page["evaluations"] + second_page["evaluations"]
            } == {item["evaluation_id"] for item in evaluations["evaluations"]}
            summary = evaluations["evaluations"][0]
            assert (
                summary["provenance"]["publisher_principal"] == "lighteval-production"
            )
            assert summary["provenance"]["configured_selectors"] == ["gsm8k"]
            assert summary["provenance"]["resolved_selectors"] == ["gsm8k"]
            assert summary["provenance"]["skipped_selectors"] == []
            assert summary["task"]["selector"] == "gsm8k"
            assert summary["task"]["weight_sha256"] in {"a" * 64, "b" * 64}
            assert summary["model"]["wkv_mode"] in {"fp16", "fp32io16"}
            assert summary["aggregates"]["exact_match_stderr"] == 0.01

            evaluation_id = receipts[0]["evaluation_id"]
            page = (
                await client.get(
                    f"/api/evaluations/{evaluation_id}/samples",
                    params={"offset": 0, "limit": 1},
                )
            ).json()
            assert page["total"] == 2
            assert page["next_offset"] == 1
            assert len(page["items"][0]["model_response"]["text"]) == 2
            second_page = (
                await client.get(
                    f"/api/evaluations/{evaluation_id}/samples",
                    params={"offset": 1, "limit": 1},
                )
            ).json()
            assert second_page["next_offset"] is None
            assert second_page["items"][0]["sample_index"] == 1

            new_run = {**campaign, "run_key": "9" * 64}
            next_campaign = await client.post(
                "/api/v1/evaluation-campaigns",
                content=_body(new_run),
                headers=_campaign_headers(new_run),
            )
            assert next_campaign.status_code == 201
            assert next_campaign.json()["campaign_id"] != campaign_id

            assert (
                await client.put(
                    "/api/v1/evaluation-publications/legacy",
                    content=b"{}",
                )
            ).status_code == 404
    finally:
        await app.state.database.stop()


async def test_server_refuses_legacy_or_unversioned_evaluation_schema(
    database_settings: DatabaseSettings,
) -> None:
    connection = await asyncpg.connect(
        host=database_settings.host,
        port=database_settings.port,
        user=database_settings.user,
        password=database_settings.password,
        database=database_settings.database,
    )
    try:
        await connection.execute("CREATE TABLE evaluation_campaign (id integer)")
    finally:
        await connection.close()

    app = create_app(database_settings, publication_tokens={TOKEN: "ci"})
    with pytest.raises(asyncpg.PostgresError, match="schema detected"):
        await app.state.database.start()
    assert app.state.database.pool is None
