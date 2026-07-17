from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from tortoise.exceptions import IntegrityError
from tortoise.transactions import in_transaction

from scoreboard_server.cores.normalize import (
    normalize_model_name,
    parse_model_tags,
    sanitize_json,
)
from scoreboard_server.db.connection import init_db
from scoreboard_server.db.models import (
    Completion,
    EvalRecord,
    EvaluationPublication,
    Score,
    Task,
)
from scoreboard_server.db.settings import DatabaseSettings


class PublicationConflict(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    task_id: int
    disposition: str


class EvaluationPublicationRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings

    async def publish(
        self,
        *,
        run_id: str,
        publisher_subject: str,
        idempotency_key: str,
        request_digest: str,
        payload: Mapping[str, Any],
    ) -> PublicationReceipt:
        await init_db(self._settings)
        for attempt in range(2):
            existing = await self._existing_receipt(
                run_id=run_id,
                publisher_subject=publisher_subject,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
            )
            if existing is not None:
                return existing
            try:
                return await self._publish_once(
                    run_id=run_id,
                    publisher_subject=publisher_subject,
                    idempotency_key=idempotency_key,
                    request_digest=request_digest,
                    payload=payload,
                )
            except IntegrityError:
                if attempt == 1:
                    raise
        raise AssertionError("unreachable publication retry state")

    async def _existing_receipt(
        self,
        *,
        run_id: str,
        publisher_subject: str,
        idempotency_key: str,
        request_digest: str,
    ) -> PublicationReceipt | None:
        by_run = await EvaluationPublication.filter(run_id=run_id).first()
        by_key = await EvaluationPublication.filter(
            publisher_subject=publisher_subject,
            idempotency_key=idempotency_key,
        ).first()
        rows = [row for row in (by_run, by_key) if row is not None]
        if not rows:
            return None
        if any(
            row.run_id != run_id
            or row.publisher_subject != publisher_subject
            or row.idempotency_key != idempotency_key
            or row.request_digest != request_digest
            for row in rows
        ):
            raise PublicationConflict(
                "publication identity already has different content"
            )
        return PublicationReceipt(int(rows[0].task_id), "unchanged")

    async def _publish_once(
        self,
        *,
        run_id: str,
        publisher_subject: str,
        idempotency_key: str,
        request_digest: str,
        payload: Mapping[str, Any],
    ) -> PublicationReceipt:
        identity = payload["identity"]
        task_identity = identity["task"]
        model_identity = identity["model"]
        accounting = payload["accounting"]
        manifest = payload["manifest"]
        completed_at = datetime.fromisoformat(str(manifest["completed_at"]))
        if completed_at.tzinfo is not None:
            completed_at = completed_at.astimezone(timezone.utc).replace(tzinfo=None)

        async with in_transaction("default") as connection:
            benchmark_id = await _upsert_benchmark(
                connection,
                name=f"{task_identity['suite']}/{task_identity['task']}@{task_identity['version']}",
                split=str(task_identity["split"]),
                num_samples=int(accounting["selected"]),
            )
            model_id = await _upsert_model(
                connection, str(model_identity["served_name"])
            )
            task = await Task.create(
                using_db=connection,
                config_path=None,
                evaluator=f"lighteval:{task_identity['suite']}/{task_identity['task']}",
                is_param_search=False,
                is_tmp=(
                    identity["eligibility"] != "official" or not identity["comparable"]
                ),
                created_at=completed_at,
                status="Running",
                git_hash=str(identity["evaluator"]["product_revision"]),
                model_id=model_id,
                benchmark_id=benchmark_id,
                description=f"evaluation_run_id={run_id}",
                sampling_config=sanitize_json(
                    {
                        "display_metric_key": payload["primary_metric"],
                        "lighteval_identity": identity,
                        "sampling_config": {
                            "answer": {
                                "max_new_tokens": payload["samples"][0]["generation"][
                                    "generation_limit"
                                ],
                                "stop_tokens": [0],
                            }
                        },
                    }
                ),
                log_path="",
            )
            binary_metric = next(
                metric
                for metric in task_identity["metrics"]
                if metric["name"] == payload["primary_metric"]
            )["binary_correctness"]
            for sample in payload["samples"]:
                context = {
                    "stages": [
                        {
                            "prompt": sample["prompt"],
                            "completion": sample["raw_completion"],
                            "stop_reason": sample["generation"]["terminal_reason"],
                        }
                    ],
                    "sampling_config": task.sampling_config,
                    "sample_id": sample["sample_id"],
                    "evidence": sample,
                }
                completion = await Completion.create(
                    using_db=connection,
                    task=task,
                    context=context,
                    sample_index=int(sample["sample_index"]),
                    avg_repeat_index=int(sample["attempt"]) - 1,
                    pass_index=0,
                    created_at=completed_at,
                    status="Completed",
                )
                reference_answer = sample.get("reference_answer")
                if binary_metric and reference_answer is not None:
                    passed = sample["metrics"][payload["primary_metric"]] == 1.0
                    await EvalRecord.create(
                        using_db=connection,
                        completion=completion,
                        answer=str(sample["scored_completion"])[:65_536],
                        ref_answer=str(reference_answer)[:4_096],
                        is_passed=passed,
                        fail_reason="" if passed else "primary metric mismatch",
                        created_at=completed_at,
                    )
            await Score.create(
                using_db=connection,
                task=task,
                cot_mode="CoT" if task_identity["cot_mode"] == "cot" else "NoCoT",
                metrics=sanitize_json(payload["native_metrics"]),
                created_at=completed_at,
            )
            await EvaluationPublication.create(
                using_db=connection,
                run_id=run_id,
                task=task,
                publisher_subject=publisher_subject,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                identity_digest=str(manifest["identity_digest"]),
                manifest_digest=str(manifest["digest"]),
                terminal_status="completed",
                identity_payload=identity,
                accounting_payload=sanitize_json(
                    {
                        **accounting,
                        "generated_samples": int(payload["generated_samples"]),
                        "truncated_samples": int(payload["truncated_samples"]),
                    }
                ),
                rejections_payload=payload["rejections"],
                performance_payload=payload["performance"],
                created_at=completed_at,
            )
            task.status = "Completed"
            await task.save(using_db=connection, update_fields=["status"])
            return PublicationReceipt(int(task.task_id), "created")


async def _upsert_benchmark(
    connection: Any, *, name: str, split: str, num_samples: int
) -> int:
    rows = await connection.execute_query_dict(
        """
        INSERT INTO benchmark (benchmark_name, benchmark_split, url, status, num_samples)
        VALUES ($1, $2, NULL, 'Completed', $3)
        ON CONFLICT (benchmark_name, benchmark_split)
        DO UPDATE SET num_samples = GREATEST(benchmark.num_samples, EXCLUDED.num_samples)
        RETURNING benchmark_id
        """,
        [name, split, num_samples],
    )
    return int(rows[0]["benchmark_id"])


async def _upsert_model(connection: Any, model_name: str) -> int:
    normalized = normalize_model_name(model_name)
    arch, data_version, num_params = parse_model_tags(normalized)
    rows = await connection.execute_query_dict(
        """
        INSERT INTO model (data_version, arch_version, num_params, model_name)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (arch_version, data_version, num_params, model_name)
        DO UPDATE SET model_name = EXCLUDED.model_name
        RETURNING model_id
        """,
        [data_version, arch, num_params, normalized],
    )
    return int(rows[0]["model_id"])
