from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .connection import Database, DatabaseSession


class RunCommandRepository:
    """Primitive SQL commands used by application-owned transactions."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def insert_run(
        self, session: DatabaseSession, values: Sequence[Any]
    ) -> dict[str, Any] | None:
        return await session.fetchone(
            """INSERT INTO evaluation_runs(
                run_id, publisher_subject, idempotency_key, create_digest,
                identity_json, identity_digest, suite, task_name, task_version,
                split_name, fewshot, cot_mode, repair_strategy, model_name,
                checkpoint_digest, provider_revision, config_digest, dataset_digest,
                eligibility, comparable, expected_sample_count, sample_set_digest,
                status, revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                'planned', 1, ?, ?)
            ON CONFLICT DO NOTHING RETURNING *""",
            values,
        )

    async def by_idempotency(
        self, session: DatabaseSession, subject: str, key: str
    ) -> dict[str, Any] | None:
        return await session.fetchone(
            "SELECT * FROM evaluation_runs WHERE publisher_subject=? AND idempotency_key=?",
            (subject, key),
        )

    async def by_id(
        self, session: DatabaseSession, run_id: str
    ) -> dict[str, Any] | None:
        return await session.fetchone(
            "SELECT * FROM evaluation_runs WHERE run_id=?", (run_id,)
        )

    async def resume(
        self,
        session: DatabaseSession,
        *,
        run_id: str,
        subject: str,
        revision: int,
        now: str,
    ) -> dict[str, Any] | None:
        return await session.fetchone(
            """UPDATE evaluation_runs SET status='running', revision=revision+1, updated_at=?
            WHERE run_id=? AND publisher_subject=? AND revision=?
              AND status IN ('planned', 'partial', 'failed') RETURNING *""",
            (now, run_id, subject, revision),
        )

    async def receipt(
        self, session: DatabaseSession, run_id: str, key: str
    ) -> dict[str, Any] | None:
        return await session.fetchone(
            "SELECT request_digest, response_status, response_revision FROM ingest_receipts "
            "WHERE run_id=? AND idempotency_key=?",
            (run_id, key),
        )

    async def start_finalizing(
        self,
        session: DatabaseSession,
        *,
        run_id: str,
        subject: str,
        revision: int,
        request_digest: str,
        now: str,
    ) -> dict[str, Any] | None:
        return await session.fetchone(
            """UPDATE evaluation_runs SET status='finalizing', revision=revision+1,
                ingest_digest=?, updated_at=?
            WHERE run_id=? AND publisher_subject=? AND revision=? AND status='running'
            RETURNING *""",
            (request_digest, now, run_id, subject, revision),
        )

    async def attempt_digest(
        self,
        session: DatabaseSession,
        *,
        run_id: str,
        sample_id: str,
        attempt: int,
    ) -> str | None:
        row = await session.fetchone(
            "SELECT evidence_digest FROM sample_evidence "
            "WHERE run_id=? AND sample_id=? AND attempt=?",
            (run_id, sample_id, attempt),
        )
        return row["evidence_digest"] if row else None

    async def insert_sample(
        self, session: DatabaseSession, values: Sequence[Any]
    ) -> None:
        await session.execute(
            """INSERT INTO sample_evidence(
                run_id, sample_id, attempt, status, prompt, raw_completion,
                scored_completion, generation_json, scoring_json, metrics_json,
                provenance_json, evidence_digest, error_code, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )

    async def upsert_aggregate(
        self, session: DatabaseSession, values: Sequence[Any]
    ) -> None:
        await session.execute(
            """INSERT INTO run_aggregates(
                run_id, accounting_json, metrics_json, truncated_samples,
                generated_samples, manifest_digest, payload_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                accounting_json=excluded.accounting_json,
                metrics_json=excluded.metrics_json,
                truncated_samples=excluded.truncated_samples,
                generated_samples=excluded.generated_samples,
                manifest_digest=excluded.manifest_digest,
                payload_digest=excluded.payload_digest""",
            values,
        )

    async def replace_metrics(
        self,
        session: DatabaseSession,
        *,
        run_id: str,
        metrics: dict[str, float],
        primary_metric: str | None,
    ) -> None:
        await session.execute("DELETE FROM run_metrics WHERE run_id=?", (run_id,))
        for name, value in metrics.items():
            await session.execute(
                """INSERT INTO run_metrics(run_id, metric_name, metric_value, is_primary)
                VALUES (?, ?, ?, ?)""",
                (run_id, name, value, int(name == primary_metric)),
            )

    async def finish(
        self,
        session: DatabaseSession,
        *,
        run_id: str,
        expected_revision: int,
        status: str,
        now: str,
    ) -> dict[str, Any] | None:
        return await session.fetchone(
            """UPDATE evaluation_runs SET status=?, revision=revision+1, updated_at=?,
                completed_at=? WHERE run_id=? AND status='finalizing' AND revision=?
                RETURNING *""",
            (
                status,
                now,
                now if status == "completed" else None,
                run_id,
                expected_revision,
            ),
        )

    async def insert_receipt(
        self, session: DatabaseSession, values: Sequence[Any]
    ) -> None:
        await session.execute(
            """INSERT INTO ingest_receipts(
                run_id, idempotency_key, request_digest, response_status,
                response_revision, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            values,
        )

    async def upsert_performance(
        self,
        session: DatabaseSession,
        *,
        run_id: str,
        expected_revision: int,
        metrics_json: str,
        now: str,
    ) -> dict[str, Any] | None:
        if expected_revision == 0:
            return await session.fetchone(
                """INSERT INTO run_performance(run_id, revision, metrics_json, updated_at)
                VALUES (?, 1, ?, ?) ON CONFLICT(run_id) DO NOTHING RETURNING *""",
                (run_id, metrics_json, now),
            )
        return await session.fetchone(
            """UPDATE run_performance SET revision=revision+1, metrics_json=?, updated_at=?
            WHERE run_id=? AND revision=? RETURNING *""",
            (metrics_json, now, run_id, expected_revision),
        )
