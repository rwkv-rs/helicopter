from __future__ import annotations

import json
from typing import Any

from scoreboard_server.cores.identity import decode_cursor, encode_cursor

from .connection import Database, DatabaseSession


class RunQueryRepository:
    """Read-only SQL projections; no run mutation or state transition lives here."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def get(self, run_id: str) -> dict[str, Any]:
        async with self.database.read() as session:
            row = await self._required_run(session, run_id)
            aggregate = await session.fetchone(
                "SELECT accounting_json, metrics_json, truncated_samples, generated_samples, "
                "manifest_digest FROM run_aggregates WHERE run_id=?",
                (run_id,),
            )
            performance = await session.fetchone(
                "SELECT revision, metrics_json, updated_at FROM run_performance WHERE run_id=?",
                (run_id,),
            )
        return {
            "run_id": row["run_id"],
            "identity": json.loads(row["identity_json"]),
            "status": row["status"],
            "revision": row["revision"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "aggregate": (
                {
                    "accounting": json.loads(aggregate["accounting_json"]),
                    "metrics": json.loads(aggregate["metrics_json"]),
                    "truncated_samples": aggregate["truncated_samples"],
                    "generated_samples": aggregate["generated_samples"],
                    "manifest_digest": aggregate["manifest_digest"],
                }
                if aggregate
                else None
            ),
            "performance": (
                {
                    "revision": performance["revision"],
                    "metrics": json.loads(performance["metrics_json"]),
                    "updated_at": performance["updated_at"],
                }
                if performance
                else None
            ),
        }

    async def samples(
        self, run_id: str, *, limit: int, after: str | None
    ) -> dict[str, Any]:
        parameters: list[Any] = [run_id]
        condition = ""
        if after:
            sample_id, attempt_text = decode_cursor(after)
            attempt = int(attempt_text)
            if attempt <= 0:
                raise ValueError("sample cursor attempt must be positive")
            condition = " AND (sample_id > ? OR (sample_id = ? AND attempt > ?))"
            parameters.extend((sample_id, sample_id, attempt))
        parameters.append(limit + 1)
        async with self.database.read() as session:
            await self._required_run(session, run_id)
            rows = await session.fetchall(
                f"SELECT * FROM sample_evidence WHERE run_id=?{condition} "
                "ORDER BY sample_id, attempt LIMIT ?",
                parameters,
            )
        page = rows[:limit]
        items = [
            {
                "sample_id": row["sample_id"],
                "attempt": row["attempt"],
                "status": row["status"],
                "prompt": row["prompt"],
                "raw_completion": row["raw_completion"],
                "scored_completion": row["scored_completion"],
                "generation": json.loads(row["generation_json"]),
                "scoring": json.loads(row["scoring_json"]),
                "metrics": json.loads(row["metrics_json"]),
                "error_code": row["error_code"],
                "error_message": row["error_message"],
                "provenance": json.loads(row["provenance_json"]),
            }
            for row in page
        ]
        return {
            "items": items,
            "next_cursor": (
                encode_cursor(page[-1]["sample_id"], str(page[-1]["attempt"]))
                if len(rows) > limit
                else None
            ),
        }

    async def history(
        self, *, limit: int, cursor: str | None, model: str | None
    ) -> dict[str, Any]:
        conditions = ["1=1"]
        parameters: list[Any] = []
        if model:
            conditions.append("model_name=?")
            parameters.append(model)
        if cursor:
            updated_at, run_id = decode_cursor(cursor)
            conditions.append("(updated_at < ? OR (updated_at = ? AND run_id < ?))")
            parameters.extend((updated_at, updated_at, run_id))
        parameters.append(limit + 1)
        async with self.database.read() as session:
            rows = await session.fetchall(
                f"""SELECT run_id, suite, task_name, task_version, model_name, cot_mode,
                    repair_strategy, eligibility, comparable, status, revision,
                    created_at, updated_at, completed_at
                    FROM evaluation_runs WHERE {" AND ".join(conditions)}
                    ORDER BY updated_at DESC, run_id DESC LIMIT ?""",
                parameters,
            )
        page = rows[:limit]
        for row in page:
            row["comparable"] = bool(row["comparable"])
        return {
            "items": page,
            "next_cursor": (
                encode_cursor(page[-1]["updated_at"], page[-1]["run_id"])
                if len(rows) > limit
                else None
            ),
        }

    async def leaderboard(
        self, *, limit: int, cursor: str | None, model: str | None
    ) -> dict[str, Any]:
        conditions = [
            "r.status='completed'",
            "r.eligibility='official'",
            "r.comparable=1",
            "m.is_primary=1",
        ]
        parameters: list[Any] = []
        if model:
            conditions.append("r.model_name=?")
            parameters.append(model)
        outer_conditions = ["identity_rank=1"]
        if cursor:
            completed_at, run_id = decode_cursor(cursor)
            outer_conditions.append(
                "(completed_at < ? OR (completed_at = ? AND run_id < ?))"
            )
            parameters.extend((completed_at, completed_at, run_id))
        parameters.append(limit + 1)
        async with self.database.read() as session:
            rows = await session.fetchall(
                f"""WITH ranked AS (
                    SELECT r.run_id, r.suite, r.task_name, r.task_version, r.split_name,
                        r.fewshot, r.model_name, r.dataset_digest, r.cot_mode,
                        r.repair_strategy, r.completed_at, m.metric_name, m.metric_value,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.identity_digest
                            ORDER BY r.completed_at DESC, r.run_id DESC
                        ) AS identity_rank
                    FROM evaluation_runs r JOIN run_metrics m ON m.run_id=r.run_id
                    WHERE {" AND ".join(conditions)}
                ) SELECT run_id, suite, task_name, task_version, split_name, fewshot,
                    model_name, dataset_digest, cot_mode, repair_strategy, completed_at,
                    metric_name, metric_value FROM ranked
                    WHERE {" AND ".join(outer_conditions)}
                    ORDER BY completed_at DESC, run_id DESC LIMIT ?""",
                parameters,
            )
        page = rows[:limit]
        return {
            "items": page,
            "next_cursor": (
                encode_cursor(page[-1]["completed_at"], page[-1]["run_id"])
                if len(rows) > limit
                else None
            ),
        }

    async def metadata(self) -> dict[str, Any]:
        async with self.database.read() as session:
            counts = await session.fetchone(
                "SELECT COUNT(*) AS run_count, COUNT(DISTINCT model_name) AS model_count "
                "FROM evaluation_runs WHERE status='completed' AND eligibility='official' "
                "AND comparable=1"
            )
            models = await session.fetchall(
                "SELECT DISTINCT model_name FROM evaluation_runs WHERE status='completed' "
                "AND eligibility='official' AND comparable=1 ORDER BY model_name"
            )
        return {
            **(counts or {"run_count": 0, "model_count": 0}),
            "models": [row["model_name"] for row in models],
        }

    @staticmethod
    async def _required_run(session: DatabaseSession, run_id: str) -> dict[str, Any]:
        row = await session.fetchone(
            "SELECT * FROM evaluation_runs WHERE run_id=?", (run_id,)
        )
        if row is None:
            raise KeyError(run_id)
        return row
