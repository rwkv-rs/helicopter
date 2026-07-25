from __future__ import annotations

from datetime import datetime, timezone
import uuid

from scoreboard_server.dtos.api.evaluation_results import (
    AnswerOutcome,
    EvaluationList,
    EvaluationPublication,
    EvaluationSummary,
    PublicationReceipt,
    SampleDetail,
    SampleGroup,
    SampleGroups,
    sample_outcome,
)
from .connection import Database


class PublicationConflictError(Exception):
    pass


OUTCOMES: tuple[AnswerOutcome, ...] = (
    "correct",
    "incorrect",
    "unanswered",
    "undetermined",
)


class ScoreboardRepository:
    def __init__(self, database: Database):
        self.database = database

    async def publish(
        self,
        *,
        publication_id: str,
        digest: str,
        source: str,
        publication: EvaluationPublication,
    ) -> PublicationReceipt:
        pool = self.database.require_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                publication_id,
            )
            existing = await connection.fetchrow(
                """
                SELECT id, content_digest
                FROM evaluation_result
                WHERE publication_id = $1
                """,
                publication_id,
            )
            if existing is not None:
                if existing["content_digest"] != digest:
                    raise PublicationConflictError(publication_id)
                return PublicationReceipt(
                    evaluation_id=str(existing["id"]),
                    publication_id=publication_id,
                    content_digest=digest,
                    disposition="unchanged",
                )

            evaluation_id = uuid.uuid4()
            await connection.execute(
                """
                INSERT INTO evaluation_result (
                    id, publication_id, content_digest, source_run_id, source,
                    visibility, task_name, task_config, artifact, model, benchmark,
                    evaluation, comparisons, sampling_config, primary_metric,
                    aggregates, diagnostics
                ) VALUES (
                    $1, $2, $3, $4, $5, 'non_official', $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16
                )
                """,
                evaluation_id,
                publication_id,
                digest,
                publication.source_run_id,
                source,
                publication.task_name,
                publication.task_config,
                publication.artifact.model_dump(mode="json"),
                publication.model.model_dump(mode="json"),
                publication.benchmark.model_dump(mode="json"),
                publication.evaluation.model_dump(mode="json"),
                [
                    coordinate.model_dump(mode="json")
                    for coordinate in publication.comparisons
                ],
                publication.sampling_config,
                publication.primary_metric,
                publication.aggregates,
                publication.diagnostics.model_dump(mode="json"),
            )
            if publication.details:
                await connection.executemany(
                    """
                    INSERT INTO evaluation_sample (
                        evaluation_id, sample_index, outcome, doc, metric,
                        model_response
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    [
                        (
                            evaluation_id,
                            index,
                            sample_outcome(detail, publication.primary_metric),
                            detail.doc,
                            detail.metric,
                            detail.model_response,
                        )
                        for index, detail in enumerate(publication.details)
                    ],
                )
        return PublicationReceipt(
            evaluation_id=str(evaluation_id),
            publication_id=publication_id,
            content_digest=digest,
            disposition="created",
        )

    async def list_evaluations(self, *, limit: int = 1000) -> EvaluationList:
        pool = self.database.require_pool()
        rows = await pool.fetch(
            """
            SELECT id, publication_id, source_run_id, source, visibility,
                   created_at, task_name, model, benchmark, evaluation,
                   comparisons, sampling_config, primary_metric, aggregates,
                   diagnostics
            FROM evaluation_result
            ORDER BY created_at DESC, id
            LIMIT $1
            """,
            limit,
        )
        evaluations = [
            EvaluationSummary(
                evaluation_id=str(row["id"]),
                publication_id=row["publication_id"],
                source_run_id=row["source_run_id"],
                source=row["source"],
                visibility=row["visibility"],
                created_at=row["created_at"].isoformat(),
                task_name=row["task_name"],
                model=row["model"],
                benchmark=row["benchmark"],
                evaluation=row["evaluation"],
                comparisons=row["comparisons"],
                sampling_config=row["sampling_config"],
                primary_metric=row["primary_metric"],
                aggregates=row["aggregates"],
                diagnostics=row["diagnostics"],
            )
            for row in rows
        ]
        return EvaluationList(
            evaluations=evaluations,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    async def sample_groups(
        self, evaluation_id: uuid.UUID, *, limit: int
    ) -> SampleGroups | None:
        pool = self.database.require_pool()
        result = await pool.fetchrow(
            "SELECT primary_metric FROM evaluation_result WHERE id = $1",
            evaluation_id,
        )
        if result is None:
            return None
        rows = await pool.fetch(
            """
            WITH ranked AS (
                SELECT sample_index, outcome, doc, metric, model_response,
                       count(*) OVER (PARTITION BY outcome) AS outcome_total,
                       row_number() OVER (
                           PARTITION BY outcome ORDER BY sample_index
                       ) AS outcome_rank
                FROM evaluation_sample
                WHERE evaluation_id = $1
            )
            SELECT sample_index, outcome, doc, metric, model_response,
                   outcome_total
            FROM ranked
            WHERE outcome_rank <= $2
            ORDER BY outcome, sample_index
            """,
            evaluation_id,
            limit,
        )
        grouped: dict[AnswerOutcome, SampleGroup] = {
            outcome: SampleGroup(outcome=outcome, total=0, items=[])
            for outcome in OUTCOMES
        }
        for row in rows:
            outcome: AnswerOutcome = row["outcome"]
            group = grouped[outcome]
            group.total = row["outcome_total"]
            group.items.append(
                SampleDetail(
                    id=f"{evaluation_id}:{row['sample_index']}",
                    sample_index=row["sample_index"],
                    outcome=outcome,
                    doc=row["doc"],
                    metric=row["metric"],
                    model_response=row["model_response"],
                )
            )
        return SampleGroups(
            evaluation_id=str(evaluation_id),
            primary_metric=result["primary_metric"],
            groups=grouped,
        )
