from __future__ import annotations

from datetime import datetime
import uuid

from scoreboard_server.dtos.api.evaluation_results import (
    AnswerOutcome,
    CampaignCreate,
    CampaignProvenance,
    CampaignReceipt,
    CampaignStatus,
    EvaluationList,
    EvaluationSummary,
    FinalizeReceipt,
    SampleDetail,
    SamplePage,
    TaskPublication,
    TaskReceipt,
    sample_outcome,
)
from .connection import Database


class PublicationConflictError(Exception):
    pass


class CampaignCompletenessError(Exception):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(", ".join(missing))


class CampaignContractError(Exception):
    pass


class ScoreboardRepository:
    def __init__(self, database: Database):
        self.database = database

    async def create_campaign(
        self,
        *,
        campaign: CampaignCreate,
        publisher_principal: str,
    ) -> CampaignReceipt:
        pool = self.database.require_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                campaign.run_key,
            )
            existing = await connection.fetchrow(
                """
                SELECT id, status, config_digest, registry_digest,
                       eval_contract_digest, lighteval_version,
                       configured_selectors, resolved_selectors,
                       skipped_selectors, expected_tasks, publisher_principal
                FROM evaluation_campaign
                WHERE run_key = $1
                """,
                campaign.run_key,
            )
            expected = [
                task.model_dump(mode="json") for task in campaign.expected_tasks
            ]
            if existing is not None:
                if existing["publisher_principal"] != publisher_principal:
                    raise CampaignContractError(
                        "run key belongs to another publisher principal"
                    )
                values = {
                    "config_digest": campaign.config_digest,
                    "registry_digest": campaign.registry_digest,
                    "eval_contract_digest": campaign.eval_contract_digest,
                    "lighteval_version": campaign.lighteval_version,
                    "configured_selectors": campaign.configured_selectors,
                    "resolved_selectors": campaign.resolved_selectors,
                    "skipped_selectors": campaign.skipped_selectors,
                    "expected_tasks": expected,
                }
                mismatched = [
                    key for key, value in values.items() if existing[key] != value
                ]
                if mismatched:
                    raise CampaignContractError(
                        "run key contract mismatch: " + ", ".join(mismatched)
                    )
                task_rows = await connection.fetch(
                    """
                    SELECT task_identity, content_digest
                    FROM evaluation_task
                    WHERE campaign_id = $1
                    """,
                    existing["id"],
                )
                return CampaignReceipt(
                    campaign_id=str(existing["id"]),
                    disposition="unchanged",
                    status=existing["status"],
                    expected_task_count=len(expected),
                    acknowledged_task_digests={
                        row["task_identity"]: row["content_digest"] for row in task_rows
                    },
                )

            campaign_id = uuid.uuid4()
            await connection.execute(
                """
                INSERT INTO evaluation_campaign (
                    id, run_key, status, config_digest, registry_digest,
                    eval_contract_digest, lighteval_version,
                    configured_selectors, resolved_selectors, skipped_selectors,
                    expected_tasks, publisher_principal
                ) VALUES (
                    $1, $2, 'incomplete', $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
                """,
                campaign_id,
                campaign.run_key,
                campaign.config_digest,
                campaign.registry_digest,
                campaign.eval_contract_digest,
                campaign.lighteval_version,
                campaign.configured_selectors,
                campaign.resolved_selectors,
                campaign.skipped_selectors,
                expected,
                publisher_principal,
            )
            return CampaignReceipt(
                campaign_id=str(campaign_id),
                disposition="created",
                status="incomplete",
                expected_task_count=len(expected),
                acknowledged_task_digests={},
            )

    async def campaign_status(
        self,
        campaign_id: uuid.UUID,
        *,
        publisher_principal: str,
    ) -> CampaignStatus | None:
        pool = self.database.require_pool()
        campaign = await pool.fetchrow(
            """
            SELECT status, expected_tasks
            FROM evaluation_campaign
            WHERE id = $1 AND publisher_principal = $2
            """,
            campaign_id,
            publisher_principal,
        )
        if campaign is None:
            return None
        task_rows = await pool.fetch(
            """
            SELECT task_identity, content_digest
            FROM evaluation_task
            WHERE campaign_id = $1
            """,
            campaign_id,
        )
        digests = {row["task_identity"]: row["content_digest"] for row in task_rows}
        expected = [task["identity"] for task in campaign["expected_tasks"]]
        return CampaignStatus(
            campaign_id=str(campaign_id),
            status=campaign["status"],
            expected_task_count=len(expected),
            acknowledged_task_digests=digests,
            missing_task_identities=[
                identity for identity in expected if identity not in digests
            ],
        )

    async def publish_task(
        self,
        *,
        campaign_id: uuid.UUID,
        task_identity: str,
        digest: str,
        publication: TaskPublication,
        publisher_principal: str,
    ) -> TaskReceipt:
        pool = self.database.require_pool()
        async with pool.acquire() as connection, connection.transaction():
            lock_key = f"{campaign_id}:{task_identity}"
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                lock_key,
            )
            campaign = await connection.fetchrow(
                """
                SELECT status, expected_tasks, lighteval_version
                FROM evaluation_campaign
                WHERE id = $1 AND publisher_principal = $2
                """,
                campaign_id,
                publisher_principal,
            )
            if campaign is None:
                raise CampaignContractError("campaign not found")
            if campaign["status"] != "incomplete":
                raise CampaignContractError("complete campaign is immutable")
            expected = {task["identity"]: task for task in campaign["expected_tasks"]}
            if task_identity not in expected:
                raise CampaignContractError(
                    "task identity is not in campaign expected set"
                )
            if publication.campaign_id != str(campaign_id):
                raise CampaignContractError("body campaign_id does not match path")
            if publication.task.identity != task_identity:
                raise CampaignContractError("body task identity does not match path")
            if publication.task.model_dump(mode="json") != expected[task_identity]:
                raise CampaignContractError(
                    "task metadata does not match campaign expected set"
                )
            if publication.artifact.lighteval_version != campaign["lighteval_version"]:
                raise CampaignContractError(
                    "artifact LightEval version does not match campaign"
                )

            existing = await connection.fetchrow(
                """
                SELECT id, content_digest
                FROM evaluation_task
                WHERE campaign_id = $1 AND task_identity = $2
                """,
                campaign_id,
                task_identity,
            )
            if existing is not None:
                if existing["content_digest"] != digest:
                    raise PublicationConflictError(task_identity)
                return TaskReceipt(
                    evaluation_id=str(existing["id"]),
                    task_identity=task_identity,
                    content_digest=digest,
                    disposition="unchanged",
                )

            evaluation_id = uuid.uuid4()
            await connection.execute(
                """
                INSERT INTO evaluation_task (
                    id, campaign_id, task_identity, content_digest, task,
                    artifact, task_config, model, sampling_config,
                    primary_metric, aggregates, diagnostics
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                )
                """,
                evaluation_id,
                campaign_id,
                task_identity,
                digest,
                publication.task.model_dump(mode="json"),
                publication.artifact.model_dump(mode="json"),
                publication.task_config,
                publication.model.model_dump(mode="json"),
                publication.sampling_config,
                publication.primary_metric,
                publication.aggregates,
                publication.diagnostics.model_dump(mode="json"),
            )
            if publication.details:
                await connection.executemany(
                    """
                    INSERT INTO evaluation_sample (
                        evaluation_id, sample_index, document_index, outcome, doc, metric,
                        model_response
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    [
                        (
                            evaluation_id,
                            detail.sample_index,
                            detail.document_index,
                            sample_outcome(detail, publication.primary_metric),
                            detail.doc,
                            detail.metric,
                            detail.model_response,
                        )
                        for detail in publication.details
                    ],
                )
            return TaskReceipt(
                evaluation_id=str(evaluation_id),
                task_identity=task_identity,
                content_digest=digest,
                disposition="created",
            )

    async def finalize_campaign(
        self,
        campaign_id: uuid.UUID,
        *,
        publisher_principal: str,
    ) -> FinalizeReceipt:
        pool = self.database.require_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                str(campaign_id),
            )
            campaign = await connection.fetchrow(
                """
                SELECT status, expected_tasks
                FROM evaluation_campaign
                WHERE id = $1 AND publisher_principal = $2
                FOR UPDATE
                """,
                campaign_id,
                publisher_principal,
            )
            if campaign is None:
                raise CampaignContractError("campaign not found")
            expected = [task["identity"] for task in campaign["expected_tasks"]]
            task_rows = await connection.fetch(
                """
                SELECT task_identity
                FROM evaluation_task
                WHERE campaign_id = $1
                """,
                campaign_id,
            )
            actual = {row["task_identity"] for row in task_rows}
            missing = [identity for identity in expected if identity not in actual]
            if missing:
                raise CampaignCompletenessError(missing)
            if actual != set(expected):
                raise CampaignContractError(
                    "campaign contains task identities outside expected set"
                )
            if campaign["status"] == "incomplete":
                await connection.execute(
                    """
                    UPDATE evaluation_campaign
                    SET status = 'complete', completed_at = now()
                    WHERE id = $1
                    """,
                    campaign_id,
                )
            return FinalizeReceipt(
                campaign_id=str(campaign_id),
                status="complete",
                task_count=len(expected),
            )

    async def list_evaluations(
        self,
        *,
        completed_before: datetime,
        offset: int = 0,
        limit: int = 1000,
    ) -> EvaluationList:
        pool = self.database.require_pool()
        total = await pool.fetchval(
            """
            SELECT count(*)
            FROM evaluation_task AS t
            JOIN evaluation_campaign AS c ON c.id = t.campaign_id
            WHERE c.status = 'complete' AND c.completed_at <= $1
            """,
            completed_before,
        )
        rows = await pool.fetch(
            """
            SELECT t.id, t.campaign_id, t.task_identity, t.created_at,
                   c.completed_at, c.publisher_principal, c.config_digest,
                   c.registry_digest, c.eval_contract_digest,
                   c.lighteval_version, c.configured_selectors,
                   c.resolved_selectors, c.skipped_selectors,
                   t.task, t.artifact, t.task_config,
                   t.model, t.sampling_config, t.primary_metric,
                   t.aggregates, t.diagnostics
            FROM evaluation_task AS t
            JOIN evaluation_campaign AS c ON c.id = t.campaign_id
            WHERE c.status = 'complete' AND c.completed_at <= $1
            ORDER BY c.completed_at DESC, c.id DESC, t.task_identity
            OFFSET $2 LIMIT $3
            """,
            completed_before,
            offset,
            limit,
        )
        next_offset = offset + len(rows)
        return EvaluationList(
            evaluations=[
                EvaluationSummary(
                    evaluation_id=str(row["id"]),
                    campaign_id=str(row["campaign_id"]),
                    task_identity=row["task_identity"],
                    created_at=row["created_at"].isoformat(),
                    completed_at=row["completed_at"].isoformat(),
                    task=row["task"],
                    artifact=row["artifact"],
                    task_config=row["task_config"],
                    model=row["model"],
                    sampling_config=row["sampling_config"],
                    primary_metric=row["primary_metric"],
                    aggregates=row["aggregates"],
                    diagnostics=row["diagnostics"],
                    provenance=CampaignProvenance(
                        config_digest=row["config_digest"],
                        registry_digest=row["registry_digest"],
                        eval_contract_digest=row["eval_contract_digest"],
                        lighteval_version=row["lighteval_version"],
                        configured_selectors=row["configured_selectors"],
                        resolved_selectors=row["resolved_selectors"],
                        skipped_selectors=row["skipped_selectors"],
                        publisher_principal=row["publisher_principal"],
                    ),
                )
                for row in rows
            ],
            generated_at=completed_before.isoformat(),
            total=total,
            offset=offset,
            limit=limit,
            next_offset=next_offset if next_offset < total else None,
        )

    async def sample_page(
        self,
        evaluation_id: uuid.UUID,
        *,
        offset: int,
        limit: int,
        outcome: AnswerOutcome | None,
    ) -> SamplePage | None:
        pool = self.database.require_pool()
        result = await pool.fetchrow(
            """
            SELECT t.primary_metric
            FROM evaluation_task AS t
            JOIN evaluation_campaign AS c ON c.id = t.campaign_id
            WHERE t.id = $1 AND c.status = 'complete'
            """,
            evaluation_id,
        )
        if result is None:
            return None
        total = await pool.fetchval(
            """
            SELECT count(*)
            FROM evaluation_sample
            WHERE evaluation_id = $1
              AND ($2::text IS NULL OR outcome = $2)
            """,
            evaluation_id,
            outcome,
        )
        rows = await pool.fetch(
            """
            SELECT sample_index, document_index, outcome, doc, metric, model_response
            FROM evaluation_sample
            WHERE evaluation_id = $1
              AND ($2::text IS NULL OR outcome = $2)
            ORDER BY sample_index
            OFFSET $3 LIMIT $4
            """,
            evaluation_id,
            outcome,
            offset,
            limit,
        )
        next_offset = offset + len(rows)
        return SamplePage(
            evaluation_id=str(evaluation_id),
            primary_metric=result["primary_metric"],
            total=total,
            offset=offset,
            limit=limit,
            next_offset=next_offset if next_offset < total else None,
            items=[
                SampleDetail(
                    id=f"{evaluation_id}:{row['sample_index']}",
                    sample_index=row["sample_index"],
                    document_index=row["document_index"],
                    outcome=row["outcome"],
                    doc=row["doc"],
                    metric=row["metric"],
                    model_response=row["model_response"],
                )
                for row in rows
            ],
        )
