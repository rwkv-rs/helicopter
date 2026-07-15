from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any

from .contracts import CreateRunRequest, IngestRunRequest, PerformancePatchRequest
from .cores.identity import canonical_json, digest_json
from .db.connection import Database, DatabaseSession
from .db.run_commands import RunCommandRepository
from .db.run_queries import RunQueryRepository
from .errors import DomainError


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunApplication:
    """Own idempotency, state transitions, validation, aggregation and transactions."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.commands = RunCommandRepository(database)
        self.queries = RunQueryRepository(database)

    async def create(
        self, *, subject: str, idempotency_key: str, request: CreateRunRequest
    ) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        create_digest = digest_json(payload)
        identity = request.identity.model_dump(mode="json")
        identity_json = canonical_json(identity)
        identity_digest = digest_json(identity)
        task = request.identity.task
        model = request.identity.model
        provider = request.identity.provider
        now = _now()
        values = (
            request.run_id,
            subject,
            idempotency_key,
            create_digest,
            identity_json,
            identity_digest,
            task.suite,
            task.task,
            task.version,
            task.split,
            task.fewshot,
            task.cot_mode,
            task.repair_strategy,
            model.served_name,
            model.checkpoint_sha256,
            provider.server_revision,
            request.identity.config_digest,
            task.dataset_digest,
            request.identity.eligibility,
            int(request.identity.comparable),
            len(request.expected_sample_ids),
            request.sample_set_digest,
            now,
            now,
        )
        async with self.database.transaction(immediate=True) as session:
            inserted = await self.commands.insert_run(session, values)
            row = inserted or await self.commands.by_idempotency(
                session, subject, idempotency_key
            )
            if row is None:
                collision = await self.commands.by_id(session, request.run_id)
                if collision is not None:
                    raise DomainError(
                        "run_id_conflict",
                        "run_id already belongs to another request",
                        409,
                    )
                raise RuntimeError("run insert did not produce a resource")
            if (
                row["run_id"] != request.run_id
                or row["create_digest"] != create_digest
                or row["identity_json"] != identity_json
            ):
                raise DomainError(
                    "idempotency_conflict",
                    "idempotency key was already used with a different run payload",
                    409,
                )
            return {**row, "disposition": "created" if inserted else "unchanged"}

    async def resume(
        self, *, run_id: str, subject: str, revision: int
    ) -> dict[str, Any]:
        async with self.database.transaction(immediate=True) as session:
            updated = await self.commands.resume(
                session,
                run_id=run_id,
                subject=subject,
                revision=revision,
                now=_now(),
            )
            if updated is not None:
                return updated
            row = await self._owned_run(session, run_id, subject)
            if int(row["revision"]) != revision:
                raise DomainError("stale_revision", "If-Match revision is stale", 412)
            raise DomainError(
                "invalid_run_state", f"run cannot resume from {row['status']}", 409
            )

    async def ingest(
        self,
        *,
        run_id: str,
        subject: str,
        revision: int,
        idempotency_key: str,
        request: IngestRunRequest,
    ) -> dict[str, Any]:
        request_digest = digest_json(request.model_dump(mode="json"))
        now = _now()
        async with self.database.transaction(immediate=True) as session:
            row = await self._owned_run(session, run_id, subject)
            receipt = await self.commands.receipt(session, run_id, idempotency_key)
            if receipt is not None:
                if receipt["request_digest"] != request_digest:
                    raise DomainError(
                        "idempotency_conflict",
                        "ingest idempotency key was already used with different evidence",
                        409,
                    )
                return {
                    **row,
                    "status": receipt["response_status"],
                    "revision": receipt["response_revision"],
                    "disposition": "unchanged",
                }
            self._validate_ingest(row, request)
            finalizing = await self.commands.start_finalizing(
                session,
                run_id=run_id,
                subject=subject,
                revision=revision,
                request_digest=request_digest,
                now=now,
            )
            if finalizing is None:
                current = await self._owned_run(session, run_id, subject)
                if int(current["revision"]) != revision:
                    raise DomainError(
                        "stale_revision", "If-Match revision is stale", 412
                    )
                raise DomainError(
                    "invalid_run_state", "only running evaluations can be ingested", 409
                )
            await self._persist_evidence(
                session, run_id=run_id, request=request, request_digest=request_digest
            )
            terminal = await self.commands.finish(
                session,
                run_id=run_id,
                expected_revision=revision + 1,
                status=request.terminal_status,
                now=now,
            )
            if terminal is None:
                raise RuntimeError(
                    "terminal state transition failed after evidence ingest"
                )
            await self.commands.insert_receipt(
                session,
                (
                    run_id,
                    idempotency_key,
                    request_digest,
                    terminal["status"],
                    terminal["revision"],
                    now,
                ),
            )
            return {**terminal, "disposition": "created"}

    async def patch_performance(
        self,
        *,
        run_id: str,
        subject: str,
        revision: int,
        request: PerformancePatchRequest,
    ) -> dict[str, Any]:
        async with self.database.transaction(immediate=True) as session:
            row = await self._owned_run(session, run_id, subject)
            if row["status"] != "completed":
                raise DomainError(
                    "invalid_run_state",
                    "performance can only be attached to completed runs",
                    409,
                )
            updated = await self.commands.upsert_performance(
                session,
                run_id=run_id,
                expected_revision=revision,
                metrics_json=canonical_json(request.metrics),
                now=_now(),
            )
            if updated is None:
                raise DomainError("stale_revision", "If-Match revision is stale", 412)
            return {
                "run_id": run_id,
                "revision": updated["revision"],
                "disposition": "created",
            }

    async def get(self, run_id: str) -> dict[str, Any]:
        return await self._query(lambda: self.queries.get(run_id))

    async def samples(
        self, run_id: str, *, limit: int, after: str | None
    ) -> dict[str, Any]:
        return await self._query(
            lambda: self.queries.samples(run_id, limit=limit, after=after)
        )

    async def history(
        self, *, limit: int, cursor: str | None, model: str | None
    ) -> dict[str, Any]:
        return await self._query(
            lambda: self.queries.history(limit=limit, cursor=cursor, model=model)
        )

    async def leaderboard(
        self, *, limit: int, cursor: str | None, model: str | None
    ) -> dict[str, Any]:
        return await self._query(
            lambda: self.queries.leaderboard(limit=limit, cursor=cursor, model=model)
        )

    async def metadata(self) -> dict[str, Any]:
        return await self.queries.metadata()

    @staticmethod
    async def _query(call):
        try:
            return await call()
        except KeyError as error:
            raise DomainError(
                "run_not_found", "evaluation run was not found", 404
            ) from error
        except (TypeError, ValueError) as error:
            raise DomainError(
                "invalid_cursor", "invalid pagination cursor", 422
            ) from error

    @staticmethod
    def _validate_ingest(row: dict[str, Any], request: IngestRunRequest) -> None:
        latest_ids = sorted({sample.sample_id for sample in request.samples})
        sample_set_digest = hashlib.sha256("\n".join(latest_ids).encode()).hexdigest()
        if (
            len(latest_ids) != int(row["expected_sample_count"])
            or sample_set_digest != row["sample_set_digest"]
        ):
            raise DomainError(
                "sample_set_mismatch",
                "ingest sample identities do not match run creation",
                409,
            )
        identity = json.loads(row["identity_json"])
        task = identity["task"]
        if request.manifest.identity_digest != row["identity_digest"]:
            raise DomainError(
                "manifest_identity_mismatch",
                "manifest identity digest differs from run",
                409,
            )
        accounting = request.accounting.model_dump(mode="json")
        if request.manifest.accounting_digest != digest_json(accounting):
            raise DomainError(
                "manifest_accounting_mismatch",
                "manifest accounting digest differs from ingest",
                409,
            )
        contracts = {item["name"]: item for item in task["metrics"]}
        if request.metrics and request.primary_metric != task["primary_metric"]:
            raise DomainError(
                "metric_contract_mismatch",
                "primary metric differs from run identity",
                409,
            )
        if not request.metrics and request.primary_metric is not None:
            raise DomainError(
                "metric_contract_mismatch",
                "empty aggregate cannot name a primary metric",
                409,
            )
        if set(request.metrics) - set(contracts):
            raise DomainError(
                "metric_contract_mismatch",
                "aggregate contains an undeclared metric",
                409,
            )
        latest_samples = request.evidence_summary().latest_samples
        for sample in request.samples:
            if (
                sample.scoring is not None
                and sample.scoring.scorer_revision != task["scorer_revision"]
            ):
                raise DomainError(
                    "scorer_identity_mismatch",
                    "sample scorer revision differs from run identity",
                    409,
                )
            RunApplication._validate_metric_values(sample.metrics, contracts)
        RunApplication._validate_metric_values(request.metrics, contracts)
        for name, aggregate_value in request.metrics.items():
            values = [
                sample.metrics[name]
                for sample in latest_samples.values()
                if sample.status == "scored" and name in sample.metrics
            ]
            if not values:
                raise DomainError(
                    "aggregation_mismatch",
                    f"metric {name} has no scored sample evidence",
                    409,
                )
            rule = contracts[name]["aggregation"]
            expected = sum(values) if rule == "sum" else sum(values) / len(values)
            if not math.isclose(
                aggregate_value, expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise DomainError(
                    "aggregation_mismatch",
                    f"metric {name} differs from declared aggregation",
                    409,
                )

    @staticmethod
    def _validate_metric_values(
        metrics: dict[str, float], contracts: dict[str, dict[str, Any]]
    ) -> None:
        if set(metrics) - set(contracts):
            raise DomainError(
                "metric_contract_mismatch", "sample contains an undeclared metric", 409
            )
        for name, value in metrics.items():
            contract = contracts[name]
            if not contract["minimum"] <= value <= contract["maximum"]:
                raise DomainError(
                    "metric_out_of_range",
                    f"metric {name} is outside its declared range",
                    409,
                )
            if contract["binary_correctness"] and value not in {0.0, 1.0}:
                raise DomainError(
                    "metric_contract_mismatch", f"metric {name} must be binary", 409
                )

    async def _persist_evidence(
        self,
        session: DatabaseSession,
        *,
        run_id: str,
        request: IngestRunRequest,
        request_digest: str,
    ) -> None:
        for sample in request.samples:
            evidence = sample.model_dump(mode="json")
            evidence_digest = digest_json(evidence)
            existing = await self.commands.attempt_digest(
                session,
                run_id=run_id,
                sample_id=sample.sample_id,
                attempt=sample.attempt,
            )
            if existing is not None:
                if existing != evidence_digest:
                    raise DomainError(
                        "sample_attempt_conflict",
                        "sample attempt identity already contains different evidence",
                        409,
                    )
                continue
            await self.commands.insert_sample(
                session,
                (
                    run_id,
                    sample.sample_id,
                    sample.attempt,
                    sample.status,
                    sample.prompt,
                    sample.raw_completion,
                    sample.scored_completion,
                    canonical_json(evidence.get("generation")),
                    canonical_json(evidence.get("scoring")),
                    canonical_json(sample.metrics),
                    canonical_json(sample.provenance),
                    evidence_digest,
                    sample.error_code,
                    sample.error_message,
                ),
            )
        await self.commands.upsert_aggregate(
            session,
            (
                run_id,
                canonical_json(request.accounting.model_dump(mode="json")),
                canonical_json(request.metrics),
                request.truncated_samples,
                request.generated_samples,
                request.manifest.digest,
                request_digest,
            ),
        )
        await self.commands.replace_metrics(
            session,
            run_id=run_id,
            metrics=request.metrics,
            primary_metric=request.primary_metric,
        )

    async def _owned_run(
        self, session: DatabaseSession, run_id: str, subject: str
    ) -> dict[str, Any]:
        row = await self.commands.by_id(session, run_id)
        if row is None:
            raise DomainError("run_not_found", "evaluation run was not found", 404)
        if row["publisher_subject"] != subject:
            raise DomainError("run_forbidden", "run belongs to another publisher", 403)
        return row
