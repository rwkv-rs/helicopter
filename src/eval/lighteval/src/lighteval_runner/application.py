from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from .contracts import EvaluationOutcome, EvaluationRequest
from .data_sources.acquisition import (
    DatasetSnapshot,
    DatasetSource,
    SnapshotRow,
    materialize_snapshot,
    select_snapshot_rows,
    verify_asset_manifest,
)
from .execution import (
    DISABLED_CACHE_POLICY_REVISION,
    Eligibility,
    ExecutionPlan,
    ModelIdentity,
    ProviderIdentity,
    RunLifecycle,
    RunStatus,
    SampleAccounting,
    TaskIdentity,
    create_execution_plan,
)
from .provider.attestation import (
    AttestationDecision,
    ProviderAttestation,
    fetch_provider_attestation,
    validate_attestation,
)
from .provider.endpoint import OpenAIEndpoint
from .registry import TaskDefinition, get_task_definition
from .results.artifacts import RunArtifacts, record_publication_attempt, verify_manifest
from .results.performance import summarize_run_performance
from .results.records import RunSummary
from .results.scoreboard import (
    PublicationResult,
    build_scoreboard_identity,
    publish_artifact,
    retry_publication,
)
from .sample_pipeline import SampleEvaluator, SampleExecutionContext
from .task_runtime import TaskRuntime


@dataclass(frozen=True, slots=True)
class _DatasetInput:
    source: DatasetSource
    path: Path
    asset_manifest_digest: str


@dataclass(frozen=True, slots=True)
class _ProviderPreflight:
    actual: ProviderAttestation | None
    observed: ProviderAttestation
    decision: AttestationDecision
    eligibility: Eligibility


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    definition: TaskDefinition
    runtime: TaskRuntime
    snapshot: DatasetSnapshot
    selected: tuple[SnapshotRow, ...]
    provider: _ProviderPreflight
    plan: ExecutionPlan
    config_digest: str
    asset_manifest_digest: str


@dataclass(frozen=True, slots=True)
class _RunEvidence:
    samples: tuple[dict[str, Any], ...]
    formatter_rejections: tuple[dict[str, str], ...]
    primary_metrics: tuple[float, ...]
    generated_samples: int
    truncated_samples: int
    cancelled: bool


@dataclass(frozen=True, slots=True)
class _RunAggregation:
    accounting: SampleAccounting
    status: RunStatus
    metrics: dict[str, float]
    rejections: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class _FinalizedRun:
    manifest_path: Path
    status: RunStatus
    metrics: dict[str, float]
    generated_samples: int
    truncated_samples: int


def run_evaluation(request: EvaluationRequest) -> EvaluationOutcome:
    """Own one complete immutable evaluation run from preflight to terminal manifest."""
    definition = get_task_definition(
        request.task, allow_proxy=request.allow_non_comparable
    )
    artifacts = RunArtifacts(Path(request.output_root), run_id=uuid4().hex)
    try:
        return _execute_evaluation(request, definition, artifacts)
    except Exception as error:
        if artifacts.finalized:
            raise
        terminal_status = (
            RunStatus.INVALID if isinstance(error, ValueError) else RunStatus.FAILED
        )
        identity = {
            "task": definition.contract.identity,
            "dataset_digest": definition.snapshot_sha256,
            "model": request.model,
            "checkpoint_sha256": request.checkpoint_sha256,
            "config_digest": request.config_digest or None,
            "eligibility": "invalid",
        }
        artifacts.write_json(
            "failure.json",
            {
                "status": terminal_status.value,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        manifest_path = artifacts.finalize(
            status=terminal_status,
            identity_digest=_digest(identity),
            identities={"run": identity},
            accounting=SampleAccounting(0, 0, 0, 0, 0, 0, 0),
        )
        return EvaluationOutcome(
            run_id=artifacts.run_id,
            run_status=terminal_status.value,
            manifest_path=manifest_path,
            publication_status="not_requested",
            summary={"error": str(error)},
        )


def _execute_evaluation(
    request: EvaluationRequest,
    definition: TaskDefinition,
    artifacts: RunArtifacts,
) -> EvaluationOutcome:
    prepared = _prepare_run(request, definition, artifacts.run_id)
    evidence = _execute_samples(request, prepared)
    aggregation = _aggregate_evidence(prepared, evidence)
    finalized = _commit_run(request, prepared, evidence, aggregation, artifacts)
    publication = _publish_if_requested(request, finalized)
    return EvaluationOutcome(
        run_id=prepared.plan.run_id,
        run_status=finalized.status.value,
        manifest_path=finalized.manifest_path,
        publication_status=publication.status,
        publication_error=publication.error,
        publication_retry_identity=(
            publication.retry_identity if request.publish_to_scoreboard else None
        ),
        summary={
            **finalized.metrics,
            "generated_samples": finalized.generated_samples,
            "truncated_samples": finalized.truncated_samples,
            "truncation_rate": _truncation_rate(
                finalized.truncated_samples, finalized.generated_samples
            ),
        },
    )


def _prepare_run(
    request: EvaluationRequest, definition: TaskDefinition, run_id: str
) -> _PreparedRun:
    dataset_input = _resolve_dataset_input(request, definition)
    provider = _preflight_provider(request, definition)
    config_digest = request.config_digest or _fallback_config_digest(request)
    task_identity = _task_identity(
        request,
        definition,
        provider.eligibility,
        definition.snapshot_sha256,
    )
    plan = create_execution_plan(
        request,
        task_identity=task_identity,
        model_identity=provider.observed.model,
        provider_identity=provider.observed.provider,
        config_digest=config_digest,
        asset_manifest_digest=dataset_input.asset_manifest_digest,
        task_generation_limit=definition.generation_limit,
        run_id=run_id,
    )
    snapshot = materialize_snapshot(
        dataset_input.path,
        dataset_input.source,
        validate_row=lambda row: _validate_dataset_row(
            row, definition.required_columns
        ),
    )
    if snapshot.source_rows != definition.expected_rows:
        raise ValueError(
            f"canonical snapshot row count mismatch: expected {definition.expected_rows}, "
            f"found {snapshot.source_rows}"
        )
    return _PreparedRun(
        definition=definition,
        runtime=definition.load_runtime(),
        snapshot=snapshot,
        selected=select_snapshot_rows(snapshot, request.max_samples),
        provider=provider,
        plan=plan,
        config_digest=config_digest,
        asset_manifest_digest=dataset_input.asset_manifest_digest,
    )


def _resolve_dataset_input(
    request: EvaluationRequest, definition: TaskDefinition
) -> _DatasetInput:
    if (
        request.snapshot_sha256 is not None
        and request.snapshot_sha256 != definition.snapshot_sha256
    ):
        raise ValueError(
            "snapshot digest is not the trusted digest for the canonical task"
        )
    source = DatasetSource(
        repository=definition.dataset_repository,
        revision=definition.dataset_revision,
        source_file=definition.source_file,
        sha256=definition.snapshot_sha256,
    )
    if definition.bundled_resource is not None:
        return _DatasetInput(
            source,
            definition.bundled_resource.resolve(),
            definition.bundled_resource.manifest_digest,
        )
    if request.snapshot_path is None or request.snapshot_manifest_path is None:
        raise ValueError(
            "external canonical task requires snapshot and helicopter-dev asset manifest"
        )
    return _DatasetInput(
        source,
        request.snapshot_path,
        verify_asset_manifest(
            request.snapshot_manifest_path,
            request.snapshot_path,
            source,
            asset_name=definition.asset_name,
        ),
    )


def _preflight_provider(
    request: EvaluationRequest, definition: TaskDefinition
) -> _ProviderPreflight:
    expected = ProviderAttestation(
        ModelIdentity(
            served_name=request.model,
            checkpoint_sha256=request.checkpoint_sha256,
            tokenizer_revision=request.tokenizer_revision,
            chat_template_revision=request.chat_template_revision,
        ),
        ProviderIdentity(
            server_revision=request.expected_server_revision,
            wkv_mode=request.wkv_mode,
            precision=request.precision,
            gemm_policy=request.gemm_policy,
            launch_contract=request.launch_contract,
        ),
        ("openai-chat", "output-token-ids", "terminal-reason", "prompt-evidence"),
    )
    with httpx.Client(timeout=30.0) as client:
        actual = fetch_provider_attestation(
            base_url=request.endpoint_url, client=client
        )
    decision = validate_attestation(
        expected,
        actual,
        official=definition.contract.can_publish_official
        and not request.allow_non_comparable,
        allow_non_comparable=request.allow_non_comparable,
    )
    eligibility = _effective_eligibility(request, definition, decision)
    return _ProviderPreflight(actual, actual or expected, decision, eligibility)


def _effective_eligibility(
    request: EvaluationRequest,
    definition: TaskDefinition,
    decision: AttestationDecision,
) -> Eligibility:
    if definition.contract.eligibility is not Eligibility.OFFICIAL:
        return definition.contract.eligibility
    if request.allow_non_comparable or decision.mismatches:
        return Eligibility.PROXY
    if request.max_samples is not None or request.generation_limit_override is not None:
        return Eligibility.SANITY
    return Eligibility.OFFICIAL


def _execute_samples(
    request: EvaluationRequest, prepared: _PreparedRun
) -> _RunEvidence:
    samples: list[dict[str, Any]] = []
    formatter_rejections: list[dict[str, str]] = []
    primary_metrics: list[float] = []
    generated_samples = 0
    truncated_samples = 0
    cancelled = False
    endpoint = OpenAIEndpoint(
        base_url=request.endpoint_url,
        api_key=request.endpoint_api_key,
    )
    evaluator = SampleEvaluator(
        definition=prepared.definition,
        runtime=prepared.runtime,
        endpoint=endpoint,
        context=SampleExecutionContext(
            model=request.model,
            generation_limit=prepared.plan.generation_limit.final,
            cot_mode=request.cot_mode,
            math_repair_strategy=request.math_repair_strategy,
            cache_namespace=prepared.plan.cache_namespace,
            config_digest=prepared.config_digest,
            asset_manifest_digest=prepared.asset_manifest_digest,
            provider_attestation=prepared.provider.actual,
        ),
    )
    try:
        for index, row in enumerate(prepared.selected):
            try:
                result = evaluator.evaluate(row.row_id, row.payload)
            except KeyboardInterrupt:
                cancelled = True
                samples.extend(
                    evaluator.cancelled(pending.row_id)
                    for pending in prepared.selected[index:]
                )
                break
            if result.formatter_rejection is not None:
                formatter_rejections.append(result.formatter_rejection)
            if result.evidence is not None:
                samples.append(result.evidence)
            if result.primary_metric is not None:
                primary_metrics.append(result.primary_metric)
            generated_samples += int(result.generated)
            truncated_samples += int(result.truncated)
    finally:
        endpoint.close()
    return _RunEvidence(
        tuple(samples),
        tuple(formatter_rejections),
        tuple(primary_metrics),
        generated_samples,
        truncated_samples,
        cancelled,
    )


def _aggregate_evidence(
    prepared: _PreparedRun, evidence: _RunEvidence
) -> _RunAggregation:
    status_counts = {
        name: sum(sample["status"] == name for sample in evidence.samples)
        for name in (
            "scored",
            "model_invalid",
            "provider_error",
            "cache_error",
            "scorer_error",
            "harness_error",
            "cancelled",
        )
    }
    accounting = SampleAccounting(
        source_rows=prepared.snapshot.source_rows,
        dataset_accepted=len(prepared.snapshot.accepted_rows),
        dataset_rejected=len(prepared.snapshot.rejected_rows),
        selected=len(prepared.selected),
        formatter_accepted=len(evidence.samples),
        formatter_rejected=len(evidence.formatter_rejections),
        **status_counts,
    )
    accounting.validate()
    completed = (
        not prepared.snapshot.rejected_rows
        and not evidence.formatter_rejections
        and status_counts["scored"] == len(prepared.selected)
        and bool(evidence.primary_metrics)
    )
    invalid_official_input = (
        prepared.provider.eligibility is Eligibility.OFFICIAL
        and bool(prepared.snapshot.rejected_rows or evidence.formatter_rejections)
    )
    status = _terminal_status(
        cancelled=evidence.cancelled,
        invalid_official_input=invalid_official_input,
        completed=completed,
        has_scores=bool(evidence.primary_metrics),
    )
    metrics = (
        {
            prepared.definition.contract.metric.primary_metric: (
                prepared.definition.contract.metric.aggregate(
                    list(evidence.primary_metrics)
                )
            )
        }
        if evidence.primary_metrics
        else {}
    )
    rejections = (
        tuple(
            {
                "stage": "dataset",
                "row_identity": item.row_id,
                "reason_code": item.reason,
                "message": item.reason,
            }
            for item in prepared.snapshot.rejected_rows
        )
        + evidence.formatter_rejections
    )
    return _RunAggregation(accounting, status, metrics, rejections)


def _terminal_status(
    *,
    cancelled: bool,
    invalid_official_input: bool,
    completed: bool,
    has_scores: bool,
) -> RunStatus:
    if cancelled:
        return RunStatus.CANCELLED
    if invalid_official_input:
        return RunStatus.INVALID
    if completed:
        return RunStatus.COMPLETED
    if has_scores:
        return RunStatus.PARTIAL
    return RunStatus.FAILED


def _commit_run(
    request: EvaluationRequest,
    prepared: _PreparedRun,
    evidence: _RunEvidence,
    aggregation: _RunAggregation,
    artifacts: RunArtifacts,
) -> _FinalizedRun:
    lifecycle = RunLifecycle()
    lifecycle.advance(RunStatus.RUNNING)
    if aggregation.status is RunStatus.CANCELLED:
        lifecycle.advance(RunStatus.CANCELLED)
    else:
        lifecycle.advance(RunStatus.FINALIZING)
        lifecycle.advance(aggregation.status)
    identity = build_scoreboard_identity(
        prepared.plan,
        prepared.definition,
        prepared.provider.observed,
        prepared.provider.decision,
    )
    identity_digest = _digest(identity)
    artifacts.write_json("samples.json", list(evidence.samples))
    artifacts.write_json(
        "config.json",
        request.config_evidence
        or {
            "effective_config_digest": prepared.config_digest,
            "source": "direct-application-request",
        },
    )
    artifacts.write_json(
        "summary.json",
        {
            "status": aggregation.status.value,
            "metrics": aggregation.metrics,
            "truncated_samples": evidence.truncated_samples,
            "generated_samples": evidence.generated_samples,
            "truncation_rate": _truncation_rate(
                evidence.truncated_samples, evidence.generated_samples
            ),
            "rejections": list(aggregation.rejections),
        },
    )
    artifacts.write_json(
        "performance.json", summarize_run_performance(evidence.samples).to_payload()
    )
    manifest_path = artifacts.finalize(
        status=aggregation.status,
        identity_digest=identity_digest,
        identities={
            "run": identity,
            "cache_namespace": prepared.plan.cache_namespace,
            "cache_policy_revision": DISABLED_CACHE_POLICY_REVISION,
            "asset_manifest_digest": prepared.plan.asset_manifest_digest,
        },
        accounting=aggregation.accounting,
    )
    verify_manifest(manifest_path)
    _validate_completed_run(prepared, evidence, aggregation)
    return _FinalizedRun(
        manifest_path,
        aggregation.status,
        aggregation.metrics,
        evidence.generated_samples,
        evidence.truncated_samples,
    )


def _validate_completed_run(
    prepared: _PreparedRun,
    evidence: _RunEvidence,
    aggregation: _RunAggregation,
) -> None:
    if aggregation.status is not RunStatus.COMPLETED:
        return
    RunSummary(
        run_id=prepared.plan.run_id,
        status=aggregation.status,
        eligibility=prepared.provider.eligibility,
        accounting=aggregation.accounting,
        metrics=aggregation.metrics,
        truncated_samples=evidence.truncated_samples,
        generated_samples=evidence.generated_samples,
    ).validate_completed(
        provider_valid=(
            prepared.provider.actual is not None
            or prepared.provider.eligibility is Eligibility.PROXY
        ),
        aggregation_valid=(
            set(aggregation.metrics)
            == {prepared.definition.contract.metric.primary_metric}
            and len(evidence.primary_metrics) == aggregation.accounting.scored
        ),
        manifest_committed=True,
    )


def _publish_if_requested(
    request: EvaluationRequest, finalized: _FinalizedRun
) -> PublicationResult:
    if not request.publish_to_scoreboard:
        return PublicationResult("not_requested", finalized.manifest_path.parent.name)
    publication = publish_artifact(
        manifest_path=finalized.manifest_path,
        base_url=str(request.scoreboard_url),
        bearer_token=str(request.scoreboard_token),
    )
    record_publication_attempt(
        finalized.manifest_path,
        {
            "status": publication.status,
            "retry_identity": publication.retry_identity,
            "error": publication.error,
        },
    )
    return publication


def retry_scoreboard_publication(
    *, manifest_path: Path, scoreboard_url: str, scoreboard_token: str
) -> EvaluationOutcome:
    manifest = verify_manifest(manifest_path)
    publication = retry_publication(
        manifest_path=manifest_path,
        base_url=scoreboard_url,
        bearer_token=scoreboard_token,
    )
    record_publication_attempt(
        manifest_path,
        {
            "status": publication.status,
            "retry_identity": publication.retry_identity,
            "error": publication.error,
        },
    )
    return EvaluationOutcome(
        run_id=manifest.run_id,
        run_status=manifest.status.value,
        manifest_path=manifest_path,
        publication_status=publication.status,
        publication_error=publication.error,
        publication_retry_identity=publication.retry_identity,
    )


def _fallback_config_digest(request: EvaluationRequest) -> str:
    return _digest(
        {
            "task": request.task,
            "max_samples": request.max_samples,
            "generation_limit_override": request.generation_limit_override,
            "generation_limit_override_source": request.generation_limit_override_source,
            "endpoint_url": request.endpoint_url,
            "publish_to_scoreboard": request.publish_to_scoreboard,
        }
    )


def _task_identity(
    request: EvaluationRequest,
    definition: TaskDefinition,
    eligibility: Eligibility,
    dataset_digest: str,
) -> TaskIdentity:
    task_name, version = definition.contract.identity.rsplit("@", 1)
    suite, task = task_name.split("/", 1)
    return TaskIdentity(
        suite=suite,
        task=task,
        version=version,
        dataset_digest=dataset_digest,
        split=definition.dataset_split,
        fewshot=0,
        prompt_revision=definition.contract.prompt_revision,
        scorer_revision=definition.contract.scorer_revision,
        generation_contract=definition.contract.generation_contract,
        cot_mode=request.cot_mode,
        repair_strategy=(
            request.math_repair_strategy
            if definition.family == "math"
            else "not-applicable"
        ),
        eligibility=eligibility,
    )


def _validate_dataset_row(row: dict[str, Any], required: frozenset[str]) -> str | None:
    missing = sorted(required - set(row))
    return f"missing_columns:{','.join(missing)}" if missing else None


def _truncation_rate(truncated: int, generated: int) -> float | None:
    return truncated / generated if generated else None


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()
