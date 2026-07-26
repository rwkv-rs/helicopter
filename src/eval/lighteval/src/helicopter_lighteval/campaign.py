from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

from .artifacts import publications_from_shard
from .config import EvaluationEnvironment
from .http_client import ScoreboardClient, ScoreboardError
from .manifest import (
    MANIFEST_VERSION,
    CampaignManifest,
    ManifestError,
    ManifestStore,
    campaign_directory,
    create_campaign_directory,
    ensure_private_staging_root,
    remove_acknowledged_shard,
    remove_empty_campaign,
    validate_campaign_child,
    write_control_metadata,
)
from .plan import EvaluationPlan, EvaluationUnit


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _failure_summary(error: BaseException) -> str:
    if isinstance(error, (ManifestError, ScoreboardError)):
        return str(error)
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _task_identity(unit: EvaluationUnit, task_identity: str) -> str:
    return f"{unit.weight.sha256}:{unit.wkv_mode}:{task_identity}"


def _expected_tasks(plan: EvaluationPlan) -> list[dict[str, object]]:
    expected: list[dict[str, object]] = []
    for unit in plan.units:
        for task in plan.registry.tasks:
            expected.append(
                {
                    "identity": _task_identity(unit, task.identity),
                    "weight_sha256": unit.weight.sha256,
                    "weight_display_name": unit.weight.display_name,
                    "wkv_mode": unit.wkv_mode,
                    "selector": task.selector,
                    "task_name": task.identity,
                    "task_version": task.version,
                    "module_family": task.module_family,
                    "module": task.module,
                    "dataset": task.dataset,
                    "subset": task.subset,
                    "evaluation_splits": list(task.evaluation_splits),
                    "languages": list(task.languages),
                    "upstream_tags": list(task.upstream_tags),
                }
            )
    return expected


def _resume_key(plan: EvaluationPlan) -> str:
    raw = json.dumps(
        {
            "config": plan.config_digest,
            "weights": list(dict.fromkeys(unit.weight.sha256 for unit in plan.units)),
            "registry": plan.registry.digest,
            "eval_contract": plan.eval_contract_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _campaign_payload(plan: EvaluationPlan, resume_key: str) -> dict[str, object]:
    return {
        "schema_version": "lighteval-campaign-v2",
        "resume_key": resume_key,
        "config_digest": plan.config_digest,
        "registry_digest": plan.registry.digest,
        "eval_contract_digest": plan.eval_contract_digest,
        "lighteval_version": plan.registry.lighteval_version,
        "configured_selectors": list(plan.registry.configured_selectors),
        "resolved_selectors": list(plan.registry.resolved_selectors),
        "skipped_selectors": list(plan.registry.skipped_selectors),
        "expected_tasks": _expected_tasks(plan),
    }


def _new_manifest(
    plan: EvaluationPlan, resume_key: str, campaign_id: str
) -> CampaignManifest:
    return CampaignManifest(
        version=MANIFEST_VERSION,
        resume_key=resume_key,
        campaign_id=campaign_id,
        config_digest=plan.config_digest,
        registry_digest=plan.registry.digest,
        eval_contract_digest=plan.eval_contract_digest,
        weight_sha256=list(dict.fromkeys(unit.weight.sha256 for unit in plan.units)),
        configured_selectors=list(plan.registry.configured_selectors),
        resolved_selectors=list(plan.registry.resolved_selectors),
        skipped_selectors=list(plan.registry.skipped_selectors),
        registry_task_identities=[task.identity for task in plan.registry.tasks],
    )


def _check_manifest_contract(
    manifest: CampaignManifest,
    plan: EvaluationPlan,
    resume_key: str,
) -> None:
    expected = {
        "resume_key": resume_key,
        "config_digest": plan.config_digest,
        "registry_digest": plan.registry.digest,
        "eval_contract_digest": plan.eval_contract_digest,
    }
    mismatched = [
        name for name, value in expected.items() if getattr(manifest, name) != value
    ]
    if mismatched:
        raise ManifestError(
            "local manifest does not match evaluation contract: "
            + ", ".join(mismatched)
        )


def _validate_manifest_plan(
    manifest: CampaignManifest,
    plan: EvaluationPlan,
) -> None:
    if manifest.configured_selectors != list(plan.registry.configured_selectors):
        raise ManifestError("campaign manifest configured selectors do not match plan")
    if manifest.resolved_selectors != list(plan.registry.resolved_selectors):
        raise ManifestError("campaign manifest resolved selectors do not match plan")
    if manifest.skipped_selectors != list(plan.registry.skipped_selectors):
        raise ManifestError("campaign manifest skipped selectors do not match plan")
    expected_weights = list(dict.fromkeys(unit.weight.sha256 for unit in plan.units))
    if manifest.weight_sha256 != expected_weights:
        raise ManifestError("campaign manifest weight snapshot does not match plan")
    expected_registry = [task.identity for task in plan.registry.tasks]
    if manifest.registry_task_identities != expected_registry:
        raise ManifestError("campaign manifest registry snapshot does not match plan")
    expected_tasks = {
        _task_identity(unit, task.identity)
        for unit in plan.units
        for task in plan.registry.tasks
    }
    expected_shards = {
        f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
        for unit in plan.units
        for shard in unit.shards
    }
    expected_units = {f"{unit.weight.sha256}:{unit.wkv_mode}" for unit in plan.units}
    for name in (
        "pending_task_digests",
        "acknowledged_task_digests",
    ):
        unexpected = set(getattr(manifest, name)) - expected_tasks
        if unexpected:
            raise ManifestError(f"campaign manifest {name} contains unknown tasks")
    for name in (
        "shard_paths",
        "attempted_shard_paths",
        "failed_shard_paths",
        "failed_shard_errors",
    ):
        unexpected = set(getattr(manifest, name)) - expected_shards
        if unexpected:
            raise ManifestError(f"campaign manifest {name} contains unknown shards")
    if set(manifest.runtime_paths) - expected_units:
        raise ManifestError("campaign manifest runtime_paths contains unknown units")
    for unit_key, runtime_path in manifest.runtime_paths.items():
        weight_sha256, wkv_mode = unit_key.split(":", 1)
        expected_runtime_path = str(Path("runtime") / weight_sha256 / wkv_mode)
        if runtime_path != expected_runtime_path:
            raise ManifestError(
                "campaign manifest runtime path does not match its unit"
            )
    if set(manifest.shard_paths) & set(manifest.attempted_shard_paths):
        raise ManifestError(
            "campaign manifest has overlapping complete and attempted shards"
        )
    if set(manifest.failed_shard_errors) != set(manifest.failed_shard_paths):
        raise ManifestError(
            "campaign manifest failure records do not match failed shards"
        )
    if set(manifest.model_executions) - expected_units:
        raise ManifestError("campaign manifest model_executions contains unknown units")
    shard_state_keys = (
        set(manifest.shard_paths)
        | set(manifest.attempted_shard_paths)
        | set(manifest.failed_shard_paths)
    )
    for shard_key in shard_state_keys:
        weight_sha256, wkv_mode, _ = shard_key.split(":", 2)
        if f"{weight_sha256}:{wkv_mode}" not in manifest.model_executions:
            raise ManifestError(
                "campaign manifest shard state lacks model execution metadata"
            )
    paths = [
        *manifest.shard_paths.values(),
        *manifest.attempted_shard_paths.values(),
        *(
            path
            for shard_paths in manifest.failed_shard_paths.values()
            for path in shard_paths
        ),
        *manifest.runtime_paths.values(),
    ]
    if len(paths) != len(set(paths)):
        raise ManifestError("campaign manifest reuses a staging path")


def _canonical_campaign_id(raw_campaign_id: object) -> str:
    try:
        campaign_id = str(uuid.UUID(str(raw_campaign_id)))
    except (ValueError, TypeError) as error:
        raise ScoreboardError("Scoreboard returned an invalid campaign id") from error
    if campaign_id != raw_campaign_id:
        raise ScoreboardError("Scoreboard campaign id is not canonical")
    return campaign_id


def _validate_status(
    *,
    status: dict[str, object],
    campaign_id: str,
    plan: EvaluationPlan,
) -> tuple[str, dict[str, str], list[str]]:
    if status.get("campaign_id") != campaign_id:
        raise ScoreboardError("campaign status id does not match request")
    state = status.get("status")
    if state not in {"incomplete", "complete"}:
        raise ScoreboardError("campaign status is invalid")
    task_count = status.get("expected_task_count")
    if isinstance(task_count, bool) or task_count != plan.expected_task_count:
        raise ScoreboardError("campaign status task count does not match plan")
    backend_digests = status.get("acknowledged_task_digests")
    missing = status.get("missing_task_identities")
    if not isinstance(backend_digests, dict) or not all(
        isinstance(identity, str)
        and isinstance(digest, str)
        and _DIGEST.fullmatch(digest) is not None
        for identity, digest in backend_digests.items()
    ):
        raise ScoreboardError("campaign status lacks valid task digests")
    if (
        not isinstance(missing, list)
        or not all(isinstance(identity, str) for identity in missing)
        or len(missing) != len(set(missing))
    ):
        raise ScoreboardError("campaign status lacks valid missing tasks")
    expected = {
        item["identity"]
        for item in _expected_tasks(plan)
        if isinstance(item["identity"], str)
    }
    if set(backend_digests) | set(missing) != expected:
        raise ScoreboardError("campaign status task identities do not match plan")
    if set(backend_digests) & set(missing):
        raise ScoreboardError("campaign status reports overlapping task states")
    if state == "complete" and (
        missing or len(backend_digests) != plan.expected_task_count
    ):
        raise ScoreboardError("complete campaign is missing task publications")
    return state, backend_digests, missing


def _validate_campaign_receipt(
    receipt: dict[str, object],
    *,
    plan: EvaluationPlan,
) -> str:
    campaign_id = _canonical_campaign_id(receipt.get("campaign_id"))
    if receipt.get("status") != "incomplete":
        raise ScoreboardError("new campaign receipt must be incomplete")
    if receipt.get("disposition") not in {"created", "resumed"}:
        raise ScoreboardError("new campaign receipt disposition is invalid")
    task_count = receipt.get("expected_task_count")
    if isinstance(task_count, bool) or task_count != plan.expected_task_count:
        raise ScoreboardError("new campaign receipt task count does not match plan")
    acknowledged = receipt.get("acknowledged_task_digests")
    expected = {
        item["identity"]
        for item in _expected_tasks(plan)
        if isinstance(item["identity"], str)
    }
    if not isinstance(acknowledged, dict) or not all(
        isinstance(identity, str)
        and identity in expected
        and isinstance(digest, str)
        and _DIGEST.fullmatch(digest) is not None
        for identity, digest in acknowledged.items()
    ):
        raise ScoreboardError("new campaign receipt lacks task digests")
    return campaign_id


def _reconcile(
    manifest: CampaignManifest,
    backend_digests: dict[str, str],
) -> None:
    for identity, digest in backend_digests.items():
        pending = manifest.pending_task_digests.get(identity)
        acknowledged = manifest.acknowledged_task_digests.get(identity)
        local = pending or acknowledged
        if local is not None and local != digest:
            raise ManifestError(
                f"backend digest conflicts with local task digest: {identity}"
            )
        manifest.acknowledged_task_digests[identity] = digest
        manifest.pending_task_digests.pop(identity, None)


def _cleanup_completed_shards(
    *,
    plan: EvaluationPlan,
    manifest: CampaignManifest,
    environment: EvaluationEnvironment,
) -> None:
    for unit in plan.units:
        for shard in unit.shards:
            key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
            shard_path = manifest.shard_paths.get(key)
            identities = [_task_identity(unit, task.identity) for task in shard.tasks]
            if all(
                identity in manifest.acknowledged_task_digests
                for identity in identities
            ):
                if shard_path is not None:
                    remove_acknowledged_shard(
                        staging_root=environment.staging_root,
                        campaign_id=manifest.campaign_id,
                        shard_path=shard_path,
                    )
                    manifest.shard_paths.pop(key, None)
                for failed_path in manifest.failed_shard_paths.pop(key, []):
                    remove_acknowledged_shard(
                        staging_root=environment.staging_root,
                        campaign_id=manifest.campaign_id,
                        shard_path=failed_path,
                    )
                manifest.failed_shard_errors.pop(key, None)


def _cleanup_interrupted_attempts(
    *,
    manifest: CampaignManifest,
    environment: EvaluationEnvironment,
) -> None:
    for shard_key, shard_path in tuple(manifest.attempted_shard_paths.items()):
        remove_acknowledged_shard(
            staging_root=environment.staging_root,
            campaign_id=manifest.campaign_id,
            shard_path=shard_path,
        )
        manifest.attempted_shard_paths.pop(shard_key)


def _cleanup_interrupted_runtimes(
    *,
    manifest: CampaignManifest,
    environment: EvaluationEnvironment,
) -> None:
    for unit_key, runtime_path in tuple(manifest.runtime_paths.items()):
        remove_acknowledged_shard(
            staging_root=environment.staging_root,
            campaign_id=manifest.campaign_id,
            shard_path=runtime_path,
        )
        manifest.runtime_paths.pop(unit_key)


def _record_runtime_started(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    campaign_dir: Path,
    unit: EvaluationUnit,
    runtime_dir: Path,
) -> None:
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    candidate, relative = validate_campaign_child(campaign_dir, runtime_dir)
    if candidate.is_symlink():
        raise ManifestError("model runtime path must not be a symlink")
    relative_path = str(relative)
    expected_path = str(Path("runtime") / unit.weight.sha256 / unit.wkv_mode)
    if relative_path != expected_path:
        raise ManifestError(f"model runtime path is not deterministic: {unit_key}")
    known = manifest.runtime_paths.get(unit_key)
    if known is not None and known != relative_path:
        raise ManifestError(f"model runtime path changed: {unit_key}")
    manifest.runtime_paths[unit_key] = relative_path
    store.save(manifest)


def _record_runtime_finished(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    unit: EvaluationUnit,
) -> None:
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    if unit_key not in manifest.runtime_paths:
        raise ManifestError(f"model runtime completion was not registered: {unit_key}")
    manifest.runtime_paths.pop(unit_key)
    store.save(manifest)


def _control_metadata(
    *,
    plan: EvaluationPlan,
    campaign_id: str,
) -> dict[str, object]:
    return {
        "schema_version": "lighteval-control-v2",
        "campaign_id": campaign_id,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_digest": plan.config_digest,
        "implementation_digest": plan.implementation_digest,
        "registry_digest": plan.registry.digest,
        "eval_contract_digest": plan.eval_contract_digest,
        "prompt_template": plan.prompt_template,
        "configured_selectors": list(plan.registry.configured_selectors),
        "resolved_selectors": list(plan.registry.resolved_selectors),
        "skipped_selectors": list(plan.registry.skipped_selectors),
        "weight_sha256": list(dict.fromkeys(unit.weight.sha256 for unit in plan.units)),
        "wkv_modes": ["fp16", "fp32io16"],
        "expected_task_count": plan.expected_task_count,
        "content_location": "scoreboard-database-only",
    }


def _finish_local_campaign(
    *,
    plan: EvaluationPlan,
    environment: EvaluationEnvironment,
    store: ManifestStore,
    manifest: CampaignManifest,
) -> Path:
    _cleanup_completed_shards(
        plan=plan,
        manifest=manifest,
        environment=environment,
    )
    store.save(manifest)
    if manifest.shard_paths:
        raise ManifestError("acknowledged campaign still has shard artifacts")
    if manifest.failed_shard_paths:
        raise ManifestError("acknowledged campaign still has failed shard artifacts")
    if manifest.failed_shard_errors:
        raise ManifestError("acknowledged campaign still has shard failure records")
    if manifest.runtime_paths:
        raise ManifestError("acknowledged campaign still has model runtime artifacts")
    remove_empty_campaign(environment.staging_root, manifest.campaign_id)
    metadata_path = write_control_metadata(
        staging_root=environment.staging_root,
        campaign_id=manifest.campaign_id,
        metadata=_control_metadata(
            plan=plan,
            campaign_id=manifest.campaign_id,
        ),
    )
    store.delete()
    return metadata_path


def _publish_shard(
    *,
    client: ScoreboardClient,
    store: ManifestStore,
    manifest: CampaignManifest,
    plan: EvaluationPlan,
    environment: EvaluationEnvironment,
    unit: EvaluationUnit,
    shard,
    shard_dir,
    model_execution: dict[str, object],
) -> None:
    publications = publications_from_shard(
        shard_dir=shard_dir,
        campaign_id=manifest.campaign_id,
        unit=unit,
        shard=shard,
        model_execution=model_execution,
        registry_tasks=plan.registry.tasks,
    )
    for identity, payload, digest in publications:
        known = manifest.pending_task_digests.get(identity)
        acknowledged = manifest.acknowledged_task_digests.get(identity)
        if known is not None and known != digest:
            raise ManifestError(f"recomputed task digest changed: {identity}")
        if acknowledged is not None:
            if acknowledged != digest:
                raise ManifestError(f"acknowledged task digest changed: {identity}")
            continue
        manifest.pending_task_digests[identity] = digest
        store.save(manifest)
        task_receipt = client.publish_task(
            campaign_id=manifest.campaign_id,
            task_identity=identity,
            payload=payload,
            digest=digest,
        )
        if (
            task_receipt.get("content_digest") != digest
            or task_receipt.get("task_identity") != identity
            or task_receipt.get("disposition") not in {"created", "unchanged"}
        ):
            raise ScoreboardError(f"invalid publication receipt for {identity}")
        manifest.acknowledged_task_digests[identity] = digest
        manifest.pending_task_digests.pop(identity, None)
        store.save(manifest)
    _cleanup_completed_shards(
        plan=plan,
        manifest=manifest,
        environment=environment,
    )
    store.save(manifest)


def _record_shard_artifact(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    campaign_dir: Path,
    unit: EvaluationUnit,
    shard,
    shard_dir: Path,
    model_execution: dict[str, object],
) -> None:
    shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    candidate, relative = validate_campaign_child(campaign_dir, shard_dir)
    if not relative.parts or candidate.is_symlink() or not candidate.is_dir():
        raise ManifestError("evaluated shard path is not a safe directory")
    known_path = manifest.shard_paths.get(shard_key)
    if known_path is not None and known_path != str(relative):
        raise ManifestError(f"shard artifact path changed: {shard_key}")
    attempted_path = manifest.attempted_shard_paths.get(shard_key)
    if attempted_path != str(relative):
        raise ManifestError(
            f"shard artifact was not the registered attempt: {shard_key}"
        )
    known_execution = manifest.model_executions.get(unit_key)
    if known_execution is not None and known_execution != model_execution:
        raise ManifestError(f"model execution metadata changed: {unit_key}")
    manifest.shard_paths[shard_key] = str(relative)
    manifest.attempted_shard_paths.pop(shard_key)
    manifest.model_executions[unit_key] = model_execution
    store.save(manifest)


def _record_shard_attempt(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    campaign_dir: Path,
    unit: EvaluationUnit,
    shard,
    shard_dir: Path,
    model_execution: dict[str, object],
) -> None:
    shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
    unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
    _, relative = validate_campaign_child(campaign_dir, shard_dir)
    relative_path = str(relative)
    if shard_key in manifest.shard_paths:
        raise ManifestError(f"cannot start an already completed shard: {shard_key}")
    known_attempt = manifest.attempted_shard_paths.get(shard_key)
    if known_attempt is not None and known_attempt != relative_path:
        raise ManifestError(f"shard attempt path changed: {shard_key}")
    known_execution = manifest.model_executions.get(unit_key)
    if known_execution is not None and known_execution != model_execution:
        raise ManifestError(f"model execution metadata changed: {unit_key}")
    manifest.attempted_shard_paths[shard_key] = relative_path
    manifest.model_executions[unit_key] = model_execution
    store.save(manifest)


def _record_shard_failure(
    *,
    store: ManifestStore,
    manifest: CampaignManifest,
    campaign_dir: Path,
    unit: EvaluationUnit,
    failure,
) -> None:
    shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{failure.shard.shard_id}"
    candidate, relative = validate_campaign_child(
        campaign_dir,
        failure.path,
    )
    if not candidate.is_dir() or candidate.is_symlink():
        raise ManifestError("failed shard path is not a safe directory")
    relative_path = str(relative)
    if manifest.attempted_shard_paths.get(shard_key) != relative_path:
        raise ManifestError(f"failed shard was not the registered attempt: {shard_key}")
    manifest.attempted_shard_paths.pop(shard_key)
    paths = manifest.failed_shard_paths.setdefault(shard_key, [])
    if relative_path not in paths:
        paths.append(relative_path)
    manifest.failed_shard_errors[shard_key] = failure.error_type
    store.save(manifest)


def run_campaign(
    *,
    plan: EvaluationPlan,
    environment: EvaluationEnvironment,
) -> int:
    ensure_private_staging_root(environment.staging_root)
    if plan.registry.skipped_selectors:
        print(
            "skipped benchmark selectors unavailable in this LightEval release: "
            + ", ".join(plan.registry.skipped_selectors)
        )
    client = ScoreboardClient(environment)
    resume_key = _resume_key(plan)
    store = ManifestStore(environment.staging_root, resume_key)
    try:
        manifest = store.load()
    except ManifestError as error:
        quarantined = store.quarantine()
        print(
            "isolated an unreadable local campaign manifest without touching "
            f"run content: {quarantined}; reason: {error}"
        )
        manifest = None
    if manifest is not None:
        try:
            _check_manifest_contract(manifest, plan, resume_key)
            _validate_manifest_plan(manifest, plan)
        except ManifestError:
            quarantined = store.quarantine()
            print(
                "isolated a local manifest that does not match the current "
                f"evaluation contract: {quarantined}"
            )
            manifest = None
    if manifest is not None:
        campaign_id = _canonical_campaign_id(manifest.campaign_id)
        state, backend_digests, _ = _validate_status(
            status=client.campaign_status(campaign_id),
            campaign_id=campaign_id,
            plan=plan,
        )
        _reconcile(manifest, backend_digests)
        _cleanup_interrupted_attempts(
            manifest=manifest,
            environment=environment,
        )
        _cleanup_interrupted_runtimes(
            manifest=manifest,
            environment=environment,
        )
        store.save(manifest)
        if state == "complete":
            metadata_path = _finish_local_campaign(
                plan=plan,
                environment=environment,
                store=store,
                manifest=manifest,
            )
            print(
                f"recovered completed campaign {campaign_id}; evaluation content "
                f"retained only by Scoreboard; control metadata: {metadata_path}"
            )
            # A matching local manifest proves this invocation is recovering
            # the prior command after backend finalization, not requesting a
            # new evaluation. Once local cleanup finishes, that command is
            # complete. A later invocation has no manifest and therefore
            # creates a fresh campaign instead of treating this one as cache.
            return 0

    receipt = client.create_campaign(
        _campaign_payload(plan, resume_key),
        resume_key,
    )
    campaign_id = _validate_campaign_receipt(receipt, plan=plan)
    if manifest is None:
        existing_campaign_dir = campaign_directory(
            environment.staging_root,
            campaign_id,
        )
        if existing_campaign_dir.exists():
            if existing_campaign_dir.is_symlink() or not existing_campaign_dir.is_dir():
                raise ManifestError("resumed campaign has an unsafe local run path")
            if any(existing_campaign_dir.iterdir()):
                raise ManifestError(
                    "resumed campaign has local run content without a matching "
                    "manifest; refusing to guess, reuse, or delete it"
                )
        manifest = _new_manifest(plan, resume_key, campaign_id)
    elif manifest.campaign_id != campaign_id:
        raise ManifestError(
            "local manifest campaign does not match resumed backend campaign"
        )
    state, backend_digests, _ = _validate_status(
        status=client.campaign_status(campaign_id),
        campaign_id=campaign_id,
        plan=plan,
    )
    if state != "incomplete":
        raise ScoreboardError("campaign became complete before evaluation started")
    _reconcile(manifest, backend_digests)
    _validate_manifest_plan(manifest, plan)
    _cleanup_interrupted_attempts(
        manifest=manifest,
        environment=environment,
    )
    _cleanup_interrupted_runtimes(
        manifest=manifest,
        environment=environment,
    )
    _cleanup_completed_shards(
        plan=plan,
        manifest=manifest,
        environment=environment,
    )
    store.save(manifest)

    failures: list[str] = []
    from .lighteval_adapter import UnsafeModelCleanupError, evaluate_unit

    campaign_dir = create_campaign_directory(
        environment.staging_root,
        campaign_id,
    )
    for unit in plan.units:
        unit_key = f"{unit.weight.sha256}:{unit.wkv_mode}"
        for shard in unit.shards:
            shard_key = f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
            relative = manifest.shard_paths.get(shard_key)
            model_execution = manifest.model_executions.get(unit_key)
            if relative is None or model_execution is None:
                continue
            shard_dir, _ = validate_campaign_child(
                campaign_dir,
                campaign_dir / relative,
            )
            if not shard_dir.is_dir() or shard_dir.is_symlink():
                raise ManifestError(
                    f"manifest shard artifact is unavailable: {relative}"
                )
            try:
                _publish_shard(
                    client=client,
                    store=store,
                    manifest=manifest,
                    plan=plan,
                    environment=environment,
                    unit=unit,
                    shard=shard,
                    shard_dir=shard_dir,
                    model_execution=model_execution,
                )
            except ManifestError:
                raise
            except Exception as error:
                failures.append(
                    f"{unit.weight.display_name}/{unit.wkv_mode}/"
                    f"{shard.shard_id} resume: {_failure_summary(error)}"
                )
        pending_shards = [
            shard
            for shard in unit.shards
            if (
                f"{unit.weight.sha256}:{unit.wkv_mode}:{shard.shard_id}"
                not in manifest.shard_paths
                and any(
                    _task_identity(unit, task.identity)
                    not in manifest.acknowledged_task_digests
                    for task in shard.tasks
                )
            )
        ]
        if not pending_shards:
            continue

        def on_shard_started(shard, shard_dir, model_execution):
            _record_shard_attempt(
                store=store,
                manifest=manifest,
                campaign_dir=campaign_dir,
                unit=unit,
                shard=shard,
                shard_dir=shard_dir,
                model_execution=model_execution,
            )

        def on_shard_completed(evaluation):
            shard, shard_dir, model_execution = evaluation
            _record_shard_artifact(
                store=store,
                manifest=manifest,
                campaign_dir=campaign_dir,
                unit=unit,
                shard=shard,
                shard_dir=shard_dir,
                model_execution=model_execution,
            )
            try:
                _publish_shard(
                    client=client,
                    store=store,
                    manifest=manifest,
                    plan=plan,
                    environment=environment,
                    unit=unit,
                    shard=shard,
                    shard_dir=shard_dir,
                    model_execution=model_execution,
                )
            except ManifestError:
                raise
            except Exception as error:
                failures.append(
                    f"{unit.weight.display_name}/{unit.wkv_mode}/"
                    f"{shard.shard_id} publication: "
                    f"{_failure_summary(error)}"
                )

        def on_shard_failed(failure):
            _record_shard_failure(
                store=store,
                manifest=manifest,
                campaign_dir=campaign_dir,
                unit=unit,
                failure=failure,
            )
            failures.append(
                f"{unit.weight.display_name}/{unit.wkv_mode}/{failure.message}"
            )

        def on_runtime_started(runtime_dir):
            _record_runtime_started(
                store=store,
                manifest=manifest,
                campaign_dir=campaign_dir,
                unit=unit,
                runtime_dir=runtime_dir,
            )

        def on_runtime_finished():
            _record_runtime_finished(
                store=store,
                manifest=manifest,
                unit=unit,
            )

        try:
            evaluate_unit(
                unit=unit,
                shards=tuple(pending_shards),
                campaign_dir=campaign_dir,
                on_shard_started=on_shard_started,
                on_shard_completed=on_shard_completed,
                on_shard_failed=on_shard_failed,
                on_runtime_started=on_runtime_started,
                on_runtime_finished=on_runtime_finished,
            )
        except UnsafeModelCleanupError as error:
            raise ManifestError(
                "model lifecycle could not be proven safe; campaign stopped "
                f"before starting another weight or WKV mode: {error}"
            ) from error
        except ManifestError:
            raise
        except Exception as error:
            failures.append(
                f"{unit.weight.display_name}/{unit.wkv_mode}: {_failure_summary(error)}"
            )
            continue

    final_state, final_digests, missing = _validate_status(
        status=client.campaign_status(campaign_id),
        campaign_id=campaign_id,
        plan=plan,
    )
    if final_state != "incomplete":
        raise ScoreboardError("campaign completed without an explicit finalize")
    _reconcile(manifest, final_digests)
    _cleanup_completed_shards(
        plan=plan,
        manifest=manifest,
        environment=environment,
    )
    store.save(manifest)
    if failures or missing:
        for failure in failures:
            print(f"evaluation unit failed: {failure}")
        print(
            f"campaign incomplete: {len(missing) if isinstance(missing, list) else 'unknown'} tasks missing"
        )
        return 1

    finalized = client.finalize(campaign_id)
    if (
        finalized.get("campaign_id") != campaign_id
        or finalized.get("status") != "complete"
        or finalized.get("task_count") != plan.expected_task_count
    ):
        raise ScoreboardError("Scoreboard did not finalize campaign")
    finalized_state, finalized_digests, finalized_missing = _validate_status(
        status=client.campaign_status(campaign_id),
        campaign_id=campaign_id,
        plan=plan,
    )
    if finalized_state != "complete" or finalized_missing:
        raise ScoreboardError("Scoreboard finalized campaign inconsistently")
    _reconcile(manifest, finalized_digests)
    metadata_path = _finish_local_campaign(
        plan=plan,
        environment=environment,
        store=store,
        manifest=manifest,
    )
    print(
        f"campaign {campaign_id} complete; evaluation content retained only by "
        f"Scoreboard; control metadata: {metadata_path}"
    )
    return 0
