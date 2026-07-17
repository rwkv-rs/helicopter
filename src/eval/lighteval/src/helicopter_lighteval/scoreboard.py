"""HTTP-only publication of a completed LightEval run.

This module intentionally does not know about the scoreboard database, ORM,
repository layer, or frontend.  It validates the local run evidence, projects
the existing strict publication DTO, and sends one idempotent HTTP request.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import httpx


class ScoreboardPublicationError(RuntimeError):
    """The completed run cannot be published or the HTTP boundary failed."""


@dataclass(frozen=True, slots=True)
class PublicationResult:
    status: str
    retry_identity: str
    error: str | None = None
    task_id: int | None = None


def publish_manifest(
    *,
    manifest_path: Path,
    scoreboard_url: str,
    bearer_token: str,
    client: httpx.Client | None = None,
) -> PublicationResult:
    manifest = verify_manifest(manifest_path)
    payload = build_publication_payload(manifest_path, manifest=manifest)
    manifest_digest = _sha256(manifest_path.read_bytes())
    retry_identity = f"publish:{manifest_digest}"
    if not scoreboard_url.strip() or not bearer_token.strip():
        raise ScoreboardPublicationError("scoreboard URL and bearer token are required")

    owns_client = client is None
    http_client = client or httpx.Client(timeout=30.0)
    base_url = scoreboard_url.rstrip("/")
    body = gzip.compress(_canonical_bytes(payload))
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Content-Encoding": "gzip",
        "Idempotency-Key": retry_identity,
    }
    try:
        response = http_client.put(
            f"{base_url}/api/v1/evaluation-publications/{manifest['run_id']}",
            content=body,
            headers=headers,
        )
    except httpx.HTTPError as error:
        result = PublicationResult(
            "failed", retry_identity, f"scoreboard transport failed: {error}"
        )
        _write_receipt(manifest_path, result)
        if owns_client:
            http_client.close()
        return result
    if response.status_code not in {200, 201}:
        result = PublicationResult(
            "failed",
            retry_identity,
            f"scoreboard publication failed with HTTP {response.status_code}",
        )
        _write_receipt(manifest_path, result)
        if owns_client:
            http_client.close()
        return result
    try:
        receipt = response.json()
    except ValueError as error:
        if owns_client:
            http_client.close()
        raise ScoreboardPublicationError("scoreboard response is not JSON") from error
    if not isinstance(receipt, dict) or receipt.get("status") != "completed":
        if owns_client:
            http_client.close()
        raise ScoreboardPublicationError(
            "scoreboard returned an inconsistent publication receipt"
        )
    if receipt.get("run_id") != manifest["run_id"]:
        if owns_client:
            http_client.close()
        raise ScoreboardPublicationError(
            "scoreboard receipt run_id does not match manifest"
        )
    task_id = receipt.get("task_id")
    if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id <= 0:
        if owns_client:
            http_client.close()
        raise ScoreboardPublicationError("scoreboard returned an invalid task_id")
    result = PublicationResult("published", retry_identity, task_id=task_id)
    _write_receipt(manifest_path, result, response=receipt)
    if owns_client:
        http_client.close()
    return result


def retry_publication(
    *,
    manifest_path: Path,
    scoreboard_url: str,
    bearer_token: str,
    client: httpx.Client | None = None,
) -> PublicationResult:
    """Retry the same manifest with the same digest-derived idempotency key."""

    return publish_manifest(
        manifest_path=manifest_path,
        scoreboard_url=scoreboard_url,
        bearer_token=bearer_token,
        client=client,
    )


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScoreboardPublicationError("manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ScoreboardPublicationError("manifest must be a JSON object")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id
    ):
        raise ScoreboardPublicationError("manifest run_id is unsafe")
    if manifest_path.parent.name != run_id:
        raise ScoreboardPublicationError("manifest run_id does not match its directory")
    if payload.get("status") != "completed":
        raise ScoreboardPublicationError("only completed manifests are publishable")
    identity = payload.get("identities", {}).get("run")
    if not isinstance(identity, dict):
        raise ScoreboardPublicationError("manifest run identity is missing")
    if set(identity) != {
        "task",
        "model",
        "provider",
        "evaluator",
        "config_digest",
        "eligibility",
        "comparable",
    }:
        raise ScoreboardPublicationError(
            "manifest identity does not match scoreboard DTO"
        )
    identity_digest = payload.get("identity_digest")
    if identity_digest != _digest(identity):
        raise ScoreboardPublicationError(
            "manifest identity digest does not match identity"
        )
    accounting = payload.get("accounting")
    if not isinstance(accounting, dict):
        raise ScoreboardPublicationError("manifest accounting is missing")
    _validate_accounting(accounting)
    completed_at = payload.get("completed_at")
    if not isinstance(completed_at, str):
        raise ScoreboardPublicationError("manifest completed_at is missing")
    try:
        parsed_time = datetime.fromisoformat(completed_at)
    except ValueError as error:
        raise ScoreboardPublicationError("manifest completed_at is invalid") from error
    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise ScoreboardPublicationError("manifest completed_at must include timezone")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ScoreboardPublicationError("manifest artifacts are missing")
    seen: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ScoreboardPublicationError("manifest artifact entry is invalid")
        relative = entry.get("relative_path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ScoreboardPublicationError("manifest artifact entry is invalid")
        seen.add(relative)
        path = manifest_path.parent / relative
        try:
            encoded = path.read_bytes()
        except OSError as error:
            raise ScoreboardPublicationError(
                f"manifest artifact is missing: {relative}"
            ) from error
        if len(encoded) != size or _sha256(encoded) != digest:
            raise ScoreboardPublicationError(f"manifest checksum mismatch: {relative}")
    if "terminal_evidence.json" not in seen:
        raise ScoreboardPublicationError(
            "terminal_evidence.json is required for publication"
        )
    return payload


def build_publication_payload(
    manifest_path: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_data = (
        dict(manifest) if manifest is not None else verify_manifest(manifest_path)
    )
    evidence_path = manifest_path.parent / "terminal_evidence.json"
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScoreboardPublicationError(
            "terminal evidence is not valid JSON"
        ) from error
    if not isinstance(evidence, dict):
        raise ScoreboardPublicationError("terminal evidence must be an object")
    identity = manifest_data["identities"]["run"]
    if not isinstance(identity, dict):
        raise ScoreboardPublicationError("publication identity is missing")
    samples = evidence.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ScoreboardPublicationError(
            "completed publication requires scored samples"
        )
    metrics = evidence.get("metrics")
    if not isinstance(metrics, dict):
        raise ScoreboardPublicationError("terminal evidence metrics are missing")
    native_metrics = evidence.get("native_metrics")
    if not isinstance(native_metrics, dict):
        raise ScoreboardPublicationError("terminal native metrics are missing")
    primary_metric = identity.get("task", {}).get("primary_metric")
    if not isinstance(primary_metric, str) or primary_metric not in metrics:
        raise ScoreboardPublicationError("primary metric evidence is missing")
    if set(metrics) != {primary_metric}:
        raise ScoreboardPublicationError(
            "publication must contain only the primary metric"
        )
    if native_metrics.get(primary_metric) != metrics[primary_metric]:
        raise ScoreboardPublicationError(
            "native metrics do not preserve the primary aggregate"
        )
    accounting = manifest_data["accounting"]
    generated = evidence.get("generated_samples")
    truncated = evidence.get("truncated_samples")
    if (
        not isinstance(generated, int)
        or generated <= 0
        or not isinstance(truncated, int)
    ):
        raise ScoreboardPublicationError("terminal sample accounting is invalid")
    if generated != len(samples) or truncated < 0 or truncated > generated:
        raise ScoreboardPublicationError("terminal sample accounting does not close")
    published_samples = [_publication_sample(sample) for sample in samples]
    if truncated != sum(
        int(sample["generation"]["truncated"]) for sample in published_samples
    ):
        raise ScoreboardPublicationError("truncated sample evidence does not close")
    manifest_digest = _sha256(manifest_path.read_bytes())
    return {
        "identity": identity,
        "samples": published_samples,
        "accounting": accounting,
        "rejections": [],
        "metrics": {primary_metric: metrics[primary_metric]},
        "native_metrics": native_metrics,
        "primary_metric": primary_metric,
        "truncated_samples": truncated,
        "generated_samples": generated,
        "performance": evidence.get("performance", {}),
        "manifest": {
            "digest": manifest_digest,
            "identity_digest": manifest_data["identity_digest"],
            "accounting_digest": _digest(accounting),
            "terminal_status": "completed",
            "checksums_verified": True,
            "completed_at": manifest_data["completed_at"],
        },
        "terminal_status": "completed",
    }


def _publication_sample(sample: Any) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise ScoreboardPublicationError("sample evidence must be an object")
    generation = sample.get("generation")
    scoring = sample.get("scoring")
    metrics = sample.get("metrics")
    if (
        not isinstance(generation, dict)
        or not isinstance(scoring, dict)
        or not isinstance(metrics, dict)
    ):
        raise ScoreboardPublicationError("sample evidence is incomplete")
    required = (
        "sample_index",
        "sample_id",
        "attempt",
        "status",
        "prompt",
        "raw_completion",
        "scored_completion",
        "error_code",
        "error_message",
        "reference_answer",
        "provenance",
    )
    if any(field not in sample for field in required):
        raise ScoreboardPublicationError("sample evidence is missing a required field")
    generation_projection = {
        "output_token_count": generation.get("output_token_count"),
        "finish_reason": generation.get("finish_reason"),
        "stop_reason": generation.get("stop_reason"),
        "terminal_reason": generation.get("terminal_reason"),
        "truncated": generation.get("truncated"),
        "generation_limit": generation.get("generation_limit"),
        "request_id": generation.get("request_id"),
        "usage": generation.get("usage"),
    }
    scoring_projection = {
        "scorer_revision": scoring.get("scorer_revision"),
        "repair_strategy": scoring.get("repair_strategy"),
        "repair_action": scoring.get("repair_action"),
    }
    if generation_projection["usage"] is not None and not isinstance(
        generation_projection["usage"], dict
    ):
        raise ScoreboardPublicationError("sample usage evidence is invalid")
    return {
        "sample_index": sample["sample_index"],
        "sample_id": sample["sample_id"],
        "attempt": sample["attempt"],
        "status": "scored",
        "prompt": sample["prompt"],
        "raw_completion": sample["raw_completion"],
        "scored_completion": sample["scored_completion"],
        "generation": generation_projection,
        "scoring": scoring_projection,
        "metrics": metrics,
        "error_code": None,
        "error_message": None,
        "reference_answer": sample["reference_answer"],
        "provenance": sample["provenance"],
    }


def _validate_accounting(accounting: Mapping[str, Any]) -> None:
    fields = (
        "source_rows",
        "dataset_accepted",
        "dataset_rejected",
        "selected",
        "formatter_accepted",
        "formatter_rejected",
        "scored",
        "model_invalid",
        "provider_error",
        "cache_error",
        "scorer_error",
        "harness_error",
        "cancelled",
    )
    if any(field not in accounting for field in fields):
        raise ScoreboardPublicationError("manifest accounting is incomplete")
    if any(
        not isinstance(accounting[field], int)
        or isinstance(accounting[field], bool)
        or accounting[field] < 0
        for field in fields
    ):
        raise ScoreboardPublicationError("manifest accounting values are invalid")
    if (
        accounting["source_rows"]
        != accounting["dataset_accepted"] + accounting["dataset_rejected"]
    ):
        raise ScoreboardPublicationError("source row accounting does not close")
    if (
        accounting["selected"]
        != accounting["formatter_accepted"] + accounting["formatter_rejected"]
    ):
        raise ScoreboardPublicationError("selected accounting does not close")
    terminal = sum(accounting[field] for field in fields[6:])
    if accounting["formatter_accepted"] != terminal:
        raise ScoreboardPublicationError("terminal accounting does not close")


def _write_receipt(
    manifest_path: Path,
    result: PublicationResult,
    *,
    response: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "run_id": manifest_path.parent.name,
        "status": result.status,
        "retry_identity": result.retry_identity,
        "task_id": result.task_id,
        "error": result.error,
        "response": dict(response) if response is not None else None,
    }
    (manifest_path.parent / "publication.json").write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
