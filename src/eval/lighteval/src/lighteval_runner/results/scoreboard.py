from __future__ import annotations

import hashlib
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

from ..execution import ExecutionPlan, RunStatus
from ..provider.attestation import (
    AttestationDecision,
    Comparability,
    ProviderAttestation,
)
from ..registry import TaskDefinition
from .artifacts import verify_manifest


class ScoreboardPublicationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PublicationResult:
    status: str
    retry_identity: str
    error: str | None = None
    task_id: int | None = None


@dataclass(frozen=True, slots=True)
class PublicationArtifact:
    run_id: str
    identity: Mapping[str, Any]
    identity_digest: str
    manifest_path: Path
    terminal_status: RunStatus
    accounting: Mapping[str, int]
    samples: tuple[dict[str, Any], ...]
    metrics: Mapping[str, float]
    primary_metric: str | None
    rejections: tuple[dict[str, str], ...]
    truncated_samples: int
    generated_samples: int
    performance: Mapping[str, Any]

    @property
    def manifest_digest(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    @property
    def retry_identity(self) -> str:
        return f"publish:{self.manifest_digest}"

    def publication_payload(self) -> dict[str, Any]:
        accounting = dict(self.accounting)
        manifest = verify_manifest(self.manifest_path)
        if manifest.completed_at is None:
            raise ScoreboardPublicationError(
                "legacy artifact has no completion time and cannot be published"
            )
        return {
            "identity": dict(self.identity),
            "samples": [_publication_sample(sample) for sample in self.samples],
            "accounting": accounting,
            "rejections": list(self.rejections),
            "metrics": dict(self.metrics),
            "primary_metric": self.primary_metric,
            "truncated_samples": self.truncated_samples,
            "generated_samples": self.generated_samples,
            "performance": dict(self.performance),
            "manifest": {
                "digest": self.manifest_digest,
                "identity_digest": self.identity_digest,
                "accounting_digest": _digest(accounting),
                "terminal_status": self.terminal_status.value,
                "checksums_verified": True,
                "completed_at": manifest.completed_at,
            },
            "terminal_status": self.terminal_status.value,
        }


class ScoreboardClient:
    """HTTP-only publication boundary; this package never imports persistence code."""

    def __init__(
        self,
        *,
        base_url: str,
        bearer_token: str,
        client: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url.strip() or not bearer_token.strip():
            raise ValueError("scoreboard URL and bearer token are required")
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Accept": "application/json",
        }
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def publish(
        self,
        *,
        run_id: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> PublicationResult:
        receipt = self._request(
            "PUT",
            f"/api/v1/evaluation-publications/{run_id}",
            content=gzip.compress(_canonical_json(payload)),
            headers={
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
            expected={200, 201},
        )
        if receipt.get("status") != "completed" or receipt.get("run_id") != run_id:
            raise ScoreboardPublicationError(
                "scoreboard returned an inconsistent publication receipt"
            )
        task_id = receipt.get("task_id")
        if not isinstance(task_id, int) or task_id <= 0:
            raise ScoreboardPublicationError(
                "scoreboard returned an invalid publication task identity"
            )
        return PublicationResult("published", idempotency_key, task_id=task_id)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        expected: set[int],
    ) -> dict[str, Any]:
        merged = {**self._headers, **(headers or {})}
        try:
            response = self._client.request(
                method,
                f"{self._base_url}{path}",
                headers=merged,
                json=json if content is None else None,
                content=content,
            )
        except httpx.HTTPError as error:
            raise ScoreboardPublicationError(
                f"scoreboard transport failed: {error}"
            ) from error
        if response.status_code not in expected:
            code = "unknown_error"
            try:
                body = response.json()
                error = body.get("error") or body.get("detail") or {}
                code = str(error.get("code", code))
            except ValueError:
                pass
            raise ScoreboardPublicationError(
                f"scoreboard {method} {path} failed with {response.status_code} ({code})"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise ScoreboardPublicationError(
                "scoreboard response is not JSON"
            ) from error
        if not isinstance(payload, dict):
            raise ScoreboardPublicationError("scoreboard response must be an object")
        return payload


def build_scoreboard_identity(
    plan: ExecutionPlan,
    definition: TaskDefinition,
    attestation: ProviderAttestation,
    decision: AttestationDecision,
    *,
    product_revision: str,
    product_dirty: bool,
) -> dict[str, Any]:
    metric = definition.contract.metric
    comparable = decision.comparability is Comparability.OFFICIAL
    return {
        "task": {
            "suite": plan.task_identity.suite,
            "task": plan.task_identity.task,
            "version": plan.task_identity.version,
            "split": plan.task_identity.split,
            "fewshot": plan.task_identity.fewshot,
            "prompt_revision": plan.task_identity.prompt_revision,
            "scorer_revision": plan.task_identity.scorer_revision,
            "generation_contract": plan.task_identity.generation_contract,
            "cot_mode": plan.task_identity.cot_mode,
            "repair_strategy": plan.task_identity.repair_strategy,
            "dataset_digest": plan.task_identity.dataset_digest,
            "primary_metric": metric.primary_metric,
            "metrics": [
                {
                    "name": metric.primary_metric,
                    "minimum": metric.minimum,
                    "maximum": metric.maximum,
                    "aggregation": metric.aggregation_rule,
                    "binary_correctness": (
                        metric.binary_correctness_metric == metric.primary_metric
                    ),
                }
            ],
        },
        "model": asdict(attestation.model),
        "provider": {
            **asdict(attestation.provider),
            "attestation_digest": _digest(asdict(attestation)),
            "attestation_verified": comparable,
            "attestation_present": "missing_attestation" not in decision.mismatches,
            "attestation_mismatches": list(decision.mismatches),
        },
        "evaluator": {
            "product_revision": product_revision,
            "dirty": product_dirty,
        },
        "config_digest": plan.config_digest,
        "eligibility": plan.task_identity.eligibility.value,
        "comparable": comparable,
    }


def publish_artifact(
    *,
    base_url: str,
    bearer_token: str,
    manifest_path: Path,
) -> PublicationResult:
    artifact = load_publication_artifact(manifest_path)
    return _send_publication(
        base_url=base_url,
        bearer_token=bearer_token,
        artifact=artifact,
    )


def retry_publication(
    *, manifest_path: Path, base_url: str, bearer_token: str
) -> PublicationResult:
    return publish_artifact(
        manifest_path=manifest_path,
        base_url=base_url,
        bearer_token=bearer_token,
    )


def load_publication_artifact(manifest_path: Path) -> PublicationArtifact:
    manifest = verify_manifest(manifest_path)
    try:
        samples = json.loads(
            (manifest_path.parent / "samples.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (manifest_path.parent / "summary.json").read_text(encoding="utf-8")
        )
        performance = json.loads(
            (manifest_path.parent / "performance.json").read_text(encoding="utf-8")
        )
        identity = manifest.identities["run"]
        metrics = summary["metrics"]
        rejections = summary["rejections"]
        primary_metric = identity["task"]["primary_metric"] if metrics else None
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise ScoreboardPublicationError(
            "run artifact cannot be reconstructed for publication"
        ) from error
    if (
        not isinstance(samples, list)
        or not isinstance(metrics, dict)
        or not isinstance(rejections, list)
        or not isinstance(performance, dict)
    ):
        raise ScoreboardPublicationError(
            "run artifact publication payload has an invalid schema"
        )
    return PublicationArtifact(
        run_id=manifest.run_id,
        identity=identity,
        identity_digest=manifest.identity_digest,
        manifest_path=manifest_path,
        terminal_status=manifest.status,
        accounting=manifest.accounting,
        samples=tuple(samples),
        metrics=metrics,
        primary_metric=primary_metric,
        rejections=tuple(rejections),
        truncated_samples=summary.get("truncated_samples", 0),
        generated_samples=summary.get("generated_samples", 0),
        performance=performance,
    )


def _send_publication(
    *,
    base_url: str,
    bearer_token: str,
    artifact: PublicationArtifact,
) -> PublicationResult:
    client = ScoreboardClient(base_url=base_url, bearer_token=bearer_token)
    try:
        return client.publish(
            run_id=artifact.run_id,
            payload=artifact.publication_payload(),
            idempotency_key=artifact.retry_identity,
        )
    except ScoreboardPublicationError as error:
        return PublicationResult("failed", artifact.retry_identity, str(error))
    finally:
        client.close()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode()


def _publication_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    generation = dict(sample["generation"])
    for field in (
        "raw_completion",
        "output_token_ids",
        "prompt_text",
        "prompt_token_ids",
    ):
        generation.pop(field, None)
    scoring = dict(sample["scoring"])
    scoring.pop("raw_completion", None)
    scoring.pop("scored_completion", None)
    return {
        "sample_index": sample["sample_index"],
        "sample_id": sample["sample_id"],
        "attempt": sample["attempt"],
        "status": sample["status"],
        "prompt": sample["prompt"],
        "raw_completion": sample["raw_completion"],
        "scored_completion": sample["scored_completion"],
        "generation": generation,
        "scoring": scoring,
        "metrics": sample["metrics"],
        "error_code": sample["error_code"],
        "error_message": sample["error_message"],
        "reference_answer": sample["reference_answer"],
        "provenance": sample["provenance"],
    }
