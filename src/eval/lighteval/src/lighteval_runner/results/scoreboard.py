from __future__ import annotations

import hashlib
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

    @property
    def manifest_digest(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    @property
    def retry_identity(self) -> str:
        return f"ingest:{self.manifest_digest}"

    def ingest_payload(self) -> dict[str, Any]:
        accounting = dict(self.accounting)
        return {
            "samples": list(self.samples),
            "accounting": accounting,
            "rejections": list(self.rejections),
            "metrics": dict(self.metrics),
            "primary_metric": self.primary_metric,
            "truncated_samples": self.truncated_samples,
            "generated_samples": self.generated_samples,
            "manifest": {
                "digest": self.manifest_digest,
                "identity_digest": self.identity_digest,
                "accounting_digest": _digest(accounting),
                "terminal_status": self.terminal_status.value,
                "checksums_verified": True,
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
        create_payload: Mapping[str, Any],
        ingest_payload: Mapping[str, Any],
        ingest_identity: str,
    ) -> PublicationResult:
        created = self._request(
            "POST",
            "/api/v1/runs",
            json=create_payload,
            headers={"Idempotency-Key": f"create:{run_id}"},
            expected={200, 201},
        )
        revision = _positive_revision(created)
        state = created.get("status")
        if state in {"planned", "partial", "failed"}:
            resumed = self._request(
                "POST",
                f"/api/v1/runs/{run_id}/resume",
                headers={"If-Match": str(revision)},
                expected={200},
            )
            revision = _positive_revision(resumed)
        elif state not in {"running", "completed", "invalid", "cancelled"}:
            raise ScoreboardPublicationError(
                f"scoreboard returned unsupported run state: {state}"
            )
        terminal = self._request(
            "PUT",
            f"/api/v1/runs/{run_id}/ingest",
            json=ingest_payload,
            headers={"If-Match": str(revision), "Idempotency-Key": ingest_identity},
            expected={200, 201},
        )
        if terminal.get("status") != ingest_payload.get("terminal_status"):
            raise ScoreboardPublicationError(
                "scoreboard returned a different terminal status"
            )
        return PublicationResult("published", ingest_identity)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        expected: set[int],
    ) -> dict[str, Any]:
        merged = {**self._headers, **(headers or {})}
        try:
            response = self._client.request(
                method, f"{self._base_url}{path}", headers=merged, json=json
            )
        except httpx.HTTPError as error:
            raise ScoreboardPublicationError(
                f"scoreboard transport failed: {error}"
            ) from error
        if response.status_code not in expected:
            code = "unknown_error"
            try:
                code = str(response.json().get("error", {}).get("code", code))
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
            create_payload={
                "run_id": artifact.run_id,
                "identity": artifact.identity,
                "expected_sample_ids": [
                    sample["sample_id"] for sample in artifact.samples
                ],
            },
            ingest_payload=artifact.ingest_payload(),
            ingest_identity=artifact.retry_identity,
        )
    except ScoreboardPublicationError as error:
        return PublicationResult("failed", artifact.retry_identity, str(error))
    finally:
        client.close()


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_revision(payload: Mapping[str, Any]) -> int:
    revision = payload.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0:
        raise ScoreboardPublicationError(
            "scoreboard response is missing a positive revision"
        )
    return revision
