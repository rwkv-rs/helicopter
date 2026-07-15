from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
import re
from typing import Any, Mapping

from scoreboard_server.db.evaluation_publications import (
    EvaluationPublicationRepository,
    PublicationConflict,
    PublicationReceipt,
)
from scoreboard_server.dtos.api.evaluation_publications import (
    EvaluationPublicationRequest,
)


MAX_PUBLICATION_TRANSFER_BYTES = 16 * 1024 * 1024
MAX_PUBLICATION_BYTES = 64 * 1024 * 1024
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class PublicationAuthenticationError(RuntimeError):
    pass


class PublicationAuthorizationError(RuntimeError):
    pass


class PublicationPayloadError(ValueError):
    pass


class PublicationPayloadTooLarge(PublicationPayloadError):
    pass


class PublicationConflictError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TokenGrant:
    subject: str
    roles: frozenset[str]


class EvaluationPublicationService:
    def __init__(
        self,
        repository: EvaluationPublicationRepository,
        grants: Mapping[str, TokenGrant],
    ) -> None:
        self._repository = repository
        self._grants = dict(grants)

    async def publish(
        self,
        *,
        run_id: str,
        authorization: str | None,
        idempotency_key: str,
        request: EvaluationPublicationRequest,
    ) -> PublicationReceipt:
        subject = self._publisher_subject(authorization)
        if not _RUN_ID.fullmatch(run_id):
            raise PublicationPayloadError("run_id is not a normalized identifier")
        payload = request.model_dump(mode="json")
        encoded = _canonical_json(payload)
        if len(encoded) > MAX_PUBLICATION_BYTES:
            raise PublicationPayloadTooLarge(
                "publication exceeds the request size limit"
            )
        manifest = payload["manifest"]
        if idempotency_key != f"publish:{manifest['digest']}":
            raise PublicationPayloadError(
                "idempotency key must be derived from the manifest digest"
            )
        if _digest(payload["identity"]) != manifest["identity_digest"]:
            raise PublicationPayloadError("identity digest does not match payload")
        if _digest(payload["accounting"]) != manifest["accounting_digest"]:
            raise PublicationPayloadError("accounting digest does not match payload")
        metric = next(
            item
            for item in payload["identity"]["task"]["metrics"]
            if item["name"] == payload["primary_metric"]
        )
        values = [payload["metrics"][payload["primary_metric"]]] + [
            sample["metrics"][payload["primary_metric"]]
            for sample in payload["samples"]
        ]
        if any(
            value < metric["minimum"] or value > metric["maximum"] for value in values
        ):
            raise PublicationPayloadError(
                "primary metric is outside its declared range"
            )
        sample_values = values[1:]
        expected_aggregate = (
            sum(sample_values) / len(sample_values)
            if metric["aggregation"] == "mean"
            else sum(sample_values)
        )
        if not math.isclose(values[0], expected_aggregate, rel_tol=0.0, abs_tol=1e-12):
            raise PublicationPayloadError(
                "primary metric does not match its declared sample aggregation"
            )
        if metric["binary_correctness"] and any(
            sample["reference_answer"] is None
            or sample["metrics"][payload["primary_metric"]] not in {0.0, 1.0}
            for sample in payload["samples"]
        ):
            raise PublicationPayloadError(
                "binary metrics require reference answers and exact boolean values"
            )
        try:
            return await self._repository.publish(
                run_id=run_id,
                publisher_subject=subject,
                idempotency_key=idempotency_key,
                request_digest=_digest({"run_id": run_id, "payload": payload}),
                payload=payload,
            )
        except PublicationConflict as error:
            raise PublicationConflictError(str(error)) from error

    def authenticate(self, authorization: str | None) -> str:
        return self._publisher_subject(authorization)

    def _publisher_subject(self, authorization: str | None) -> str:
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            raise PublicationAuthenticationError("bearer token is required")
        presented = authorization[len(prefix) :]
        grant = next(
            (
                candidate
                for token, candidate in self._grants.items()
                if hmac.compare_digest(token, presented)
            ),
            None,
        )
        if grant is None:
            raise PublicationAuthenticationError("bearer token is invalid")
        if "publisher" not in grant.roles:
            raise PublicationAuthorizationError("publisher role is required")
        return grant.subject


def publication_grants_from_env() -> dict[str, TokenGrant]:
    raw = os.environ.get("SCOREBOARD_AUTH_TOKENS", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("SCOREBOARD_AUTH_TOKENS must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("SCOREBOARD_AUTH_TOKENS must be a JSON object")
    grants: dict[str, TokenGrant] = {}
    for token, value in payload.items():
        if not isinstance(token, str) or not token or not isinstance(value, dict):
            raise ValueError("scoreboard auth token entries are invalid")
        subject = value.get("subject")
        roles = value.get("roles")
        if (
            not isinstance(subject, str)
            or not subject
            or not isinstance(roles, list)
            or not all(isinstance(role, str) and role for role in roles)
        ):
            raise ValueError("scoreboard auth token grant is invalid")
        grants[token] = TokenGrant(subject, frozenset(roles))
    return grants


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()
