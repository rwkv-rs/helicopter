from __future__ import annotations

import json
import secrets

from pydantic import ValidationError

from scoreboard_server.db.repository import (
    PublicationConflictError,
    ScoreboardRepository,
)
from scoreboard_server.dtos.api.evaluation_results import (
    EvaluationPublication,
    PublicationReceipt,
    content_digest,
)


class PublicationAuthenticationError(RuntimeError):
    pass


class PublicationPayloadError(ValueError):
    def __init__(self, detail: object):
        super().__init__(str(detail))
        self.detail = detail


def publication_tokens_from_env(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("SCOREBOARD_PUBLICATION_TOKENS must be JSON") from error
    if not isinstance(value, dict) or not all(
        isinstance(token, str)
        and token
        and isinstance(source, str)
        and source
        for token, source in value.items()
    ):
        raise RuntimeError(
            "SCOREBOARD_PUBLICATION_TOKENS must map non-empty tokens to sources"
        )
    return value


class EvaluationPublicationService:
    def __init__(
        self,
        repository: ScoreboardRepository,
        publication_tokens: dict[str, str],
    ) -> None:
        self.repository = repository
        self.publication_tokens = publication_tokens

    def source_for_authorization(self, authorization: str) -> str:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise PublicationAuthenticationError("Bearer token required")
        for candidate, source in self.publication_tokens.items():
            if secrets.compare_digest(candidate, token):
                return source
        raise PublicationAuthenticationError("invalid publication token")

    async def publish(
        self,
        *,
        publication_id: str,
        authorization: str,
        idempotency_key: str | None,
        raw: dict,
    ) -> PublicationReceipt:
        source = self.source_for_authorization(authorization)
        digest = content_digest(raw)
        if idempotency_key != f"publish:{digest}":
            raise PublicationPayloadError(
                "Idempotency-Key does not match canonical content digest"
            )
        try:
            publication = EvaluationPublication.model_validate(raw)
        except ValidationError as error:
            raise PublicationPayloadError(error.errors()) from error
        return await self.repository.publish(
            publication_id=publication_id,
            digest=digest,
            source=source,
            publication=publication,
        )


__all__ = [
    "EvaluationPublicationService",
    "PublicationAuthenticationError",
    "PublicationConflictError",
    "PublicationPayloadError",
    "publication_tokens_from_env",
]
