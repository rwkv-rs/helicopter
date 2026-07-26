from __future__ import annotations

import json
import secrets
import uuid

from pydantic import ValidationError

from scoreboard_server.db.repository import (
    CampaignCompletenessError,
    CampaignContractError,
    PublicationConflictError,
    ScoreboardRepository,
)
from scoreboard_server.dtos.api.evaluation_results import (
    CampaignCreate,
    CampaignReceipt,
    CampaignStatus,
    FinalizeReceipt,
    PublicationPreflight,
    TaskPublication,
    TaskReceipt,
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
        isinstance(token, str) and token and isinstance(principal, str) and principal
        for token, principal in value.items()
    ):
        raise RuntimeError(
            "SCOREBOARD_PUBLICATION_TOKENS must map non-empty tokens "
            "to publisher principals"
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

    def principal_for_authorization(self, authorization: str) -> str:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise PublicationAuthenticationError("Bearer token required")
        for candidate, principal in self.publication_tokens.items():
            if secrets.compare_digest(candidate, token):
                return principal
        raise PublicationAuthenticationError("invalid publication token")

    def preflight(self, *, authorization: str) -> PublicationPreflight:
        return PublicationPreflight(
            status="ready",
            publisher_principal=self.principal_for_authorization(authorization),
            schema_version="lighteval-campaign-v3",
            lighteval_version="0.13.0",
        )

    @staticmethod
    def _validate(model: type, raw: dict):
        try:
            return model.model_validate(raw)
        except ValidationError as error:
            raise PublicationPayloadError(
                error.errors(include_context=False, include_input=False)
            ) from error

    async def create_campaign(
        self,
        *,
        authorization: str,
        idempotency_key: str | None,
        raw: dict,
    ) -> CampaignReceipt:
        principal = self.principal_for_authorization(authorization)
        campaign: CampaignCreate = self._validate(CampaignCreate, raw)
        if idempotency_key != f"campaign:{campaign.run_key}":
            raise PublicationPayloadError(
                "Idempotency-Key does not match campaign run key"
            )
        try:
            return await self.repository.create_campaign(
                campaign=campaign,
                publisher_principal=principal,
            )
        except CampaignContractError as error:
            raise PublicationPayloadError(str(error)) from error

    async def campaign_status(
        self,
        *,
        campaign_id: uuid.UUID,
        authorization: str,
    ) -> CampaignStatus | None:
        principal = self.principal_for_authorization(authorization)
        return await self.repository.campaign_status(
            campaign_id,
            publisher_principal=principal,
        )

    async def publish_task(
        self,
        *,
        campaign_id: uuid.UUID,
        task_identity: str,
        authorization: str,
        idempotency_key: str | None,
        raw: dict,
    ) -> TaskReceipt:
        principal = self.principal_for_authorization(authorization)
        digest = content_digest(raw)
        if idempotency_key != f"publish:{digest}":
            raise PublicationPayloadError(
                "Idempotency-Key does not match canonical content digest"
            )
        publication: TaskPublication = self._validate(TaskPublication, raw)
        try:
            return await self.repository.publish_task(
                campaign_id=campaign_id,
                task_identity=task_identity,
                digest=digest,
                publication=publication,
                publisher_principal=principal,
            )
        except CampaignContractError as error:
            raise PublicationPayloadError(str(error)) from error

    async def finalize_campaign(
        self,
        *,
        campaign_id: uuid.UUID,
        authorization: str,
        idempotency_key: str | None,
    ) -> FinalizeReceipt:
        principal = self.principal_for_authorization(authorization)
        if idempotency_key != f"finalize:{campaign_id}":
            raise PublicationPayloadError("Idempotency-Key does not match campaign id")
        try:
            return await self.repository.finalize_campaign(
                campaign_id,
                publisher_principal=principal,
            )
        except CampaignContractError as error:
            raise PublicationPayloadError(str(error)) from error


__all__ = [
    "CampaignCompletenessError",
    "EvaluationPublicationService",
    "PublicationAuthenticationError",
    "PublicationConflictError",
    "PublicationPayloadError",
    "publication_tokens_from_env",
]
