from __future__ import annotations

import json
from typing import Never
import uuid
import zlib

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from scoreboard_server.services.api.evaluation_publications import (
    CampaignCompletenessError,
    EvaluationPublicationService,
    PublicationAuthenticationError,
    PublicationConflictError,
    PublicationPayloadError,
)


MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


async def _publication_json(request: Request) -> dict:
    content_length = request.headers.get("Content-Length")
    if content_length is not None:
        try:
            parsed_content_length = int(content_length)
            if parsed_content_length < 0:
                raise ValueError
            if parsed_content_length > MAX_COMPRESSED_BYTES:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    "compressed publication exceeds size limit",
                )
        except ValueError as error:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "invalid Content-Length"
            ) from error
    content_type = request.headers.get("Content-Type", "").partition(";")[0].lower()
    if content_type != "application/json":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "publication Content-Type must be application/json",
        )
    encoding = request.headers.get("Content-Encoding", "").lower()
    if encoding != "gzip":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "publication Content-Encoding must be gzip",
        )
    chunks: list[bytes] = []
    compressed_size = 0
    async for chunk in request.stream():
        compressed_size += len(chunk)
        if compressed_size > MAX_COMPRESSED_BYTES:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                "compressed publication exceeds size limit",
            )
        chunks.append(chunk)
    body = b"".join(chunks)
    try:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        decoded = decompressor.decompress(body, MAX_UNCOMPRESSED_BYTES + 1)
        decoded += decompressor.flush(max(1, MAX_UNCOMPRESSED_BYTES + 1 - len(decoded)))
    except zlib.error as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid gzip body") from error
    if (
        len(decoded) > MAX_UNCOMPRESSED_BYTES
        or decompressor.unconsumed_tail
        or decompressor.unused_data
        or not decompressor.eof
    ):
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "publication exceeds uncompressed size limit",
        )
    body = decoded
    if len(body) > MAX_UNCOMPRESSED_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "publication exceeds uncompressed size limit",
        )
    try:
        value = json.loads(body, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON body") from error
    if not isinstance(value, dict):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "body must be an object"
        )
    return value


def _raise_publication_error(error: Exception) -> Never:
    if isinstance(error, PublicationAuthenticationError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(error)) from error
    if isinstance(error, PublicationPayloadError):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, error.detail
        ) from error
    if isinstance(error, PublicationConflictError):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"task identity already contains different content: {error}",
        ) from error
    if isinstance(error, CampaignCompletenessError):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"message": "campaign is incomplete", "missing": error.missing},
        ) from error
    raise error


def register(app: FastAPI, service: EvaluationPublicationService) -> None:
    @app.get("/api/v1/evaluation-publication-preflight")
    async def publication_preflight(request: Request):
        try:
            return service.preflight(
                authorization=request.headers.get("Authorization", ""),
            )
        except Exception as error:
            _raise_publication_error(error)

    @app.post("/api/v1/evaluation-campaigns")
    async def create_campaign(
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        try:
            service.principal_for_authorization(
                request.headers.get("Authorization", "")
            )
            receipt = await service.create_campaign(
                authorization=request.headers.get("Authorization", ""),
                idempotency_key=idempotency_key,
                raw=await _publication_json(request),
            )
        except Exception as error:
            _raise_publication_error(error)
        response_status = (
            status.HTTP_201_CREATED
            if receipt.disposition == "created"
            else status.HTTP_200_OK
        )
        return JSONResponse(
            status_code=response_status,
            content=receipt.model_dump(mode="json"),
        )

    @app.get("/api/v1/evaluation-campaigns/{campaign_id}")
    async def campaign_status(campaign_id: uuid.UUID, request: Request):
        try:
            result = await service.campaign_status(
                campaign_id=campaign_id,
                authorization=request.headers.get("Authorization", ""),
            )
        except Exception as error:
            _raise_publication_error(error)
        if result is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "campaign not found")
        return result

    @app.put("/api/v1/evaluation-campaigns/{campaign_id}/tasks/{task_identity:path}")
    async def publish_task(
        campaign_id: uuid.UUID,
        task_identity: str,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        try:
            service.principal_for_authorization(
                request.headers.get("Authorization", "")
            )
            receipt = await service.publish_task(
                campaign_id=campaign_id,
                task_identity=task_identity,
                authorization=request.headers.get("Authorization", ""),
                idempotency_key=idempotency_key,
                raw=await _publication_json(request),
            )
        except Exception as error:
            _raise_publication_error(error)
        response_status = (
            status.HTTP_201_CREATED
            if receipt.disposition == "created"
            else status.HTTP_200_OK
        )
        return JSONResponse(
            status_code=response_status,
            content=receipt.model_dump(mode="json"),
        )

    @app.post("/api/v1/evaluation-campaigns/{campaign_id}/finalize")
    async def finalize_campaign(
        campaign_id: uuid.UUID,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        try:
            return await service.finalize_campaign(
                campaign_id=campaign_id,
                authorization=request.headers.get("Authorization", ""),
                idempotency_key=idempotency_key,
            )
        except Exception as error:
            _raise_publication_error(error)
