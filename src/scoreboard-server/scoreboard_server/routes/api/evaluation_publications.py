from __future__ import annotations

import json
import zlib

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from scoreboard_server.services.api.evaluation_publications import (
    EvaluationPublicationService,
    PublicationAuthenticationError,
    PublicationConflictError,
    PublicationPayloadError,
)


MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024


async def _publication_json(request: Request) -> dict:
    content_length = request.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_COMPRESSED_BYTES:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    "compressed publication exceeds size limit",
                )
        except ValueError as error:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "invalid Content-Length"
            ) from error
    body = await request.body()
    if len(body) > MAX_COMPRESSED_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "compressed publication exceeds size limit",
        )
    encoding = request.headers.get("Content-Encoding", "identity").lower()
    if encoding == "gzip":
        try:
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            decoded = decompressor.decompress(body, MAX_UNCOMPRESSED_BYTES + 1)
            decoded += decompressor.flush(
                max(1, MAX_UNCOMPRESSED_BYTES + 1 - len(decoded))
            )
        except zlib.error as error:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "invalid gzip body"
            ) from error
        if (
            len(decoded) > MAX_UNCOMPRESSED_BYTES
            or decompressor.unconsumed_tail
            or not decompressor.eof
        ):
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "publication exceeds uncompressed size limit",
            )
        body = decoded
    elif encoding != "identity":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported Content-Encoding: {encoding}",
        )
    if len(body) > MAX_UNCOMPRESSED_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "publication exceeds uncompressed size limit",
        )
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON body") from error
    if not isinstance(value, dict):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "body must be an object")
    return value


def register(app: FastAPI, service: EvaluationPublicationService) -> None:
    @app.put("/api/v1/evaluation-publications/{publication_id:path}")
    async def publish_evaluation(
        publication_id: str,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        raw = await _publication_json(request)
        try:
            receipt = await service.publish(
                publication_id=publication_id,
                authorization=request.headers.get("Authorization", ""),
                idempotency_key=idempotency_key,
                raw=raw,
            )
        except PublicationAuthenticationError as error:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, str(error)
            ) from error
        except PublicationPayloadError as error:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, error.detail
            ) from error
        except PublicationConflictError as error:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"publication id already contains different content: {error}",
            ) from error
        response_status = (
            status.HTTP_201_CREATED
            if receipt.disposition == "created"
            else status.HTTP_200_OK
        )
        return JSONResponse(
            status_code=response_status,
            content=receipt.model_dump(mode="json"),
        )
