from __future__ import annotations

import asyncio
import json
import re
from typing import Any
import zlib

from fastapi import FastAPI, Header, HTTPException, Response

from scoreboard_server.dtos.api.evaluation_publications import (
    EvaluationPublicationRequest,
    EvaluationPublicationResponse,
)
from scoreboard_server.services.api.evaluation_publications import (
    EvaluationPublicationService,
    PublicationAuthenticationError,
    PublicationAuthorizationError,
    PublicationConflictError,
    PublicationPayloadError,
    PublicationPayloadTooLarge,
    MAX_PUBLICATION_BYTES,
    MAX_PUBLICATION_TRANSFER_BYTES,
)


_PUBLICATION_PATH = re.compile(
    r"/api/v1/evaluation-publications/[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
)


class PublicationRequestBoundary:
    def __init__(self, app: Any, *, service: EvaluationPublicationService) -> None:
        self._app = app
        self._service = service
        self._publication_slot = asyncio.Semaphore(1)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "PUT"
            or not _PUBLICATION_PATH.fullmatch(scope.get("path", ""))
        ):
            await self._app(scope, receive, send)
            return
        async with self._publication_slot:
            await self._handle_publication(scope, receive, send)

    async def _handle_publication(self, scope: dict, receive: Any, send: Any) -> None:
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        try:
            self._service.authenticate(headers.get("authorization"))
        except PublicationAuthenticationError as error:
            await _send_error(send, 401, "unauthorized", str(error))
            return
        except PublicationAuthorizationError as error:
            await _send_error(send, 403, "forbidden", str(error))
            return
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                await _send_error(
                    send, 400, "invalid_content_length", "content length is invalid"
                )
                return
            if declared_size > MAX_PUBLICATION_TRANSFER_BYTES:
                await _send_error(
                    send,
                    413,
                    "publication_too_large",
                    "publication exceeds the request size limit",
                )
                return
        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                await self._app(scope, receive, send)
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_PUBLICATION_TRANSFER_BYTES:
                await _send_error(
                    send,
                    413,
                    "publication_too_large",
                    "publication exceeds the request size limit",
                )
                return
            more_body = bool(message.get("more_body"))

        encoding = headers.get("content-encoding", "identity").lower()
        if encoding == "gzip":
            try:
                decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
                decoded = decompressor.decompress(
                    bytes(body), MAX_PUBLICATION_BYTES + 1
                )
                decoded += decompressor.flush(
                    max(1, MAX_PUBLICATION_BYTES + 1 - len(decoded))
                )
            except zlib.error:
                await _send_error(
                    send, 400, "invalid_gzip", "publication gzip body is invalid"
                )
                return
            if (
                len(decoded) > MAX_PUBLICATION_BYTES
                or decompressor.unconsumed_tail
                or not decompressor.eof
            ):
                await _send_error(
                    send,
                    413,
                    "publication_too_large",
                    "publication exceeds the decompressed request size limit",
                )
                return
            body = bytearray(decoded)
        elif encoding != "identity":
            await _send_error(
                send,
                415,
                "unsupported_content_encoding",
                "publication content encoding must be gzip or identity",
            )
            return

        scope["headers"] = [
            (key, value)
            for key, value in scope.get("headers", [])
            if key.lower() not in {b"content-encoding", b"content-length"}
        ] + [(b"content-length", str(len(body)).encode())]

        delivered = False

        async def replay_body() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self._app(scope, replay_body, send)


def register(app: FastAPI, service: EvaluationPublicationService) -> None:
    app.add_middleware(PublicationRequestBoundary, service=service)

    @app.put(
        "/api/v1/evaluation-publications/{run_id}",
        response_model=EvaluationPublicationResponse,
    )
    async def publish_evaluation(
        run_id: str,
        request: EvaluationPublicationRequest,
        response: Response,
        authorization: str | None = Header(default=None),
        idempotency_key: str = Header(..., alias="Idempotency-Key"),
    ) -> EvaluationPublicationResponse:
        try:
            receipt = await service.publish(
                run_id=run_id,
                authorization=authorization,
                idempotency_key=idempotency_key,
                request=request,
            )
        except PublicationAuthenticationError as error:
            raise _http_error(401, "unauthorized", str(error)) from error
        except PublicationAuthorizationError as error:
            raise _http_error(403, "forbidden", str(error)) from error
        except PublicationPayloadTooLarge as error:
            raise _http_error(413, "publication_too_large", str(error)) from error
        except PublicationPayloadError as error:
            raise _http_error(422, "invalid_publication", str(error)) from error
        except PublicationConflictError as error:
            raise _http_error(409, "publication_conflict", str(error)) from error
        response.status_code = 201 if receipt.disposition == "created" else 200
        return EvaluationPublicationResponse(
            run_id=run_id,
            task_id=receipt.task_id,
            status="completed",
            disposition=receipt.disposition,
        )


def _http_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


async def _send_error(send: Any, status: int, code: str, message: str) -> None:
    body = json.dumps({"detail": {"code": code, "message": message}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
