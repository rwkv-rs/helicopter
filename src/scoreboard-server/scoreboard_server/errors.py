from __future__ import annotations

from uuid import uuid4
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class DomainError(Exception):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


logger = logging.getLogger(__name__)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_handler(request: Request, error: DomainError) -> JSONResponse:
        return _response(request, error.status_code, error.code, error.message)

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, error: HTTPException) -> JSONResponse:
        detail = error.detail if isinstance(error.detail, dict) else {}
        response = _response(
            request,
            error.status_code,
            str(detail.get("code", "http_error")),
            str(detail.get("message", error.detail)),
        )
        response.headers.update(error.headers or {})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "type": item.get("type"),
                "location": list(item.get("loc", ())),
                "message": item.get("msg"),
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "Request validation failed",
                    "request_id": request.state.request_id,
                    "details": details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def internal_handler(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unhandled scoreboard request failure", exc_info=error)
        return _response(request, 500, "internal_error", "Internal server error")


def _response(
    request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
    )
