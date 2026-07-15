from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, Response, status

from .auth import require_role
from .contracts import (
    CreateRunRequest,
    HealthResponse,
    HistoryPage,
    IngestResponse,
    IngestRunRequest,
    ErrorEnvelope,
    MigrationResponse,
    LeaderboardPage,
    MetadataResponse,
    RunResponse,
    RunDetail,
    SamplePage,
    PerformancePatchRequest,
    PerformancePatchResponse,
)
from .errors import DomainError
from .settings import AuthPrincipal


_ERROR_RESPONSES = {
    code: {"model": ErrorEnvelope}
    for code in (400, 401, 403, 404, 409, 412, 422, 500, 503)
}
router = APIRouter(prefix="/api/v1", responses=_ERROR_RESPONSES)
Publisher = Annotated[AuthPrincipal, Depends(require_role("publisher"))]
EvidenceReader = Annotated[AuthPrincipal, Depends(require_role("evidence_reader"))]
Administrator = Annotated[AuthPrincipal, Depends(require_role("admin"))]


def _runs(request: Request):
    return request.app.state.runs


async def _require_ready(request: Request) -> None:
    await request.app.state.schema.require_ready()


def _required_header(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise DomainError("missing_header", f"{name} header is required", 400)
    return value.strip()


def _revision(value: str | None) -> int:
    raw = _required_header(value, "If-Match").strip('"')
    try:
        revision = int(raw)
    except ValueError as error:
        raise DomainError(
            "invalid_revision", "If-Match must contain an integer revision", 400
        ) from error
    if revision < 1:
        raise DomainError("invalid_revision", "If-Match revision must be positive", 400)
    return revision


def _performance_revision(value: str) -> int:
    raw = value.strip().strip('"')
    try:
        revision = int(raw)
    except ValueError as error:
        raise DomainError(
            "invalid_revision", "If-Match must contain an integer revision", 400
        ) from error
    if revision < 0:
        raise DomainError(
            "invalid_revision", "performance revision must be non-negative", 400
        )
    return revision


def _mutation_response(
    row: dict, *, ingest: bool = False
) -> RunResponse | IngestResponse:
    model = IngestResponse if ingest else RunResponse
    return model(
        run_id=row["run_id"],
        status=row["status"],
        revision=int(row["revision"]),
        disposition=row["disposition"],
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = await request.app.state.schema.state()
    return HealthResponse(
        status="ok" if state == "ready" else "degraded", schema_state=state
    )


@router.post("/admin/migrations", response_model=MigrationResponse)
async def migrate(request: Request, principal: Administrator) -> MigrationResponse:
    disposition = await request.app.state.schema.migrate(subject=principal.subject)
    return MigrationResponse(disposition=disposition, schema_state="ready")


@router.get(
    "/meta", response_model=MetadataResponse, dependencies=[Depends(_require_ready)]
)
async def metadata(request: Request) -> MetadataResponse:
    return await _runs(request).metadata()


@router.get(
    "/leaderboard",
    response_model=LeaderboardPage,
    dependencies=[Depends(_require_ready)],
)
async def leaderboard(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: str | None = None,
    model: str | None = None,
) -> LeaderboardPage:
    return await _runs(request).leaderboard(limit=limit, cursor=cursor, model=model)


@router.get(
    "/history", response_model=HistoryPage, dependencies=[Depends(_require_ready)]
)
async def history(
    request: Request,
    _principal: EvidenceReader,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: str | None = None,
    model: str | None = None,
) -> HistoryPage:
    return await _runs(request).history(limit=limit, cursor=cursor, model=model)


@router.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_require_ready)],
)
async def create_run(
    request: Request,
    body: CreateRunRequest,
    principal: Publisher,
    response: Response,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> RunResponse:
    row = await _runs(request).create(
        subject=principal.subject,
        idempotency_key=_required_header(idempotency_key, "Idempotency-Key"),
        request=body,
    )
    response.headers["ETag"] = f'"{row["revision"]}"'
    if row["disposition"] == "unchanged":
        response.status_code = status.HTTP_200_OK
    return _mutation_response(row)


@router.post(
    "/runs/{run_id}/resume",
    response_model=RunResponse,
    dependencies=[Depends(_require_ready)],
)
async def resume_run(
    run_id: str,
    request: Request,
    principal: Publisher,
    response: Response,
    if_match: Annotated[str, Header(alias="If-Match")],
) -> RunResponse:
    row = await _runs(request).resume(
        run_id=run_id, subject=principal.subject, revision=_revision(if_match)
    )
    row["disposition"] = "created"
    response.headers["ETag"] = f'"{row["revision"]}"'
    return _mutation_response(row)


@router.put(
    "/runs/{run_id}/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(_require_ready)],
)
async def ingest_run(
    run_id: str,
    request: Request,
    body: IngestRunRequest,
    principal: Publisher,
    response: Response,
    if_match: Annotated[str, Header(alias="If-Match")],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> IngestResponse:
    row = await _runs(request).ingest(
        run_id=run_id,
        subject=principal.subject,
        revision=_revision(if_match),
        idempotency_key=_required_header(idempotency_key, "Idempotency-Key"),
        request=body,
    )
    response.headers["ETag"] = f'"{row["revision"]}"'
    return _mutation_response(row, ingest=True)


@router.put(
    "/runs/{run_id}/performance",
    response_model=PerformancePatchResponse,
    dependencies=[Depends(_require_ready)],
)
async def patch_performance(
    run_id: str,
    request: Request,
    body: PerformancePatchRequest,
    principal: Publisher,
    response: Response,
    if_match: Annotated[str, Header(alias="If-Match")],
) -> PerformancePatchResponse:
    result = await _runs(request).patch_performance(
        run_id=run_id,
        subject=principal.subject,
        revision=_performance_revision(if_match),
        request=body,
    )
    response.headers["ETag"] = f'"{result["revision"]}"'
    return PerformancePatchResponse(**result)


@router.get(
    "/runs/{run_id}", response_model=RunDetail, dependencies=[Depends(_require_ready)]
)
async def get_run(
    run_id: str, request: Request, _principal: EvidenceReader
) -> RunDetail:
    return RunDetail.model_validate(await _runs(request).get(run_id))


@router.get(
    "/runs/{run_id}/samples",
    response_model=SamplePage,
    dependencies=[Depends(_require_ready)],
)
async def get_samples(
    run_id: str,
    request: Request,
    _principal: EvidenceReader,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: str | None = None,
) -> SamplePage:
    return SamplePage.model_validate(
        await _runs(request).samples(run_id, limit=limit, after=cursor)
    )
