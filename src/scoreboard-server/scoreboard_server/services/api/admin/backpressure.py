from __future__ import annotations

from scoreboard_server.dtos.api.admin.backpressure import AdminBackpressureResponse
from scoreboard_server.services.api.admin.eval.status import admin_eval_status_response


def admin_backpressure_response(*, infer_base_url: str | None) -> AdminBackpressureResponse:
    status = admin_eval_status_response()
    return {"infer_base_url": infer_base_url or "", "available_gpus": [], "models": [], "error": status["error"]}
