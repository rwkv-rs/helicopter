from __future__ import annotations

from scoreboard_server.dtos.api.admin.eval.options import AdminEvalOptionsResponse


def admin_eval_options_response() -> AdminEvalOptionsResponse:
    return {"jobs": [], "domains": [], "model_select": [], "worker_profile": [], "protocol": [], "run_mode": []}
