from __future__ import annotations

from scoreboard_server.dtos.api.admin.eval.status import AdminEvalStatusResponse


SCHEDULER_CONTROL_ERROR = "Scheduler control is not part of the migrated scoreboard server."

_DISABLED_STATUS: AdminEvalStatusResponse = {
    "status": "idle",
    "desired_state": None,
    "run_id": None,
    "error": SCHEDULER_CONTROL_ERROR,
    "started_at_unix_ms": None,
    "updated_at_unix_ms": None,
    "finished_at_unix_ms": None,
    "pending_jobs": 0,
    "running_jobs": 0,
    "completed_jobs": 0,
    "failed_jobs": 0,
    "tasks_total": 0,
    "progress_percent": 0,
    "queue_head": [],
    "active_jobs": [],
    "available_gpus": [],
    "request": None,
}


def admin_eval_status_response() -> AdminEvalStatusResponse:
    return dict(_DISABLED_STATUS)
