from __future__ import annotations

from typing import TypedDict


class AdminHealthResponse(TypedDict):
    status: str
    active: bool
    auth_required: bool
