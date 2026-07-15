from __future__ import annotations

from typing import Any, TypeAlias, TypedDict


class CapturePageRequest(TypedDict, total=False):
    url: str | None
    width: int | None
    height: int | None


CapturePageResponse: TypeAlias = dict[str, Any]
