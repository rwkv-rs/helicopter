from __future__ import annotations

"""Deployment-facing health checks for the model and web search endpoints."""

import json
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import HarnessSettings


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    ok: bool
    url: str
    status: int | None = None
    message: str = ""
    model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_preflight(settings: HarnessSettings, *, timeout: float | None = None) -> list[ProbeResult]:
    """Probe both endpoints using the same URL, model, and auth as a real run."""

    probe_timeout = float(timeout if timeout is not None else settings.timeout)
    return [
        probe_model(
            settings.model_url,
            model=settings.model,
            api_key=settings.api_key,
            timeout=probe_timeout,
        ),
        probe_search(
            settings.resolved_search_url,
            backend=settings.search_backend,
            timeout=probe_timeout,
        ),
    ]


def probe_model(base_url: str, *, model: str, api_key: str | None, timeout: float) -> ProbeResult:
    url = f"{base_url.rstrip('/')}/models"
    request = Request(url, headers=_auth_headers(api_key), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            payload = _read_json(response.read())
    except HTTPError as exc:
        return ProbeResult("model", False, url, status=int(exc.code), message=_http_error_message(exc))
    except (URLError, TimeoutError, OSError) as exc:
        return ProbeResult("model", False, url, message=str(exc))

    available = _model_ids(payload)
    if available and model not in available:
        return ProbeResult(
            "model",
            False,
            url,
            status=status,
            model=model,
            message=f"requested model is not advertised; available={available[:8]}",
        )
    return ProbeResult(
        "model",
        200 <= status < 300,
        url,
        status=status,
        model=model,
        message="ready" if 200 <= status < 300 else f"unexpected HTTP status {status}",
    )


def probe_search(url: str, *, backend: str, timeout: float) -> ProbeResult:
    params = {"q": "RWKV"}
    if backend == "searxng":
        params["format"] = "json"
    probe_url = f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"
    request = Request(probe_url, headers={"User-Agent": "rwkv-web-harness/0.1"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            response.read(256)
    except HTTPError as exc:
        return ProbeResult("search", False, probe_url, status=int(exc.code), message=_http_error_message(exc))
    except (URLError, TimeoutError, OSError) as exc:
        return ProbeResult("search", False, probe_url, message=str(exc))
    return ProbeResult(
        "search",
        200 <= status < 300,
        probe_url,
        status=status,
        message="reachable" if 200 <= status < 300 else f"unexpected HTTP status {status}",
    )


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"User-Agent": "rwkv-web-harness/0.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _read_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _model_ids(payload: Any) -> list[str]:
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [str(row["id"]) for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)]


def _http_error_message(exc: HTTPError) -> str:
    try:
        raw_detail = exc.read()
    except (AttributeError, OSError):
        raw_detail = b""
    detail = raw_detail.decode("utf-8", errors="replace").strip()
    return f"HTTP {exc.code}" + (f": {detail[:240]}" if detail else "")
