from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any
from urllib.parse import urlparse


_SRC_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _SRC_ROOT.parent
_CLIENT_DIR = _SRC_ROOT / "scoreboard-client"
_SCRIPT = _CLIENT_DIR / "scripts" / "capture-page.mjs"
_OUTPUT_DIR = _REPO_ROOT / "tmp" / "scoreboard-screenshots"
_DEFAULT_URL = "http://127.0.0.1:3000/"


def _local_url(raw_url: Any) -> str:
    url = str(raw_url or _DEFAULT_URL).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("screenshot URL must use http or https")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("screenshot URL must point to localhost or 127.0.0.1")
    return url


def _positive_int(raw_value: Any, fallback: int, *, low: int, high: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return fallback
    return min(max(value, low), high)


def capture_page(*, url: Any = None, width: Any = None, height: Any = None) -> dict[str, Any]:
    target_url = _local_url(url)
    viewport_width = _positive_int(width, 1440, low=900, high=2400)
    viewport_height = _positive_int(height, 1200, low=700, high=1800)

    if not _SCRIPT.is_file():
        raise FileNotFoundError(f"screenshot script not found: {_SCRIPT}")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = _OUTPUT_DIR / f"dashboard-{stamp}.png"

    result = subprocess.run(
        [
            "node",
            str(_SCRIPT),
            target_url,
            str(output_path),
            str(viewport_width),
            str(viewport_height),
        ],
        cwd=str(_CLIENT_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(detail)

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {}

    return {
        "path": str(output_path),
        "url": target_url,
        "width": viewport_width,
        "height": viewport_height,
        "page_height": payload.get("pageHeight"),
        "full_page": True,
    }
