from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.0


class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> str:
        """Generate one model turn from a fully rendered prompt."""


class ModelBackendError(RuntimeError):
    """Raised when the local model backend cannot produce a response."""


class RWKVLocalBackend:
    """Call a local RWKV-compatible completion server.

    The server is expected to be local and OpenAI-compatible, for example the
    existing RWKV vLLM endpoint. No remote provider SDK is required; requests
    use Python's standard library and an optional bearer token.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        api_key: str | None = None,
        endpoint: str = "/completions",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("RWKV_MODEL_API_KEY")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def generate(self, request: GenerationRequest) -> str:
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ModelBackendError(f"model backend returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise ModelBackendError(f"cannot reach local model backend {self.url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ModelBackendError(f"model backend timed out after {self.timeout:.1f}s") from exc

        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelBackendError(f"model backend returned invalid JSON: {raw[:500]}") from exc
        text = _extract_text(data)
        if not text:
            raise ModelBackendError("model backend response did not contain completion text")
        return text


def _extract_text(data: Any) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    if isinstance(text, str):
        return text.strip()
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return str(message["content"]).strip()
    return ""
