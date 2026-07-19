from __future__ import annotations

"""Shared configuration for the local web harness.

The CLI and the embeddable API should validate the same runtime contract.  Keeping
the defaults here prevents the single-task and batch commands from drifting apart.
"""

import os
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlparse


Interface = Literal["chat", "completion", "rwkv-json", "g1h"]
SearchBackend = Literal["html", "searxng"]

INTERFACE_CHOICES: tuple[Interface, ...] = ("chat", "completion", "rwkv-json", "g1h")
SEARCH_BACKEND_CHOICES: tuple[SearchBackend, ...] = ("html", "searxng")

DEFAULT_MODEL_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_NAME = "RWKV"
DEFAULT_MODEL_INTERFACE: Interface = "chat"
DEFAULT_SEARCH_BACKEND: SearchBackend = "html"
DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_CONTEXT_CHARS = 12_000
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_MAX_PAGE_CHARS = 6_000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 120.0


def resolve_search_url(search_backend: SearchBackend, configured: str | None = None) -> str:
    """Return the configured search URL or the backend's local-first default."""

    if configured and configured.strip():
        return configured.strip()
    if search_backend == "searxng":
        return "http://127.0.0.1:8080/search"
    return "https://lite.duckduckgo.com/lite/"


@dataclass(frozen=True, slots=True)
class HarnessSettings:
    """Validated settings shared by CLI runs and embedded callers."""

    model_url: str = DEFAULT_MODEL_URL
    model: str = DEFAULT_MODEL_NAME
    api_key: str | None = None
    interface: Interface = DEFAULT_MODEL_INTERFACE
    endpoint: str | None = None
    search_url: str | None = None
    search_backend: SearchBackend = DEFAULT_SEARCH_BACKEND
    max_steps: int = DEFAULT_MAX_STEPS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_page_chars: int = DEFAULT_MAX_PAGE_CHARS
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        _validate_http_url(self.model_url, "model_url")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.interface not in INTERFACE_CHOICES:
            raise ValueError(f"unsupported interface: {self.interface}")
        if self.search_backend not in SEARCH_BACKEND_CHOICES:
            raise ValueError(f"unsupported search backend: {self.search_backend}")
        if self.endpoint is not None and not self.endpoint.strip():
            raise ValueError("endpoint must not be empty when provided")
        for name in ("max_steps", "max_context_chars", "max_new_tokens", "max_page_chars"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    @classmethod
    def from_namespace(cls, args: Any) -> "HarnessSettings":
        """Build settings from an argparse namespace without duplicating defaults."""

        return cls(
            model_url=str(getattr(args, "model_url", DEFAULT_MODEL_URL) or DEFAULT_MODEL_URL),
            model=str(getattr(args, "model", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME),
            api_key=getattr(args, "api_key", None),
            interface=cast(
                Interface,
                str(getattr(args, "interface", DEFAULT_MODEL_INTERFACE) or DEFAULT_MODEL_INTERFACE),
            ),
            endpoint=getattr(args, "endpoint", None),
            search_url=getattr(args, "search_url", None),
            search_backend=cast(
                SearchBackend,
                str(getattr(args, "search_backend", DEFAULT_SEARCH_BACKEND) or DEFAULT_SEARCH_BACKEND),
            ),
            max_steps=int(getattr(args, "max_steps", DEFAULT_MAX_STEPS) or DEFAULT_MAX_STEPS),
            max_context_chars=int(
                getattr(args, "max_context_chars", DEFAULT_MAX_CONTEXT_CHARS) or DEFAULT_MAX_CONTEXT_CHARS
            ),
            max_new_tokens=int(getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS) or DEFAULT_MAX_NEW_TOKENS),
            max_page_chars=int(getattr(args, "max_page_chars", DEFAULT_MAX_PAGE_CHARS) or DEFAULT_MAX_PAGE_CHARS),
            temperature=float(getattr(args, "temperature", DEFAULT_TEMPERATURE)),
            timeout=float(getattr(args, "timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT),
        )

    @classmethod
    def from_env(cls) -> "HarnessSettings":
        """Build the default runtime from the documented environment variables."""

        return cls(
            model_url=os.environ.get("RWKV_MODEL_URL", DEFAULT_MODEL_URL),
            model=os.environ.get("RWKV_MODEL_NAME", DEFAULT_MODEL_NAME),
            api_key=os.environ.get("RWKV_MODEL_API_KEY"),
            interface=cast(Interface, os.environ.get("RWKV_MODEL_INTERFACE", DEFAULT_MODEL_INTERFACE)),
            search_url=os.environ.get("RWKV_WEB_SEARCH_URL"),
            search_backend=cast(
                SearchBackend,
                os.environ.get("RWKV_WEB_SEARCH_BACKEND", DEFAULT_SEARCH_BACKEND),
            ),
        )

    @property
    def resolved_search_url(self) -> str:
        return resolve_search_url(self.search_backend, self.search_url)


def _validate_http_url(value: str, field_name: str) -> None:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute http(s) URL")
