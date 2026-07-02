from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class InferRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def normalize_api_base(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise InferRequestError(
            f"infer request failed: HTTP {exc.code}: {detail}",
            status_code=int(exc.code),
            detail=detail,
        ) from exc
    except urllib.error.URLError as exc:
        raise InferRequestError(f"infer request failed: {exc.reason}", detail=str(exc.reason)) from exc


def _is_context_length_error(exc: InferRequestError) -> bool:
    if exc.status_code not in {400, 413}:
        return False
    detail = str(exc.detail or exc).lower()
    return (
        "maximum context" in detail
        or "maximum context length" in detail
        or "reduce the length of the input prompt" in detail
        or "input_tokens" in detail
    )


def _is_response_format_retryable(exc: InferRequestError) -> bool:
    if exc.status_code not in {400, 422, 500}:
        return False
    detail = str(exc.detail or exc).lower()
    return (
        "response_format" in detail
        or "guided" in detail
        or "json mode" in detail
        or (exc.status_code == 500 and "internal server error" in detail)
    )


def _post_with_context_retry(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: str | None,
) -> dict[str, Any]:
    current = dict(payload)
    while True:
        try:
            return post_json(url, current, timeout_s=timeout_s, api_key=api_key)
        except InferRequestError as exc:
            max_tokens = int(current.get("max_tokens") or 0)
            next_tokens = max(1, max_tokens // 2)
            if _is_context_length_error(exc) and next_tokens < max_tokens:
                current["max_tokens"] = next_tokens
                continue
            if "response_format" in current and _is_response_format_retryable(exc):
                current.pop("response_format", None)
                continue
            raise


def chat_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if presence_penalty is not None:
        payload["presence_penalty"] = float(presence_penalty)
    if frequency_penalty is not None:
        payload["frequency_penalty"] = float(frequency_penalty)
    if seed is not None:
        payload["seed"] = int(seed)
    if stop:
        payload["stop"] = list(stop)
    response = _post_with_context_retry(
        f"{normalize_api_base(base_url)}/chat/completions",
        payload,
        timeout_s=timeout_s,
        api_key=api_key,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("infer response missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("infer response choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("infer response missing message")
    return str(message.get("content") or "")


def text_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if presence_penalty is not None:
        payload["presence_penalty"] = float(presence_penalty)
    if frequency_penalty is not None:
        payload["frequency_penalty"] = float(frequency_penalty)
    if seed is not None:
        payload["seed"] = int(seed)
    if stop:
        payload["stop"] = list(stop)
    response = _post_with_context_retry(
        f"{normalize_api_base(base_url)}/completions",
        payload,
        timeout_s=timeout_s,
        api_key=api_key,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("infer response missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("infer response choice is not an object")
    if "text" in choice:
        return str(choice.get("text") or "")
    message = choice.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    raise RuntimeError("infer response choice missing text")
