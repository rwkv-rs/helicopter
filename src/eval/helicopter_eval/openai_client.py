from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


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
        raise RuntimeError(f"infer request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"infer request failed: {exc.reason}") from exc


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
    response = post_json(
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
    response = post_json(
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
