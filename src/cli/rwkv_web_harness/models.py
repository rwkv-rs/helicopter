from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import INTERFACE_CHOICES


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.0
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class GenerationResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None


class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse | str:
        """Generate one model turn from a rendered prompt or chat messages."""


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
        interface: str = "chat",
        endpoint: str | None = None,
    ) -> None:
        if interface not in INTERFACE_CHOICES:
            choices = ", ".join(INTERFACE_CHOICES)
            raise ValueError(f"interface must be one of: {choices}")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = float(timeout)
        self.api_key = api_key or os.environ.get("RWKV_MODEL_API_KEY")
        self.interface = interface
        default_endpoint = "/chat/completions" if interface == "chat" else "/completions"
        selected_endpoint = endpoint or default_endpoint
        self.endpoint = selected_endpoint if selected_endpoint.startswith("/") else f"/{selected_endpoint}"

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
        }
        if self.interface == "chat":
            payload["messages"] = request.messages or [{"role": "user", "content": request.prompt}]
            if request.tools:
                payload["tools"] = request.tools
                payload["tool_choice"] = "auto"
        else:
            payload["prompt"] = request.prompt
        if request.stop:
            payload["stop"] = request.stop
        if request.stop_token_ids:
            payload["stop_token_ids"] = request.stop_token_ids
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
        response = _extract_response(data)
        if not response.content and not response.tool_calls:
            raise ModelBackendError("model backend response did not contain text or tool calls")
        return response


def _extract_response(data: Any) -> GenerationResponse:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return GenerationResponse(content="")
    first = choices[0]
    if not isinstance(first, dict):
        return GenerationResponse(content="")
    text = first.get("text")
    if isinstance(text, str):
        content_text = text.strip()
        tool_calls = _extract_text_tool_calls(content_text)
        return GenerationResponse(
            content="" if tool_calls else content_text,
            tool_calls=tuple(tool_calls),
            finish_reason=_finish_reason(first),
        )
    message = first.get("message")
    if not isinstance(message, dict):
        return GenerationResponse(content="")
    content = message.get("content")
    content_text = content.strip() if isinstance(content, str) else ""
    tool_calls: list[ToolCall] = []
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for index, raw_call in enumerate(raw_tool_calls):
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            raw_arguments = function.get("arguments", {})
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    continue
            else:
                arguments = raw_arguments
            if not isinstance(arguments, dict):
                continue
            call_id = str(raw_call.get("id") or f"call_{index + 1}")
            tool_calls.append(ToolCall(call_id=call_id, name=name.strip(), arguments=dict(arguments)))
    if not tool_calls:
        tool_calls = _extract_text_tool_calls(content_text)
        if tool_calls:
            content_text = ""
    return GenerationResponse(
        content=content_text,
        tool_calls=tuple(tool_calls),
        finish_reason=_finish_reason(first),
    )


def _finish_reason(choice: dict[str, Any]) -> str | None:
    value = choice.get("finish_reason")
    return value if isinstance(value, str) else None


def _extract_text_tool_calls(text: str) -> list[ToolCall]:
    """Adapt RWKV JSON/agentic text when the model skips vLLM's parser marker.

    Some g1h checkpoints emit a JSON plan with ``commands[].keystrokes`` rather
    than the ``**Tool Call:**`` marker understood by vLLM's ``rwkv`` parser.
    This deliberately supports only the local web tools and only the first
    command, so an arbitrary model string cannot become an arbitrary action.
    """

    normalized = text.replace("\x60\x60\x60json", "").replace("\x60\x60", "")
    if normalized.lstrip().startswith(('"name"', '"tool"', '"action"', '"function"')):
        payload = _decode_json_candidate("{" + normalized.lstrip())
        parsed = _coerce_payload_tool_call(payload, index=0)
        if parsed is not None:
            return [parsed]
    for start in (index for index, char in enumerate(normalized) if char in "{["):
        payload = _decode_json_candidate(normalized[start:])
        parsed = _coerce_payload_tool_call(payload, index=start)
        if parsed is not None:
            return [parsed]
    return []


def _coerce_payload_tool_call(payload: Any, *, index: int) -> ToolCall | None:
    if not isinstance(payload, (dict, list)):
        return None
    candidates: list[Any] = []
    if isinstance(payload, list):
        candidates.extend(payload)
    elif isinstance(payload.get("tool_calls"), list):
        candidates.extend(payload["tool_calls"])
    elif isinstance(payload.get("commands"), list):
        candidates.extend(payload["commands"])
    elif "final_answer" in payload:
        final_answer = payload["final_answer"]
        if isinstance(final_answer, dict):
            final_arguments = final_answer
        else:
            final_arguments = {"answer": final_answer}
            if "citations" in payload:
                final_arguments["citations"] = payload["citations"]
        candidates.append({"name": "final_answer", "arguments": final_arguments})
    elif "answer" in payload:
        candidates.append({"name": "final_answer", "arguments": payload})
    elif "name" in payload or "function" in payload:
        candidates.append(payload)
    for offset, candidate in enumerate(candidates):
        parsed = _coerce_text_tool_call(candidate, index=index + offset)
        if parsed is not None:
            return parsed
    return None


def _leading_json_value(text: str) -> Any:
    normalized = text.strip()
    if "</think>" in normalized:
        after_think = normalized.rsplit("</think>", 1)[1].strip()
        normalized = after_think or normalized.rsplit("</think>", 1)[0].strip()
    if "```" in normalized:
        normalized = normalized.replace("```json", "").replace("```", "").strip()
    prefilled = normalized.lstrip()
    if prefilled.startswith(('"name"', '"tool"', '"action"', '"function"')):
        normalized = "{" + prefilled
        start = 0
    else:
        start = min((index for index in (normalized.find("{"), normalized.find("[")) if index >= 0), default=-1)
        if start < 0:
            return None
    return _decode_json_candidate(normalized[start:])


def _decode_json_candidate(text: str) -> Any:
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
        return value
    except json.JSONDecodeError:
        repaired = _repair_json(text)
        if repaired is None:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(repaired)
        except json.JSONDecodeError:
            return None
        return value


def _repair_json(text: str) -> str | None:
    """Repair only unmatched JSON brackets in a model-generated call."""

    output: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            output.append(char)
        elif char in "{[":
            stack.append(char)
            output.append(char)
        elif char in "}]":
            expected = "{" if char == "}" else "["
            if stack and stack[-1] == expected:
                stack.pop()
                output.append(char)
        else:
            output.append(char)
    if in_string:
        return None
    output.extend("}" if opener == "{" else "]" for opener in reversed(stack))
    return "".join(output)


def _coerce_text_tool_call(candidate: Any, *, index: int) -> ToolCall | None:
    if not isinstance(candidate, dict):
        return None
    # A g1h checkpoint sometimes echoes the JSON observation returned by the
    # harness.  It is not a new function call; treating it as one can create a
    # policy loop after the final network tool has already succeeded.
    if (
        candidate.get("ok") in {True, False}
        and isinstance(candidate.get("message"), str)
        and isinstance(candidate.get("tool"), str)
        and "arguments" not in candidate
        and "input" not in candidate
        and "parameters" not in candidate
    ):
        return None
    function = candidate.get("function") if isinstance(candidate.get("function"), dict) else candidate
    name = function.get("name") or function.get("tool") or function.get("tool_name")
    arguments = function.get("arguments", function.get("input", function.get("parameters", {})))
    if not isinstance(name, str) or name.strip() not in {"web_search", "open_url", "find_in_page", "final_answer"}:
        keystrokes = candidate.get("keystrokes")
        if not isinstance(keystrokes, str):
            return None
        try:
            tokens = shlex.split(keystrokes.strip().splitlines()[0])
        except ValueError:
            return None
        if not tokens or tokens[0] not in {"web_search", "open_url", "find_in_page"}:
            return None
        name = tokens[0]
        if name == "web_search":
            arguments = {"query": " ".join(tokens[1:])}
        elif name == "open_url":
            arguments = {"url": tokens[1]} if len(tokens) > 1 else {}
        else:
            arguments = {"source_id": tokens[1], "pattern": " ".join(tokens[2:])} if len(tokens) > 2 else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return ToolCall(call_id=str(candidate.get("id") or f"text_call_{index + 1}"), name=name.strip(), arguments=arguments)
