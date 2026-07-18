from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    answer: str
    citations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedTurn:
    action: Action | None = None
    final: FinalAnswer | None = None
    error: str | None = None


_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.IGNORECASE | re.DOTALL)


def parse_turn(text: str) -> ParsedTurn:
    """Parse the deliberately small text protocol emitted by the RWKV agent."""

    final_match = _FINAL_RE.search(text)
    tool_match = _TOOL_RE.search(text)
    if final_match and tool_match:
        return ParsedTurn(error="model emitted both a tool call and a final answer")
    if final_match:
        return ParsedTurn(final=_parse_final(final_match.group(1).strip()))
    if tool_match:
        return ParsedTurn(action=_parse_action(tool_match.group(1).strip()))
    return ParsedTurn(error="expected <tool_call>...</tool_call> or <final_answer>...</final_answer>")


def _parse_action(raw: str) -> Action:
    data = _parse_json_object(raw, "tool call")
    name = data.get("name", data.get("action"))
    arguments = data.get("arguments", data.get("args", {}))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must be a JSON object")
    return Action(name=name.strip(), arguments=dict(arguments))


def _parse_final(raw: str) -> FinalAnswer:
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"final answer JSON is invalid: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError("final answer JSON must be an object")
        answer = data.get("answer", data.get("text"))
        citations = data.get("citations", [])
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("final answer must contain a non-empty answer")
        if isinstance(citations, str):
            citations = [citations]
        if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
            raise ValueError("final answer citations must be a string list")
        return FinalAnswer(answer=answer.strip(), citations=tuple(citations))
    if not raw:
        raise ValueError("final answer cannot be empty")
    return FinalAnswer(answer=raw)


def _parse_json_object(raw: str, label: str) -> dict[str, Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON is invalid: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON must be an object")
    return data
