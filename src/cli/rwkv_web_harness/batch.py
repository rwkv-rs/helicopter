from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .runner import RunResult


NETWORK_TOOLS = frozenset({"web_search", "open_url", "find_in_page"})


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    question: str
    expected_tools: tuple[str, ...] = ("web_search",)
    require_citation: bool = True
    tags: tuple[str, ...] = ()

    @classmethod
    def from_row(cls, row: dict[str, Any], *, line_number: int) -> "TaskSpec":
        task_id = row.get("task_id", row.get("id"))
        question = row.get("question", row.get("instruction", row.get("task")))
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"task line {line_number} must contain a non-empty task_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"task {task_id!r} must contain question, instruction, or task")

        raw_tools = row.get("expected_tools", ["web_search"])
        if isinstance(raw_tools, str):
            raw_tools = [raw_tools]
        if not isinstance(raw_tools, list) or not all(isinstance(item, str) for item in raw_tools):
            raise ValueError(f"task {task_id!r} expected_tools must be a string list")
        expected_tools = tuple(item.strip() for item in raw_tools if item.strip())
        unknown_tools = set(expected_tools) - NETWORK_TOOLS
        if unknown_tools:
            raise ValueError(f"task {task_id!r} contains unsupported expected tools: {sorted(unknown_tools)}")

        require_citation = row.get("require_citation", True)
        if not isinstance(require_citation, bool):
            raise ValueError(f"task {task_id!r} require_citation must be boolean")
        raw_tags = row.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        if not isinstance(raw_tags, list) or not all(isinstance(item, str) for item in raw_tags):
            raise ValueError(f"task {task_id!r} tags must be a string list")
        return cls(
            task_id=task_id.strip(),
            question=question.strip(),
            expected_tools=expected_tools,
            require_citation=require_citation,
            tags=tuple(item.strip() for item in raw_tags if item.strip()),
        )


def load_task_specs(path: str | Path) -> list[TaskSpec]:
    task_path = Path(path)
    specs: list[TaskSpec] = []
    seen: set[str] = set()
    with task_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {task_path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"task line {line_number} must be a JSON object")
            spec = TaskSpec.from_row(row, line_number=line_number)
            if spec.task_id in seen:
                raise ValueError(f"duplicate task_id: {spec.task_id}")
            seen.add(spec.task_id)
            specs.append(spec)
    if not specs:
        raise ValueError(f"task file is empty: {task_path}")
    return specs


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    if not trace_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid trace JSON in {trace_path}:{line_number}: {exc.msg}") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def validate_case(
    result: RunResult | dict[str, Any],
    spec: TaskSpec,
    trace_path: str | Path,
) -> dict[str, Any]:
    result_data = result.as_dict() if isinstance(result, RunResult) else dict(result)
    events = read_trace(trace_path)
    calls = [
        str(event.get("tool"))
        for event in events
        if event.get("event") == "tool_call" and isinstance(event.get("tool"), str)
    ]
    successful_results = [
        event
        for event in events
        if event.get("event") == "tool_result"
        and event.get("tool") in NETWORK_TOOLS
        and event.get("ok") is True
    ]
    evidence = [
        event
        for event in successful_results
        if _has_network_payload(event.get("tool"), event.get("data"))
    ]
    expected_present = _is_subsequence(spec.expected_tools, calls)
    expected_success = all(
        any(event.get("tool") == tool for event in successful_results)
        for tool in spec.expected_tools
    )
    citations = result_data.get("citations") or []
    checks = {
        "completed": result_data.get("status") == "completed",
        "trace_present": bool(events),
        "expected_tools_in_order": expected_present,
        "expected_tools_succeeded": expected_success,
        "network_evidence": bool(evidence),
        "citation_present": bool(citations) if spec.require_citation else True,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "actual_tools": calls,
        "successful_network_tools": [event.get("tool") for event in evidence],
        "network_evidence_count": len(evidence),
        "trace_events": len(events),
        "error": None if passed else _validation_error(checks),
    }


def summarize_cases(cases: Iterable[dict[str, Any]], *, suite: str) -> dict[str, Any]:
    case_list = list(cases)
    passed = sum(1 for case in case_list if case.get("validation", {}).get("passed") is True)
    completed = sum(1 for case in case_list if case.get("result", {}).get("status") == "completed")
    network_verified = sum(
        1 for case in case_list if case.get("validation", {}).get("checks", {}).get("network_evidence") is True
    )
    return {
        "suite": suite,
        "total": len(case_list),
        "passed": passed,
        "failed": len(case_list) - passed,
        "completed": completed,
        "network_verified": network_verified,
        "pass_rate": (passed / len(case_list)) if case_list else 0.0,
        "cases": case_list,
    }


def _has_network_payload(tool: Any, data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if tool == "web_search":
        return isinstance(data.get("query"), str) and isinstance(data.get("results"), list)
    if tool == "open_url":
        return isinstance(data.get("url"), str) and isinstance(data.get("content"), str) and bool(data["content"].strip())
    if tool == "find_in_page":
        return isinstance(data.get("source_id"), str) and isinstance(data.get("matches"), list)
    return False


def _is_subsequence(expected: tuple[str, ...], actual: list[str]) -> bool:
    if not expected:
        return True
    position = 0
    for tool in actual:
        if tool == expected[position]:
            position += 1
            if position == len(expected):
                return True
    return False


def _validation_error(checks: dict[str, bool]) -> str:
    return ", ".join(name for name, passed in checks.items() if not passed) or "validation failed"
