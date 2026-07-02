from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any

from .apibank import decode_tool_calls
from .openai_client import chat_completion
from .sampling import apply_limit_or_sample, dataset_sample_suffix
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


MCP_BENCH_PASS_THRESHOLD = 7.0
MCP_BENCH_MAX_HISTORY_CHARS = 24000
MCP_BENCH_MAX_RESULT_CHARS = 6000
MCP_BENCH_MAX_ERROR_CHARS = 2000
MCP_BENCH_MAX_TOOL_SCHEMA_CHARS = 6000
MCP_BENCH_TASK_FILES: tuple[str, ...] = (
    "mcpbench_tasks_single_runner_format.json",
    "mcpbench_tasks_multi_2server_runner_format.json",
    "mcpbench_tasks_multi_3server_runner_format.json",
)
MCP_BENCH_DATASET_FILES: dict[str, tuple[str, ...] | None] = {
    "mcp_bench": None,
    "mcp_bench_single": ("mcpbench_tasks_single_runner_format.json",),
    "mcp_bench_multi_2server": ("mcpbench_tasks_multi_2server_runner_format.json",),
    "mcp_bench_multi_3server": ("mcpbench_tasks_multi_3server_runner_format.json",),
}


@dataclass(frozen=True, slots=True)
class McpBenchTaskSpec:
    task_id: str
    task_description: str
    fuzzy_description: str = ""
    dependency_analysis: str = ""
    distraction_servers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class McpBenchItem:
    task_file: str
    server_name: str
    combination_name: str
    combination_type: str
    servers: tuple[str, ...]
    task: McpBenchTaskSpec
    runtime_root: str | None = None
    tasks_root: str | None = None
    official_source_root: str | None = None


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    server: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.server}:{self.tool}" if self.server else self.tool


@dataclass(frozen=True, slots=True)
class PlanningDecision:
    should_continue: bool
    tool_calls: tuple[PlannedToolCall, ...]
    final_answer: str = ""


@dataclass(frozen=True, slots=True)
class McpBenchExecutionResult:
    tool: str
    server: str
    parameters: dict[str, Any]
    round_num: int
    planned_layer: int | None
    success: bool
    result: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class McpBenchEvaluation:
    task_completion_score: float = 0.0
    tool_selection_score: float = 0.0
    planning_effectiveness_and_efficiency_score: float = 0.0
    task_fulfillment: float = 0.0
    grounding: float = 0.0
    tool_appropriateness: float = 0.0
    parameter_accuracy: float = 0.0
    dependency_awareness: float = 0.0
    parallelism_and_efficiency: float = 0.0
    input_schema_compliance: float | None = None
    valid_tool_name_rate: float | None = None
    execution_success_rate: float | None = None
    planning_json_compliance: float | None = None


@dataclass(frozen=True, slots=True)
class McpBenchPreflightReport:
    ok: bool
    runtime_root: str
    worker_script: str
    checked_servers: tuple[str, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class McpBenchResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    final_answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
    evaluation: McpBenchEvaluation | None
    trace: tuple[dict[str, Any], ...]

    def to_scoreboard(self) -> ScoreboardEvalResult:
        answer = {
            "final_answer": self.final_answer,
            "task_id": self.task_id,
            "trace": list(self.trace),
            "evaluation": _evaluation_to_dict(self.evaluation),
        }
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=json.dumps(answer, ensure_ascii=False, sort_keys=True),
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    model: str
    api_key: str
    base_url: str = ""


@dataclass(frozen=True, slots=True)
class McpBenchRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    source_path: str | None = None
    source_root: str | None = None
    runtime_root: str | None = None
    worker_script: str | None = None
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    decision_max_tokens: int = 1024
    final_max_tokens: int = 1024
    max_rounds: int = 8
    timeout_s: float = 600.0
    tool_router_mode: str = "off"
    tool_router_max_tools: int = 16
    tool_router_trigger_tool_count: int = 20
    tool_router_trigger_catalog_chars: int = 6000
    tool_router_context_chars: int = 6000
    tool_router_description_chars: int = 240
    long_context_router_mode: str = "off"
    long_context_min_chars: int = 4000
    long_context_chunk_chars: int = 1200
    long_context_overlap_lines: int = 2
    long_context_max_evidence_chunks: int = 4
    long_context_max_evidence_chars: int = 6000
    skip_runtime_preflight: bool = False
    scoreboard_dataset: str | None = None
    job_name: str = "function_mcp_bench"
    job_id: str | None = None
    runner: str = "helicopter_eval.mcp_bench"
    cot_mode: str = "CoT"


class McpBenchWorkerClient:
    def __init__(self, *, runtime_root: str | Path, worker_script: str | Path) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.worker_script = Path(worker_script).expanduser().resolve()
        python_bin = self.runtime_root / ".venv" / "bin" / "python"
        if not python_bin.is_file():
            raise FileNotFoundError(f"missing MCP-Bench runtime python: {python_bin}")
        self._proc = subprocess.Popen(
            [str(python_bin), str(self.worker_script), "--runtime-root", str(self.runtime_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, name="McpBenchWorkerStderr", daemon=True)
        self._stderr_thread.start()
        self._closed = False

    def open_task(self, item: McpBenchItem) -> dict[str, Any]:
        response = self._request("open_task", {"servers": list(item.servers)})
        available_tools = response.get("available_tools")
        if not isinstance(available_tools, dict):
            raise RuntimeError("worker returned invalid available_tools payload")
        return available_tools

    def call_tool(self, full_tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request("call_tool", {"tool_name": full_tool_name, "arguments": dict(arguments)})
        if not isinstance(response, dict):
            raise RuntimeError("worker returned invalid tool response")
        return response

    def evaluate(self, request: Mapping[str, Any]) -> McpBenchEvaluation:
        response = self._request("evaluate", {"request": dict(request)})
        if not isinstance(response, dict):
            raise RuntimeError("worker returned invalid evaluation payload")
        return McpBenchEvaluation(
            task_completion_score=float(response.get("task_completion_score", 0.0)),
            tool_selection_score=float(response.get("tool_selection_score", 0.0)),
            planning_effectiveness_and_efficiency_score=float(
                response.get("planning_effectiveness_and_efficiency_score", 0.0)
            ),
            task_fulfillment=float(response.get("task_fulfillment", 0.0)),
            grounding=float(response.get("grounding", 0.0)),
            tool_appropriateness=float(response.get("tool_appropriateness", 0.0)),
            parameter_accuracy=float(response.get("parameter_accuracy", 0.0)),
            dependency_awareness=float(response.get("dependency_awareness", 0.0)),
            parallelism_and_efficiency=float(response.get("parallelism_and_efficiency", 0.0)),
            input_schema_compliance=_float_or_none(response.get("input_schema_compliance")),
            valid_tool_name_rate=_float_or_none(response.get("valid_tool_name_rate")),
            execution_success_rate=_float_or_none(response.get("execution_success_rate")),
            planning_json_compliance=_float_or_none(response.get("planning_json_compliance")),
        )

    def close_task(self) -> None:
        if self._closed:
            return
        try:
            self._request("close_task", {})
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None:
                try:
                    self._request("shutdown", {})
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5.0)

    def _request(self, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("worker client already closed")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("worker pipes are unavailable")
        wire = json.dumps({"action": action, "payload": dict(payload)}, ensure_ascii=False)
        try:
            self._proc.stdin.write(wire + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._worker_failure_message("worker stdin closed")) from exc

        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(self._worker_failure_message("worker stdout closed"))
            raw = line.strip()
            if not raw:
                continue
            try:
                response = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(response, dict):
                continue
            if not response.get("ok", False):
                error = str(response.get("error") or "unknown worker error")
                raise RuntimeError(self._worker_failure_message(error))
            data = response.get("data")
            return data if isinstance(data, dict) else {}

    def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            self._stderr_lines.append(line.rstrip("\n"))

    def _worker_failure_message(self, message: str) -> str:
        stderr = "\n".join(self._stderr_lines)
        if stderr:
            return f"{message}\nworker stderr:\n{stderr}"
        return message


def load_mcp_bench_task_items(
    tasks_root: str | Path,
    runtime_root: str | Path,
    *,
    file_names: Sequence[str] | None = None,
) -> list[McpBenchItem]:
    root = Path(tasks_root).expanduser().resolve()
    runtime = Path(runtime_root).expanduser().resolve()
    selected_file_names = tuple(file_names or MCP_BENCH_TASK_FILES)
    items: list[McpBenchItem] = []
    for file_name in selected_file_names:
        payload = json.loads((root / file_name).read_text(encoding="utf-8"))
        for group in payload.get("server_tasks", []):
            if not isinstance(group, dict):
                continue
            for task_payload in group.get("tasks", []) or []:
                if not isinstance(task_payload, dict):
                    continue
                items.append(
                    McpBenchItem(
                        task_file=file_name,
                        server_name=str(group.get("server_name") or ""),
                        combination_name=str(group.get("combination_name") or ""),
                        combination_type=str(group.get("combination_type") or ""),
                        servers=tuple(str(item) for item in (group.get("servers") or [])),
                        task=McpBenchTaskSpec(
                            task_id=str(task_payload.get("task_id") or ""),
                            task_description=str(task_payload.get("task_description") or ""),
                            fuzzy_description=str(task_payload.get("fuzzy_description") or ""),
                            dependency_analysis=str(task_payload.get("dependency_analysis") or ""),
                            distraction_servers=tuple(
                                str(item) for item in (task_payload.get("distraction_servers") or [])
                            ),
                        ),
                        runtime_root=str(runtime),
                        tasks_root=str(root),
                        official_source_root=str(root.parent.parent if root.name == "tasks" else root.parent),
                    )
                )
    return items


def load_mcp_bench_manifest_records(path: str | Path) -> list[McpBenchItem]:
    items: list[McpBenchItem] = []
    target = Path(path).expanduser().resolve()
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            task_payload = payload.get("task") or {}
            if not isinstance(task_payload, dict):
                task_payload = {}
            items.append(
                McpBenchItem(
                    task_file=str(payload.get("task_file") or ""),
                    server_name=str(payload.get("server_name") or ""),
                    combination_name=str(payload.get("combination_name") or ""),
                    combination_type=str(payload.get("combination_type") or ""),
                    servers=tuple(str(item) for item in (payload.get("servers") or [])),
                    task=McpBenchTaskSpec(
                        task_id=str(task_payload.get("task_id") or payload.get("task_id") or ""),
                        task_description=str(task_payload.get("task_description") or ""),
                        fuzzy_description=str(task_payload.get("fuzzy_description") or payload.get("instruction") or ""),
                        dependency_analysis=str(task_payload.get("dependency_analysis") or ""),
                        distraction_servers=tuple(
                            str(item) for item in (task_payload.get("distraction_servers") or [])
                        ),
                    ),
                    runtime_root=str(payload.get("runtime_root") or ""),
                    tasks_root=str(payload.get("tasks_root") or ""),
                    official_source_root=str(payload.get("official_source_root") or ""),
                )
            )
    if not items:
        raise ValueError(f"MCP-Bench manifest is empty: {target}")
    return items


def load_samples(config: McpBenchRunConfig) -> list[McpBenchItem]:
    if config.split != "test":
        raise ValueError("MCP-Bench only provides test split")
    if config.dataset_name not in MCP_BENCH_DATASET_FILES:
        raise ValueError(f"unknown MCP-Bench dataset: {config.dataset_name}")
    items = _load_items(config)
    items = apply_limit_or_sample(
        items,
        limit=config.limit,
        sample_size=config.sample_size,
        sample_seed=config.sample_seed,
    )
    if not items:
        raise ValueError("MCP-Bench run selected zero samples")
    return items


def preflight_mcp_bench_runtime(
    items: Sequence[McpBenchItem],
    *,
    runtime_root: str | Path,
    worker_script: str | Path,
    open_first_task: bool = False,
    raise_on_error: bool = False,
) -> McpBenchPreflightReport:
    runtime = Path(runtime_root).expanduser().resolve()
    worker = Path(worker_script).expanduser().resolve()
    errors: list[str] = []
    if not items:
        errors.append("mcp_bench_no_items")
    python_bin = runtime / ".venv" / "bin" / "python"
    commands_path = runtime / "mcp_servers" / "commands.json"
    if not python_bin.is_file():
        errors.append(f"missing_runtime_python:{python_bin}")
    if not worker.is_file():
        errors.append(f"missing_worker_script:{worker}")
    commands: Mapping[str, Any] = {}
    if not commands_path.is_file():
        errors.append(f"missing_commands_json:{commands_path}")
    else:
        try:
            payload = json.loads(commands_path.read_text(encoding="utf-8"))
            commands = payload if isinstance(payload, Mapping) else {}
            if not commands:
                errors.append(f"invalid_commands_json:{commands_path}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid_commands_json:{commands_path}:{exc}")

    checked_servers = tuple(sorted({server for item in items for server in item.servers if server}))
    for server_name in checked_servers:
        raw = commands.get(server_name) if isinstance(commands, Mapping) else None
        if not isinstance(raw, Mapping):
            errors.append(f"missing_server_config:{server_name}")
            continue
        if not str(raw.get("cmd") or "").strip():
            errors.append(f"missing_server_command:{server_name}")
        cwd = _resolve_mcp_server_cwd(runtime, str(raw.get("cwd") or ""))
        if not cwd.exists():
            errors.append(f"missing_server_cwd:{server_name}:{cwd}")

    if not errors and open_first_task and items:
        client = McpBenchWorkerClient(runtime_root=runtime, worker_script=worker)
        try:
            client.open_task(items[0])
            client.close_task()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"worker_open_task_failed:{exc}")
        finally:
            client.close()

    report = McpBenchPreflightReport(
        ok=not errors,
        runtime_root=str(runtime),
        worker_script=str(worker),
        checked_servers=checked_servers,
        errors=tuple(errors),
    )
    if errors and raise_on_error:
        raise RuntimeError("MCP-Bench runtime preflight failed: " + "; ".join(errors))
    return report


def presented_task(item: McpBenchItem) -> str:
    fuzzy = item.task.fuzzy_description.strip()
    return fuzzy or item.task.task_description.strip()


def build_mcp_task_user_message(item: McpBenchItem) -> str:
    return f"Task:\n{presented_task(item)}"


def route_mcp_tools(
    item: McpBenchItem,
    available_tools: Mapping[str, Mapping[str, Any]],
    prompt_messages: Sequence[Mapping[str, Any]],
    config: McpBenchRunConfig,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    mode = str(config.tool_router_mode or "off").strip().lower()
    catalog_chars = len(render_tool_catalog(available_tools))
    trace: dict[str, Any] = {
        "mode": mode,
        "enabled": mode != "off",
        "input_tool_count": len(available_tools),
        "input_catalog_chars": catalog_chars,
        "selected_tool_count": len(available_tools),
        "selected_tools": sorted(_mcp_tool_full_name(tool) for tool in available_tools.values()),
        "triggered": False,
    }
    if mode == "off":
        return dict(available_tools), trace
    if mode != "lexical":
        raise ValueError(f"unsupported MCP tool_router_mode={config.tool_router_mode!r}; expected off or lexical")
    triggered = (
        len(available_tools) > max(1, int(config.tool_router_trigger_tool_count))
        or catalog_chars > max(1, int(config.tool_router_trigger_catalog_chars))
    )
    trace["triggered"] = triggered
    if not triggered:
        return dict(available_tools), trace
    query = _mcp_router_query(item, prompt_messages, max_chars=max(1, int(config.tool_router_context_chars)))
    query_terms = _lexical_terms(query)
    scored: list[tuple[float, str, Mapping[str, Any]]] = []
    for key, tool in available_tools.items():
        scored.append((_tool_score(tool, query_terms), str(key), tool))
    scored.sort(key=lambda row: (-row[0], _mcp_tool_full_name(row[2])))
    selected_rows = scored[: max(1, int(config.tool_router_max_tools))]
    selected: dict[str, Mapping[str, Any]] = {key: tool for _score, key, tool in selected_rows}
    trace.update(
        {
            "selected_tool_count": len(selected),
            "selected_tools": [_mcp_tool_full_name(tool) for _score, _key, tool in selected_rows],
            "scores": [
                {"tool": _mcp_tool_full_name(tool), "score": score}
                for score, _key, tool in selected_rows[: max(1, int(config.tool_router_max_tools))]
            ],
        }
    )
    return selected, trace


def compact_mcp_messages_for_long_context(
    item: McpBenchItem,
    messages: Sequence[Mapping[str, str]],
    config: McpBenchRunConfig,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    mode = str(config.long_context_router_mode or "off").strip().lower()
    trace: dict[str, Any] = {
        "mode": mode,
        "enabled": mode != "off",
        "compacted_messages": 0,
        "selected_chunks": 0,
    }
    if mode == "off":
        return [dict(message) for message in messages], trace
    if mode != "lexical":
        raise ValueError(
            f"unsupported MCP long_context_router_mode={config.long_context_router_mode!r}; expected off or lexical"
        )
    query_terms = _lexical_terms(presented_task(item) + "\n" + _format_messages(messages[-4:]))
    compacted: list[dict[str, str]] = []
    chunk_count = 0
    for message in messages:
        role = str(message.get("role") or "message")
        content = str(message.get("content") or "")
        if len(content) < max(1, int(config.long_context_min_chars)):
            compacted.append({"role": role, "content": content})
            continue
        evidence, selected = _select_long_context_evidence(
            content,
            query_terms=query_terms,
            chunk_chars=max(1, int(config.long_context_chunk_chars)),
            overlap_lines=max(0, int(config.long_context_overlap_lines)),
            max_chunks=max(1, int(config.long_context_max_evidence_chunks)),
            max_chars=max(1, int(config.long_context_max_evidence_chars)),
        )
        compacted.append(
            {
                "role": role,
                "content": (
                    "[long context compacted]\n"
                    f"original_chars={len(content)} selected_chunks={len(selected)}\n"
                    f"{evidence}"
                ),
            }
        )
        trace["compacted_messages"] = int(trace["compacted_messages"]) + 1
        chunk_count += len(selected)
    trace["selected_chunks"] = chunk_count
    return compacted, trace


def build_planning_prompt(
    item: McpBenchItem,
    available_tools: Mapping[str, Mapping[str, Any]],
    prompt_messages: Sequence[Mapping[str, Any]],
    *,
    tool_description_chars: int = 400,
) -> str:
    schema = {
        "type": "object",
        "required": ["name", "arguments"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"},
        },
    }
    return (
        "You are solving an MCP-Bench interactive task.\n\n"
        "Available MCP tools:\n"
        f"{render_tool_catalog(available_tools, description_chars=tool_description_chars)}\n\n"
        "Output JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "Return exactly one JSON object. Use a listed tool name in the form `server:tool`, "
        "or use `final_answer` when no more MCP tool calls are needed. "
        "Do not invent tool names, arguments, or tool results. Return no prose or markdown.\n\n"
        "Conversation:\n"
        f"{_format_messages(prompt_messages)}\n\n"
        "Next action:"
    )


def build_final_answer_prompt(item: McpBenchItem, accumulated_information: str) -> str:
    history = _trim_text(accumulated_information, MCP_BENCH_MAX_HISTORY_CHARS) or "No tool evidence was gathered."
    return (
        "You are the final answer synthesizer for an MCP benchmark agent.\n"
        "Use only the gathered evidence below. Do not invent missing facts.\n"
        "Return only the final answer requested by the task. If the task requires JSON, return valid JSON.\n\n"
        f"Task:\n{presented_task(item)}\n\n"
        f"Function output history:\n{history}\n\n"
        "Final answer:"
    )


def parse_planning_decision(response: str) -> PlanningDecision:
    calls = decode_tool_calls(response)
    if not calls:
        raise ValueError("model returned no planning tool call")
    call = calls[0]
    name = str(call.get("name") or "").strip()
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    if name == "final_answer":
        return PlanningDecision(
            should_continue=False,
            tool_calls=(),
            final_answer=str(arguments.get("answer") or "").strip(),
        )
    server = ""
    tool = name
    if ":" in tool:
        server, tool = [part.strip() for part in tool.split(":", 1)]
    return PlanningDecision(
        should_continue=True,
        tool_calls=(PlannedToolCall(server=server, tool=tool, arguments=dict(arguments)),),
    )


def normalize_planned_tool_call(
    call: PlannedToolCall,
    available_tools: Mapping[str, Mapping[str, Any]],
) -> PlannedToolCall:
    server = call.server.strip()
    tool = call.tool.strip()
    if not server and ":" in tool:
        server, tool = [part.strip() for part in tool.split(":", 1)]
    full_name = f"{server}:{tool}" if server else tool
    if server and full_name in available_tools:
        return PlannedToolCall(server=server, tool=tool, arguments=dict(call.arguments))
    if not server:
        matches = sorted(name for name in available_tools if name.endswith(f":{tool}"))
        if len(matches) == 1:
            matched_server, matched_tool = matches[0].split(":", 1)
            return PlannedToolCall(server=matched_server, tool=matched_tool, arguments=dict(call.arguments))
    raise ValueError(f"planned tool `{full_name}` was not found in available tools")


def generate_sample(
    sample_index: int,
    item: McpBenchItem,
    config: McpBenchRunConfig,
    *,
    worker: McpBenchWorkerClient,
    judge: JudgeConfig,
) -> McpBenchResult:
    available_tools = worker.open_task(item)
    prompts: list[str] = []
    completions: list[str] = []
    steps: list[dict[str, Any]] = []
    execution_results: list[McpBenchExecutionResult] = []
    prompt_messages: list[dict[str, str]] = [{"role": "user", "content": build_mcp_task_user_message(item)}]
    accumulated_information = ""
    final_answer = ""
    fail_reason = ""
    evaluation: McpBenchEvaluation | None = None
    is_passed = False
    total_planned_tools = 0
    valid_planned_tools = 0
    executed_rounds = 0
    try:
        for round_num in range(1, max(1, int(config.max_rounds)) + 1):
            planning_messages, long_context_trace = compact_mcp_messages_for_long_context(item, prompt_messages, config)
            routed_tools, tool_router_trace = route_mcp_tools(item, available_tools, planning_messages, config)
            decision_prompt = build_planning_prompt(
                item,
                routed_tools,
                planning_messages,
                tool_description_chars=max(40, int(config.tool_router_description_chars)),
            )
            decision_output = chat_completion(
                base_url=config.base_url,
                model=config.model,
                prompt=decision_prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.decision_max_tokens,
                timeout_s=config.timeout_s,
                response_format={"type": "json_object"},
            )
            prompts.append(decision_prompt)
            completions.append(decision_output)
            try:
                decision = parse_planning_decision(decision_output)
            except Exception as exc:  # noqa: BLE001
                fail_reason = f"parse_error:{exc}"
                steps.append(
                    {
                        "round_num": round_num,
                        "decision": {"raw": decision_output, "parse_error": str(exc)},
                        "executions": [],
                        "tool_router": tool_router_trace,
                        "long_context": long_context_trace,
                    }
                )
                break

            if decision.should_continue:
                prompt_messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"name": decision.tool_calls[0].full_name, "arguments": decision.tool_calls[0].arguments},
                            ensure_ascii=False,
                        ),
                    }
                )
            else:
                final_answer = decision.final_answer.strip()
                prompt_messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"name": "final_answer", "arguments": {"answer": final_answer}},
                            ensure_ascii=False,
                        ),
                    }
                )

            round_executions: list[McpBenchExecutionResult] = []
            if decision.should_continue:
                for planned_layer, raw_call in enumerate(decision.tool_calls):
                    total_planned_tools += 1
                    try:
                        normalized = normalize_planned_tool_call(raw_call, routed_tools)
                        valid_planned_tools += 1
                        tool_response = worker.call_tool(normalized.full_name, normalized.arguments)
                        success = bool(tool_response.get("success", False))
                        round_executions.append(
                            McpBenchExecutionResult(
                                tool=normalized.full_name,
                                server=normalized.server,
                                parameters=dict(normalized.arguments),
                                round_num=round_num,
                                planned_layer=planned_layer,
                                success=success,
                                result=str(tool_response.get("result") or "") or None,
                                error=str(tool_response.get("error") or "") or None,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        tool_name = raw_call.full_name if raw_call.server.strip() else raw_call.tool
                        round_executions.append(
                            McpBenchExecutionResult(
                                tool=tool_name,
                                server=raw_call.server.strip() or "unknown",
                                parameters=dict(raw_call.arguments),
                                round_num=round_num,
                                planned_layer=planned_layer,
                                success=False,
                                error=str(exc),
                            )
                        )
            for execution in round_executions:
                prompt_messages.append({"role": "user", "content": render_function_output_user_block(execution)})
            steps.append(
                {
                    "round_num": round_num,
                    "decision": {
                        "should_continue": decision.should_continue,
                        "final_answer": decision.final_answer,
                        "tool_calls": [
                            {"server": call.server, "tool": call.tool, "arguments": dict(call.arguments)}
                            for call in decision.tool_calls
                        ],
                    },
                    "executions": [_execution_to_dict(entry) for entry in round_executions],
                    "tool_router": tool_router_trace,
                    "long_context": long_context_trace,
                }
            )
            if round_executions:
                accumulated_information = append_round_summary(
                    accumulated_information,
                    round_num,
                    round_executions,
                )
                execution_results.extend(round_executions)
                executed_rounds = round_num
            if not decision.should_continue or not round_executions:
                break

        if not fail_reason and not final_answer:
            final_prompt = build_final_answer_prompt(item, accumulated_information)
            final_output = chat_completion(
                base_url=config.base_url,
                model=config.model,
                prompt=final_prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.final_max_tokens,
                timeout_s=config.timeout_s,
            )
            prompts.append(final_prompt)
            completions.append(final_output)
            final_answer = final_output.strip()
        if not fail_reason and not final_answer:
            fail_reason = "mcp_bench produced no final answer"
        if not fail_reason:
            planning_json_compliance = valid_planned_tools / total_planned_tools if total_planned_tools > 0 else 1.0
            evaluation = worker.evaluate(
                {
                    "judge_config": {
                        "api_key": judge.api_key,
                        "base_url": judge.base_url,
                        "model": judge.model,
                    },
                    "task": presented_task(item),
                    "final_solution": final_answer,
                    "total_rounds": executed_rounds,
                    "available_tools": available_tools,
                    "planning_json_compliance": planning_json_compliance,
                    "accumulated_information": accumulated_information,
                    "concrete_task_description": item.task.task_description if item.task.fuzzy_description.strip() else "",
                    "dependency_analysis": item.task.dependency_analysis,
                    "execution_results": [_execution_to_dict(entry) for entry in execution_results],
                }
            )
            is_passed = collapse_mcp_bench_pass(evaluation)
            if not is_passed:
                fail_reason = summarize_mcp_bench_evaluation(evaluation)
    finally:
        worker.close_task()

    return McpBenchResult(
        sample_index=sample_index,
        task_id=item.task.task_id,
        prompt=_join_round_artifacts(prompts),
        completion=json.dumps({"completions": list(completions), "trace": steps}, ensure_ascii=False),
        final_answer=final_answer,
        reference_answer=build_mcp_bench_ref_answer(item),
        is_passed=is_passed,
        fail_reason="" if is_passed else fail_reason,
        evaluation=evaluation,
        trace=tuple(steps),
    )


def evaluate_samples(samples: Sequence[McpBenchItem], config: McpBenchRunConfig) -> list[McpBenchResult]:
    runtime_root = resolve_runtime_root(config, samples)
    worker_script = resolve_worker_script(config)
    judge = resolve_judge_config(config)
    worker = McpBenchWorkerClient(runtime_root=runtime_root, worker_script=worker_script)
    try:
        return [
            generate_sample(sample_index, sample, config, worker=worker, judge=judge)
            for sample_index, sample in enumerate(samples)
        ]
    finally:
        worker.close()


def scoreboard_dataset_name(config: McpBenchRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    dataset += dataset_sample_suffix(sample_size=config.sample_size, sample_seed=config.sample_seed)
    return dataset


def job_id(config: McpBenchRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: McpBenchRunConfig) -> dict[str, Any]:
    return {
        "decision": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.decision_max_tokens,
        },
        "final": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.final_max_tokens,
        },
        "router": mcp_router_config_payload(config),
    }


def task_sampling_config(config: McpBenchRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "max_rounds": config.max_rounds,
        "prompt_profile": "helicopter_mcp_bench_json_tool_call",
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[McpBenchResult], *, config: McpBenchRunConfig, repo_root: Path) -> int:
    evaluations = [result.evaluation for result in results if result.evaluation is not None]
    task_id = asyncio.run(
        write_scoreboard_results(
            [result.to_scoreboard() for result in results],
            config=ScoreboardWriteConfig(
                dataset=scoreboard_dataset_name(config),
                model=config.model,
                job_name=config.job_name,
                job_id=job_id(config),
                benchmark=config.benchmark,
                runner=config.runner,
                cot_mode=config.cot_mode,
                sampling_config=task_sampling_config(config),
                completion_sampling_config=completion_sampling_config(config),
                extra_metrics={
                    "avg_task_completion_score": _avg(
                        evaluation.task_completion_score for evaluation in evaluations
                    ),
                    "avg_tool_selection_score": _avg(evaluation.tool_selection_score for evaluation in evaluations),
                    "avg_planning_efficiency_score": _avg(
                        evaluation.planning_effectiveness_and_efficiency_score for evaluation in evaluations
                    ),
                    "evaluated_samples": len(evaluations),
                },
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_mcp_bench(config: McpBenchRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    runtime_root = resolve_runtime_root(config, samples)
    worker_script = resolve_worker_script(config)
    resolve_judge_config(config)
    if not config.skip_runtime_preflight:
        preflight_mcp_bench_runtime(
            samples,
            runtime_root=runtime_root,
            worker_script=worker_script,
            open_first_task=True,
            raise_on_error=True,
        )
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "source_dataset": config.dataset_name,
        "runtime_root": str(runtime_root),
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: McpBenchRunConfig) -> dict[str, Any]:
    samples = load_samples(config)
    runtime_root = resolve_runtime_root(config, samples)
    worker_script = resolve_worker_script(config)
    preflight = preflight_mcp_bench_runtime(
        samples,
        runtime_root=runtime_root,
        worker_script=worker_script,
        open_first_task=False,
        raise_on_error=False,
    )
    source = _resolve_items_path(config)
    return {
        "benchmark": config.benchmark,
        "source": str(source) if source is not None else "official_mcp_bench_source",
        "split": config.split,
        "source_dataset": config.dataset_name,
        "limit": config.limit,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "available_samples": len(samples),
        "base_url": config.base_url,
        "model": config.model,
        "runtime_root": str(runtime_root),
        "worker_script": str(worker_script),
        "runtime_ready": preflight.ok,
        "runtime_errors": list(preflight.errors),
        "checked_servers": list(preflight.checked_servers),
        "judge_required": True,
        "router": mcp_router_config_payload(config),
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


def resolve_runtime_root(config: McpBenchRunConfig, samples: Sequence[McpBenchItem]) -> Path:
    for raw in (
        config.runtime_root,
        os.getenv("HELICOPTER_MCP_BENCH_RUNTIME_ROOT"),
        os.getenv("RWKV_MCP_BENCH_RUNTIME_ROOT"),
        os.getenv("MCP_BENCH_RUNTIME_ROOT"),
    ):
        if raw:
            return Path(raw).expanduser().resolve()
    for item in samples:
        if item.runtime_root:
            path = Path(item.runtime_root).expanduser()
            if path.exists():
                return path.resolve()
    source = _resolve_official_source_root(config)
    if source is not None:
        runtime = source / "runtime"
        return runtime.resolve() if runtime.exists() else source.resolve()
    return Path("/home/chase/GitHub/rwkv-rs/examples/rwkv-lm-eval/datasets/mcp_bench/runtime").resolve()


def resolve_worker_script(config: McpBenchRunConfig) -> Path:
    for raw in (
        config.worker_script,
        os.getenv("HELICOPTER_MCP_BENCH_WORKER_SCRIPT"),
        os.getenv("RWKV_MCP_BENCH_WORKER_SCRIPT"),
        os.getenv("MCP_BENCH_WORKER_SCRIPT"),
    ):
        if raw:
            return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parent / "mcp_bench_worker.py").resolve()


def resolve_judge_config(config: McpBenchRunConfig) -> JudgeConfig:
    model = config.judge_model or os.getenv("HELICOPTER_JUDGE_MODEL") or os.getenv("JUDGE_MODEL")
    api_key = config.judge_api_key or os.getenv("HELICOPTER_JUDGE_API_KEY") or os.getenv("JUDGE_API_KEY")
    base_url = config.judge_base_url or os.getenv("HELICOPTER_JUDGE_BASE_URL") or os.getenv("JUDGE_BASE_URL") or ""
    if not model or not api_key:
        raise ValueError("MCP-Bench requires HELICOPTER_JUDGE_MODEL/JUDGE_MODEL and HELICOPTER_JUDGE_API_KEY/JUDGE_API_KEY")
    return JudgeConfig(model=str(model), api_key=str(api_key), base_url=str(base_url))


def _load_items(config: McpBenchRunConfig) -> list[McpBenchItem]:
    source_path = _resolve_items_path(config)
    if source_path is None:
        raise FileNotFoundError(
            "MCP-Bench data not found. Set HELICOPTER_MCP_BENCH_DATA_ROOT, "
            "HELICOPTER_MCP_BENCH_SOURCE_ROOT, or source_path."
        )
    if _path_looks_like_manifest(source_path):
        return load_mcp_bench_manifest_records(source_path)
    if source_path.name in MCP_BENCH_TASK_FILES:
        runtime = resolve_runtime_root(config, ())
        return load_mcp_bench_task_items(source_path.parent, runtime, file_names=(source_path.name,))
    source_root = source_path
    tasks_root, runtime_root = _resolve_source_roots(source_root)
    return load_mcp_bench_task_items(
        tasks_root,
        runtime_root,
        file_names=MCP_BENCH_DATASET_FILES[config.dataset_name],
    )


def _resolve_items_path(config: McpBenchRunConfig) -> Path | None:
    if config.source_path:
        candidate = Path(config.source_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"MCP-Bench source_path does not exist: {candidate}")
    for candidate in _manifest_candidates(config):
        if candidate.is_file():
            return candidate.resolve()
    source = _resolve_official_source_root(config)
    return source.resolve() if source is not None else None


def _manifest_candidates(config: McpBenchRunConfig) -> list[Path]:
    dataset_env = config.dataset_name.upper()
    candidates: list[Path] = []
    for key in (
        f"HELICOPTER_{dataset_env}_MANIFEST",
        f"RWKV_{dataset_env}_MANIFEST",
        f"{dataset_env}_MANIFEST",
        "HELICOPTER_MCP_BENCH_MANIFEST",
        "RWKV_MCP_BENCH_MANIFEST",
        "MCP_BENCH_MANIFEST",
    ):
        raw = os.getenv(key)
        if raw:
            candidates.append(Path(raw).expanduser())
    for raw in (
        config.source_root,
        os.getenv("HELICOPTER_MCP_BENCH_DATA_ROOT"),
        os.getenv("RWKV_MCP_BENCH_DATA_ROOT"),
        os.getenv("MCP_BENCH_DATA_ROOT"),
    ):
        if raw:
            candidates.append(Path(raw).expanduser() / config.dataset_name / "test.jsonl")
    candidates.extend(
        [
            _repo_root() / "data" / config.dataset_name / "test.jsonl",
            Path("/home/chase/GitHub/rwkv-skills/data") / config.dataset_name / "test.jsonl",
            Path("/home/chase/rwkv-skills/data") / config.dataset_name / "test.jsonl",
            Path("/tmp/rwkv-skills/data") / config.dataset_name / "test.jsonl",
        ]
    )
    return candidates


def _resolve_official_source_root(config: McpBenchRunConfig) -> Path | None:
    for raw in (
        config.source_root,
        os.getenv("HELICOPTER_MCP_BENCH_SOURCE_ROOT"),
        os.getenv("RWKV_MCP_BENCH_SOURCE_ROOT"),
        os.getenv("MCP_BENCH_SOURCE_ROOT"),
    ):
        if raw:
            candidate = Path(raw).expanduser()
            if _mcp_bench_required_paths(config.dataset_name, candidate):
                return candidate
    candidates = [
        Path("/home/chase/GitHub/rwkv-rs/examples/rwkv-lm-eval/datasets/mcp_bench"),
        Path("/home/chase/GitHub/rwkv-skills/references/mcp-bench"),
        Path("/home/chase/GitHub/mcp-bench"),
        Path("/tmp/rwkv-official-refs/mcp-bench"),
        Path("/tmp/ref-mcp-bench"),
    ]
    for candidate in candidates:
        if _mcp_bench_required_paths(config.dataset_name, candidate):
            return candidate
    return None


def _mcp_bench_required_paths(dataset_name: str, source_root: Path) -> bool:
    if dataset_name not in MCP_BENCH_DATASET_FILES:
        return False
    tasks_root, _runtime_root = _resolve_source_roots(source_root)
    file_names = MCP_BENCH_DATASET_FILES[dataset_name] or MCP_BENCH_TASK_FILES
    return all((tasks_root / file_name).is_file() for file_name in file_names)


def _resolve_source_roots(source_root: Path) -> tuple[Path, Path]:
    root = source_root.expanduser().resolve()
    legacy_tasks_root = root / "tasks"
    runtime_tasks_root = root / "runtime" / "tasks"
    tasks_root = legacy_tasks_root if legacy_tasks_root.exists() else runtime_tasks_root
    runtime_root = root / "runtime"
    return tasks_root, runtime_root if runtime_root.exists() else root


def _path_looks_like_manifest(path: Path) -> bool:
    if path.suffix != ".jsonl":
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            return isinstance(payload, Mapping) and "task" in payload and "instruction" in payload
    return False


def _resolve_mcp_server_cwd(runtime_root: Path, raw_cwd: str) -> Path:
    if not raw_cwd:
        return runtime_root
    if raw_cwd.startswith("../"):
        return (runtime_root / "mcp_servers" / raw_cwd[3:]).resolve()
    return (runtime_root / raw_cwd).resolve()


def append_round_summary(
    accumulated_information: str,
    round_num: int,
    executions: Sequence[McpBenchExecutionResult],
) -> str:
    lines = [accumulated_information, "", f"--- Summary of Round {round_num} ---"]
    for execution in executions:
        params = json.dumps(execution.parameters, ensure_ascii=False) if execution.parameters else "{}"
        if execution.success:
            rendered = _truncate_text(execution.result or "", MCP_BENCH_MAX_RESULT_CHARS)
            lines.append(
                f"Tool `{execution.tool}` with Parameter {params} on {execution.server} succeeded. Result: {rendered}"
            )
        else:
            rendered = _truncate_text(execution.error or "", MCP_BENCH_MAX_ERROR_CHARS)
            lines.append(
                f"Tool `{execution.tool}` with Parameter {params} on {execution.server} failed. Error: {rendered}"
            )
    return _trim_text("\n".join(part for part in lines if part is not None), MCP_BENCH_MAX_HISTORY_CHARS)


def render_function_output_user_block(execution: McpBenchExecutionResult) -> str:
    return (
        "Function output:\n"
        f"{json.dumps(_execution_to_dict(execution), ensure_ascii=False, sort_keys=True)}"
    )


def render_tool_catalog(available_tools: Mapping[str, Mapping[str, Any]], *, description_chars: int = 400) -> str:
    rendered: list[dict[str, Any]] = []
    for tool in available_tools.values():
        server = str(tool.get("server") or "")
        name = str(tool.get("name") or "").strip()
        schema = tool.get("input_schema")
        arguments: Any = {}
        if isinstance(schema, Mapping):
            properties = schema.get("properties")
            arguments = dict(properties) if isinstance(properties, Mapping) else dict(schema)
        rendered.append(
            {
                "name": f"{server}:{name}" if server else name,
                "description": _truncate_text(
                    str(tool.get("description") or "").strip() or "No description available",
                    max(40, int(description_chars)),
                ),
                "arguments": arguments,
            }
        )
    if not any(str(item.get("name") or "") == "final_answer" for item in rendered):
        rendered.append(
            {
                "name": "final_answer",
                "description": "Use this when the task is complete.",
                "arguments": {"answer": {"type": "string"}},
            }
        )
    return json.dumps(sorted(rendered, key=lambda item: str(item.get("name") or "")), ensure_ascii=False, indent=2)


def collapse_mcp_bench_pass(evaluation: McpBenchEvaluation) -> bool:
    return (
        evaluation.task_completion_score >= MCP_BENCH_PASS_THRESHOLD
        and evaluation.tool_selection_score >= MCP_BENCH_PASS_THRESHOLD
        and evaluation.planning_effectiveness_and_efficiency_score >= MCP_BENCH_PASS_THRESHOLD
    )


def summarize_mcp_bench_evaluation(evaluation: McpBenchEvaluation) -> str:
    return (
        f"task_completion_score={evaluation.task_completion_score:.2f}, "
        f"tool_selection_score={evaluation.tool_selection_score:.2f}, "
        f"planning_effectiveness_and_efficiency_score={evaluation.planning_effectiveness_and_efficiency_score:.2f}, "
        f"valid_tool_name_rate={_fmt_optional(evaluation.valid_tool_name_rate)}, "
        f"execution_success_rate={_fmt_optional(evaluation.execution_success_rate)}, "
        f"planning_json_compliance={_fmt_optional(evaluation.planning_json_compliance)}"
    )


def build_mcp_bench_ref_answer(item: McpBenchItem) -> str:
    return (
        f"task_id={item.task.task_id}\n"
        f"task_file={item.task_file}\n"
        f"server_name={item.server_name}\n"
        f"servers={', '.join(item.servers)}\n"
        f"combination_type={item.combination_type}\n"
        "runtime_origin=official_mcp_bench_runtime\n"
        "evaluator=official_mcp_bench_evaluator_phase2"
    )


def mcp_router_config_payload(config: McpBenchRunConfig) -> dict[str, Any]:
    return {
        "tool_router": {
            "mode": str(config.tool_router_mode or "off"),
            "max_tools": int(config.tool_router_max_tools),
            "trigger_tool_count": int(config.tool_router_trigger_tool_count),
            "trigger_catalog_chars": int(config.tool_router_trigger_catalog_chars),
            "context_chars": int(config.tool_router_context_chars),
            "description_chars": int(config.tool_router_description_chars),
        },
        "long_context": {
            "mode": str(config.long_context_router_mode or "off"),
            "min_chars": int(config.long_context_min_chars),
            "chunk_chars": int(config.long_context_chunk_chars),
            "overlap_lines": int(config.long_context_overlap_lines),
            "max_evidence_chunks": int(config.long_context_max_evidence_chunks),
            "max_evidence_chars": int(config.long_context_max_evidence_chars),
        },
    }


def _mcp_router_query(item: McpBenchItem, messages: Sequence[Mapping[str, Any]], *, max_chars: int) -> str:
    context = _trim_text(_format_messages(messages), max(0, int(max_chars)))
    return f"{presented_task(item)}\n{item.task.dependency_analysis}\n{context}"


def _mcp_tool_full_name(tool: Mapping[str, Any]) -> str:
    server = str(tool.get("server") or "").strip()
    name = str(tool.get("name") or "").strip()
    return f"{server}:{name}" if server else name


def _tool_score(tool: Mapping[str, Any], query_terms: set[str]) -> float:
    full_name = _mcp_tool_full_name(tool)
    schema = tool.get("input_schema") if isinstance(tool.get("input_schema"), Mapping) else {}
    text = "\n".join(
        [
            full_name,
            str(tool.get("description") or ""),
            json.dumps(schema, ensure_ascii=False, sort_keys=True),
        ]
    )
    terms = _lexical_terms(text)
    if not query_terms:
        return 0.0
    overlap = query_terms & terms
    name_terms = _lexical_terms(full_name)
    return float(len(overlap)) + (2.0 * len(query_terms & name_terms))


def _lexical_terms(text: str) -> set[str]:
    terms = {
        token
        for token in re.findall(r"[A-Za-z0-9_]{2,}", str(text or "").lower())
        if token not in _LEXICAL_STOPWORDS
    }
    return terms


_LEXICAL_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "when",
    "then",
    "task",
    "tool",
    "tools",
    "user",
    "assistant",
    "function",
    "output",
    "arguments",
    "name",
    "type",
    "string",
    "object",
    "true",
    "false",
    "none",
    "null",
}


def _select_long_context_evidence(
    text: str,
    *,
    query_terms: set[str],
    chunk_chars: int,
    overlap_lines: int,
    max_chunks: int,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    chunks = _split_text_chunks(text, chunk_chars=chunk_chars, overlap_lines=overlap_lines)
    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        score = float(len(_lexical_terms(chunk) & query_terms))
        scored.append((score, index, chunk))
    scored.sort(key=lambda row: (-row[0], row[1]))
    selected_rows = sorted(scored[: max(1, int(max_chunks))], key=lambda row: row[1])
    parts: list[str] = []
    metadata: list[dict[str, Any]] = []
    used = 0
    for score, index, chunk in selected_rows:
        header = f"[evidence chunk {index} score={score:.1f}]"
        body_budget = max(0, int(max_chars) - used - len(header) - 2)
        if body_budget <= 0:
            break
        body = _truncate_text(chunk, body_budget)
        parts.append(f"{header}\n{body}")
        used += len(parts[-1]) + 2
        metadata.append({"index": index, "score": score, "chars": len(chunk)})
    return "\n\n".join(parts), metadata


def _split_text_chunks(text: str, *, chunk_chars: int, overlap_lines: int) -> list[str]:
    lines = str(text or "").splitlines()
    if not lines:
        return [str(text or "")]
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for line in lines:
        line_chars = len(line) + 1
        if current and current_chars + line_chars > max(1, int(chunk_chars)):
            chunks.append("\n".join(current))
            current = current[-max(0, int(overlap_lines)) :] if overlap_lines else []
            current_chars = sum(len(item) + 1 for item in current)
        current.append(line)
        current_chars += line_chars
    if current:
        chunks.append("\n".join(current))
    return chunks or [str(text or "")]


def _execution_to_dict(entry: McpBenchExecutionResult) -> dict[str, Any]:
    return {
        "tool": entry.tool,
        "server": entry.server,
        "parameters": dict(entry.parameters),
        "round_num": int(entry.round_num),
        "planned_layer": entry.planned_layer,
        "success": bool(entry.success),
        "result": entry.result,
        "error": entry.error,
    }


def _evaluation_to_dict(evaluation: McpBenchEvaluation | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "task_completion_score": evaluation.task_completion_score,
        "tool_selection_score": evaluation.tool_selection_score,
        "planning_effectiveness_and_efficiency_score": evaluation.planning_effectiveness_and_efficiency_score,
        "task_fulfillment": evaluation.task_fulfillment,
        "grounding": evaluation.grounding,
        "tool_appropriateness": evaluation.tool_appropriateness,
        "parameter_accuracy": evaluation.parameter_accuracy,
        "dependency_awareness": evaluation.dependency_awareness,
        "parallelism_and_efficiency": evaluation.parallelism_and_efficiency,
        "input_schema_compliance": evaluation.input_schema_compliance,
        "valid_tool_name_rate": evaluation.valid_tool_name_rate,
        "execution_success_rate": evaluation.execution_success_rate,
        "planning_json_compliance": evaluation.planning_json_compliance,
    }


def _format_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role") or "message")
        rendered.append(f"{role}: {message.get('content') or ''}")
    return "\n".join(rendered)


def _join_round_artifacts(items: Sequence[str]) -> str:
    return "\n\n".join(f"--- round {index + 1} ---\n{item}" for index, item in enumerate(items))


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)] + "\n...[truncated]..."


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = [
    "MCP_BENCH_PASS_THRESHOLD",
    "McpBenchEvaluation",
    "McpBenchItem",
    "McpBenchRunConfig",
    "McpBenchTaskSpec",
    "build_planning_prompt",
    "dry_run_summary",
    "evaluate_samples",
    "load_mcp_bench_manifest_records",
    "load_mcp_bench_task_items",
    "load_samples",
    "parse_planning_decision",
    "preflight_mcp_bench_runtime",
    "run_mcp_bench",
]
