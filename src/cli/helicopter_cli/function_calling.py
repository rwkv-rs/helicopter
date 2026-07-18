from __future__ import annotations

import concurrent.futures
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from .commands import CommandPlan, build_infer_plan, is_local_base_url, local_openai_base_url
from .config import resolve_model_entry, table
from .env import env_value, pick
from .eval_run import (
    DEFAULT_SERVER_TIMEOUT_S,
    format_plan_for_display,
    ingest_scoreboard_results,
    port_from_base_url,
    server_is_healthy,
    stop_server,
    wait_for_server,
)
from .lighteval_rwkv_skills_tasks import (
    APIBANK_LEVEL1_PATH,
    APIBANK_LEVEL2_PATH,
    BFCL_URL_BASE,
    BFCL_V3_SINGLE_TURN_FILES,
    COMPLEXFUNCBENCH_URL,
    TOOLALPACA_REAL_PATH,
    TOOLALPACA_SIMULATED_PATH,
    _apibank_arguments_match,
    _apibank_expected_call,
    _apibank_expected_result,
    _bfcl_calls_match,
    _bfcl_ground_truth,
    _bfcl_messages,
    _bfcl_specific_expected_calls,
    _complexfuncbench_calls_match,
    _complexfuncbench_conversations,
    _complexfuncbench_expected_turn,
    _complexfuncbench_tools,
    _complexfuncbench_turn_text,
    _normalize_bfcl_answer,
    _parse_apibank_turn_index,
    _toolalpaca_expected_calls,
    _toolalpaca_tools,
    _toolalpaca_execution_matches,
    ApiBankSandbox,
    ToolAlpacaSandbox,
)


FC_TASKS = (
    "bfcl_simple_python",
    "bfcl_multiple",
    "bfcl_v3",
    "bfcl_exec_simple",
    "bfcl_exec_multiple",
    "bfcl_exec_parallel",
    "bfcl_exec_parallel_multiple",
    "apibank_level1",
    "apibank_level2",
    "complexfuncbench_official",
    "toolalpaca_eval_simulated",
    "toolalpaca_eval_real",
)

BFCL_TASK_FILES: dict[str, str | list[str]] = {
    "bfcl_simple_python": "BFCL_v3_simple.json",
    "bfcl_multiple": "BFCL_v3_multiple.json",
    "bfcl_v3": list(BFCL_V3_SINGLE_TURN_FILES),
    "bfcl_exec_simple": "BFCL_v3_exec_simple.json",
    "bfcl_exec_multiple": "BFCL_v3_exec_multiple.json",
    "bfcl_exec_parallel": "BFCL_v3_exec_parallel.json",
    "bfcl_exec_parallel_multiple": "BFCL_v3_exec_parallel_multiple.json",
}


@dataclass(frozen=True)
class FunctionCallingSample:
    task_name: str
    sample_id: str
    kind: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    specific: dict[str, Any]
    parallel_tool_calls: bool = True


@dataclass(frozen=True)
class FunctionCallingRunResult:
    task_name: str
    sample_id: str
    score: float
    actual_calls: list[dict[str, Any]]
    raw_response: dict[str, Any] | None
    error: str | None = None
    elapsed_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def _read_text_source(source: str) -> str:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=120) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def _load_json_records(source: str) -> list[dict[str, Any]]:
    text = _read_text_source(source).strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _local_task_sources(task_name: str, *, root: Path | None) -> list[str]:
    if root is None:
        return []
    candidates = [
        root / "data" / task_name / "test.jsonl",
        root.parent / "rwkv-skills" / "data" / task_name / "test.jsonl",
    ]
    aliases = {
        "apibank_level1": ("apibank_l1",),
        "apibank_level2": ("apibank_l2",),
        "complexfuncbench_official": ("complexfuncbench_subset",),
    }
    for alias in aliases.get(task_name, ()):
        candidates.extend(
            [
                root / "data" / alias / "test.jsonl",
                root.parent / "rwkv-skills" / "data" / alias / "test.jsonl",
            ]
        )
    seen: set[str] = set()
    sources: list[str] = []
    for path in candidates:
        resolved = str(path)
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        sources.append(resolved)
    return sources


def _task_sources(task_name: str, *, root: Path | None = None) -> list[str]:
    local_sources = _local_task_sources(task_name, root=root)
    if local_sources:
        return local_sources
    if task_name in BFCL_TASK_FILES:
        files = BFCL_TASK_FILES[task_name]
        values = [files] if isinstance(files, str) else files
        return [f"{BFCL_URL_BASE}/{item}" for item in values]
    if task_name == "apibank_level1":
        return [APIBANK_LEVEL1_PATH]
    if task_name == "apibank_level2":
        return [APIBANK_LEVEL2_PATH]
    if task_name == "complexfuncbench_official":
        return [COMPLEXFUNCBENCH_URL]
    if task_name == "toolalpaca_eval_simulated":
        return [TOOLALPACA_SIMULATED_PATH]
    if task_name == "toolalpaca_eval_real":
        return [TOOLALPACA_REAL_PATH]
    raise SystemExit(f"unknown function-calling task: {task_name}")


def _plain_messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _openai_tools(tools: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return output
    for item in tools:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") if isinstance(item.get("function"), Mapping) else item
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}, "required": []}
        output.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "parameters": dict(parameters),
                },
            }
        )
    return output


def _build_bfcl_sample(line: dict[str, Any], task_name: str) -> FunctionCallingSample | None:
    functions = line.get("function")
    expected_calls = _normalize_bfcl_answer(_bfcl_ground_truth(line, task_name))
    if not functions or not expected_calls:
        return None
    messages = [
        {
            "role": "system",
            "content": (
                "You are solving a function-calling benchmark. Use the provided tools for the next "
                "assistant action. Return tool calls through the tool calling interface."
            ),
        },
        *_bfcl_messages(line.get("question")),
    ]
    if len(messages) == 1:
        return None
    return FunctionCallingSample(
        task_name=task_name,
        sample_id=str(line.get("id") or ""),
        kind="bfcl",
        messages=messages,
        tools=_openai_tools(functions),
        specific={
            "expected_calls_json": json.dumps(expected_calls, ensure_ascii=False, sort_keys=True),
            "sample_id": str(line.get("id") or ""),
            "execution_result_type": line.get("execution_result_type"),
        },
    )


def _build_apibank_sample(line: dict[str, Any], task_name: str) -> FunctionCallingSample | None:
    instruction = str(line.get("instruction") or "").strip()
    tools_json = str(line.get("tools_json") or "").strip()
    expected_call_json = str(line.get("expected_call_json") or "").strip()
    if not instruction or not tools_json or not expected_call_json:
        return None
    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError:
        return None
    turn_index = line.get("turn_index")
    if turn_index is None:
        turn_index = _parse_apibank_turn_index(line.get("task_id"))
    return FunctionCallingSample(
        task_name=task_name,
        sample_id=str(line.get("task_id") or ""),
        kind="apibank",
        messages=_plain_messages(
            (
                "You are solving API-Bank. Choose exactly one API call for the next assistant "
                "action. For dates with month/day or relative dates and no explicit year, use "
                "year 2023. Return the call through the tool calling interface."
            ),
            instruction,
        ),
        tools=_openai_tools(tools),
        specific={
            "expected_call_json": expected_call_json,
            "expected_result_json": str(line.get("expected_result_json") or "null").strip(),
            "sample_id": str(line.get("task_id") or ""),
            "source_path": str(line.get("source_path") or ""),
            "turn_index": "" if turn_index is None else str(turn_index),
        },
        parallel_tool_calls=False,
    )


def _build_complexfuncbench_sample(line: dict[str, Any], task_name: str) -> FunctionCallingSample | None:
    tools = _complexfuncbench_tools(line)
    expected_turn = _complexfuncbench_expected_turn(line)
    conversations = _complexfuncbench_conversations(line)
    if not tools or expected_turn is None or not conversations:
        return None
    target_turn_index, expected_calls = expected_turn
    if not expected_calls:
        return None
    visible_turns = conversations[:target_turn_index]
    transcript = "\n".join(_complexfuncbench_turn_text(turn) for turn in visible_turns)
    sample_id = str(line.get("task_id") or line.get("id") or f"complexfuncbench_{target_turn_index}")
    return FunctionCallingSample(
        task_name=task_name,
        sample_id=f"{sample_id}__turn_{target_turn_index}",
        kind="complexfuncbench",
        messages=_plain_messages(
            (
                "You are solving ComplexFuncBench. Choose the next assistant tool call or calls "
                "for the current state. Return calls through the tool calling interface."
            ),
            f"Conversation so far:\n{transcript}",
        ),
        tools=_openai_tools(tools),
        specific={
            "expected_calls_json": json.dumps(expected_calls, ensure_ascii=False, sort_keys=True),
            "sample_id": f"{sample_id}__turn_{target_turn_index}",
            "target_turn_index": str(target_turn_index),
        },
    )


def _build_toolalpaca_sample(line: dict[str, Any], task_name: str) -> FunctionCallingSample | None:
    instruction = str(line.get("instruction") or "").strip()
    tools = line.get("tools")
    expected_calls = line.get("expected_tool_calls")
    if not instruction or not isinstance(tools, list) or not isinstance(expected_calls, list):
        return None
    return FunctionCallingSample(
        task_name=task_name,
        sample_id=str(line.get("task_id") or ""),
        kind="toolalpaca",
        messages=_plain_messages(
            (
                "You are solving ToolAlpaca. Choose the tool call sequence needed to satisfy the "
                "user request. Return calls through the tool calling interface."
            ),
            instruction,
        ),
        tools=_openai_tools(tools),
        specific={
            "expected_tool_calls_json": json.dumps(expected_calls, ensure_ascii=False, sort_keys=True),
            "sample_id": str(line.get("task_id") or ""),
            "tools_json": json.dumps(tools, ensure_ascii=False, sort_keys=True),
        },
    )


def _build_intermediate_sample(line: dict[str, Any], task_name: str, kind: str) -> FunctionCallingSample | None:
    instruction = str(line.get("instruction") or "").strip()
    tools = line.get("tools")
    expected_calls = line.get("expected_tool_calls")
    if not instruction or not isinstance(tools, list) or not isinstance(expected_calls, list) or not expected_calls:
        return None
    sample_kind = "toolalpaca" if kind == "toolalpaca" else kind
    if sample_kind == "apibank":
        sample_kind = "bfcl"
    return FunctionCallingSample(
        task_name=task_name,
        sample_id=str(line.get("task_id") or line.get("id") or ""),
        kind=sample_kind,
        messages=_plain_messages(
            (
                "You are solving a function-calling benchmark. Choose the tool call or calls "
                "needed to satisfy the user request. Return calls through the tool calling interface."
            ),
            instruction,
        ),
        tools=_openai_tools(tools),
        specific={
            "expected_calls_json": json.dumps(expected_calls, ensure_ascii=False, sort_keys=True),
            "expected_tool_calls_json": json.dumps(expected_calls, ensure_ascii=False, sort_keys=True),
            "sample_id": str(line.get("task_id") or line.get("id") or ""),
            "tools_json": json.dumps(tools, ensure_ascii=False, sort_keys=True),
        },
    )


BUILDERS: dict[str, Callable[[dict[str, Any], str], FunctionCallingSample | None]] = {
    "bfcl": _build_bfcl_sample,
    "apibank": _build_apibank_sample,
    "complexfuncbench": _build_complexfuncbench_sample,
    "toolalpaca": _build_toolalpaca_sample,
}


def task_kind(task_name: str) -> str:
    if task_name.startswith("bfcl_") or task_name == "bfcl_v3":
        return "bfcl"
    if task_name.startswith("apibank_"):
        return "apibank"
    if task_name.startswith("complexfuncbench_"):
        return "complexfuncbench"
    if task_name.startswith("toolalpaca_"):
        return "toolalpaca"
    raise SystemExit(f"unknown function-calling task: {task_name}")


def load_samples(task_name: str, *, max_samples: int | None = None, root: Path | None = None) -> list[FunctionCallingSample]:
    kind = task_kind(task_name)
    builder = BUILDERS[kind]
    samples: list[FunctionCallingSample] = []
    for source in _task_sources(task_name, root=root):
        for line in _load_json_records(source):
            sample = _build_intermediate_sample(line, task_name, kind) or builder(line, task_name)
            if sample is not None:
                samples.append(sample)
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
    return samples


def parse_task_names(value: str | None) -> list[str]:
    if value is None or not value.strip() or value.strip().lower() == "all":
        return list(FC_TASKS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in FC_TASKS]
    if unknown:
        raise SystemExit(f"unknown function-calling task(s): {', '.join(unknown)}")
    return names


def normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") if isinstance(item.get("function"), Mapping) else item
        name = str(function.get("name") or item.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments", item.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {}
            arguments = parsed
        if not isinstance(arguments, Mapping):
            arguments = {}
        calls.append({"name": name, "arguments": dict(arguments)})
    return calls


def tool_calls_from_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first = choices[0]
    if not isinstance(first, Mapping):
        return []
    message = first.get("message")
    if not isinstance(message, Mapping):
        return []
    return normalize_tool_calls(message.get("tool_calls"))


def score_calls(sample: FunctionCallingSample, calls: list[dict[str, Any]]) -> float:
    if sample.kind == "bfcl":
        expected_calls = _bfcl_specific_expected_calls(sample.specific)
        return 1.0 if expected_calls and _bfcl_calls_match(calls, expected_calls) else 0.0
    if sample.kind == "apibank":
        return _score_apibank_calls(sample.specific, calls)
    if sample.kind == "complexfuncbench":
        expected_calls = _bfcl_specific_expected_calls(sample.specific)
        return 1.0 if expected_calls and _complexfuncbench_calls_match(calls, expected_calls) else 0.0
    if sample.kind == "toolalpaca":
        return _score_toolalpaca_calls(sample.specific, calls)
    return 0.0


def _score_apibank_calls(specific: dict[str, Any], calls: list[dict[str, Any]]) -> float:
    expected = _apibank_expected_call(specific)
    if not expected or not calls:
        return 0.0
    expected_name = str(expected.get("name") or "").strip()
    actual = calls[0]
    actual_name = str(actual.get("name") or "").strip()
    actual_args = actual.get("arguments")
    if not expected_name or actual_name != expected_name or not isinstance(actual_args, Mapping):
        return 0.0
    sandbox = ApiBankSandbox()
    sandbox.replay_history(
        str(specific.get("source_path") or ""),
        _parse_apibank_turn_index(specific.get("turn_index")),
    )
    expected_args = expected.get("arguments")
    if not isinstance(expected_args, Mapping):
        expected_args = {}
    arguments_match = _apibank_arguments_match(sandbox, expected_name, actual_args, expected_args)
    call_result = sandbox.api_call(actual_name, dict(actual_args))
    if not call_result.success:
        return 1.0 if arguments_match else 0.0
    try:
        ok = sandbox.check_api_call_correctness(
            actual_name,
            call_result.result,
            _apibank_expected_result(specific, expected_name),
        )
    except Exception:
        ok = False
    return 1.0 if ok or arguments_match else 0.0


def _score_toolalpaca_calls(specific: Mapping[str, Any], calls: list[dict[str, Any]]) -> float:
    tools = _toolalpaca_tools(specific)
    expected_calls = _toolalpaca_expected_calls(specific)
    if not tools:
        return 0.0
    sandbox = ToolAlpacaSandbox(tools)
    expected_results = sandbox.execute_sequence(expected_calls)
    actual_results = sandbox.execute_sequence(calls)
    required_expected = [item for item in expected_results if not item.optional]
    denominator = max(1, len(required_expected))
    passed = 0
    actual_index = 0
    for expected in expected_results:
        if actual_index >= len(actual_results):
            continue
        actual = actual_results[actual_index]
        if _toolalpaca_execution_matches(actual, expected):
            if not expected.optional:
                passed += 1
            actual_index += 1
        elif expected.optional:
            continue
        else:
            actual_index += 1
    if not expected_results:
        return 1.0 if not actual_results else 0.0
    return float(passed / denominator)


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _post_json(url: str, payload: Mapping[str, Any], *, api_key: str | None, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _usage_int(response: Mapping[str, Any] | None, key: str) -> int | None:
    if not isinstance(response, Mapping):
        return None
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(key)
    if not isinstance(value, int):
        return None
    return value if value >= 0 else None


def evaluate_sample(
    sample: FunctionCallingSample,
    *,
    base_url: str,
    model_name: str,
    api_key: str | None,
    max_new_tokens: int,
    timeout_s: float,
) -> FunctionCallingRunResult:
    payload = {
        "model": model_name,
        "messages": sample.messages,
        "tools": sample.tools,
        "tool_choice": "auto",
        "parallel_tool_calls": sample.parallel_tool_calls,
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }
    started = time.monotonic()
    try:
        response = _post_json(_chat_completions_url(base_url), payload, api_key=api_key, timeout_s=timeout_s)
        elapsed_seconds = time.monotonic() - started
        calls = tool_calls_from_response(response)
        score = score_calls(sample, calls)
        prompt_tokens = _usage_int(response, "prompt_tokens")
        completion_tokens = _usage_int(response, "completion_tokens")
        total_tokens = _usage_int(response, "total_tokens")
        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        return FunctionCallingRunResult(
            sample.task_name,
            sample.sample_id,
            score,
            calls,
            response,
            elapsed_seconds=elapsed_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    except Exception as error:  # noqa: BLE001 - failures are recorded per sample.
        elapsed_seconds = time.monotonic() - started
        return FunctionCallingRunResult(
            sample.task_name,
            sample.sample_id,
            0.0,
            [],
            None,
            str(error),
            elapsed_seconds=elapsed_seconds,
        )


def run_samples(
    samples: list[FunctionCallingSample],
    *,
    base_url: str,
    model_name: str,
    api_key: str | None,
    max_new_tokens: int,
    timeout_s: float,
    concurrent_requests: int,
) -> list[FunctionCallingRunResult]:
    worker_count = max(1, int(concurrent_requests))
    if worker_count == 1:
        return [
            evaluate_sample(
                sample,
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                max_new_tokens=max_new_tokens,
                timeout_s=timeout_s,
            )
            for sample in samples
        ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                evaluate_sample,
                sample,
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                max_new_tokens=max_new_tokens,
                timeout_s=timeout_s,
            )
            for sample in samples
        ]
        return [future.result() for future in futures]


def model_name_for_fc(args: Any, config: dict[str, Any]) -> str:
    model = resolve_model_entry(config, args.model)
    return str(
        pick(
            getattr(args, "served_model_name", None),
            getattr(args, "lighteval_model_name", None),
            model.get("served_model_name"),
            model.get("requested_name"),
            args.model,
        )
    )


def infer_args_namespace(args: Any, *, port: str | None) -> Any:
    import argparse

    return argparse.Namespace(
        model=args.model,
        dry_run=getattr(args, "dry_run", False),
        wkv_mode=getattr(args, "wkv_mode", None),
        emb_device=getattr(args, "emb_device", None),
        host=None,
        port=port,
        served_model_name=model_name_for_fc(args, getattr(args, "_config", {})),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", None),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
        max_model_len=None,
        max_num_seqs=getattr(args, "max_num_seqs", None),
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", None),
        enable_auto_tool_choice=pick(getattr(args, "enable_auto_tool_choice", None), True),
        vllm_env=getattr(args, "vllm_env", None),
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


def _latency_summary(results: list[FunctionCallingRunResult]) -> dict[str, float | None]:
    latencies = [result.elapsed_seconds for result in results if result.elapsed_seconds is not None]
    return {
        "mean": (sum(latencies) / len(latencies)) if latencies else None,
        "p50": _percentile(latencies, 0.50),
        "p90": _percentile(latencies, 0.90),
        "p95": _percentile(latencies, 0.95),
        "p99": _percentile(latencies, 0.99),
    }


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _aggregate_performance(results: list[FunctionCallingRunResult], *, elapsed_seconds: float) -> dict[str, Any]:
    prompt_tokens = _sum_optional([result.prompt_tokens for result in results])
    completion_tokens = _sum_optional([result.completion_tokens for result in results])
    total_tokens = _sum_optional([result.total_tokens for result in results])
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return {
        "elapsed_seconds": elapsed_seconds,
        "requests": len(results),
        "successful_requests": sum(1 for result in results if result.error is None),
        "failed_requests": sum(1 for result in results if result.error is not None),
        "requests_per_second": (len(results) / elapsed_seconds) if elapsed_seconds > 0 else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": (total_tokens / elapsed_seconds) if total_tokens is not None and elapsed_seconds > 0 else None,
        "e2e_latency_seconds": _latency_summary(results),
    }


def _aggregate_results(results: list[FunctionCallingRunResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[FunctionCallingRunResult]] = {}
    for result in results:
        grouped.setdefault(result.task_name, []).append(result)
    aggregated: dict[str, dict[str, Any]] = {}
    for task_name, rows in grouped.items():
        count = len(rows)
        tool_call_missing = sum(1 for row in rows if not row.actual_calls)
        errors = sum(1 for row in rows if row.error)
        aggregated[task_name] = {
            "accuracy": sum(row.score for row in rows) / count if count else 0.0,
            "samples": count,
            "tool_call_missing_rate": tool_call_missing / count if count else 0.0,
            "error_rate": errors / count if count else 0.0,
            "e2e_latency_seconds": _latency_summary(rows),
            "prompt_tokens": _sum_optional([row.prompt_tokens for row in rows]),
            "completion_tokens": _sum_optional([row.completion_tokens for row in rows]),
            "total_tokens": _sum_optional([row.total_tokens for row in rows]),
        }
    return aggregated


def _first_response_choice(response: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(response, Mapping):
        return {}
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    return first if isinstance(first, Mapping) else {}


def _compact_response_message(response: Mapping[str, Any] | None) -> dict[str, Any]:
    choice = _first_response_choice(response)
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in ("role", "content", "reasoning_content"):
        value = message.get(key)
        if value not in (None, ""):
            compact[key] = value
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        compact["tool_calls"] = tool_calls
    if choice.get("finish_reason") not in (None, ""):
        compact["finish_reason"] = choice.get("finish_reason")
    return compact


def _write_results(
    *,
    output_dir: Path,
    stamp: str,
    model_name: str,
    task_names: list[str],
    samples: list[FunctionCallingSample],
    results: list[FunctionCallingRunResult],
    elapsed_seconds: float,
) -> Path:
    run_dir = output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    details_path = run_dir / "details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "task": result.task_name,
                        "sample_id": result.sample_id,
                        "score": result.score,
                        "actual_calls": result.actual_calls,
                        "response_message": _compact_response_message(result.raw_response),
                        "error": result.error,
                        "elapsed_seconds": result.elapsed_seconds,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.total_tokens,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    results_path = run_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "mode": "openai_tool_calls",
                "tasks": task_names,
                "sample_count": len(samples),
                "details": str(details_path),
                "performance": _aggregate_performance(results, elapsed_seconds=elapsed_seconds),
                "results": _aggregate_results(results),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return results_path


def run_function_calling_eval(args: Any, *, root: Path, env: dict[str, str], config: dict[str, Any]) -> int:
    task_names = parse_task_names(getattr(args, "tasks", None))
    fc_config = table(config, "function_calling")
    lighteval_config = table(config, "lighteval")
    model_name = model_name_for_fc(args, config)
    base_url = local_openai_base_url(config, env, args)
    api_key = str(pick(env.get("HELICOPTER_EVAL_API_KEY"), env.get("OPENAI_API_KEY"), fc_config.get("api_key"), "EMPTY"))
    if api_key == "EMPTY" and not is_local_base_url(base_url):
        api_key = ""
    output_dir = Path(
        str(
            pick(
                getattr(args, "output_dir", None),
                env_value(env, "HELICOPTER_FC_OUTPUT_DIR"),
                fc_config.get("output_dir"),
                "results/function_calling",
            )
        )
    )
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    max_samples = getattr(args, "max_samples", None)
    max_new_tokens = int(
        pick(
            env_value(env, "HELICOPTER_FC_MAX_NEW_TOKENS"),
            fc_config.get("max_new_tokens"),
            lighteval_config.get("max_new_tokens"),
            768,
        )
    )
    concurrent_requests = int(
        pick(
            env_value(env, "HELICOPTER_FC_CONCURRENT_REQUESTS"),
            fc_config.get("concurrent_requests"),
            lighteval_config.get("concurrent_requests"),
            1,
        )
    )
    timeout_s = float(
        pick(
            env_value(env, "HELICOPTER_FC_REQUEST_TIMEOUT"),
            fc_config.get("request_timeout"),
            120.0,
        )
    )

    manage_server = not getattr(args, "no_server", False) and is_local_base_url(base_url)
    infer_plan: CommandPlan | None = None
    if manage_server:
        args._config = config
        infer_plan = build_infer_plan(
            infer_args_namespace(args, port=port_from_base_url(base_url)),
            root=root,
            env=env,
            config=config,
        )

    if args.dry_run:
        if infer_plan is not None:
            print(format_plan_for_display(infer_plan))
        print(
            "function-calling: "
            f"model={model_name} base_url={base_url} tasks={','.join(task_names)} "
            "mode=openai_tool_calls"
        )
        return 0

    samples: list[FunctionCallingSample] = []
    for task_name in task_names:
        samples.extend(load_samples(task_name, max_samples=max_samples, root=root))
    if not samples:
        raise SystemExit("no function-calling samples loaded")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    server_process: subprocess.Popen[bytes] | None = None
    server_log: Path | None = None
    if manage_server and infer_plan is not None:
        if server_is_healthy(base_url):
            print(f"function-calling: reusing healthy server at {base_url}")
        else:
            server_log = output_dir / "server_logs" / f"vllm_{stamp}.log"
            server_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"function-calling: starting vLLM server (log: {server_log})")
            with server_log.open("wb") as log_file:
                server_process = subprocess.Popen(
                    infer_plan.command,
                    cwd=str(infer_plan.cwd) if infer_plan.cwd else None,
                    env=infer_plan.env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            wait_for_server(
                base_url,
                process=server_process,
                log_path=server_log,
                timeout_s=float(
                    pick(
                        env_value(env, "HELICOPTER_FC_SERVER_TIMEOUT"),
                        fc_config.get("server_timeout"),
                        DEFAULT_SERVER_TIMEOUT_S,
                    )
                ),
            )
            print(f"function-calling: server healthy at {base_url}")

    started = time.monotonic()
    try:
        results = run_samples(
            samples,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key or None,
            max_new_tokens=max_new_tokens,
            timeout_s=timeout_s,
            concurrent_requests=concurrent_requests,
        )
    finally:
        elapsed = time.monotonic() - started
        if server_process is not None and not getattr(args, "keep_server", False):
            print("function-calling: stopping vLLM server")
            stop_server(server_process)
        elif server_process is not None:
            print(f"function-calling: leaving vLLM server running (pid {server_process.pid})")

    results_path = _write_results(
        output_dir=output_dir,
        stamp=stamp,
        model_name=model_name,
        task_names=task_names,
        samples=samples,
        results=results,
        elapsed_seconds=elapsed,
    )
    if getattr(args, "scoreboard", False):
        ingest_scoreboard_results(
            result_files=[str(results_path)],
            model_name=model_name,
            root=root,
            env=env,
            job_name="function_calling",
        )
    aggregate = _aggregate_results(results)
    for task_name in task_names:
        if task_name in aggregate:
            print(f"function-calling: {task_name} accuracy={aggregate[task_name]['accuracy']:.4f}")
    print(f"function-calling: finished {len(samples)} samples in {elapsed:.1f}s; results: {results_path}")
    return 0
