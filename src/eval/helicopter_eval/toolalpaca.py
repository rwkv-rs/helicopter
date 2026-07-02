from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterator, Mapping, Sequence

from .apibank import decode_tool_calls
from .openai_client import chat_completion
from .sampling import apply_limit_or_sample, dataset_sample_suffix
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


TOOLALPACA_REPO_URL = "https://github.com/tangqiaoyu/ToolAlpaca.git"
TOOLALPACA_CACHE_ROOT_NAME = "ToolAlpaca"
TOOLALPACA_FILES = {
    "toolalpaca_eval_simulated": "eval_simulated.json",
    "toolalpaca_eval_real": "eval_real.json",
}
TOOLALPACA_REF_KEY = "__toolalpaca_ref__"
TOOLALPACA_OPTIONAL_KEY = "__toolalpaca_optional__"
TOOLALPACA_AUTH_PARAMS_BY_API = {
    "apilayer weatherstack": frozenset({"access_key"}),
    "wolframalpha": frozenset({"appid"}),
    "currencybeacon": frozenset({"api_key"}),
}
HTTP_BODY_METHODS = {"post", "put", "patch"}


@dataclass(frozen=True, slots=True)
class ToolAlpacaExpectedCall:
    name: str
    arguments: dict[str, Any]
    optional: bool = False


@dataclass(frozen=True, slots=True)
class ToolAlpacaActionResult:
    action: str
    action_input: dict[str, Any]
    success: bool
    optional: bool = False
    request: dict[str, Any] = field(default_factory=dict)
    response: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolAlpacaSample:
    sample_index: int
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_tool_calls: tuple[ToolAlpacaExpectedCall, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolAlpacaResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str

    def to_scoreboard(self) -> ScoreboardEvalResult:
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=self.answer,
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class ToolAlpacaRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    source_root: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout_s: float = 600.0
    scoreboard_dataset: str | None = None
    job_name: str = "function_toolalpaca"
    job_id: str | None = None
    runner: str = "helicopter_eval.toolalpaca"
    cot_mode: str = "CoT"


class ToolAlpacaSandbox:
    execution_mode = "local_toolalpaca_request_sandbox"

    def execute_sequence(
        self,
        sample: ToolAlpacaSample,
        calls: Sequence[Mapping[str, Any]],
    ) -> list[ToolAlpacaActionResult]:
        results: list[ToolAlpacaActionResult] = []
        for call in calls:
            results.append(self.execute_call(sample, call, history=results))
        return results

    def execute_call(
        self,
        sample: ToolAlpacaSample,
        call: Mapping[str, Any],
        *,
        history: Sequence[ToolAlpacaActionResult],
    ) -> ToolAlpacaActionResult:
        normalized = _normalize_toolalpaca_call(call, history)
        if normalized is None:
            return ToolAlpacaActionResult(
                action=str(call.get("name") or call.get("Action") or "").strip(),
                action_input={},
                success=False,
                error="arguments_not_object",
            )
        action, resolved_arguments, optional = normalized
        tool = {str(item.get("name") or ""): item for item in sample.tools}.get(action)
        if tool is None and action == "getDetails":
            tool = _get_details_tool_schema()
        if tool is None:
            request = {
                "action": action,
                "method": "",
                "path": "",
                "path_params": {},
                "query": {},
                "body": dict(_json_safe(resolved_arguments)),
                "headers": {},
                "cookies": {},
                "builtin": False,
            }
            return ToolAlpacaActionResult(
                action=action,
                action_input=dict(_json_safe(resolved_arguments)),
                success=True,
                optional=optional,
                request=request,
                response=_synthetic_toolalpaca_response(action, request, history),
            )
        try:
            request = _build_toolalpaca_request(sample, tool, dict(resolved_arguments))
        except ValueError as exc:
            return ToolAlpacaActionResult(
                action=action,
                action_input=dict(_json_safe(resolved_arguments)),
                success=False,
                optional=optional,
                error=str(exc),
            )
        return ToolAlpacaActionResult(
            action=action,
            action_input=dict(_json_safe(resolved_arguments)),
            success=True,
            optional=optional,
            request=request,
            response=_synthetic_toolalpaca_response(action, request, history),
        )


def resolve_toolalpaca_source_root(config: ToolAlpacaRunConfig) -> Path:
    candidates: list[Path] = []
    for raw in (
        config.source_root,
        os.getenv("TOOLALPACA_SOURCE_ROOT"),
        os.getenv("RWKV_TOOLALPACA_SOURCE_ROOT"),
    ):
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            Path("/home/chase/GitHub/rwkv-skills/references/ToolAlpaca/data"),
            Path("/home/chase/GitHub/ToolAlpaca/data"),
            Path("/home/chase/ToolAlpaca/data"),
            Path("/tmp/rwkv-official-refs/ToolAlpaca/data"),
            Path("/tmp/ToolAlpaca/data"),
        ]
    )
    for candidate in candidates:
        resolved = _normalize_toolalpaca_root(candidate)
        if _toolalpaca_required_paths(resolved):
            return resolved
    return _ensure_toolalpaca_cache() / "data"


def load_samples(config: ToolAlpacaRunConfig) -> list[ToolAlpacaSample]:
    if config.split != "test":
        raise ValueError("ToolAlpaca only provides test split")
    if config.dataset_name not in TOOLALPACA_FILES:
        raise ValueError(f"unknown ToolAlpaca dataset: {config.dataset_name}")
    source_root = resolve_toolalpaca_source_root(config)
    source_path = source_root / TOOLALPACA_FILES[config.dataset_name]
    rows = load_toolalpaca_rows_from_source(source_path, dataset_name=config.dataset_name)
    samples: list[ToolAlpacaSample] = []
    for row in rows:
        if config.limit is not None and config.sample_size is None and len(samples) >= int(config.limit):
            break
        samples.append(
            ToolAlpacaSample(
                sample_index=len(samples),
                task_id=str(row["task_id"]),
                instruction=str(row["instruction"]),
                tools=tuple(dict(item) for item in row["tools"]),
                expected_tool_calls=tuple(
                    ToolAlpacaExpectedCall(
                        name=str(item.get("name") or ""),
                        arguments=dict(item.get("arguments") or {}),
                        optional=_truthy(dict(item.get("arguments") or {}).get(TOOLALPACA_OPTIONAL_KEY, False)),
                    )
                    for item in row["expected_tool_calls"]
                ),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return apply_limit_or_sample(
        samples,
        limit=config.limit,
        sample_size=config.sample_size,
        sample_seed=config.sample_seed,
        sort_key=lambda sample: sample.sample_index,
    )


def load_toolalpaca_rows_from_source(path: str | Path, *, dataset_name: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ToolAlpaca source must be a JSON array: {source}")

    rows: list[dict[str, Any]] = []
    for api_index, api_info in enumerate(payload):
        if not isinstance(api_info, Mapping):
            continue
        api_name = str(api_info.get("Name") or api_info.get("API") or f"api_{api_index}")
        if _toolalpaca_should_skip_api(dataset_name, api_name):
            continue
        tools = toolalpaca_tools(api_info)
        metadata_base: dict[str, Any] = {
            "source_format": "official_toolalpaca",
            "api_name": api_name,
            "api_index": api_index,
            "source_path": str(source),
            "execution_backend": _toolalpaca_execution_backend(dataset_name),
        }
        server_url = _toolalpaca_api_server_url(api_info)
        if server_url:
            metadata_base["api_server_url"] = server_url
        instructions = _coerce_list(api_info.get("Instructions"))
        golden_answers = _coerce_list(api_info.get("Golden_Answers"))
        for question_index, instruction in enumerate(instructions):
            if question_index >= len(golden_answers):
                continue
            instruction_text = str(instruction or "").strip()
            if not instruction_text:
                continue
            rows.append(
                {
                    "task_id": f"{dataset_name}__{_slug(api_name)}_{question_index:03d}",
                    "instruction": instruction_text,
                    "tools": tools,
                    "expected_tool_calls": normalize_toolalpaca_golden_answer(golden_answers[question_index]),
                    "metadata": {**metadata_base, "question_index": question_index},
                }
            )
    return rows


def build_prompt(sample: ToolAlpacaSample) -> str:
    tools = json.dumps(sample.tools, ensure_ascii=False, indent=2, sort_keys=True)
    schema = json.dumps(
        {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "arguments"],
                "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are solving a ToolAlpaca tool-use task.\n\n"
        "Tools:\n"
        f"{tools}\n\n"
        "Output JSON schema:\n"
        f"{schema}\n\n"
        "Return exactly one JSON array that validates against the schema. "
        "Include every required tool call in execution order. Use only listed tool names. "
        "Return no prose, no markdown, and no extra text outside the JSON value.\n\n"
        f"User request:\n{sample.instruction}\n\n"
        "Tool calls:"
    )


def evaluate_completion(
    sample: ToolAlpacaSample,
    completion: str,
    *,
    sandbox: ToolAlpacaSandbox | None = None,
) -> tuple[bool, str, list[dict[str, Any]], dict[str, Any]]:
    try:
        decoded_calls = decode_tool_calls(completion)
    except Exception as exc:  # noqa: BLE001
        return False, f"parse_error:{exc}", [], {"parse_error": str(exc)}
    passed, reason, details = evaluate_toolalpaca_calls(sample, decoded_calls, sandbox=sandbox)
    return passed, reason, decoded_calls, details


def evaluate_toolalpaca_calls(
    sample: ToolAlpacaSample,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    sandbox: ToolAlpacaSandbox | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    sandbox = sandbox or ToolAlpacaSandbox()
    expected_calls = [{"name": item.name, "arguments": dict(item.arguments)} for item in sample.expected_tool_calls]
    actual_calls = [
        {"name": str(item.get("name") or ""), "arguments": dict(item.get("arguments") or {})}
        for item in decoded_calls
    ]
    expected_results = sandbox.execute_sequence(sample, expected_calls)
    actual_results = sandbox.execute_sequence(sample, actual_calls)
    details: dict[str, Any] = {
        "execution_mode": sandbox.execution_mode,
        "expected_tool_calls": expected_calls,
        "decoded_tool_calls": actual_calls,
        "expected_execution_results": [_result_payload(item) for item in expected_results],
        "decoded_execution_results": [_result_payload(item) for item in actual_results],
        "call_matches": [],
    }

    required_expected = [item for item in expected_results if not item.optional]
    passed_count = 0
    failure_bits: list[str] = []
    actual_index = 0
    for expected_index, expected in enumerate(expected_results):
        if actual_index >= len(actual_results):
            if expected.optional:
                details["call_matches"].append(
                    {"expected_index": expected_index, "decoded_index": None, "ok": True, "reason": "optional_skipped"}
                )
                continue
            details["call_matches"].append(
                {"expected_index": expected_index, "decoded_index": None, "ok": False, "reason": "missing_call"}
            )
            failure_bits.append(f"call_{expected_index}:missing_call")
            continue

        actual = actual_results[actual_index]
        ok, reason = _execution_matches(actual, expected)
        if ok:
            if not expected.optional:
                passed_count += 1
            details["call_matches"].append(
                {"expected_index": expected_index, "decoded_index": actual_index, "ok": True, "reason": reason}
            )
            actual_index += 1
            continue
        if expected.optional:
            details["call_matches"].append(
                {
                    "expected_index": expected_index,
                    "decoded_index": actual_index,
                    "ok": True,
                    "reason": "optional_skipped",
                    "candidate_reason": reason,
                }
            )
            continue
        details["call_matches"].append(
            {"expected_index": expected_index, "decoded_index": actual_index, "ok": False, "reason": reason}
        )
        failure_bits.append(f"call_{expected_index}:{reason}")
        actual_index += 1

    while actual_index < len(actual_results):
        details["call_matches"].append(
            {"expected_index": None, "decoded_index": actual_index, "ok": False, "reason": "unexpected_extra_call"}
        )
        failure_bits.append(f"call_{actual_index}:unexpected_extra_call")
        actual_index += 1

    is_passed = passed_count == len(required_expected) and not failure_bits
    if not expected_results:
        is_passed = len(actual_results) == 0
    return bool(is_passed), "; ".join(failure_bits), details


def generate_completion(sample: ToolAlpacaSample, config: ToolAlpacaRunConfig) -> ToolAlpacaResult:
    prompt = build_prompt(sample)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    passed, fail_reason, decoded_calls, _details = evaluate_completion(sample, completion)
    reference = [{"name": item.name, "arguments": item.arguments} for item in sample.expected_tool_calls]
    return ToolAlpacaResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=completion,
        answer=json.dumps(decoded_calls, ensure_ascii=False, sort_keys=True),
        reference_answer=json.dumps(reference, ensure_ascii=False, sort_keys=True),
        is_passed=passed,
        fail_reason=fail_reason,
    )


def evaluate_samples(samples: Sequence[ToolAlpacaSample], config: ToolAlpacaRunConfig) -> list[ToolAlpacaResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: ToolAlpacaRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    dataset += dataset_sample_suffix(sample_size=config.sample_size, sample_seed=config.sample_seed)
    return dataset


def job_id(config: ToolAlpacaRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: ToolAlpacaRunConfig) -> dict[str, Any]:
    return {
        "tool": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: ToolAlpacaRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_toolalpaca",
        "execution_backend": "local_toolalpaca_request_sandbox",
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[ToolAlpacaResult], *, config: ToolAlpacaRunConfig, repo_root: Path) -> int:
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
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_toolalpaca(config: ToolAlpacaRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "source_dataset": config.dataset_name,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: ToolAlpacaRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "source": "git+https://github.com/tangqiaoyu/ToolAlpaca.git#data",
        "split": config.split,
        "source_dataset": config.dataset_name,
        "limit": config.limit,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


def normalize_toolalpaca_golden_answer(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("Action") or item.get("action") or "").strip()
        optional = False
        if action.lower().startswith("[optional]"):
            optional = True
            action = action.split("]", 1)[-1].strip()
        arguments = parse_toolalpaca_action_input(item.get("Action_Input", item.get("action_input", {})))
        if optional:
            arguments[TOOLALPACA_OPTIONAL_KEY] = True
        calls.append({"name": action, "arguments": dict(arguments)})
    return calls


def parse_toolalpaca_action_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        normalized = _normalize_toolalpaca_placeholders(dict(raw))
        return dict(normalized) if isinstance(normalized, Mapping) else {}
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    repaired = _repair_toolalpaca_action_input_text(text)
    for candidate in (text, repaired, _quote_unquoted_toolalpaca_refs(repaired)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            normalized = _normalize_toolalpaca_placeholders(parsed)
            return dict(normalized) if isinstance(normalized, Mapping) else {}
        return {}
    return _parse_loose_toolalpaca_object(text)


def toolalpaca_tools(api_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    api_name = str(api_info.get("Name") or api_info.get("API") or "")
    openapi_spec = _toolalpaca_openapi_spec(api_info)
    server_url = _openapi_server_url(openapi_spec)
    descriptions = api_info.get("Function_Description")
    projection = api_info.get("Function_Projection")
    tools: list[dict[str, Any]] = []
    if isinstance(descriptions, Mapping):
        for name, description in descriptions.items():
            name_text = str(name).strip()
            if not name_text or name_text == "components":
                continue
            method = ""
            path = ""
            if isinstance(projection, Mapping):
                projected = projection.get(name)
                if isinstance(projected, Sequence) and not isinstance(projected, (str, bytes, bytearray)):
                    path = str(projected[0]) if len(projected) > 0 else ""
                    method = str(projected[1]) if len(projected) > 1 else ""
            metadata: dict[str, Any] = {"path": path, "method": method, "api_name": api_name}
            if server_url:
                metadata["server_url"] = server_url
            operation = _toolalpaca_openapi_operation(openapi_spec, path, method)
            if operation:
                metadata["operation"] = dict(operation)
            tools.append(
                {
                    "name": name_text,
                    "description": _normalize_text(str(description or "")),
                    "parameters": _strip_toolalpaca_auth_parameters(
                        api_name,
                        _toolalpaca_parameters_from_description(str(description or "")),
                    ),
                    "metadata": metadata,
                }
            )
    if _toolalpaca_api_uses_action(api_info, "getDetails"):
        tools.append(_get_details_tool_schema(api_name=api_name))
    return tools


def _build_toolalpaca_request(
    sample: ToolAlpacaSample,
    tool: Mapping[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    name = str(tool.get("name") or "").strip()
    metadata = tool.get("metadata") if isinstance(tool.get("metadata"), Mapping) else {}
    parameters = tool.get("parameters") if isinstance(tool.get("parameters"), Mapping) else {}
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), Mapping) else {}
    required = {str(item) for item in _coerce_list(parameters.get("required"))}
    if metadata.get("tool_type") == "toolalpaca_builtin" or name == "getDetails":
        return _build_builtin_request(name, arguments, properties, required)

    operation = _load_operation(sample, metadata)
    if operation:
        op_required, op_properties = _operation_parameter_schema(operation)
        required.update(op_required)
        properties = {**op_properties, **dict(properties)}

    canonical_arguments: dict[str, Any] = {}
    unknown_arguments: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == TOOLALPACA_OPTIONAL_KEY:
            continue
        property_name = _resolve_property_name(str(key), properties)
        schema = properties.get(property_name) if property_name else None
        if schema is None:
            unknown_arguments[str(key)] = _json_safe(value)
            continue
        if _is_absent(value) and property_name not in required:
            continue
        canonical_arguments[str(property_name)] = _coerce_argument_value(str(property_name), value, schema)

    missing = sorted(key for key in required if key not in canonical_arguments or _is_absent(canonical_arguments[key]))
    path = str(metadata.get("path") or "")
    required_path_arguments = [key for key in missing if f"{{{key}}}" in path]
    if required_path_arguments:
        raise ValueError(f"missing_required_arguments({', '.join(required_path_arguments)})")
    if missing and not canonical_arguments:
        raise ValueError(f"missing_required_arguments({', '.join(missing)})")

    method = str(metadata.get("method") or "get").lower()
    param_locations = _operation_param_locations(operation)
    path_params: dict[str, Any] = {}
    query: dict[str, Any] = {}
    headers: dict[str, Any] = {}
    cookies: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for key, value in canonical_arguments.items():
        location = param_locations.get(key)
        if location == "path" or f"{{{key}}}" in path:
            path_params[key] = value
        elif location == "header":
            headers[key] = value
        elif location == "cookie":
            cookies[key] = value
        elif location == "query" or method not in HTTP_BODY_METHODS:
            query[key] = value
        else:
            body[key] = value

    rendered_path = path
    for key, value in path_params.items():
        rendered_path = rendered_path.replace(f"{{{key}}}", str(value))
    unresolved_path_params = re.findall(r"\{([^{}]+)\}", rendered_path)
    if unresolved_path_params:
        raise ValueError(f"missing_path_arguments({', '.join(sorted(unresolved_path_params))})")

    return _json_safe(
        {
            "action": name,
            "method": method,
            "path": rendered_path,
            "path_template": path,
            "path_params": path_params,
            "query": _drop_absent_values(query),
            "body": _drop_absent_values(body),
            "headers": _drop_absent_values(headers),
            "cookies": _drop_absent_values(cookies),
            "ignored_arguments": unknown_arguments,
            "builtin": False,
            "api_name": str(metadata.get("api_name") or sample.metadata.get("api_name") or ""),
            "server_url": str(metadata.get("server_url") or sample.metadata.get("api_server_url") or ""),
        }
    )


def _build_builtin_request(
    name: str,
    arguments: Mapping[str, Any],
    properties: Mapping[str, Any],
    required: set[str],
) -> dict[str, Any]:
    canonical_arguments: dict[str, Any] = {}
    for key, value in arguments.items():
        if key not in properties:
            continue
        if _is_absent(value) and key not in required:
            continue
        canonical_arguments[str(key)] = _coerce_argument_value(str(key), value, properties[key])
    missing = sorted(key for key in required if key not in canonical_arguments or _is_absent(canonical_arguments[key]))
    if missing:
        raise ValueError(f"missing_required_arguments({', '.join(missing)})")
    return _json_safe(
        {
            "action": name,
            "method": "",
            "path": "",
            "path_template": "",
            "path_params": {},
            "query": {},
            "body": canonical_arguments,
            "headers": {},
            "cookies": {},
            "ignored_arguments": {},
            "builtin": True,
            "api_name": "",
            "server_url": "",
        }
    )


def _load_operation(sample: ToolAlpacaSample, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    operation = metadata.get("operation")
    if isinstance(operation, Mapping):
        return operation
    path = str(metadata.get("path") or "")
    method = str(metadata.get("method") or "").lower()
    source_path = sample.metadata.get("source_path")
    if not path or not method or source_path is None:
        return {}
    api_info = load_toolalpaca_api_info_from_source(
        str(source_path),
        api_index=sample.metadata.get("api_index"),
        api_name=str(sample.metadata.get("api_name") or ""),
    )
    documentation = str(api_info.get("Documentation") or "") if isinstance(api_info, Mapping) else ""
    if not documentation:
        return {}
    spec = _load_openapi_spec(documentation)
    paths = spec.get("paths") if isinstance(spec.get("paths"), Mapping) else {}
    path_doc = paths.get(path) if isinstance(paths, Mapping) else {}
    operation = path_doc.get(method) if isinstance(path_doc, Mapping) else {}
    return operation if isinstance(operation, Mapping) else {}


def load_toolalpaca_api_info_from_source(
    source_path: str,
    *,
    api_index: Any,
    api_name: str,
) -> Mapping[str, Any]:
    payload = _load_toolalpaca_source_payload(source_path)
    try:
        index = int(api_index)
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(payload) and isinstance(payload[index], Mapping):
        return payload[index]
    for item in payload:
        if isinstance(item, Mapping) and str(item.get("Name") or item.get("API") or "") == api_name:
            return item
    return {}


def _load_toolalpaca_source_payload(source_path: str) -> tuple[Any, ...]:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    return tuple(payload) if isinstance(payload, list) else ()


def _operation_parameter_schema(operation: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    required: set[str] = set()
    properties: dict[str, Any] = {}
    for param_doc in _coerce_list(operation.get("parameters")):
        if not isinstance(param_doc, Mapping) or not param_doc.get("name"):
            continue
        name = str(param_doc.get("name") or "")
        schema = param_doc.get("schema") if isinstance(param_doc.get("schema"), Mapping) else {}
        properties[name] = {**dict(schema or {}), "description": str(param_doc.get("description") or "")}
        if param_doc.get("required"):
            required.add(name)
    body_schema = _request_body_schema(operation)
    body_props = body_schema.get("properties") if isinstance(body_schema.get("properties"), Mapping) else {}
    for key, schema in body_props.items():
        properties[str(key)] = dict(schema) if isinstance(schema, Mapping) else {}
    required.update(str(item) for item in _coerce_list(body_schema.get("required")))
    return required, properties


def _operation_param_locations(operation: Mapping[str, Any]) -> dict[str, str]:
    locations: dict[str, str] = {}
    for param_doc in _coerce_list(operation.get("parameters")):
        if isinstance(param_doc, Mapping) and param_doc.get("name"):
            locations[str(param_doc.get("name"))] = str(param_doc.get("in") or "query")
    return locations


def _request_body_schema(operation: Mapping[str, Any]) -> Mapping[str, Any]:
    request_body = operation.get("requestBody") if isinstance(operation.get("requestBody"), Mapping) else {}
    content = request_body.get("content") if isinstance(request_body.get("content"), Mapping) else {}
    json_content = content.get("application/json") if isinstance(content.get("application/json"), Mapping) else {}
    schema = json_content.get("schema") if isinstance(json_content.get("schema"), Mapping) else {}
    return schema if isinstance(schema, Mapping) else {}


def _resolve_property_name(key: str, properties: Mapping[str, Any]) -> str | None:
    if key in properties:
        return key
    lowered = key.lower()
    for candidate in properties.keys():
        text = str(candidate)
        if text.lower() == lowered:
            return text
        if text.lower() == f"{lowered}s":
            return text
        if lowered.endswith("s") and text.lower() == lowered[:-1]:
            return text
    return None


def _coerce_argument_value(key: str, value: Any, schema: Any) -> Any:
    if not isinstance(schema, Mapping):
        return _json_safe(value)
    expected_type = str(schema.get("type") or "").lower()
    if expected_type == "integer":
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"argument_type_error({key}:integer)") from None
    elif expected_type == "number":
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"argument_type_error({key}:number)") from None
    elif expected_type == "boolean":
        value = _coerce_bool(key, value)
    elif expected_type == "array":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        if not isinstance(value, list):
            raise ValueError(f"argument_type_error({key}:array)")
    elif expected_type == "object":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        if not isinstance(value, Mapping):
            raise ValueError(f"argument_type_error({key}:object)")
        value = dict(value)
    elif expected_type == "string" and not isinstance(value, str):
        value = str(value)

    enum = _schema_enum(schema)
    if enum and value not in enum and not _is_absent(value):
        raise ValueError(f"argument_enum_error({key})")
    return _json_safe(value)


def _coerce_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"argument_type_error({key}:boolean)")


def _schema_enum(schema: Mapping[str, Any]) -> list[Any]:
    raw_enum = schema.get("enum")
    if isinstance(raw_enum, list):
        return list(raw_enum)
    description = str(schema.get("description") or "")
    match = re.search(r"One of:\s*\[([^\]]+)\]", description, re.IGNORECASE)
    if not match:
        return []
    return [item.strip().strip("\"'") for item in match.group(1).split(",")]


def _normalize_toolalpaca_call(
    call: Mapping[str, Any],
    history: Sequence[ToolAlpacaActionResult],
) -> tuple[str, dict[str, Any], bool] | None:
    action = str(call.get("name") or call.get("Action") or "").strip()
    raw_arguments = call.get("arguments", call.get("Action_Input", {}))
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            raw_arguments = {}
    if not isinstance(raw_arguments, Mapping):
        return None
    arguments = dict(raw_arguments)
    optional = _truthy(arguments.pop(TOOLALPACA_OPTIONAL_KEY, False))
    if action.lower().startswith("[optional]"):
        optional = True
        action = action.split("]", 1)[-1].strip()
    resolved_arguments = _resolve_toolalpaca_value(arguments, history)
    if not isinstance(resolved_arguments, Mapping):
        resolved_arguments = {}
    return action, dict(resolved_arguments), optional


def _resolve_toolalpaca_value(value: Any, history: Sequence[ToolAlpacaActionResult]) -> Any:
    if isinstance(value, Mapping):
        if TOOLALPACA_REF_KEY in value:
            return _resolve_toolalpaca_ref(str(value.get(TOOLALPACA_REF_KEY) or ""), history)
        return {str(key): _resolve_toolalpaca_value(item, history) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_toolalpaca_value(item, history) for item in value]
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{([^{}]+)\}", value.strip())
        if match:
            return _resolve_toolalpaca_ref(match.group(1).strip(), history)
    return value


def _resolve_toolalpaca_ref(ref: str, history: Sequence[ToolAlpacaActionResult]) -> Any:
    text = " ".join(str(ref).strip().split())
    if not text:
        return ""
    lowered = text.lower()
    if " from " in lowered:
        prefix, source = re.split(r"\s+from\s+", text, maxsplit=1, flags=re.IGNORECASE)
        field_name = prefix.strip()
        source_name = source.strip()
        for result in reversed(history):
            if result.action == source_name and result.success:
                extracted = _extract_response_field(result.response, field_name)
                if extracted is not None:
                    return extracted
                return _synthetic_ref_value(field_name, source_name, result.request)
        return _synthetic_ref_value(field_name, source_name, {})
    if "end date" in lowered:
        return "2023-01-31"
    if "start date" in lowered:
        return "2023-01-01"
    if "year" in lowered:
        return 2023
    if "permission" in lowered:
        return "read"
    if lowered == "string":
        return "string"
    return _synthetic_ref_value(text, "context", {})


def _extract_response_field(response: Any, field_name: str) -> Any:
    if isinstance(response, Mapping):
        if field_name in response:
            return response[field_name]
        normalized_target = _slug(field_name)
        for key, value in response.items():
            if _slug(str(key)) == normalized_target:
                return value
            nested = _extract_response_field(value, field_name)
            if nested is not None:
                return nested
    if isinstance(response, list):
        for item in response:
            nested = _extract_response_field(item, field_name)
            if nested is not None:
                return nested
    return None


def _synthetic_toolalpaca_response(
    action: str,
    request: Mapping[str, Any],
    history: Sequence[ToolAlpacaActionResult],
) -> dict[str, Any]:
    seed = _stable_seed({"action": action, "request": _comparable_request(request)})
    response: dict[str, Any] = {"ok": True, "action": action, "id": _stable_int(seed, "id")}
    response["resultId"] = _stable_id(seed, "result")
    for field_name in _likely_response_fields(action, request, history):
        response[field_name] = _synthetic_ref_value(field_name, action, request)
    return _json_safe(response)


def _likely_response_fields(
    action: str,
    request: Mapping[str, Any],
    history: Sequence[ToolAlpacaActionResult],
) -> list[str]:
    fields = {
        "id",
        "resultId",
        "userId",
        "accessToken",
        "animeId",
        "ip",
        "style",
        "format",
        "category",
        "username",
        "holidayId",
        "symbol",
        "dashboardId",
        "sourceId",
        "targetId",
    }
    for container_name in ("query", "body", "path_params"):
        container = request.get(container_name)
        if isinstance(container, Mapping):
            fields.update(str(key) for key in container)
    for result in history:
        if isinstance(result.response, Mapping):
            fields.update(str(key) for key in result.response.keys())
    if action.startswith("search"):
        fields.add(f"{_slug(action.removeprefix('search'))}Id")
    if action.startswith("create"):
        fields.add(f"{_slug(action.removeprefix('create'))}Id")
    return sorted(fields)


def _synthetic_ref_value(field_name: str, source_name: str, request: Mapping[str, Any] | None) -> Any:
    field_slug = _slug(field_name)
    seed = _stable_seed({"field": field_name, "source": source_name, "request": _comparable_request(request or {})})
    if "date" in field_slug:
        return "2023-01-31" if field_slug.startswith("end") else "2023-01-01"
    if field_slug.endswith("year") or field_slug == "year":
        return 2023
    if field_slug.endswith("id") or field_slug == "id":
        return _stable_int(seed, field_slug)
    if field_slug == "ip":
        return f"192.0.2.{_stable_int(seed, field_slug) % 250 + 1}"
    if field_slug == "symbol":
        return "BTC/USD"
    if field_slug == "format":
        return "png"
    if field_slug == "style":
        return "minimal"
    if field_slug == "category":
        return "general"
    if field_slug in {"permission", "permissions", "somepermission"}:
        return "read"
    return _stable_id(seed, field_slug)


def _execution_matches(actual: ToolAlpacaActionResult, expected: ToolAlpacaActionResult) -> tuple[bool, str]:
    if actual.action != expected.action:
        return False, f"name_mismatch(expected={expected.action}, actual={actual.action})"
    if actual.success != expected.success:
        return False, f"execution_status_mismatch(expected={expected.success}, actual={actual.success})"
    if not expected.success:
        return (actual.error == expected.error, "ok" if actual.error == expected.error else "execution_error_mismatch")
    if _comparable_request(actual.request) != _comparable_request(expected.request):
        return False, "request_mismatch"
    return True, "ok"


def _comparable_request(request: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            "action": request.get("action", ""),
            "method": request.get("method", ""),
            "path": request.get("path", ""),
            "path_params": _drop_absent_values(request.get("path_params", {})),
            "query": _drop_absent_values(request.get("query", {})),
            "body": _drop_absent_values(request.get("body", {})),
            "headers": _drop_absent_values(request.get("headers", {})),
            "cookies": _drop_absent_values(request.get("cookies", {})),
            "builtin": bool(request.get("builtin", False)),
        }
    )


def _result_payload(result: ToolAlpacaActionResult) -> dict[str, Any]:
    return {
        "Action": result.action,
        "Action_Input": _clean_action_arguments(result.action_input),
        "success": bool(result.success),
        "optional": bool(result.optional),
        "request": result.request,
        "response": result.response,
        "error": result.error or "",
    }


def _clean_action_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in arguments.items() if key != TOOLALPACA_OPTIONAL_KEY}


def _repair_toolalpaca_action_input_text(text: str) -> str:
    return re.sub(
        r"\$\{([^,{}]+?\s+from\s+[^,{}]+?),\s*\"",
        lambda match: f"${{{match.group(1).strip()}}}, \"",
        text,
    )


def _quote_unquoted_toolalpaca_refs(text: str) -> str:
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if text.startswith("${", index):
            end = text.find("}", index + 2)
            if end != -1:
                output.append(json.dumps(text[index : end + 1], ensure_ascii=False))
                index = end + 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _normalize_toolalpaca_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{([^{}]+)\}", value.strip())
        if match:
            return {TOOLALPACA_REF_KEY: match.group(1).strip()}
        return value
    if isinstance(value, list):
        return [_normalize_toolalpaca_placeholders(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_toolalpaca_placeholders(item) for key, item in value.items()}
    return value


def _parse_loose_toolalpaca_object(text: str) -> dict[str, Any]:
    body = text.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    parsed: dict[str, Any] = {}
    for pair in _split_toolalpaca_top_level(body, delimiter=","):
        if ":" not in pair:
            continue
        key_text, value_text = pair.split(":", 1)
        key = key_text.strip().strip('"').strip("'").strip()
        if key:
            parsed[key] = _parse_loose_toolalpaca_value(value_text)
    return parsed


def _parse_loose_toolalpaca_value(text: str) -> Any:
    value = text.strip().rstrip(",").strip()
    if value.startswith("${"):
        end = value.find("}")
        if end != -1:
            return {TOOLALPACA_REF_KEY: value[2:end].strip()}
    if value.startswith("[") and value.endswith("]"):
        return [_parse_loose_toolalpaca_value(item) for item in _split_toolalpaca_top_level(value[1:-1], delimiter=",")]
    if value.startswith("{") and value.endswith("}"):
        return _parse_loose_toolalpaca_object(value)
    try:
        return _normalize_toolalpaca_placeholders(json.loads(value))
    except json.JSONDecodeError:
        pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip('"').strip("'")


def _split_toolalpaca_top_level(text: str, *, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if text.startswith("${", index):
            end = text.find("}", index + 2)
            if end != -1:
                index = end + 1
                continue
        if char in "[{(":
            depth += 1
        elif char in "]})" and depth > 0:
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
        index += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _toolalpaca_execution_backend(dataset_name: str) -> str:
    if dataset_name == "toolalpaca_eval_simulated":
        return "toolalpaca_simulator"
    if dataset_name == "toolalpaca_eval_real":
        return "toolalpaca_real_http"
    return "toolalpaca_synthetic"


def _toolalpaca_should_skip_api(dataset_name: str, api_name: str) -> bool:
    return dataset_name == "toolalpaca_eval_real" and api_name.strip().lower() in TOOLALPACA_AUTH_PARAMS_BY_API


def _toolalpaca_api_server_url(api_info: Mapping[str, Any]) -> str:
    return _openapi_server_url(_toolalpaca_openapi_spec(api_info))


def _toolalpaca_openapi_spec(api_info: Mapping[str, Any]) -> Mapping[str, Any]:
    documentation = api_info.get("Documentation")
    if not isinstance(documentation, str) or not documentation.strip():
        return {}
    try:
        payload = json.loads(documentation)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _openapi_server_url(openapi_spec: Mapping[str, Any]) -> str:
    servers = openapi_spec.get("servers")
    if isinstance(servers, Sequence) and not isinstance(servers, (str, bytes, bytearray)):
        for server in servers:
            if isinstance(server, Mapping) and server.get("url"):
                return str(server.get("url") or "").strip()
    return ""


def _toolalpaca_openapi_operation(openapi_spec: Mapping[str, Any], path: str, method: str) -> Mapping[str, Any]:
    paths = openapi_spec.get("paths") if isinstance(openapi_spec.get("paths"), Mapping) else {}
    path_doc = paths.get(path) if isinstance(paths, Mapping) else {}
    operation = path_doc.get(str(method or "").lower()) if isinstance(path_doc, Mapping) else {}
    return operation if isinstance(operation, Mapping) else {}


def _strip_toolalpaca_auth_parameters(api_name: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    auth_params = TOOLALPACA_AUTH_PARAMS_BY_API.get(api_name.strip().lower())
    if not auth_params:
        return dict(parameters)
    normalized = dict(parameters)
    properties = normalized.get("properties") if isinstance(normalized.get("properties"), Mapping) else {}
    normalized["properties"] = {
        str(key): value
        for key, value in dict(properties).items()
        if str(key) not in auth_params
    }
    normalized["required"] = [
        str(item) for item in _coerce_list(normalized.get("required")) if str(item) not in auth_params
    ]
    return normalized


def _toolalpaca_api_uses_action(api_info: Mapping[str, Any], action_name: str) -> bool:
    for answer in _coerce_list(api_info.get("Golden_Answers")):
        for item in _coerce_list(answer):
            if isinstance(item, Mapping):
                action = str(item.get("Action") or item.get("action") or "").strip()
                if action == action_name:
                    return True
    return False


def _toolalpaca_parameters_from_description(description: str) -> dict[str, Any]:
    marker = "Parameters:"
    if marker not in description:
        return {"type": "object", "properties": {}, "required": []}
    after = description.split(marker, 1)[1]
    before_output = after.split("\nOutput:", 1)[0].strip()
    try:
        raw_params = json.loads(before_output)
    except json.JSONDecodeError:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        raw_params = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, value in raw_params.items():
        description_text = str(value or "")
        value_lower = description_text.lower()
        param_type = "string"
        if value_lower.startswith("integer") or ". integer" in value_lower:
            param_type = "integer"
        elif value_lower.startswith("number") or value_lower.startswith("float") or ". float" in value_lower:
            param_type = "number"
        elif value_lower.startswith("boolean") or ". boolean" in value_lower:
            param_type = "boolean"
        properties[str(key)] = {"type": param_type, "description": description_text}
        if "required." in value_lower:
            required.append(str(key))
    return {"type": "object", "properties": properties, "required": required}


def _get_details_tool_schema(*, api_name: str = "") -> dict[str, Any]:
    return {
        "name": "getDetails",
        "description": "Ask the user for missing details.",
        "parameters": {
            "type": "object",
            "properties": {"Question": {"type": "string", "description": "Required. String."}},
            "required": ["Question"],
        },
        "metadata": {"tool_type": "toolalpaca_builtin", "api_name": api_name},
    }


def _normalize_toolalpaca_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name == "data":
        return resolved
    if (resolved / "data").is_dir():
        return resolved / "data"
    return resolved


def _toolalpaca_required_paths(source_root: Path) -> bool:
    return all((source_root / file_name).exists() for file_name in TOOLALPACA_FILES.values())


def _ensure_toolalpaca_cache() -> Path:
    cache_root = Path(os.getenv("HELICOPTER_CACHE_DIR") or "~/.cache/helicopter-eval").expanduser()
    repo_root = cache_root / TOOLALPACA_CACHE_ROOT_NAME
    if _toolalpaca_required_paths(repo_root / "data"):
        return repo_root
    cache_root.mkdir(parents=True, exist_ok=True)
    with _toolalpaca_cache_lock(cache_root):
        if _toolalpaca_required_paths(repo_root / "data"):
            return repo_root
        if not repo_root.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", TOOLALPACA_REPO_URL, str(repo_root)],
                check=True,
            )
        subprocess.run(["git", "-C", str(repo_root), "sparse-checkout", "set", "data"], check=True)
        if not _toolalpaca_required_paths(repo_root / "data"):
            raise FileNotFoundError(f"ToolAlpaca sparse checkout did not create expected files in {repo_root / 'data'}")
    return repo_root


@contextmanager
def _toolalpaca_cache_lock(cache_root: Path, *, timeout_s: float = 600.0) -> Iterator[None]:
    lock_dir = cache_root / f"{TOOLALPACA_CACHE_ROOT_NAME}.lock"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for ToolAlpaca cache lock: {lock_dir}")
            time.sleep(0.25)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except FileNotFoundError:
            pass


def _load_openapi_spec(documentation: str) -> Mapping[str, Any]:
    try:
        spec = json.loads(documentation)
    except json.JSONDecodeError:
        return {}
    return spec if isinstance(spec, Mapping) else {}


def _drop_absent_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _drop_absent_values(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _is_absent(item)
        }
    if isinstance(value, list):
        return [_drop_absent_values(item) for item in value if not _is_absent(item)]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _is_absent(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _coerce_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _stable_seed(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _stable_int(seed: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(digest[:8], 16) % 900000 + 1000


def _stable_id(seed: str, salt: str) -> str:
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return f"{_slug(salt)}_{digest[:10]}"


def _slug(value: str) -> str:
    rendered = []
    for char in str(value).lower():
        rendered.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(rendered).split("_") if part) or "item"


def _normalize_text(value: str) -> str:
    return "\n".join(" ".join(line.split()) for line in str(value).splitlines()).strip()


__all__ = [
    "ToolAlpacaRunConfig",
    "ToolAlpacaSample",
    "ToolAlpacaSandbox",
    "build_prompt",
    "dry_run_summary",
    "evaluate_completion",
    "evaluate_toolalpaca_calls",
    "load_samples",
    "load_toolalpaca_rows_from_source",
    "parse_toolalpaca_action_input",
    "run_toolalpaca",
]
