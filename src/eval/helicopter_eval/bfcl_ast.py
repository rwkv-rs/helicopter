from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.request

from .apibank import decode_tool_calls
from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


BFCL_RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)
BFCL_CATEGORY_PATHS: dict[str, tuple[str, str]] = {
    "simple_python": ("BFCL_v4_simple_python.json", "possible_answer/BFCL_v4_simple_python.json"),
    "multiple": ("BFCL_v4_multiple.json", "possible_answer/BFCL_v4_multiple.json"),
    "exec_simple": (
        "unused_datasets/question/BFCL_v4_exec_simple.json",
        "unused_datasets/possible_answer/BFCL_v4_exec_simple.json",
    ),
    "exec_multiple": (
        "unused_datasets/question/BFCL_v4_exec_multiple.json",
        "unused_datasets/possible_answer/BFCL_v4_exec_multiple.json",
    ),
}


@dataclass(frozen=True, slots=True)
class ToolCallExpectation:
    name: str
    arguments: dict[str, Any]
    argument_options: dict[str, tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class BfclAstSample:
    sample_index: int
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_tool_calls: tuple[ToolCallExpectation, ...]
    category: str


@dataclass(frozen=True, slots=True)
class BfclAstResult:
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
class BfclAstRunConfig:
    base_url: str
    model: str
    benchmark: str
    category: str
    limit: int | None = None
    split: str = "test"
    source_root: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 768
    timeout_s: float = 600.0
    scoreboard_dataset: str | None = None
    job_name: str = "function_bfcl_ast"
    job_id: str | None = None
    runner: str = "helicopter_eval.bfcl_ast"
    cot_mode: str = "CoT"


def load_samples(config: BfclAstRunConfig) -> list[BfclAstSample]:
    if config.split != "test":
        raise ValueError("BFCL v4 AST datasets only provide test split")
    if config.category not in BFCL_CATEGORY_PATHS:
        raise ValueError(f"unknown BFCL AST category: {config.category}")
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")

    question_rel, answer_rel = BFCL_CATEGORY_PATHS[config.category]
    questions = _read_items(config, question_rel)
    answers = {
        str(item.get("id") or item.get("task_id") or ""): item
        for item in _read_items(config, answer_rel)
        if isinstance(item, Mapping)
    }
    samples: list[BfclAstSample] = []
    for index, item in enumerate(questions):
        if config.limit is not None and len(samples) >= int(config.limit):
            break
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or f"{config.category}_{index}")
        answer = answers.get(task_id)
        if answer is None:
            raise ValueError(f"missing BFCL possible-answer entry for {task_id}")
        instruction = _render_bfcl_question(item.get("question"))
        if not instruction:
            raise ValueError(f"BFCL row {task_id!r} is missing question content")
        samples.append(
            BfclAstSample(
                sample_index=len(samples),
                task_id=task_id,
                instruction=instruction,
                tools=tuple(_normalize_tool_schema(tool) for tool in _coerce_list(item.get("function"))),
                expected_tool_calls=tuple(_normalize_ground_truth_calls(answer.get("ground_truth"))),
                category=config.category,
            )
        )
    return samples


def build_prompt(sample: BfclAstSample) -> str:
    tools = json.dumps(sample.tools, ensure_ascii=False, indent=2, sort_keys=True)
    schema = json.dumps(
        {
            "oneOf": [
                {
                    "type": "object",
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                },
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                    },
                    "minItems": 1,
                },
            ]
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are solving a Berkeley Function Calling Leaderboard task.\n\n"
        "Tools:\n"
        f"{tools}\n\n"
        "Output JSON schema:\n"
        f"{schema}\n\n"
        "Return exactly one JSON value that validates against the schema. "
        "For one required call, return one JSON object. "
        "For multiple required calls, return a JSON array in execution order. "
        "Use only listed tool names. Return no prose, no markdown, and no extra text outside the JSON value.\n\n"
        f"User request:\n{sample.instruction}\n\n"
        "Tool call:"
    )


def evaluate_completion(sample: BfclAstSample, completion: str) -> tuple[bool, str, list[dict[str, Any]]]:
    try:
        decoded = decode_tool_calls(completion)
    except Exception as exc:  # noqa: BLE001
        return False, f"parse_error:{exc}", []
    expected = list(sample.expected_tool_calls)
    failure_bits: list[str] = []
    passed_count = 0
    max_len = max(len(expected), len(decoded))
    for index in range(max_len):
        if index >= len(expected):
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(decoded):
            failure_bits.append(f"call_{index}:missing_call")
            continue
        ok, reason = _call_matches_expectation(decoded[index], expected[index])
        if ok:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")
    is_passed = len(decoded) == len(expected) and passed_count == len(expected)
    return bool(is_passed), "; ".join(failure_bits), decoded


def generate_completion(sample: BfclAstSample, config: BfclAstRunConfig) -> BfclAstResult:
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
    passed, fail_reason, decoded = evaluate_completion(sample, completion)
    return BfclAstResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=completion,
        answer=json.dumps(decoded, ensure_ascii=False, sort_keys=True),
        reference_answer=json.dumps(
            [_expectation_payload(item) for item in sample.expected_tool_calls],
            sort_keys=True,
        ),
        is_passed=passed,
        fail_reason=fail_reason,
    )


def evaluate_samples(samples: Sequence[BfclAstSample], config: BfclAstRunConfig) -> list[BfclAstResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: BfclAstRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: BfclAstRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: BfclAstRunConfig) -> dict[str, Any]:
    return {
        "tool": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: BfclAstRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_bfcl_ast",
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[BfclAstResult], *, config: BfclAstRunConfig, repo_root: Path) -> int:
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


def run_bfcl_ast(config: BfclAstRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "category": config.category,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: BfclAstRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "source": "github://ShishirPatil/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
        "split": config.split,
        "category": config.category,
        "limit": config.limit,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


def _read_items(config: BfclAstRunConfig, rel_path: str) -> list[Any]:
    local = _local_source_file(config, rel_path)
    raw = (
        local.read_text(encoding="utf-8")
        if local is not None
        else _cached_url_text(rel_path, timeout_s=config.timeout_s)
    )
    return _read_json_or_jsonl_items(raw)


def _local_source_file(config: BfclAstRunConfig, rel_path: str) -> Path | None:
    candidates: list[Path] = []
    for raw in (
        config.source_root,
        os.getenv("RWKV_BFCL_SMALL_SOURCE_ROOT"),
        os.getenv("RWKV_BFCL_V4_SOURCE_ROOT"),
        os.getenv("BFCL_V4_SOURCE_ROOT"),
    ):
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            Path("/home/chase/GitHub/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"),
            Path("/tmp/rwkv-official-refs/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"),
            Path("/tmp/gorilla-official/berkeley-function-call-leaderboard/bfcl_eval/data"),
        ]
    )
    for root in candidates:
        resolved = _normalize_source_root(root) / rel_path
        if resolved.exists():
            return resolved
    return None


def _normalize_source_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.name == "data" and root.parent.name == "bfcl_eval":
        return root
    candidate = root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    if candidate.is_dir():
        return candidate
    candidate = root / "bfcl_eval" / "data"
    if candidate.is_dir():
        return candidate
    return root


def _cached_url_text(rel_path: str, *, timeout_s: float) -> str:
    cache_root = Path(os.getenv("HELICOPTER_CACHE_DIR") or "~/.cache/helicopter-eval").expanduser()
    cache_path = cache_root / "bfcl-v4" / rel_path
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BFCL_RAW_BASE}/{rel_path}"
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        text = response.read().decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return text


def _read_json_or_jsonl_items(raw: str) -> list[Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ValueError("unsupported BFCL JSON payload")


def _render_bfcl_question(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    turns = _coerce_list(raw)
    parts: list[str] = []
    for turn in turns:
        messages = _coerce_list(turn)
        for message in messages:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "user").strip().lower() or "user"
                content = str(message.get("content") or "").strip()
                if content:
                    parts.append(f"{role.title()}: {content}")
            elif str(message or "").strip():
                parts.append(str(message).strip())
    return "\n".join(parts).strip()


def _normalize_tool_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "name": "unknown_tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else None
    source = function or raw
    parameters = source.get("parameters") or {"type": "object", "properties": {}, "required": []}
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    parameters = dict(parameters)
    if str(parameters.get("type") or "").lower() == "dict":
        parameters["type"] = "object"
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    return {
        "name": str(source.get("name") or raw.get("name") or "unknown_tool"),
        "description": str(source.get("description") or raw.get("description") or ""),
        "parameters": parameters,
    }


def _normalize_ground_truth_calls(raw: Any) -> list[ToolCallExpectation]:
    calls: list[ToolCallExpectation] = []
    for item in _coerce_list(raw):
        if isinstance(item, str):
            name, arguments = _parse_python_call(item)
            calls.append(
                ToolCallExpectation(
                    name=name,
                    arguments=arguments,
                    argument_options={key: (value,) for key, value in arguments.items()},
                )
            )
            continue
        if not isinstance(item, Mapping):
            continue
        if "name" in item:
            calls.append(_normalize_tool_expectation(item))
            continue
        if len(item) != 1:
            continue
        name, raw_options = next(iter(item.items()))
        if not isinstance(raw_options, Mapping):
            raw_options = {}
        argument_options = {
            str(key): tuple(_coerce_list(value) or [value])
            for key, value in raw_options.items()
        }
        arguments = {key: _canonical_option_value(values) for key, values in argument_options.items()}
        calls.append(ToolCallExpectation(name=str(name), arguments=arguments, argument_options=argument_options))
    return calls


def _normalize_tool_expectation(raw: Mapping[str, Any]) -> ToolCallExpectation:
    name = str(raw.get("name") or raw.get("tool_name") or raw.get("function_name") or "").strip()
    arguments = raw.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            arguments = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    argument_options = {str(key): (value,) for key, value in dict(arguments).items()}
    return ToolCallExpectation(name=name, arguments=dict(arguments), argument_options=argument_options)


def _call_matches_expectation(actual: Mapping[str, Any], expected: ToolCallExpectation) -> tuple[bool, str]:
    actual_name = str(actual.get("name") or "").strip()
    if actual_name != expected.name:
        return False, f"name_mismatch(expected={expected.name}, actual={actual_name})"
    arguments = actual.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, "arguments_not_object"
    actual_arguments = dict(arguments)
    for key, options in expected.argument_options.items():
        if key not in actual_arguments:
            if any(_is_absent_option(option) for option in options):
                continue
            return False, f"missing_argument({key})"
        actual_value = actual_arguments[key]
        if not any(_value_matches(actual_value, option) for option in options):
            return False, f"argument_mismatch({key})"
    for key, value in actual_arguments.items():
        if key not in expected.argument_options and not _is_absent_option(value):
            return False, f"unexpected_argument({key})"
    return True, "ok"


def _value_matches(actual: Any, expected: Any) -> bool:
    if _is_absent_option(expected):
        return _is_absent_option(actual)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9
    if isinstance(actual, str) and not isinstance(expected, str):
        parsed = _try_parse_json_scalar(actual)
        if parsed is not actual:
            return _value_matches(parsed, expected)
    if isinstance(expected, str) and not isinstance(actual, str):
        parsed = _try_parse_json_scalar(expected)
        if parsed is not expected:
            return _value_matches(actual, parsed)
    if isinstance(actual, str) and isinstance(expected, str):
        return _normalize_text(actual) == _normalize_text(expected)
    return actual == expected


def _parse_python_call(text: str) -> tuple[str, dict[str, Any]]:
    parsed = ast.parse(str(text).strip(), mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"BFCL ground-truth expression is not a function call: {text}")
    name = _render_ast_call_name(parsed.body.func)
    arguments: dict[str, Any] = {}
    for keyword in parsed.body.keywords:
        if keyword.arg is not None:
            arguments[keyword.arg] = _literal_from_ast(keyword.value)
    return name, arguments


def _render_ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _render_ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.BinOp):
        left = _literal_from_ast(node.left)
        right = _literal_from_ast(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
    return ast.literal_eval(node)


def _canonical_option_value(options: Sequence[Any]) -> Any:
    for option in options:
        if not _is_absent_option(option):
            return option
    return options[0] if options else None


def _is_absent_option(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _try_parse_json_scalar(value: str) -> Any:
    text = value.strip()
    if not text or text[0] not in "[{\"-0123456789tfn":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _expectation_payload(expectation: ToolCallExpectation) -> dict[str, Any]:
    return {
        "name": expectation.name,
        "arguments": dict(expectation.arguments),
        "argument_options": {key: list(value) for key, value in expectation.argument_options.items()},
    }


def _coerce_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


__all__ = [
    "BfclAstRunConfig",
    "BfclAstSample",
    "build_prompt",
    "dry_run_summary",
    "evaluate_completion",
    "load_samples",
    "run_bfcl_ast",
]
