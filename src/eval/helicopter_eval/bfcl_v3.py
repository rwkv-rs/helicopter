from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import uuid

from .apibank import decode_tool_calls
from .bfcl_ast import _coerce_list, _normalize_tool_schema
from .bfcl_exec import render_bfcl_exec_call
from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


BFCL_ADDITIONAL_FUNCTION_PROMPT = "I have updated some more functions you can choose from. What about now?"
BFCL_SOURCE_ROOT_ENV_VARS = (
    "RWKV_BFCL_V3_SOURCE_ROOT",
    "BFCL_V3_SOURCE_ROOT",
    "RWKV_BFCL_V3_ROOT",
    "BFCL_V3_ROOT",
    "RWKV_BFCL_V3_SOURCE",
    "BFCL_V3_SOURCE",
)
BFCL_OFFICIAL_ROOT_ENV_VARS = ("RWKV_BFCL_OFFICIAL_ROOT", "BFCL_OFFICIAL_ROOT")
BFCL_POSSIBLE_ANSWER_ROOT_ENV_VARS = ("RWKV_BFCL_POSSIBLE_ANSWER_ROOT", "BFCL_POSSIBLE_ANSWER_ROOT")
BFCL_FUNC_DOC_ROOT_ENV_VARS = ("RWKV_BFCL_FUNC_DOC_ROOT", "BFCL_FUNC_DOC_ROOT")
BFCL_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


@dataclass(frozen=True, slots=True)
class BfclV3Category:
    name: str
    v3_question_file: str
    official_question_rel: str
    official_answer_rel: str


BFCL_V3_CATEGORIES: tuple[BfclV3Category, ...] = (
    BfclV3Category(
        "multi_turn_base",
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v4_multi_turn_base.json",
        "possible_answer/BFCL_v4_multi_turn_base.json",
    ),
    BfclV3Category(
        "multi_turn_composite",
        "BFCL_v3_multi_turn_composite.json",
        "unused_datasets/question/BFCL_v4_multi_turn_composite.json",
        "unused_datasets/possible_answer/BFCL_v4_multi_turn_composite.json",
    ),
    BfclV3Category(
        "multi_turn_long_context",
        "BFCL_v3_multi_turn_long_context.json",
        "BFCL_v4_multi_turn_long_context.json",
        "possible_answer/BFCL_v4_multi_turn_long_context.json",
    ),
    BfclV3Category(
        "multi_turn_miss_func",
        "BFCL_v3_multi_turn_miss_func.json",
        "BFCL_v4_multi_turn_miss_func.json",
        "possible_answer/BFCL_v4_multi_turn_miss_func.json",
    ),
    BfclV3Category(
        "multi_turn_miss_param",
        "BFCL_v3_multi_turn_miss_param.json",
        "BFCL_v4_multi_turn_miss_param.json",
        "possible_answer/BFCL_v4_multi_turn_miss_param.json",
    ),
)


@dataclass(frozen=True, slots=True)
class BfclV3Turn:
    messages: tuple[dict[str, str], ...]
    ground_truth: tuple[str, ...]
    tool_additions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class BfclV3Sample:
    sample_index: int
    task_id: str
    category: str
    turns: tuple[BfclV3Turn, ...]
    tools: tuple[dict[str, Any], ...]
    initial_config: dict[str, Any]
    involved_classes: tuple[str, ...]
    source_path: str
    official_root: str


@dataclass(frozen=True, slots=True)
class BfclV3Result:
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
class BfclV3RunConfig:
    base_url: str
    model: str
    benchmark: str
    limit: int | None = None
    split: str = "test"
    source_root: str | None = None
    official_root: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout_s: float = 600.0
    max_turns: int = 20
    scoreboard_dataset: str | None = None
    job_name: str = "function_bfcl_v3"
    job_id: str | None = None
    runner: str = "helicopter_eval.bfcl_v3"
    cot_mode: str = "CoT"


def load_samples(config: BfclV3RunConfig) -> list[BfclV3Sample]:
    if config.split != "test":
        raise ValueError("BFCL v3 only provides test split")
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")

    sources = _resolve_category_sources(config)
    samples: list[BfclV3Sample] = []
    for category, source_path, answer_path, official_root in sources:
        answers = _load_possible_answer_lookup(answer_path)
        func_doc_root = _resolve_func_doc_root(config, official_root=official_root, source_path=source_path)
        for item in _read_json_or_jsonl_items(source_path):
            if config.limit is not None and len(samples) >= int(config.limit):
                return samples
            if not isinstance(item, Mapping):
                continue
            task_id = str(item.get("id") or item.get("task_id") or f"bfcl_v3_{len(samples):04d}")
            question_turns = _normalize_question_turns(item.get("question"))
            ground_truth_turns = answers.get(task_id)
            if ground_truth_turns is None:
                raise ValueError(f"missing BFCL v3 possible-answer entry for {task_id}")
            involved_classes = tuple(
                str(value).strip()
                for value in _coerce_list(item.get("involved_classes") or item.get("classes"))
                if str(value).strip()
            )
            all_tools = _load_tools_for_classes(func_doc_root, involved_classes)
            holdout_by_turn = _normalize_holdout_functions(item.get("missed_function"))
            holdout_names = {name for names in holdout_by_turn.values() for name in names}
            initial_tools = tuple(tool for tool in all_tools if str(tool.get("name") or "") not in holdout_names)
            tools_by_name = {str(tool.get("name") or ""): tool for tool in all_tools}
            turn_count = max(len(question_turns), len(ground_truth_turns), max(holdout_by_turn.keys(), default=-1) + 1)
            turns: list[BfclV3Turn] = []
            for turn_index in range(turn_count):
                messages = question_turns[turn_index] if turn_index < len(question_turns) else ()
                ground_truth = ground_truth_turns[turn_index] if turn_index < len(ground_truth_turns) else ()
                additions = tuple(
                    tools_by_name[name]
                    for name in holdout_by_turn.get(turn_index, ())
                    if name in tools_by_name
                )
                turns.append(BfclV3Turn(messages=messages, ground_truth=ground_truth, tool_additions=additions))
            samples.append(
                BfclV3Sample(
                    sample_index=len(samples),
                    task_id=task_id,
                    category=category.name,
                    turns=tuple(turns),
                    tools=initial_tools,
                    initial_config=dict(item.get("initial_config") or {}),
                    involved_classes=involved_classes,
                    source_path=str(source_path),
                    official_root=str(official_root),
                )
            )
    return samples


def build_prompt(sample: BfclV3Sample, *, turn_index: int, active_tools: Sequence[Mapping[str, Any]], history: Sequence[Mapping[str, str]]) -> str:
    tool_payload = [
        {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": tool.get("parameters") if isinstance(tool.get("parameters"), Mapping) else {},
        }
        for tool in active_tools
    ]
    schema = {
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
            },
        ]
    }
    history_lines = "\n".join(f"{item['role'].title()}: {item['content']}" for item in history)
    user_lines = "\n".join(
        f"{message.get('role', 'user').title()}: {message.get('content', '')}"
        for message in sample.turns[turn_index].messages
        if str(message.get("content") or "").strip()
    )
    if not user_lines and sample.turns[turn_index].tool_additions:
        user_lines = f"User: {BFCL_ADDITIONAL_FUNCTION_PROMPT}"
    if not user_lines:
        user_lines = "User: No new user request. Return [] if no tool call is needed."
    return (
        "You are solving a Berkeley Function Calling Leaderboard v3 multi-turn task.\n\n"
        "Available tools:\n"
        f"{json.dumps(tool_payload, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "Output JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "Return only JSON. Use one object for one tool call, an array for multiple tool calls, or [] when no tool call is needed. "
        "Do not include markdown or prose outside the JSON value.\n\n"
        f"Conversation so far:\n{history_lines or '(empty)'}\n\n"
        f"Current turn {turn_index + 1}/{len(sample.turns)}:\n{user_lines}\n\n"
        "Tool call JSON:"
    )


def evaluate_sample(sample: BfclV3Sample, config: BfclV3RunConfig) -> BfclV3Result:
    _load_official_runtime(sample.official_root)
    active_tools = list(sample.tools)
    history: list[dict[str, str]] = []
    decoded_turns: list[list[list[str]]] = []
    completions: list[dict[str, Any]] = []
    prompts: list[str] = []
    tool_addition_count = 0

    for turn_index, turn in enumerate(sample.turns[: max(0, int(config.max_turns))]):
        if turn.tool_additions:
            active_tools = _merge_tools(active_tools, turn.tool_additions)
            tool_addition_count += len(turn.tool_additions)
        for message in turn.messages:
            content = str(message.get("content") or "").strip()
            if content:
                history.append({"role": str(message.get("role") or "user"), "content": content})
        if not turn.messages and turn.tool_additions:
            history.append({"role": "user", "content": BFCL_ADDITIONAL_FUNCTION_PROMPT})
        prompt = build_prompt(sample, turn_index=turn_index, active_tools=active_tools, history=history)
        prompts.append(prompt)
        completion = chat_completion(
            base_url=config.base_url,
            model=config.model,
            prompt=prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
        )
        try:
            calls = _decode_bfcl_v3_calls(completion)
            rendered_calls = [render_bfcl_exec_call(call) for call in calls]
            parse_error = ""
        except Exception as exc:  # noqa: BLE001
            calls = []
            rendered_calls = []
            parse_error = f"parse_error:{exc}"
        decoded_turns.append([rendered_calls] if rendered_calls else [])
        completions.append(
            {
                "turn_index": turn_index,
                "completion": completion,
                "decoded_calls": rendered_calls,
                "parse_error": parse_error,
            }
        )
        if parse_error:
            history.append({"role": "assistant", "content": completion})
        elif rendered_calls:
            history.append({"role": "assistant", "content": json.dumps(calls, ensure_ascii=False, sort_keys=True)})
        else:
            history.append({"role": "assistant", "content": "[]"})

    checker = _json_safe(_check_official(sample, decoded_turns))
    parse_errors = [str(item["parse_error"]) for item in completions if item.get("parse_error")]
    valid = bool(checker.get("valid", False)) and not parse_errors
    failure_bits: list[str] = []
    if parse_errors:
        failure_bits.extend(parse_errors)
    if not checker.get("valid", False):
        failure_bits.append(str(checker.get("error_type") or checker.get("error_message") or "official_checker_failed"))
    answer = {
        "decoded_turns": decoded_turns,
        "checker": checker,
        "tool_addition_count": tool_addition_count,
    }
    reference = {
        "ground_truth_turns": [list(turn.ground_truth) for turn in sample.turns],
        "involved_classes": list(sample.involved_classes),
        "source_path": sample.source_path,
    }
    return BfclV3Result(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=json.dumps(prompts, ensure_ascii=False),
        completion=json.dumps(completions, ensure_ascii=False),
        answer=json.dumps(answer, ensure_ascii=False, sort_keys=True),
        reference_answer=json.dumps(reference, ensure_ascii=False, sort_keys=True),
        is_passed=valid,
        fail_reason="; ".join(bit for bit in failure_bits if bit),
    )


def evaluate_samples(samples: Sequence[BfclV3Sample], config: BfclV3RunConfig) -> list[BfclV3Result]:
    return [evaluate_sample(sample, config) for sample in samples]


def scoreboard_dataset_name(config: BfclV3RunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: BfclV3RunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: BfclV3RunConfig) -> dict[str, Any]:
    return {
        "tool": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: BfclV3RunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_bfcl_v3_multi_turn",
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[BfclV3Result], *, config: BfclV3RunConfig, repo_root: Path) -> int:
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


def run_bfcl_v3(config: BfclV3RunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: BfclV3RunConfig) -> dict[str, Any]:
    sources = _resolve_category_sources(config)
    return {
        "benchmark": config.benchmark,
        "source": "github://ShishirPatil/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
        "split": config.split,
        "limit": config.limit,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
        "source_files": [str(source_path) for _category, source_path, _answer_path, _official_root in sources],
        "possible_answer_files": [str(answer_path) for _category, _source_path, answer_path, _official_root in sources],
        "official_roots": sorted({str(official_root) for _category, _source_path, _answer_path, official_root in sources}),
    }


def _resolve_category_sources(config: BfclV3RunConfig) -> tuple[tuple[BfclV3Category, Path, Path, Path], ...]:
    results: list[tuple[BfclV3Category, Path, Path, Path]] = []
    roots = _source_root_candidates(config)
    official_roots = _official_root_candidates(config)
    for category in BFCL_V3_CATEGORIES:
        source_path: Path | None = None
        source_official_root: Path | None = None
        for root in roots:
            candidate, official_root = _category_source_candidate(root, category)
            if candidate is not None and candidate.is_file():
                source_path = candidate.resolve()
                source_official_root = official_root
                break
        if source_path is None:
            raise FileNotFoundError(
                f"BFCL v3 source file not found for {category.name}; set RWKV_BFCL_V3_SOURCE_ROOT or BFCL_V3_SOURCE_ROOT"
            )

        official_root = source_official_root or _resolve_official_root_for_path(
            config,
            source_path=source_path,
            explicit_roots=official_roots,
        )
        if official_root is None:
            raise FileNotFoundError(
                "BFCL official checkout not found; set RWKV_BFCL_OFFICIAL_ROOT or BFCL_OFFICIAL_ROOT"
            )
        answer_path = _resolve_possible_answer_file(config, category, source_path=source_path, official_root=official_root)
        results.append((category, source_path, answer_path, official_root))
    return tuple(results)


def _source_root_candidates(config: BfclV3RunConfig) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for raw in (config.source_root, *(_env_value(name) for name in BFCL_SOURCE_ROOT_ENV_VARS)):
        if raw:
            candidates.append(Path(str(raw)).expanduser())
    candidates.extend(
        [
            Path("data/bfcl_v3_raw"),
            Path("/home/chase/GitHub/helicopter/data/bfcl_v3_raw"),
            Path("/home/chase/GitHub/rwkv-skills/data/bfcl_v3_raw"),
            Path("/home/chase/GitHub/rwkv-skills/references/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"),
        ]
    )
    return _dedupe_paths(candidates)


def _official_root_candidates(config: BfclV3RunConfig) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for raw in (config.official_root, *(_env_value(name) for name in BFCL_OFFICIAL_ROOT_ENV_VARS)):
        if raw:
            candidates.append(Path(str(raw)).expanduser())
    candidates.extend(
        [
            Path("/home/chase/GitHub/helicopter/references/gorilla/berkeley-function-call-leaderboard"),
            Path("/home/chase/GitHub/rwkv-skills/references/gorilla/berkeley-function-call-leaderboard"),
            Path("/tmp/gorilla-official/berkeley-function-call-leaderboard"),
        ]
    )
    return _dedupe_paths(candidates)


def _category_source_candidate(root: Path, category: BfclV3Category) -> tuple[Path | None, Path | None]:
    if root.is_file() and root.name in {category.v3_question_file, Path(category.official_question_rel).name}:
        return root, _candidate_official_root(root)
    normalized = _normalize_source_root(root)
    v3_candidate = normalized / category.v3_question_file
    if v3_candidate.is_file():
        return v3_candidate, None
    official_root = _candidate_official_root(root)
    official_data = _official_data_root(official_root) if official_root is not None else normalized
    official_candidate = official_data / category.official_question_rel
    if official_candidate.is_file():
        return official_candidate, official_root
    return None, None


def _normalize_source_root(path: Path) -> Path:
    root = path.expanduser()
    if root.name == "data" and root.parent.name == "bfcl_eval":
        return root
    for candidate in (
        root / "bfcl_eval" / "data",
        root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
    ):
        if candidate.is_dir():
            return candidate
    return root


def _resolve_official_root_for_path(
    config: BfclV3RunConfig,
    *,
    source_path: Path,
    explicit_roots: Sequence[Path],
) -> Path | None:
    for candidate in list(explicit_roots) + [source_path, *source_path.parents]:
        official_root = _candidate_official_root(candidate)
        if official_root is not None:
            return official_root.resolve()
    return None


def _candidate_official_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    current = path.expanduser()
    candidates = [current, *current.parents[:5]]
    for candidate in candidates:
        if (candidate / "bfcl_eval").is_dir():
            return candidate
        if candidate.name == "bfcl_eval" and candidate.parent.is_dir():
            return candidate.parent
    return None


def _official_data_root(official_root: Path) -> Path:
    return official_root / "bfcl_eval" / "data"


def _resolve_possible_answer_file(
    config: BfclV3RunConfig,
    category: BfclV3Category,
    *,
    source_path: Path,
    official_root: Path,
) -> Path:
    explicit_roots = [
        Path(raw).expanduser()
        for raw in (_env_value(name) for name in BFCL_POSSIBLE_ANSWER_ROOT_ENV_VARS)
        if raw
    ]
    search_files: list[Path] = []
    for root in explicit_roots:
        search_files.extend(
            [
                root / category.v3_question_file,
                root / Path(category.official_answer_rel).name,
                root / category.official_answer_rel,
            ]
        )
    for base in [source_path.parent, *source_path.parents[:4]]:
        search_files.extend(
            [
                base / "bfcl_support" / "possible_answer_v3" / category.v3_question_file,
                base / "possible_answer_v3" / category.v3_question_file,
                base / "possible_answer" / Path(category.official_answer_rel).name,
            ]
        )
    search_files.append(_official_data_root(official_root) / category.official_answer_rel)
    for candidate in search_files:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"BFCL v3 possible-answer file not found for {category.name}")


def _resolve_func_doc_root(config: BfclV3RunConfig, *, official_root: Path, source_path: Path) -> Path:
    candidates = [
        Path(raw).expanduser()
        for raw in (_env_value(name) for name in BFCL_FUNC_DOC_ROOT_ENV_VARS)
        if raw
    ]
    for base in [source_path.parent, *source_path.parents[:4]]:
        candidates.extend([base / "bfcl_support" / "multi_turn_func_doc", base / "multi_turn_func_doc"])
    candidates.append(_official_data_root(official_root) / "multi_turn_func_doc")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("BFCL v3 multi_turn_func_doc support directory not found")


def _read_json_or_jsonl_items(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ValueError(f"unsupported BFCL v3 JSON payload in {path}")


def _load_possible_answer_lookup(path: Path) -> dict[str, tuple[tuple[str, ...], ...]]:
    lookup: dict[str, tuple[tuple[str, ...], ...]] = {}
    for item in _read_json_or_jsonl_items(path):
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or "").strip()
        if not task_id:
            continue
        turns: list[tuple[str, ...]] = []
        for turn in _coerce_list(item.get("ground_truth")):
            turns.append(tuple(str(call).strip() for call in _coerce_list(turn) if str(call).strip()))
        lookup[task_id] = tuple(turns)
    return lookup


def _normalize_question_turns(raw: Any) -> tuple[tuple[dict[str, str], ...], ...]:
    turns: list[tuple[dict[str, str], ...]] = []
    for raw_turn in _coerce_list(raw):
        messages: list[dict[str, str]] = []
        for message in _coerce_list(raw_turn):
            if not isinstance(message, Mapping):
                continue
            content = str(message.get("content") or "").strip()
            if content:
                messages.append({"role": str(message.get("role") or "user"), "content": content})
        turns.append(tuple(messages))
    return tuple(turns)


def _load_tools_for_classes(func_doc_root: Path, involved_classes: Sequence[str]) -> tuple[dict[str, Any], ...]:
    tools: list[dict[str, Any]] = []
    missing: list[str] = []
    for class_name in involved_classes:
        filename = BFCL_FUNC_DOC_FILE_MAPPING.get(class_name)
        if not filename:
            missing.append(class_name)
            continue
        path = func_doc_root / filename
        if not path.is_file():
            missing.append(class_name)
            continue
        tools.extend(_normalize_tool_schema(item) for item in _read_json_or_jsonl_items(path))
    if missing:
        raise FileNotFoundError(f"missing BFCL v3 function docs for classes: {', '.join(missing)}")
    return tuple(_dedupe_tools(tools))


def _normalize_holdout_functions(raw: Any) -> dict[int, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        return {}
    normalized: dict[int, tuple[str, ...]] = {}
    for key, value in raw.items():
        try:
            turn_index = int(key)
        except (TypeError, ValueError):
            continue
        names = tuple(str(item).strip() for item in _coerce_list(value) if str(item).strip())
        if names:
            normalized[turn_index] = names
    return normalized


def _decode_bfcl_v3_calls(completion: str) -> list[dict[str, Any]]:
    text = completion.strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if text == "[]":
        return []
    return decode_tool_calls(text)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _check_official(sample: BfclV3Sample, decoded_turns: list[list[list[str]]]) -> dict[str, Any]:
    runtime = _load_official_runtime(sample.official_root)
    expected_turns = [list(turn.ground_truth) for turn in sample.turns]
    while len(decoded_turns) < len(expected_turns):
        decoded_turns.append([])
    irrelevance_checker = getattr(runtime, "multi_turn_irrelevance_checker", None)
    if irrelevance_checker is not None:
        irrelevance = irrelevance_checker(decoded_turns, expected_turns)
        if not bool(irrelevance.get("valid", False)):
            return dict(irrelevance)
    return dict(
        runtime.multi_turn_checker(
            decoded_turns,
            expected_turns,
            {
                "id": sample.task_id,
                "initial_config": dict(sample.initial_config),
                "involved_classes": list(sample.involved_classes),
            },
            "multi_turn",
            f"helicopter_bfcl_{sample.sample_index}_{uuid.uuid4().hex}",
        )
    )


def _load_official_runtime(official_root: str | Path) -> Any:
    root = Path(official_root).expanduser().resolve()
    if not (root / "bfcl_eval").is_dir():
        raise FileNotFoundError(f"BFCL official root is invalid: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
    return module


def _merge_tools(current: Sequence[Mapping[str, Any]], additions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {str(tool.get("name") or ""): dict(tool) for tool in current}
    for tool in additions:
        name = str(tool.get("name") or "")
        if name:
            merged[name] = dict(tool)
    return [tool for name, tool in merged.items() if name]


def _dedupe_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if name and name not in deduped:
            deduped[name] = dict(tool)
    return list(deduped.values())


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value and value.strip() else None


def _dedupe_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        key = expanded.resolve() if expanded.exists() else expanded
        if key not in deduped:
            deduped.append(key)
    return tuple(deduped)


__all__ = [
    "BfclV3RunConfig",
    "BfclV3Sample",
    "build_prompt",
    "dry_run_summary",
    "evaluate_sample",
    "load_samples",
    "run_bfcl_v3",
]
