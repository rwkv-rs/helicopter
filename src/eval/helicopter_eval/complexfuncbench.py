from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import urllib.request
from typing import Any

from .apibank import decode_tool_calls
from .openai_client import chat_completion
from .sampling import apply_limit_or_sample, dataset_sample_suffix
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


COMPLEXFUNCBENCH_SOURCE_URL = (
    "https://huggingface.co/datasets/zai-org/ComplexFuncBench/resolve/main/ComplexFuncBench.jsonl"
)
COMPLEXFUNCBENCH_FINAL_TOOL = {
    "name": "final_answer",
    "description": "Finish the ComplexFuncBench task with the final natural language response.",
    "parameters": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
}


@dataclass(frozen=True, slots=True)
class ComplexFuncBenchSample:
    sample_index: int
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_turns: tuple[tuple[dict[str, Any], ...], ...]
    observations: tuple[Any, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ComplexFuncBenchResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
    details: dict[str, Any] | None = None

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
class ComplexFuncBenchRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    source_path: str | None = None
    source_url: str = COMPLEXFUNCBENCH_SOURCE_URL
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout_s: float = 600.0
    max_steps: int = 8
    scoreboard_dataset: str | None = None
    job_name: str = "function_complexfuncbench"
    job_id: str | None = None
    runner: str = "helicopter_eval.complexfuncbench"
    cot_mode: str = "CoT"


def load_samples(config: ComplexFuncBenchRunConfig) -> list[ComplexFuncBenchSample]:
    path = Path(config.source_path).expanduser() if config.source_path else _ensure_source_cache(config.source_url)
    samples: list[ComplexFuncBenchSample] = []
    for row in _iter_jsonl(path):
        if not isinstance(row, Mapping):
            continue
        sample = sample_from_row(row, sample_index=len(samples), dataset_name=config.dataset_name)
        if sample is None:
            continue
        samples.append(sample)
        if config.limit is not None and config.sample_size is None and len(samples) >= config.limit:
            break
    return apply_limit_or_sample(
        samples,
        limit=config.limit,
        sample_size=config.sample_size,
        sample_seed=config.sample_seed,
        sort_key=lambda sample: sample.sample_index,
    )


def sample_from_row(
    row: Mapping[str, Any],
    *,
    sample_index: int,
    dataset_name: str = "complexfuncbench_official",
) -> ComplexFuncBenchSample | None:
    conversations = [dict(item) for item in _coerce_list(row.get("conversations")) if isinstance(item, Mapping)]
    tools = [dict(item) for item in _coerce_list(row.get("tools") or row.get("functions")) if isinstance(item, Mapping)]
    if not conversations or not tools:
        return None
    instruction = _first_user_message(conversations)
    expected_turns, observations = _conversation_turns(conversations)
    if not instruction or not expected_turns:
        return None
    if not any(str(tool.get("name") or "") == COMPLEXFUNCBENCH_FINAL_TOOL["name"] for tool in tools):
        tools = [*tools, dict(COMPLEXFUNCBENCH_FINAL_TOOL)]
    official_id = str(row.get("id") or row.get("task_id") or f"{sample_index}")
    task_id = f"{dataset_name}__{official_id}"
    return ComplexFuncBenchSample(
        sample_index=sample_index,
        task_id=task_id,
        instruction=instruction,
        tools=tuple(tools),
        expected_turns=tuple(tuple(dict(call) for call in turn) for turn in expected_turns),
        observations=tuple(observations),
        metadata={
            "official_id": official_id,
            "source": "zai-org/ComplexFuncBench",
            "runtime": "local_golden_conversation",
            "total_turn_num": len(expected_turns),
            "total_call_num": sum(len(turn) for turn in expected_turns),
        },
    )


def render_prompt(sample: ComplexFuncBenchSample) -> str:
    return render_turn_prompt(sample, history=[], current_observation=sample.instruction)


def render_turn_prompt(
    sample: ComplexFuncBenchSample,
    *,
    history: Sequence[Mapping[str, Any]],
    current_observation: Any,
    require_final_answer: bool = False,
) -> str:
    trajectory = "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in history)
    if not trajectory:
        trajectory = "[]"
    next_action = (
        'Now call {"name":"final_answer","arguments":{"answer":"..."}}.'
        if require_final_answer
        else "Predict the next required tool call turn."
    )
    return (
        "You are running a ComplexFuncBench function-calling task.\n"
        "Return only JSON, with no markdown and no extra text.\n"
        'For one call, use {"name":"ToolName","arguments":{"arg":"value"}}.\n'
        "For multiple calls in the same assistant turn, return a JSON array of those objects.\n"
        'After all required tool calls, call {"name":"final_answer","arguments":{"answer":"..."}}.\n\n'
        f"User request:\n{sample.instruction}\n\n"
        f"Available tools:\n{json.dumps(list(sample.tools), ensure_ascii=False, indent=2)}\n\n"
        f"Trajectory so far:\n{trajectory}\n\n"
        f"Current observation:\n{json.dumps(current_observation, ensure_ascii=False)}\n\n"
        f"{next_action}\n\n"
        "Assistant JSON:"
    )


def evaluate_completion(sample: ComplexFuncBenchSample, completion: str) -> tuple[bool, str, dict[str, Any]]:
    try:
        calls = decode_tool_calls(completion)
    except Exception as exc:  # noqa: BLE001
        return False, f"parse_error: {exc}", _details(sample, correct_call_num=0, finish_reason="parse_error")
    if not calls:
        return False, "empty_tool_calls", _details(sample, correct_call_num=0, finish_reason="empty_tool_calls")
    if _is_single_final_answer(calls):
        return False, "final_answer_before_tool_calls", _details(sample, correct_call_num=0, finish_reason="stop_early")

    flattened_expected = [call for turn in sample.expected_turns for call in turn]
    flattened_actual = [dict(call) for call in calls if str(call.get("name") or "") != COMPLEXFUNCBENCH_FINAL_TOOL["name"]]
    matched, reason = _match_call_sequence(flattened_actual, flattened_expected)
    details = _details(
        sample,
        correct_call_num=len(flattened_expected) if matched else _matching_prefix_len(flattened_actual, flattened_expected),
        finish_reason="matched" if matched else "mismatch",
    )
    return matched, "" if matched else reason, details


def evaluate_samples(
    samples: Sequence[ComplexFuncBenchSample],
    config: ComplexFuncBenchRunConfig,
) -> list[ComplexFuncBenchResult]:
    results: list[ComplexFuncBenchResult] = []
    for sample in samples:
        results.append(evaluate_sample(sample, config))
    return results


def evaluate_sample(sample: ComplexFuncBenchSample, config: ComplexFuncBenchRunConfig) -> ComplexFuncBenchResult:
    history: list[dict[str, Any]] = []
    prompts: list[str] = []
    completions: list[str] = []
    correct_call_num = 0
    fail_reason = ""
    current_observation: Any = sample.instruction
    max_steps = max(1, int(config.max_steps))

    for turn_index, expected_turn in enumerate(sample.expected_turns):
        if turn_index >= max_steps:
            fail_reason = "max_steps_before_all_turns"
            break
        prompt = render_turn_prompt(sample, history=history, current_observation=current_observation)
        completion = _request_completion(config, prompt)
        prompts.append(prompt)
        completions.append(completion)
        try:
            calls = decode_tool_calls(completion)
        except Exception as exc:  # noqa: BLE001
            fail_reason = f"parse_error: {exc}"
            break
        matched, reason = _match_call_sequence(calls, expected_turn)
        history.append(
            {
                "turn": turn_index,
                "assistant": _json_safe(calls),
                "matched": matched,
                "reason": reason,
            }
        )
        if not matched:
            fail_reason = reason
            break
        correct_call_num += len(expected_turn)
        current_observation = sample.observations[turn_index] if turn_index < len(sample.observations) else []
        history.append({"turn": turn_index, "observation": _json_safe(current_observation)})

    all_calls_matched = not fail_reason and correct_call_num == int(sample.metadata.get("total_call_num") or 0)
    final_answer = ""
    if all_calls_matched and len(prompts) < max_steps:
        prompt = render_turn_prompt(
            sample,
            history=history,
            current_observation=current_observation,
            require_final_answer=True,
        )
        completion = _request_completion(config, prompt)
        prompts.append(prompt)
        completions.append(completion)
        try:
            calls = decode_tool_calls(completion)
        except Exception as exc:  # noqa: BLE001
            fail_reason = f"final_answer_parse_error: {exc}"
            calls = []
        if _is_single_final_answer(calls):
            arguments = calls[0].get("arguments")
            if isinstance(arguments, Mapping):
                final_answer = str(arguments.get("answer") or arguments.get("response") or "").strip()
            if not final_answer:
                fail_reason = "empty_final_answer"
        elif not fail_reason:
            fail_reason = "missing_final_answer"
    elif all_calls_matched:
        fail_reason = "max_steps_before_final_answer"

    passed = bool(all_calls_matched and final_answer and not fail_reason)
    details = _details(
        sample,
        correct_call_num=correct_call_num,
        finish_reason="matched" if passed else (fail_reason or "failed"),
    )
    details["final_answer"] = final_answer
    return ComplexFuncBenchResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=json.dumps(prompts, ensure_ascii=False),
        completion=json.dumps(completions, ensure_ascii=False),
        answer=json.dumps(
            {"history": history, "final_answer": final_answer, "details": details},
            ensure_ascii=False,
            sort_keys=True,
        ),
        reference_answer=json.dumps(
            {"expected_tool_turns": sample.expected_turns},
            ensure_ascii=False,
            sort_keys=True,
        ),
        is_passed=passed,
        fail_reason="" if passed else fail_reason,
        details=details,
    )


def scoreboard_dataset_name(config: ComplexFuncBenchRunConfig) -> str:
    base = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit:
        base = f"{base}_limit{config.limit}"
    return base + dataset_sample_suffix(sample_size=config.sample_size, sample_seed=config.sample_seed)


def job_id(config: ComplexFuncBenchRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: ComplexFuncBenchRunConfig) -> dict[str, Any]:
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_new_tokens": config.max_tokens,
    }


def task_sampling_config(config: ComplexFuncBenchRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_complexfuncbench_local",
        "execution_backend": "local_golden_conversation",
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "sampling_config": {"tool": completion_sampling_config(config)},
    }


def write_results(
    results: Sequence[ComplexFuncBenchResult],
    *,
    config: ComplexFuncBenchRunConfig,
    repo_root: Path,
) -> int:
    total_calls = 0
    correct_calls = 0
    for result in results:
        try:
            payload = json.loads(result.answer)
        except json.JSONDecodeError:
            continue
        details = payload.get("details") if isinstance(payload, Mapping) else None
        if isinstance(details, Mapping):
            total_calls += int(details.get("total_call_num") or 0)
            correct_calls += int(details.get("correct_call_num") or 0)
    call_accuracy = correct_calls / total_calls if total_calls else 0.0
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
                extra_metrics={"call_accuracy": call_accuracy},
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_complexfuncbench(config: ComplexFuncBenchRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "source": "hf://zai-org/ComplexFuncBench",
        "runtime": "local_golden_conversation",
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: ComplexFuncBenchRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "source": "hf://zai-org/ComplexFuncBench",
        "source_url": config.source_url,
        "split": config.split,
        "limit": config.limit,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
        "runtime": "local_golden_conversation",
        "metric": "tool_call_sequence_exact_match",
    }


def _conversation_turns(conversations: Sequence[Mapping[str, Any]]) -> tuple[list[list[dict[str, Any]]], list[Any]]:
    turns: list[list[dict[str, Any]]] = []
    observations: list[Any] = []
    for item in conversations:
        raw_calls = item.get("function_call")
        if raw_calls is not None:
            calls = []
            for call in _coerce_list(raw_calls):
                if not isinstance(call, Mapping):
                    continue
                name = str(call.get("name") or "").strip()
                arguments = call.get("arguments")
                calls.append({"name": name, "arguments": dict(arguments) if isinstance(arguments, Mapping) else {}})
            if calls:
                turns.append(calls)
        elif str(item.get("role") or "") == "observation":
            observations.append(item.get("content"))
    return turns, observations


def _first_user_message(conversations: Sequence[Mapping[str, Any]]) -> str:
    for item in conversations:
        if str(item.get("role") or "") == "user":
            return str(item.get("content") or "").strip()
    return ""


def _match_call_sequence(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    if len(actual) != len(expected):
        return False, f"call_count_mismatch(expected={len(expected)}, actual={len(actual)})"
    for index, (actual_call, expected_call) in enumerate(zip(actual, expected, strict=True)):
        if str(actual_call.get("name") or "") != str(expected_call.get("name") or ""):
            return False, f"name_mismatch(index={index})"
        actual_args = actual_call.get("arguments")
        expected_args = expected_call.get("arguments")
        if not isinstance(actual_args, Mapping):
            return False, f"arguments_not_object(index={index})"
        if not isinstance(expected_args, Mapping):
            expected_args = {}
        matched, reason = _match_arguments(actual_args, expected_args)
        if not matched:
            return False, f"{reason}(index={index})"
    return True, ""


def _match_arguments(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[bool, str]:
    actual_keys = {str(key) for key in actual}
    expected_keys = {str(key) for key in expected}
    if actual_keys != expected_keys:
        return False, "argument_keys_mismatch"
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if not _value_matches(actual_value, expected_value):
            return False, f"argument_mismatch({key})"
    return True, ""


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9
    if isinstance(actual, str) and not isinstance(expected, str):
        parsed = _parse_json_scalar(actual)
        if parsed is not actual:
            return _value_matches(parsed, expected)
    if isinstance(expected, str) and not isinstance(actual, str):
        parsed = _parse_json_scalar(expected)
        if parsed is not expected:
            return _value_matches(actual, parsed)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    if isinstance(actual, Sequence) and isinstance(expected, Sequence) and not isinstance(actual, (str, bytes)):
        if len(actual) != len(expected):
            return False
        return all(_value_matches(left, right) for left, right in zip(actual, expected, strict=True))
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return _match_arguments(actual, expected)[0]
    return actual == expected


def _matching_prefix_len(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for actual_call, expected_call in zip(actual, expected, strict=False):
        matched, _reason = _match_call_sequence([actual_call], [expected_call])
        if not matched:
            break
        count += 1
    return count


def _details(sample: ComplexFuncBenchSample, *, correct_call_num: int, finish_reason: str) -> dict[str, Any]:
    total_call_num = int(sample.metadata.get("total_call_num") or 0)
    return {
        "finish_reason": finish_reason,
        "runtime": "local_golden_conversation",
        "correct_call_num": correct_call_num,
        "total_call_num": total_call_num,
        "call_accuracy": correct_call_num / total_call_num if total_call_num else 0.0,
        "total_turn_num": int(sample.metadata.get("total_turn_num") or len(sample.expected_turns)),
    }


def _is_single_final_answer(calls: Sequence[Mapping[str, Any]]) -> bool:
    return len(calls) == 1 and str(calls[0].get("name") or "") == COMPLEXFUNCBENCH_FINAL_TOOL["name"]


def _safe_decode(completion: str) -> Any:
    try:
        return decode_tool_calls(completion)
    except Exception:  # noqa: BLE001
        return str(completion)


def _request_completion(config: ComplexFuncBenchRunConfig, prompt: str) -> str:
    return chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_json_safe(item) for item in value]
        return str(value)


def _parse_json_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _coerce_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, tuple):
        return list(raw)
    return [raw]


def _iter_jsonl(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                yield json.loads(text)


def _ensure_source_cache(url: str) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "helicopter-complexfuncbench"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "ComplexFuncBench.jsonl"
    if not target.exists() or target.stat().st_size == 0:
        with urllib.request.urlopen(url, timeout=600.0) as response:
            target.write_bytes(response.read())
    return target


__all__ = [
    "COMPLEXFUNCBENCH_SOURCE_URL",
    "ComplexFuncBenchRunConfig",
    "ComplexFuncBenchSample",
    "dry_run_summary",
    "evaluate_completion",
    "evaluate_sample",
    "load_samples",
    "render_prompt",
    "render_turn_prompt",
    "run_complexfuncbench",
    "sample_from_row",
]
