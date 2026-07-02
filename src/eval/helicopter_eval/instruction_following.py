from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence
import urllib.request

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_LEADING_THINK_RE = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class InstructionFollowingSample:
    sample_index: int
    key: int
    prompt: str
    instruction_ids: tuple[str, ...]
    kwargs_list: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class InstructionFollowingResult:
    sample_index: int
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
    instruction_correct: int
    instruction_total: int
    metadata: dict[str, Any] | None = None

    def to_scoreboard(self) -> ScoreboardEvalResult:
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=self.answer,
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class InstructionFollowingRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    source_url: str
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout_s: float = 600.0
    strict: bool = True
    prompt_template: str = "User: {prompt}\n\nAssistant:"
    scoreboard_dataset: str | None = None
    job_name: str = "instruction_following"
    job_id: str | None = None
    runner: str = "helicopter_eval.instruction_following"
    cot_mode: str = "NoCoT"


def build_prompt(prompt: str, *, template: str) -> str:
    return template.format(prompt=prompt.strip())


def scoreboard_dataset_name(config: InstructionFollowingRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    if config.sample_size is not None:
        dataset = f"{dataset}_sample{int(config.sample_size)}_seed{int(config.sample_seed)}"
    return dataset


def job_id(config: InstructionFollowingRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: InstructionFollowingRunConfig) -> dict[str, Any]:
    return {
        "answer": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: InstructionFollowingRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter",
        "sampling_config": completion_sampling_config(config),
    }


def _iter_url_jsonl(url: str, *, timeout_s: float) -> Iterable[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def _as_int_key(value: Any, fallback: int) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


def _sample_metadata(
    raw_item: Mapping[str, Any],
    *,
    original_sample_index: int,
    key: int,
    instruction_ids: Sequence[str],
    config: InstructionFollowingRunConfig,
) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "dataset_name": config.dataset_name,
        "split": config.split,
        "source_url": config.source_url,
        "original_sample_index": original_sample_index,
        "source_id": str(raw_item.get("key", key)),
        "key": key,
        "instruction_ids": list(instruction_ids),
    }


def load_samples(config: InstructionFollowingRunConfig) -> list[InstructionFollowingSample]:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    if config.sample_size is not None and int(config.sample_size) < 0:
        raise ValueError("sample_size must be non-negative")
    if config.limit is not None and config.sample_size is not None:
        raise ValueError("limit and sample_size are mutually exclusive")
    if config.split != "test":
        raise ValueError(f"{config.benchmark} only supports test split")
    limit = None if config.limit is None else int(config.limit)
    samples: list[InstructionFollowingSample] = []
    for raw_item in _iter_url_jsonl(config.source_url, timeout_s=config.timeout_s):
        if limit is not None and len(samples) >= limit:
            break
        prompt = raw_item.get("prompt")
        instruction_ids = raw_item.get("instruction_id_list")
        kwargs_list = raw_item.get("kwargs")
        if not isinstance(prompt, str):
            continue
        if not isinstance(instruction_ids, list) or not all(isinstance(item, str) for item in instruction_ids):
            continue
        if not isinstance(kwargs_list, list) or not all(isinstance(item, dict) for item in kwargs_list):
            continue
        original_sample_index = len(samples)
        key = _as_int_key(raw_item.get("key"), len(samples))
        samples.append(
            InstructionFollowingSample(
                sample_index=len(samples),
                key=key,
                prompt=prompt,
                instruction_ids=tuple(instruction_ids),
                kwargs_list=tuple(dict(item) for item in kwargs_list),
                metadata=_sample_metadata(
                    raw_item,
                    original_sample_index=original_sample_index,
                    key=key,
                    instruction_ids=instruction_ids,
                    config=config,
                ),
            )
        )
    if config.sample_size is not None and int(config.sample_size) < len(samples):
        rng = random.Random(int(config.sample_seed))
        samples = sorted(rng.sample(samples, int(config.sample_size)), key=lambda item: item.sample_index)
        samples = [replace(sample, sample_index=index) for index, sample in enumerate(samples)]
    return samples


def _strip_reasoning_block(text: str) -> str:
    body = str(text or "")
    body = _THINK_BLOCK_RE.sub("", body).strip()
    if _LEADING_THINK_RE.match(body):
        return ""
    return body


def _build_loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    variants = [response]
    if len(lines) > 1:
        variants.append("\n".join(lines[1:]).strip())
        variants.append("\n".join(lines[:-1]).strip())
        variants.append("\n".join(lines[1:-1]).strip())
    stripped = response.replace("*", "")
    variants.append(stripped)
    variants.extend(text.replace("*", "") for text in variants if text)
    return [variant for variant in variants if variant]


def _instruction_registry(config: InstructionFollowingRunConfig) -> Mapping[str, Any]:
    if config.benchmark == "ifbench":
        from .instruction_following_rules.ifbench_official import instructions_registry as ifbench_registry

        return ifbench_registry.INSTRUCTION_DICT
    from .instruction_following_rules import instructions_registry

    return instructions_registry.INSTRUCTION_DICT


def score_response(
    sample: InstructionFollowingSample,
    completion: str,
    config: InstructionFollowingRunConfig,
) -> tuple[bool, int, int, str]:
    response = _strip_reasoning_block(completion)
    registry = _instruction_registry(config)
    reference_parts: list[str] = []
    follow_list: list[bool] = []
    variants = None if config.strict else _build_loose_variants(response)
    for index, instruction_id in enumerate(sample.instruction_ids):
        instruction_cls = registry[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = {key: value for key, value in sample.kwargs_list[index].items() if value is not None}
        description = instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            description = instruction.build_description(prompt=sample.prompt)
        if config.strict:
            is_following = bool(response.strip() and instruction.check_following(response))
        else:
            is_following = any(
                variant.strip() and instruction.check_following(variant)
                for variant in (variants or ())
            )
        follow_list.append(bool(is_following))
        reference_parts.append(f"{instruction_id}: {description}" if description else instruction_id)
    is_passed = bool(follow_list) and all(follow_list)
    return is_passed, sum(1 for flag in follow_list if flag), len(follow_list), "\n".join(reference_parts)


def generate_completion(
    sample: InstructionFollowingSample,
    config: InstructionFollowingRunConfig,
) -> InstructionFollowingResult:
    prompt = build_prompt(sample.prompt, template=config.prompt_template)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    is_passed, instruction_correct, instruction_total, reference_answer = score_response(sample, completion, config)
    answer = _strip_reasoning_block(completion)
    return InstructionFollowingResult(
        sample_index=sample.sample_index,
        prompt=prompt,
        completion=completion,
        answer=answer,
        reference_answer=reference_answer,
        is_passed=is_passed,
        fail_reason="" if is_passed else "one or more instruction checks failed",
        instruction_correct=instruction_correct,
        instruction_total=instruction_total,
        metadata=sample.metadata,
    )


def evaluate_samples(
    samples: Sequence[InstructionFollowingSample],
    config: InstructionFollowingRunConfig,
) -> list[InstructionFollowingResult]:
    return [generate_completion(sample, config) for sample in samples]


def aggregate_metrics(results: Sequence[InstructionFollowingResult]) -> dict[str, float]:
    prompt_accuracy = sum(1 for result in results if result.is_passed) / len(results) if results else 0.0
    instruction_total = sum(result.instruction_total for result in results)
    instruction_correct = sum(result.instruction_correct for result in results)
    instruction_accuracy = instruction_correct / instruction_total if instruction_total else 0.0
    return {
        "prompt_accuracy": prompt_accuracy,
        "instruction_accuracy": instruction_accuracy,
    }


def write_results(
    results: Sequence[InstructionFollowingResult],
    *,
    config: InstructionFollowingRunConfig,
    repo_root: Path,
) -> int:
    return int(
        asyncio.run(
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
                    extra_metrics=aggregate_metrics(results),
                ),
                repo_root=repo_root,
            )
        )
    )


def run_instruction_following(config: InstructionFollowingRunConfig, *, repo_root: Path) -> dict[str, Any]:
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
        **aggregate_metrics(results),
    }


def dry_run_summary(config: InstructionFollowingRunConfig) -> dict[str, Any]:
    payload = {
        "benchmark": config.benchmark,
        "hf_dataset": config.dataset_name,
        "hf_config": None,
        "split": config.split,
        "limit": config.limit,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
        "strict": config.strict,
    }
    if config.sample_size is not None:
        payload["sample_size"] = config.sample_size
        payload["sample_seed"] = config.sample_seed
    return payload
