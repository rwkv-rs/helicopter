from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class FreeResponseSample:
    sample_index: int
    question: str
    reference_answer: str


@dataclass(frozen=True, slots=True)
class FreeResponseResult:
    sample_index: int
    question: str
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
class FreeResponseRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    question_field: str
    answer_field: str
    limit: int | None = None
    dataset_config: str | None = None
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: float = 600.0
    answer_marker: str | None = "####"
    prompt_template: str = "Question: {question}\nAnswer:"
    scoreboard_dataset: str | None = None
    job_name: str = "free_response_judge"
    job_id: str | None = None
    runner: str = "helicopter_eval.free_response"
    cot_mode: str = "CoT"


def normalize_number(value: str) -> str:
    matches = _NUMBER_RE.findall(value.replace(",", ""))
    if not matches:
        return value.strip()
    number = matches[-1].replace(",", "").strip()
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def extract_marked_answer(text: str, marker: str | None) -> str:
    if marker and marker in text:
        text = text.rsplit(marker, 1)[1]
    return normalize_number(text)


def build_prompt(question: str, *, template: str = "Question: {question}\nAnswer:") -> str:
    return template.format(question=question.strip())


def completion_answer(completion: str, *, marker: str | None = "####") -> str:
    return extract_marked_answer(completion, marker)


def scoreboard_dataset_name(config: FreeResponseRunConfig) -> str:
    if config.scoreboard_dataset:
        return config.scoreboard_dataset
    dataset = f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: FreeResponseRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def task_sampling_config(config: FreeResponseRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter",
        "sampling_config": completion_sampling_config(config),
    }


def completion_sampling_config(config: FreeResponseRunConfig) -> dict[str, Any]:
    return {
        "answer": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def load_samples(config: FreeResponseRunConfig) -> list[FreeResponseSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("free-response eval requires the `datasets` package; install the rwkv dependency group.") from exc

    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split)
    limit = len(dataset) if config.limit is None else min(int(config.limit), len(dataset))
    samples: list[FreeResponseSample] = []
    for index, item in enumerate(dataset.select(range(limit))):
        samples.append(
            FreeResponseSample(
                sample_index=index,
                question=str(item[config.question_field]),
                reference_answer=extract_marked_answer(str(item[config.answer_field]), config.answer_marker),
            )
        )
    return samples


def generate_completion(sample: FreeResponseSample, config: FreeResponseRunConfig) -> FreeResponseResult:
    prompt = build_prompt(sample.question, template=config.prompt_template)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    answer = completion_answer(completion, marker=config.answer_marker)
    is_passed = answer == sample.reference_answer
    return FreeResponseResult(
        sample_index=sample.sample_index,
        question=sample.question,
        prompt=prompt,
        completion=completion,
        answer=answer,
        reference_answer=sample.reference_answer,
        is_passed=is_passed,
        fail_reason="" if is_passed else f"expected {sample.reference_answer}, got {answer}",
    )


def evaluate_samples(samples: Sequence[FreeResponseSample], config: FreeResponseRunConfig) -> list[FreeResponseResult]:
    return [generate_completion(sample, config) for sample in samples]


def write_results(results: Sequence[FreeResponseResult], *, config: FreeResponseRunConfig, repo_root: Path) -> int:
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


def run_free_response(config: FreeResponseRunConfig, *, repo_root: Path) -> dict[str, Any]:
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


def dry_run_summary(config: FreeResponseRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "hf_dataset": config.dataset_name,
        "hf_config": config.dataset_config,
        "split": config.split,
        "limit": config.limit,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }
