from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.request

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


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
    source_type: str = "hf"
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: float = 600.0
    answer_marker: str | None = "####"
    reference_answer_overrides: Mapping[str, str] | None = None
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
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
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


def _extract_answer_from_solution(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    match = re.search(r"The final answer is (.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    for line in reversed(text.splitlines()):
        value = line.strip()
        if value:
            return value
    return None


def _iter_qwen_math_rows(config: FreeResponseRunConfig):
    url = (
        "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/refs/heads/main/"
        f"evaluation/data/{config.dataset_name}/{config.split}.jsonl"
    )
    with urllib.request.urlopen(url, timeout=config.timeout_s) as response:
        for raw_line in response:
            payload = json.loads(raw_line.decode("utf-8"))
            if "answer" in payload:
                payload["expected_answer"] = payload.pop("answer")
            if "problem" not in payload and "question" in payload:
                payload["problem"] = payload.pop("question")
            if config.dataset_name == "olympiadbench" and "final_answer" in payload:
                answers = payload.pop("final_answer")
                if isinstance(answers, (list, tuple)) and answers:
                    payload["expected_answer"] = str(answers[0]).strip("$")
            if config.dataset_name == "minerva_math" and "solution" in payload:
                extracted = _extract_answer_from_solution(str(payload["solution"]))
                if extracted is not None:
                    payload["expected_answer"] = extracted
            yield payload


def _iter_hf_rows(config: FreeResponseRunConfig):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("free-response eval requires the `datasets` package; install the rwkv dependency group.") from exc

    return iter(load_dataset(config.dataset_name, config.dataset_config, split=config.split))


def _iter_rows(config: FreeResponseRunConfig):
    if config.source_type == "hf":
        return _iter_hf_rows(config)
    if config.source_type == "qwen_math":
        return _iter_qwen_math_rows(config)
    raise ValueError(f"unsupported free-response source_type: {config.source_type}")


def load_samples(config: FreeResponseRunConfig) -> list[FreeResponseSample]:
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    limit = None if config.limit is None else int(config.limit)
    samples: list[FreeResponseSample] = []
    for item in _iter_rows(config):
        if limit is not None and len(samples) >= limit:
            break
        question = str(item[config.question_field])
        reference = extract_marked_answer(str(item[config.answer_field]), config.answer_marker)
        if config.reference_answer_overrides and question in config.reference_answer_overrides:
            reference = str(config.reference_answer_overrides[question])
        samples.append(
            FreeResponseSample(
                sample_index=len(samples),
                question=question,
                reference_answer=reference,
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
