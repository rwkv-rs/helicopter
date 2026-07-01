from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import free_response


@dataclass(frozen=True, slots=True)
class Gsm8kRunConfig:
    base_url: str
    model: str
    limit: int | None
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: float = 600.0
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    question_field: str = "question"
    answer_field: str = "answer"
    answer_marker: str = "####"
    job_name: str = "free_response_judge"
    job_id: str = "helicopter-gsm8k"


normalize_number = free_response.normalize_number
completion_answer = free_response.completion_answer


def _free_response_config(config: Gsm8kRunConfig) -> free_response.FreeResponseRunConfig:
    return free_response.FreeResponseRunConfig(
        base_url=config.base_url,
        model=config.model,
        benchmark="gsm8k",
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        question_field=config.question_field,
        answer_field=config.answer_field,
        limit=config.limit,
        split=config.split,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
        answer_marker=config.answer_marker,
        job_name=config.job_name,
        job_id=config.job_id,
        runner="helicopter_eval.gsm8k",
    )


def reference_answer_from_gsm8k(raw_answer: str) -> str:
    return free_response.extract_marked_answer(raw_answer, "####")


def scoreboard_dataset_name(config: Gsm8kRunConfig) -> str:
    return free_response.scoreboard_dataset_name(_free_response_config(config))


def run_gsm8k(config: Gsm8kRunConfig, *, repo_root: Path) -> dict[str, Any]:
    return free_response.run_free_response(_free_response_config(config), repo_root=repo_root)


def dry_run_summary(config: Gsm8kRunConfig) -> dict[str, Any]:
    return free_response.dry_run_summary(_free_response_config(config))
