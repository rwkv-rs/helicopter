from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import urllib.request

from .openai_client import chat_completion
from .sampling import apply_limit_or_sample, dataset_sample_suffix
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


ARENA_HARD_QUESTIONS_URL = "https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl"
ARENA_HARD_BASELINE_URL = (
    "https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl"
)


@dataclass(frozen=True, slots=True)
class ArenaHardSample:
    sample_index: int
    task_id: str
    prompt: str
    category: str
    cluster: str
    baseline_answer: str


@dataclass(frozen=True, slots=True)
class ArenaHardResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
    judge_details: dict[str, Any]

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
class ArenaHardRunConfig:
    base_url: str
    model: str
    benchmark: str = "arena_hard_v2"
    source_url: str = ARENA_HARD_QUESTIONS_URL
    baseline_url: str = ARENA_HARD_BASELINE_URL
    source_path: str | None = None
    baseline_path: str | None = None
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout_s: float = 600.0
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_timeout_s: float = 120.0
    scoreboard_dataset: str | None = None
    job_name: str = "instruction_arena_hard"
    job_id: str | None = None
    runner: str = "helicopter_eval.arena_hard"
    cot_mode: str = "NoCoT"


def load_samples(config: ArenaHardRunConfig) -> list[ArenaHardSample]:
    if config.split != "test":
        raise ValueError("Arena-Hard only provides test split")
    baseline = _load_baseline_answers(config)
    samples: list[ArenaHardSample] = []
    for item in _iter_question_rows(config):
        if config.limit is not None and config.sample_size is None and len(samples) >= int(config.limit):
            break
        uid = str(item.get("uid") or item.get("question_id") or len(samples))
        prompt = str(item.get("prompt") or item.get("question") or "").strip()
        if not prompt:
            continue
        samples.append(
            ArenaHardSample(
                sample_index=len(samples),
                task_id=uid,
                prompt=prompt,
                category=str(item.get("category") or ""),
                cluster=str(item.get("cluster") or ""),
                baseline_answer=baseline.get(uid, ""),
            )
        )
    return apply_limit_or_sample(
        samples,
        limit=config.limit,
        sample_size=config.sample_size,
        sample_seed=config.sample_seed,
        sort_key=lambda sample: sample.sample_index,
    )


def build_generation_prompt(sample: ArenaHardSample) -> str:
    return f"User: {sample.prompt.strip()}\n\nAssistant:"


def generate_completion(sample: ArenaHardSample, config: ArenaHardRunConfig) -> ArenaHardResult:
    prompt = build_generation_prompt(sample)
    answer = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    try:
        passed, fail_reason, judge_details = judge_pairwise(sample, answer, config)
    except Exception as exc:  # noqa: BLE001
        passed = False
        fail_reason = f"judge failed: {exc}"
        judge_details = {}
    return ArenaHardResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=answer,
        answer=answer,
        reference_answer=sample.baseline_answer,
        is_passed=passed,
        fail_reason="" if passed else fail_reason,
        judge_details=judge_details,
    )


def judge_pairwise(
    sample: ArenaHardSample,
    model_answer: str,
    config: ArenaHardRunConfig,
) -> tuple[bool, str, dict[str, Any]]:
    if not sample.baseline_answer.strip():
        raise ValueError(f"missing Arena-Hard baseline answer for {sample.task_id}")
    judge_base_url = _resolved_judge_base_url(config)
    judge_model = _resolved_judge_model(config)
    if not judge_base_url or not judge_model:
        raise ValueError("Arena-Hard scoring requires HELICOPTER_JUDGE_BASE_URL/JUDGE_BASE_URL and HELICOPTER_JUDGE_MODEL/JUDGE_MODEL")
    forward = _judge_order(
        sample,
        answer_a=model_answer,
        answer_b=sample.baseline_answer,
        label_a="candidate",
        label_b="baseline",
        config=config,
    )
    reverse = _judge_order(
        sample,
        answer_a=sample.baseline_answer,
        answer_b=model_answer,
        label_a="baseline",
        label_b="candidate",
        config=config,
    )
    candidate_votes = int(forward["winner"] in {"candidate", "tie"}) + int(reverse["winner"] in {"candidate", "tie"})
    baseline_votes = int(forward["winner"] in {"baseline"}) + int(reverse["winner"] in {"baseline"})
    passed = candidate_votes >= baseline_votes and candidate_votes > 0
    details = {"forward": forward, "reverse": reverse, "candidate_votes": candidate_votes, "baseline_votes": baseline_votes}
    reason = "" if passed else f"baseline preferred: candidate_votes={candidate_votes}, baseline_votes={baseline_votes}"
    return passed, reason, details


def evaluate_samples(samples: Sequence[ArenaHardSample], config: ArenaHardRunConfig) -> list[ArenaHardResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: ArenaHardRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    dataset += dataset_sample_suffix(sample_size=config.sample_size, sample_seed=config.sample_seed)
    return dataset


def job_id(config: ArenaHardRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: ArenaHardRunConfig) -> dict[str, Any]:
    return {
        "answer": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        },
        "judge": {
            "model": _resolved_judge_model(config),
            "base_url": _resolved_judge_base_url(config),
            "pairwise_orders": 2,
        },
    }


def task_sampling_config(config: ArenaHardRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_arena_hard_pairwise",
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[ArenaHardResult], *, config: ArenaHardRunConfig, repo_root: Path) -> int:
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
                extra_metrics=_arena_extra_metrics(results),
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_arena_hard(config: ArenaHardRunConfig, *, repo_root: Path) -> dict[str, Any]:
    if not _resolved_judge_base_url(config) or not _resolved_judge_model(config):
        raise ValueError("Arena-Hard formal scoring requires judge base URL and judge model")
    samples = load_samples(config)
    if not samples:
        raise ValueError("Arena-Hard source is empty")
    missing_baseline = [sample.task_id for sample in samples if not sample.baseline_answer.strip()]
    if missing_baseline:
        raise ValueError(f"Arena-Hard baseline answers missing for {len(missing_baseline)} selected sample(s)")
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "judge_model": _resolved_judge_model(config),
        "total": len(results),
        "passed": passed,
        "win_or_tie_rate": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: ArenaHardRunConfig) -> dict[str, Any]:
    samples = load_samples(config)
    return {
        "benchmark": config.benchmark,
        "source": config.source_path or config.source_url,
        "baseline_source": config.baseline_path or config.baseline_url,
        "split": config.split,
        "limit": config.limit,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "base_url": config.base_url,
        "model": config.model,
        "judge_model": _resolved_judge_model(config),
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
        "sample_probe_count": len(samples),
        "baseline_available": bool(samples and all(sample.baseline_answer for sample in samples)),
    }


def _arena_extra_metrics(results: Sequence[ArenaHardResult]) -> dict[str, Any]:
    if not results:
        return {"win_or_tie_rate": 0.0}
    candidate_votes = 0
    baseline_votes = 0
    for result in results:
        candidate_votes += int(result.judge_details.get("candidate_votes") or 0)
        baseline_votes += int(result.judge_details.get("baseline_votes") or 0)
    return {
        "win_or_tie_rate": sum(1 for result in results if result.is_passed) / len(results),
        "candidate_vote_rate": candidate_votes / max(1, candidate_votes + baseline_votes),
    }


def _iter_question_rows(config: ArenaHardRunConfig) -> Iterable[dict[str, Any]]:
    if config.source_path:
        yield from _iter_jsonl_path(Path(config.source_path).expanduser())
        return
    yield from _iter_jsonl_url(config.source_url, timeout_s=config.timeout_s)


def _load_baseline_answers(config: ArenaHardRunConfig) -> dict[str, str]:
    rows = (
        list(_iter_jsonl_path(Path(config.baseline_path).expanduser()))
        if config.baseline_path
        else list(_iter_jsonl_url(config.baseline_url, timeout_s=config.timeout_s))
    )
    answers: dict[str, str] = {}
    for row in rows:
        uid = row.get("uid")
        if uid is None:
            continue
        answers[str(uid)] = _extract_baseline_answer(row)
    return answers


def _extract_baseline_answer(row: Mapping[str, Any]) -> str:
    for message in row.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, Mapping):
            return str(content.get("answer") or "")
        return str(content or "")
    return ""


def _judge_order(
    sample: ArenaHardSample,
    *,
    answer_a: str,
    answer_b: str,
    label_a: str,
    label_b: str,
    config: ArenaHardRunConfig,
) -> dict[str, Any]:
    content = chat_completion(
        base_url=str(_resolved_judge_base_url(config)),
        model=str(_resolved_judge_model(config)),
        prompt=_judge_prompt(sample.prompt, answer_a, answer_b),
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        timeout_s=config.judge_timeout_s,
        api_key=_resolved_judge_api_key(config),
        response_format={"type": "json_object"},
    )
    payload = _loads_json_object(content)
    winner_raw = str(payload.get("winner") or payload.get("choice") or "").strip().lower()
    if winner_raw in {"a", "answer_a", "model_a"}:
        winner = label_a
    elif winner_raw in {"b", "answer_b", "model_b"}:
        winner = label_b
    elif winner_raw in {"tie", "draw", "equal"}:
        winner = "tie"
    else:
        raise ValueError(f"judge returned unsupported winner {winner_raw!r}: {content!r}")
    return {"winner": winner, "raw_winner": winner_raw, "reason": str(payload.get("reason") or ""), "raw": payload}


def _judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return (
        "You are an impartial Arena-Hard judge. Compare two assistant answers to the same user question.\n"
        "Evaluate helpfulness, correctness, completeness, reasoning quality, and whether the answer follows the user request.\n"
        "Do not prefer verbosity by itself. Ignore answer order. Return only JSON with fields winner and reason.\n"
        "winner must be one of: A, B, tie.\n\n"
        f"[Question]\n{question}\n\n[Answer A]\n{answer_a}\n\n[Answer B]\n{answer_b}\n"
    )


def _loads_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, Mapping):
        raise ValueError(f"judge did not return object: {text!r}")
    return dict(payload)


def _iter_jsonl_url(url: str, *, timeout_s: float) -> Iterable[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def _iter_jsonl_path(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _resolved_judge_base_url(config: ArenaHardRunConfig) -> str | None:
    return config.judge_base_url or os.getenv("HELICOPTER_JUDGE_BASE_URL") or os.getenv("JUDGE_BASE_URL")


def _resolved_judge_model(config: ArenaHardRunConfig) -> str | None:
    return config.judge_model or os.getenv("HELICOPTER_JUDGE_MODEL") or os.getenv("JUDGE_MODEL")


def _resolved_judge_api_key(config: ArenaHardRunConfig) -> str | None:
    return config.judge_api_key or os.getenv("HELICOPTER_JUDGE_API_KEY") or os.getenv("JUDGE_API_KEY")


__all__ = [
    "ArenaHardRunConfig",
    "ArenaHardSample",
    "build_generation_prompt",
    "dry_run_summary",
    "judge_pairwise",
    "load_samples",
    "run_arena_hard",
]
