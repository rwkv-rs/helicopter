from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Iterable, Sequence


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class Gsm8kSample:
    sample_index: int
    question: str
    reference_answer: str


@dataclass(frozen=True, slots=True)
class Gsm8kResult:
    sample_index: int
    question: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str


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
    scoreboard_dataset: str | None = None
    job_name: str = "free_response_judge"
    job_id: str = "helicopter-gsm8k"


def normalize_api_base(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def reference_answer_from_gsm8k(raw_answer: str) -> str:
    marker = "####"
    if marker in raw_answer:
        raw_answer = raw_answer.rsplit(marker, 1)[1]
    return normalize_number(raw_answer)


def normalize_number(value: str) -> str:
    matches = _NUMBER_RE.findall(value.replace(",", ""))
    if not matches:
        return value.strip()
    number = matches[-1].replace(",", "").strip()
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def build_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def scoreboard_dataset_name(config: Gsm8kRunConfig) -> str:
    if config.scoreboard_dataset:
        return config.scoreboard_dataset
    dataset = f"{config.dataset_name}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def completion_answer(completion: str) -> str:
    marker = "####"
    if marker in completion:
        return normalize_number(completion.rsplit(marker, 1)[1])
    return normalize_number(completion)


def load_samples(config: Gsm8kRunConfig) -> list[Gsm8kSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in integration environments
        raise SystemExit("GSM8K eval requires the `datasets` package; install the rwkv dependency group.") from exc

    dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split)
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    limit = len(dataset) if config.limit is None else min(int(config.limit), len(dataset))
    samples: list[Gsm8kSample] = []
    for index, item in enumerate(dataset.select(range(limit))):
        samples.append(
            Gsm8kSample(
                sample_index=index,
                question=str(item["question"]),
                reference_answer=reference_answer_from_gsm8k(str(item["answer"])),
            )
        )
    return samples


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"infer request failed: HTTP {exc.code}: {detail}") from exc


def generate_completion(sample: Gsm8kSample, config: Gsm8kRunConfig) -> Gsm8kResult:
    prompt = build_prompt(sample.question)
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    response = _post_json(
        f"{normalize_api_base(config.base_url)}/chat/completions",
        payload,
        timeout_s=config.timeout_s,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("infer response missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("infer response missing message")
    completion = str(message.get("content") or "")
    answer = completion_answer(completion)
    is_passed = answer == sample.reference_answer
    return Gsm8kResult(
        sample_index=sample.sample_index,
        question=sample.question,
        prompt=prompt,
        completion=completion,
        answer=answer,
        reference_answer=sample.reference_answer,
        is_passed=is_passed,
        fail_reason="" if is_passed else f"expected {sample.reference_answer}, got {answer}",
    )


def evaluate_samples(samples: Sequence[Gsm8kSample], config: Gsm8kRunConfig) -> list[Gsm8kResult]:
    return [generate_completion(sample, config) for sample in samples]


def _scoreboard_import_root(repo_root: Path) -> None:
    scoreboard_root = repo_root / "src/scoreboard-server"
    if str(scoreboard_root) not in sys.path:
        sys.path.insert(0, str(scoreboard_root))


async def write_scoreboard_results(
    results: Sequence[Gsm8kResult],
    *,
    config: Gsm8kRunConfig,
    repo_root: Path,
) -> int:
    _scoreboard_import_root(repo_root)
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=True)
    store = ScoreboardStore(settings=settings)
    dataset = scoreboard_dataset_name(config)
    await store.ensure_benchmark_num_samples(dataset=dataset, num_samples=len(results))
    task_id = await store.get_or_create_task(
        job_name=config.job_name,
        job_id=config.job_id,
        dataset=dataset,
        model=config.model,
        is_param_search=False,
        sampling_config={
            "avg_k": 1,
            "pass_ks": [1],
            "prompt_profile": "helicopter",
            "sampling_config": {
                "answer": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_new_tokens": config.max_tokens,
                }
            },
        },
        allow_resume=True,
    )
    await store.insert_completion_payloads_batch(
        task_id=task_id,
        payloads=[
            {
                "sample_index": result.sample_index,
                "repeat_index": 0,
                "pass_index": 0,
                "prompt1": result.prompt,
                "completion1": result.completion,
                "stop_reason1": "stop",
                "sampling_config": {
                    "answer": {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "max_new_tokens": config.max_tokens,
                    }
                },
            }
            for result in results
        ],
    )
    await store.ingest_eval_payloads(
        task_id=task_id,
        payloads=[
            {
                "sample_index": result.sample_index,
                "repeat_index": 0,
                "pass_index": 0,
                "answer": result.answer,
                "ref_answer": result.reference_answer,
                "is_passed": result.is_passed,
                "fail_reason": result.fail_reason,
            }
            for result in results
        ],
    )
    score = sum(1 for result in results if result.is_passed) / len(results) if results else 0.0
    await store.record_score_payload(
        task_id=task_id,
        payload={
            "cot_mode": "CoT",
            "metrics": {"avg@1": score},
            "runner": "helicopter_eval.gsm8k",
            "benchmark": "gsm8k",
        },
    )
    await close_db()
    return int(task_id)


def run_gsm8k(config: Gsm8kRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = asyncio.run(write_scoreboard_results(results, config=config, repo_root=repo_root))
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": "gsm8k",
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: Gsm8kRunConfig) -> dict[str, Any]:
    return {
        "benchmark": "gsm8k",
        "hf_dataset": config.dataset_name,
        "hf_config": config.dataset_config,
        "split": config.split,
        "limit": config.limit,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": config.job_id,
    }
