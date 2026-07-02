from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class ScoreboardEvalResult:
    sample_index: int
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ScoreboardWriteConfig:
    dataset: str
    model: str
    job_name: str
    job_id: str
    benchmark: str
    runner: str
    cot_mode: str
    sampling_config: dict[str, Any]
    completion_sampling_config: dict[str, Any]
    extra_metrics: dict[str, Any] | None = None


def _scoreboard_import_root(repo_root: Path) -> None:
    scoreboard_root = repo_root / "src/scoreboard-server"
    if str(scoreboard_root) not in sys.path:
        sys.path.insert(0, str(scoreboard_root))


async def write_scoreboard_results(
    results: Sequence[ScoreboardEvalResult],
    *,
    config: ScoreboardWriteConfig,
    repo_root: Path,
) -> int:
    _scoreboard_import_root(repo_root)
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=True)
    try:
        store = ScoreboardStore(settings=settings)
        await store.ensure_benchmark_num_samples(dataset=config.dataset, num_samples=len(results))
        task_id = await store.get_or_create_task(
            job_name=config.job_name,
            job_id=config.job_id,
            dataset=config.dataset,
            model=config.model,
            is_param_search=False,
            sampling_config=config.sampling_config,
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
                    "sampling_config": config.completion_sampling_config,
                    "metadata": result.metadata or {},
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
        metrics = {"avg@1": score, **(config.extra_metrics or {})}
        await store.record_score_payload(
            task_id=task_id,
            payload={
                "cot_mode": config.cot_mode,
                "metrics": metrics,
                "runner": config.runner,
                "benchmark": config.benchmark,
            },
        )
        return int(task_id)
    finally:
        await close_db()
