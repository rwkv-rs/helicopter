from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen


PROMETHEUS_LINE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

PROMPT_TOKEN_METRIC_SUFFIXES = (
    ("prompt_tokens_total",),
    ("request_prompt_tokens_sum",),
)
GENERATION_TOKEN_METRIC_SUFFIXES = (
    ("generation_tokens_total",),
    ("request_generation_tokens_sum",),
)


def derive_metrics_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    path = f"{path}/metrics" if path else "/metrics"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def base_url_from_lighteval_command(command: list[str]) -> str | None:
    for item in command:
        if "base_url=" not in item:
            continue
        for part in item.split(","):
            if part.startswith("base_url="):
                return part.split("=", 1)[1]
    return None


def output_dir_from_command(command: list[str]) -> Path | None:
    for index, item in enumerate(command):
        if item == "--output-dir" and index + 1 < len(command):
            return Path(command[index + 1])
        if item.startswith("--output-dir="):
            return Path(item.split("=", 1)[1])
    return None


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_LINE_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        metrics[match.group(1)] = metrics.get(match.group(1), 0.0) + value
    return metrics


def fetch_prometheus_metrics(metrics_url: str | None, *, timeout_s: float = 2.0) -> dict[str, float] | None:
    if not metrics_url:
        return None
    try:
        with urlopen(metrics_url, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except (OSError, URLError, TimeoutError):
        return None
    return parse_prometheus_metrics(payload)


def token_delta(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
    suffix_groups: tuple[tuple[str, ...], ...],
) -> int | None:
    if before is None or after is None:
        return None
    for suffixes in suffix_groups:
        total = 0.0
        matched = False
        for name, end_value in after.items():
            if not any(name.endswith(suffix) for suffix in suffixes):
                continue
            matched = True
            total += max(0.0, end_value - before.get(name, 0.0))
        if matched:
            return int(total)
    return None


def rate_per_seconds(count: int | None, elapsed_seconds: float, scale_seconds: int) -> float | None:
    if count is None or elapsed_seconds <= 0:
        return None
    return count * scale_seconds / elapsed_seconds


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_files(output_dir: Path, started_at_epoch: float) -> list[Path]:
    if not output_dir.exists():
        return []
    files = [path for path in output_dir.rglob("results_*.json") if path.stat().st_mtime >= started_at_epoch - 1.0]
    return sorted(files)


def _detail_files(output_dir: Path, started_at_epoch: float) -> list[Path]:
    if not output_dir.exists():
        return []
    files = [path for path in output_dir.rglob("details_*.parquet") if path.stat().st_mtime >= started_at_epoch - 1.0]
    return sorted(files)


def _count_detail_rows(files: list[Path]) -> int | None:
    if not files:
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    count = 0
    for path in files:
        try:
            count += len(pd.read_parquet(path, columns=["metric"]))
        except Exception:
            return None
    return count


def summarize_lighteval_outputs(output_dir: Path | None, *, started_at_epoch: float) -> dict[str, Any]:
    if output_dir is None:
        return {
            "result_files": [],
            "detail_files": [],
            "models_completed": 0,
            "benchmarks_completed": 0,
            "samples_completed": None,
            "lighteval_reported_seconds": None,
        }

    result_files = _json_files(output_dir, started_at_epoch)
    detail_files = _detail_files(output_dir, started_at_epoch)
    model_names: set[str] = set()
    task_names: set[str] = set()
    max_samples: int | None = None
    reported_seconds: float | None = None

    for path in result_files:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config_general = payload.get("config_general", {})
        if isinstance(config_general, dict):
            model_name = config_general.get("model_name")
            if isinstance(model_name, str) and model_name:
                model_names.add(model_name)
            if max_samples is None:
                raw_max_samples = config_general.get("max_samples")
                if isinstance(raw_max_samples, int):
                    max_samples = raw_max_samples
            seconds = _safe_float(config_general.get("total_evaluation_time_secondes"))
            if seconds is not None:
                reported_seconds = (reported_seconds or 0.0) + seconds
        results = payload.get("results", {})
        if isinstance(results, dict):
            task_names.update(str(name) for name in results if name != "all")

    samples_completed = _count_detail_rows(detail_files)
    if samples_completed is None and max_samples is not None and task_names:
        samples_completed = max_samples * len(task_names)

    return {
        "result_files": [str(path) for path in result_files],
        "detail_files": [str(path) for path in detail_files],
        "models_completed": len(model_names),
        "benchmarks_completed": len(task_names),
        "samples_completed": samples_completed,
        "lighteval_reported_seconds": reported_seconds,
    }


def extract_lighteval_score_metrics(result_files: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for item in result_files:
        path = Path(item)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        all_metrics = results.get("all")
        if isinstance(all_metrics, dict):
            metrics.update(all_metrics)
            continue
        for task_metrics in results.values():
            if isinstance(task_metrics, dict):
                metrics.update(task_metrics)
    return metrics


def performance_metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "elapsed_seconds",
        "lighteval_reported_seconds",
        "samples_completed",
        "jobs_completed",
        "models_completed",
        "benchmarks_completed",
        "completed_runs",
        "prompt_tokens",
        "generation_tokens",
        "total_tokens",
        "samples_per_hour",
        "jobs_per_hour",
        "models_per_day",
        "benchmarks_per_day",
        "tokens_per_second",
        "completed_runs/day",
        "metrics_url",
        "started_at",
        "ended_at",
        "exit_code",
    )
    return {key: report.get(key) for key in keys}


async def _write_scoreboard_performance(
    *,
    task_id: str,
    report: dict[str, Any],
    root: Path,
) -> None:
    scoreboard_path = root / "src/scoreboard-server"
    if str(scoreboard_path) not in sys.path:
        sys.path.insert(0, str(scoreboard_path))

    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        service = ScoreboardStore(settings=settings)
        existing = await service.get_score_payload(task_id=task_id)
        metrics: dict[str, Any] = {}
        cot_mode = "NoCoT"
        if existing is not None:
            existing_metrics = existing.get("metrics")
            if isinstance(existing_metrics, dict):
                metrics.update(existing_metrics)
            existing_cot_mode = existing.get("cot_mode")
            if isinstance(existing_cot_mode, str):
                cot_mode = existing_cot_mode
        if not metrics:
            metrics.update(extract_lighteval_score_metrics(report.get("source_files", {}).get("results", [])))
        metrics["performance"] = performance_metrics_from_report(report)
        await service.record_score_payload(task_id=task_id, payload={"cot_mode": cot_mode, "metrics": metrics})
    finally:
        await close_db()


def write_scoreboard_performance(*, task_id: str | None, report: dict[str, Any], root: Path) -> None:
    if not task_id:
        return
    asyncio.run(_write_scoreboard_performance(task_id=task_id, report=report, root=root))


def build_performance_report(
    *,
    command: list[str],
    exit_code: int,
    output_dir: Path | None,
    started_at_epoch: float,
    ended_at_epoch: float,
    metrics_url: str | None,
    metrics_before: dict[str, float] | None,
    metrics_after: dict[str, float] | None,
) -> dict[str, Any]:
    elapsed_seconds = max(0.0, ended_at_epoch - started_at_epoch)
    summary = summarize_lighteval_outputs(output_dir, started_at_epoch=started_at_epoch)
    prompt_tokens = token_delta(metrics_before, metrics_after, PROMPT_TOKEN_METRIC_SUFFIXES)
    generation_tokens = token_delta(metrics_before, metrics_after, GENERATION_TOKEN_METRIC_SUFFIXES)
    total_tokens = None
    if prompt_tokens is not None or generation_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (generation_tokens or 0)

    completed_runs = 1 if exit_code == 0 and summary["result_files"] else 0
    jobs_completed = 1 if exit_code == 0 else 0
    samples_completed = summary["samples_completed"]
    models_completed = summary["models_completed"] if exit_code == 0 else 0
    benchmarks_completed = summary["benchmarks_completed"] if exit_code == 0 else 0
    tokens_per_second = None
    if total_tokens is not None and elapsed_seconds > 0:
        tokens_per_second = total_tokens / elapsed_seconds

    report = {
        "schema_version": 1,
        "started_at": datetime.fromtimestamp(started_at_epoch, UTC).isoformat(),
        "ended_at": datetime.fromtimestamp(ended_at_epoch, UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "exit_code": exit_code,
        "command": command,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "metrics_url": metrics_url,
        "lighteval_reported_seconds": summary["lighteval_reported_seconds"],
        "samples_completed": samples_completed,
        "jobs_completed": jobs_completed,
        "models_completed": models_completed,
        "benchmarks_completed": benchmarks_completed,
        "completed_runs": completed_runs,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "total_tokens": total_tokens,
        "samples_per_hour": rate_per_seconds(samples_completed, elapsed_seconds, 3600),
        "jobs_per_hour": rate_per_seconds(jobs_completed, elapsed_seconds, 3600),
        "models_per_day": rate_per_seconds(models_completed, elapsed_seconds, 86400),
        "benchmarks_per_day": rate_per_seconds(benchmarks_completed, elapsed_seconds, 86400),
        "tokens_per_second": tokens_per_second,
        "completed_runs/day": rate_per_seconds(completed_runs, elapsed_seconds, 86400),
        "source_files": {
            "results": summary["result_files"],
            "details": summary["detail_files"],
        },
    }
    report["rates"] = {
        key: report[key]
        for key in (
            "samples_per_hour",
            "jobs_per_hour",
            "models_per_day",
            "benchmarks_per_day",
            "tokens_per_second",
            "completed_runs/day",
        )
    }
    return report


def run_lighteval_with_performance(
    command: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str],
    root: Path,
    performance_output: Path,
    metrics_url: str | None,
    scoreboard_task_id: str | None = None,
) -> int:
    output_dir = output_dir_from_command(command)
    if output_dir is not None and not output_dir.is_absolute() and cwd is not None:
        output_dir = cwd / output_dir
    started_at_epoch = time.time()
    metrics_before = fetch_prometheus_metrics(metrics_url)
    exit_code = subprocess.call(command, cwd=str(cwd) if cwd else None, env=env)
    ended_at_epoch = time.time()
    metrics_after = fetch_prometheus_metrics(metrics_url)
    report = build_performance_report(
        command=command,
        exit_code=exit_code,
        output_dir=output_dir,
        started_at_epoch=started_at_epoch,
        ended_at_epoch=ended_at_epoch,
        metrics_url=metrics_url,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )
    performance_output.parent.mkdir(parents=True, exist_ok=True)
    performance_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_scoreboard_performance(task_id=scoreboard_task_id, report=report, root=root)
    return exit_code
