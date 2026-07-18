from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

from .commands import bool_value, local_openai_base_url
from .config import resolve_model_entry, table
from .env import env_value, pick


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

COMPLETIONS_PROFILE_DEFAULTS = {
    "prefill": {
        "prompt_tokens": 2048,
        "output_tokens": 8,
        "requests": 8,
        "concurrency": 1,
    },
    "decode": {
        "prompt_tokens": 128,
        "output_tokens": 256,
        "requests": 8,
        "concurrency": 1,
    },
}


@dataclass(frozen=True)
class CompletionRequestResult:
    ok: bool
    elapsed_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class CompletionsPerformanceConfig:
    profile: str
    model_name: str
    base_url: str
    api_key: str | None
    prompt_tokens: int
    output_tokens: int
    requests: int
    concurrency: int
    request_rate: float | None
    timeout_s: float
    ignore_eos: bool
    output_path: Path


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


def completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/completions"


def synthetic_prompt(target_tokens: int) -> str:
    count = max(1, int(target_tokens))
    return " ".join(["rwkv"] * count)


def _usage_int(response: Mapping[str, Any] | None, key: str) -> int | None:
    if not isinstance(response, Mapping):
        return None
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(key)
    if not isinstance(value, int):
        return None
    return value if value >= 0 else None


def post_completion(
    *,
    base_url: str,
    api_key: str | None,
    model_name: str,
    prompt: str,
    output_tokens: int,
    timeout_s: float,
    ignore_eos: bool,
) -> CompletionRequestResult:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": False,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(completions_url(base_url), data=data, headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - performance probes report per-request errors.
        return CompletionRequestResult(False, time.monotonic() - started, error=f"{type(error).__name__}: {error}")

    elapsed = time.monotonic() - started
    prompt_tokens = _usage_int(body, "prompt_tokens")
    completion_tokens = _usage_int(body, "completion_tokens")
    total_tokens = _usage_int(body, "total_tokens")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return CompletionRequestResult(
        True,
        elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


def latency_summary(results: list[CompletionRequestResult]) -> dict[str, float | None]:
    values = [result.elapsed_seconds for result in results if result.ok]
    return {
        "mean": (sum(values) / len(values)) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def run_completion_requests(
    *,
    base_url: str,
    api_key: str | None,
    model_name: str,
    prompt: str,
    output_tokens: int,
    requests: int,
    concurrency: int,
    request_rate: float | None,
    timeout_s: float,
    ignore_eos: bool,
) -> list[CompletionRequestResult]:
    request_count = max(1, int(requests))
    worker_count = max(1, min(int(concurrency), request_count))
    if worker_count == 1:
        results = []
        started_submit = time.monotonic()
        for index in range(request_count):
            if request_rate and index > 0:
                target = started_submit + index / request_rate
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            results.append(
                post_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    output_tokens=output_tokens,
                    timeout_s=timeout_s,
                    ignore_eos=ignore_eos,
                )
            )
        return results

    results: list[CompletionRequestResult] = []
    submitted = 0
    started_submit = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending: set[concurrent.futures.Future[CompletionRequestResult]] = set()
        while submitted < request_count or pending:
            while submitted < request_count and len(pending) < worker_count:
                if request_rate and submitted > 0:
                    target = started_submit + submitted / request_rate
                    delay = target - time.monotonic()
                    if delay > 0:
                        break
                pending.add(
                    executor.submit(
                        post_completion,
                        base_url=base_url,
                        api_key=api_key,
                        model_name=model_name,
                        prompt=prompt,
                        output_tokens=output_tokens,
                        timeout_s=timeout_s,
                        ignore_eos=ignore_eos,
                    )
                )
                submitted += 1
            if not pending:
                time.sleep(0.01)
                continue
            done, pending = concurrent.futures.wait(
                pending,
                timeout=0.05,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                results.append(future.result())
    return results


def build_completions_performance_report(
    *,
    model_name: str,
    base_url: str,
    profile: str,
    prompt_tokens_target: int,
    output_tokens: int,
    requests: int,
    concurrency: int,
    request_rate: float | None,
    timeout_s: float,
    ignore_eos: bool,
    started_at_epoch: float,
    ended_at_epoch: float,
    results: list[CompletionRequestResult],
) -> dict[str, Any]:
    elapsed_seconds = max(0.0, ended_at_epoch - started_at_epoch)
    successful = [result for result in results if result.ok]
    prompt_tokens = _sum_optional([result.prompt_tokens for result in successful])
    completion_tokens = _sum_optional([result.completion_tokens for result in successful])
    total_tokens = _sum_optional([result.total_tokens for result in successful])
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    error_counts: dict[str, int] = {}
    for result in results:
        if result.error:
            error_counts[result.error] = error_counts.get(result.error, 0) + 1
    return {
        "schema_version": 1,
        "kind": "openai_completions",
        "profile": profile,
        "started_at": datetime.fromtimestamp(started_at_epoch, UTC).isoformat(),
        "ended_at": datetime.fromtimestamp(ended_at_epoch, UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "model": model_name,
        "base_url": base_url,
        "prompt_tokens_target": prompt_tokens_target,
        "output_tokens_target": output_tokens,
        "requests": requests,
        "concurrency": concurrency,
        "request_rate": request_rate,
        "timeout_s": timeout_s,
        "ignore_eos": ignore_eos,
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "error_rate": ((len(results) - len(successful)) / len(results)) if results else None,
        "request_throughput": (len(successful) / elapsed_seconds) if elapsed_seconds > 0 else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_per_second": (prompt_tokens / elapsed_seconds) if prompt_tokens is not None and elapsed_seconds > 0 else None,
        "completion_tokens_per_second": (
            completion_tokens / elapsed_seconds
            if completion_tokens is not None and elapsed_seconds > 0
            else None
        ),
        "total_tokens_per_second": (total_tokens / elapsed_seconds) if total_tokens is not None and elapsed_seconds > 0 else None,
        "e2e_latency_seconds": latency_summary(results),
        "errors": error_counts,
    }


def _profile_default(profile: str, key: str) -> int:
    return int(COMPLETIONS_PROFILE_DEFAULTS[profile][key])


def _performance_config(config: dict[str, Any]) -> dict[str, Any]:
    value = table(config, "performance")
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise SystemExit(f"{name} must be positive")
    return parsed


def _positive_float_or_none(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise SystemExit(f"{name} must be positive")
    return parsed


def _performance_output_path(*, root: Path, profile: str, output: Any) -> Path:
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return root / "results/performance" / f"completions_{profile}_{stamp}.json"
    output_path = Path(str(output))
    return output_path if output_path.is_absolute() else root / output_path


def resolve_completions_performance_config(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> CompletionsPerformanceConfig:
    perf_config = _performance_config(config)
    profile = str(getattr(args, "profile", None) or perf_config.get("profile") or "decode")
    if profile not in COMPLETIONS_PROFILE_DEFAULTS:
        raise SystemExit(f"unsupported performance profile: {profile}")

    model = resolve_model_entry(config, args.model)
    model_name = str(
        pick(
            getattr(args, "served_model_name", None),
            model.get("served_model_name"),
            model.get("requested_name"),
            args.model,
        )
    )
    raw_api_key = str(
        pick(
            getattr(args, "api_key", None),
            env_value(env, "HELICOPTER_EVAL_API_KEY", "OPENAI_API_KEY"),
            perf_config.get("api_key"),
            "EMPTY",
        )
    )
    request_rate = _positive_float_or_none(
        pick(getattr(args, "request_rate", None), perf_config.get("request_rate")),
        name="request_rate",
    )
    timeout_s = _positive_float_or_none(
        pick(getattr(args, "timeout", None), perf_config.get("timeout"), 120.0),
        name="timeout",
    )
    assert timeout_s is not None

    return CompletionsPerformanceConfig(
        profile=profile,
        model_name=model_name,
        base_url=local_openai_base_url(config, env, args),
        api_key=None if raw_api_key == "EMPTY" else raw_api_key or None,
        prompt_tokens=_positive_int(
            pick(
                getattr(args, "prompt_tokens", None),
                perf_config.get("prompt_tokens"),
                _profile_default(profile, "prompt_tokens"),
            ),
            name="prompt_tokens",
        ),
        output_tokens=_positive_int(
            pick(
                getattr(args, "output_tokens", None),
                perf_config.get("output_tokens"),
                _profile_default(profile, "output_tokens"),
            ),
            name="output_tokens",
        ),
        requests=_positive_int(
            pick(
                getattr(args, "requests", None),
                perf_config.get("requests"),
                _profile_default(profile, "requests"),
            ),
            name="requests",
        ),
        concurrency=_positive_int(
            pick(
                getattr(args, "concurrency", None),
                perf_config.get("concurrency"),
                _profile_default(profile, "concurrency"),
            ),
            name="concurrency",
        ),
        request_rate=request_rate,
        timeout_s=timeout_s,
        ignore_eos=bool_value(pick(getattr(args, "ignore_eos", None), perf_config.get("ignore_eos"), False)),
        output_path=_performance_output_path(
            root=root,
            profile=profile,
            output=pick(getattr(args, "output", None), perf_config.get("output")),
        ),
    )


def build_completions_performance_report_from_config(
    profile_config: CompletionsPerformanceConfig,
    *,
    started_at_epoch: float,
    ended_at_epoch: float,
    results: list[CompletionRequestResult],
) -> dict[str, Any]:
    return build_completions_performance_report(
        model_name=profile_config.model_name,
        base_url=profile_config.base_url,
        profile=profile_config.profile,
        prompt_tokens_target=profile_config.prompt_tokens,
        output_tokens=profile_config.output_tokens,
        requests=profile_config.requests,
        concurrency=profile_config.concurrency,
        request_rate=profile_config.request_rate,
        timeout_s=profile_config.timeout_s,
        ignore_eos=profile_config.ignore_eos,
        started_at_epoch=started_at_epoch,
        ended_at_epoch=ended_at_epoch,
        results=results,
    )


def run_completions_performance(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> int:
    profile_config = resolve_completions_performance_config(args, root=root, env=env, config=config)

    if getattr(args, "dry_run", False):
        print(
            "eval perf: "
            f"model={profile_config.model_name} base_url={profile_config.base_url} profile={profile_config.profile} "
            f"requests={profile_config.requests} concurrency={profile_config.concurrency} "
            f"prompt_tokens={profile_config.prompt_tokens} output_tokens={profile_config.output_tokens} "
            f"output={profile_config.output_path}"
        )
        return 0

    prompt = synthetic_prompt(profile_config.prompt_tokens)
    started_at_epoch = time.time()
    results = run_completion_requests(
        base_url=profile_config.base_url,
        api_key=profile_config.api_key,
        model_name=profile_config.model_name,
        prompt=prompt,
        output_tokens=profile_config.output_tokens,
        requests=profile_config.requests,
        concurrency=profile_config.concurrency,
        request_rate=profile_config.request_rate,
        timeout_s=profile_config.timeout_s,
        ignore_eos=profile_config.ignore_eos,
    )
    ended_at_epoch = time.time()
    report = build_completions_performance_report_from_config(
        profile_config,
        started_at_epoch=started_at_epoch,
        ended_at_epoch=ended_at_epoch,
        results=results,
    )
    profile_config.output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_config.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "eval perf: "
        f"successful={report['successful_requests']} failed={report['failed_requests']} "
        f"completion_tps={report['completion_tokens_per_second']} report={profile_config.output_path}"
    )
    return 1 if report["failed_requests"] else 0


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
