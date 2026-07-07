from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from .commands import (
    CommandPlan,
    build_infer_plan,
    build_lighteval_plan,
    is_local_base_url,
    local_openai_base_url,
)
from .config import resolve_model_entry, table
from .env import pick
from .performance import (
    base_url_from_lighteval_command,
    derive_metrics_url,
    output_dir_from_command,
    run_lighteval_with_performance,
)


DEFAULT_SERVER_TIMEOUT_S = 600.0
SERVER_POLL_INTERVAL_S = 2.0


def resolve_run_tasks(args: Any, config: dict[str, Any]) -> str:
    lighteval = table(config, "lighteval")
    configured = lighteval.get("tasks")
    if isinstance(configured, list):
        configured = ",".join(str(item) for item in configured if str(item))
    tasks = pick(getattr(args, "tasks", None), configured)
    if not tasks:
        raise SystemExit(
            "no tasks given: pass a task string (e.g. 'gsm8k|0') or set [lighteval].tasks in the config"
        )
    return str(tasks)


def health_url_for(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def server_is_healthy(base_url: str, *, timeout_s: float = 2.0) -> bool:
    try:
        with urlopen(health_url_for(base_url), timeout=timeout_s) as response:
            return 200 <= response.status < 300
    except (OSError, URLError, TimeoutError, ValueError):
        return False


def _tail_lines(path: Path, count: int = 30) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-count:])


def wait_for_server(
    base_url: str,
    *,
    process: subprocess.Popen[bytes],
    log_path: Path,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"vLLM server exited early with code {process.returncode}; last log lines from {log_path}:\n"
                f"{_tail_lines(log_path)}"
            )
        if server_is_healthy(base_url):
            return
        time.sleep(SERVER_POLL_INTERVAL_S)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    raise SystemExit(
        f"vLLM server did not become healthy within {int(timeout_s)}s at {health_url_for(base_url)}; "
        f"last log lines from {log_path}:\n{_tail_lines(log_path)}"
    )


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def infer_args_namespace(args: Any, *, port: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        dry_run=getattr(args, "dry_run", False),
        wkv_mode=getattr(args, "wkv_mode", None),
        emb_device=getattr(args, "emb_device", None),
        host=None,
        port=port,
        served_model_name=getattr(args, "lighteval_model_name", None),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", None),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
        max_model_len=None,
        max_num_seqs=getattr(args, "max_num_seqs", None),
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", None),
        enable_auto_tool_choice=None,
        vllm_env=getattr(args, "vllm_env", None),
    )


def port_from_base_url(base_url: str) -> str | None:
    parsed = urlsplit(base_url)
    return str(parsed.port) if parsed.port else None


def format_plan_for_display(plan: CommandPlan) -> str:
    pieces: list[str] = []
    if plan.cwd is not None:
        pieces.extend(["cd", shlex.quote(str(plan.cwd)), "&&"])
    if plan.shown_env:
        pieces.append("env")
        for key in sorted(plan.shown_env):
            pieces.append(f"{key}={shlex.quote(plan.shown_env[key])}")
    pieces.extend(shlex.quote(item) for item in plan.command)
    return " ".join(pieces)


def scoreboard_dataset_name(task_name: str) -> str:
    """Normalize a LightEval results key (``gsm8k|0`` or ``suite|task|0``) to a benchmark name."""
    parts = [part.strip() for part in str(task_name).split("|") if part.strip()]
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return parts[-1] if parts else str(task_name)


def scoreboard_model_name(args: Any, config: dict[str, Any]) -> str:
    model = resolve_model_entry(config, args.model)
    return str(
        pick(
            getattr(args, "lighteval_model_name", None),
            model.get("served_model_name"),
            model.get("requested_name"),
            args.model,
        )
    )


async def _ingest_scoreboard_results(
    *,
    result_files: list[str],
    model_name: str,
    root: Path,
) -> list[str]:
    scoreboard_path = root / "src/scoreboard-server"
    if str(scoreboard_path) not in sys.path:
        sys.path.insert(0, str(scoreboard_path))

    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    recorded: list[str] = []
    try:
        store = ScoreboardStore(settings=settings)
        for item in result_files:
            path = Path(item)
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            results = payload.get("results")
            if not isinstance(results, dict):
                continue
            for task_name, metrics in results.items():
                if task_name == "all" or not isinstance(metrics, dict):
                    continue
                dataset = scoreboard_dataset_name(task_name)
                task_id = await store.get_or_create_task(
                    job_name="lighteval",
                    job_id=None,
                    dataset=dataset,
                    model=model_name,
                    is_param_search=False,
                    allow_resume=False,
                )
                await store.record_score_payload(
                    task_id=task_id,
                    payload={"cot_mode": "NoCoT", "metrics": metrics},
                )
                recorded.append(f"{dataset} -> task {task_id}")
    finally:
        await close_db()
    return recorded


SCOREBOARD_ENV_PREFIXES = ("SCOREBOARD_DB_", "PG")


@contextmanager
def _scoreboard_env(env: dict[str, str]):
    """Expose dotenv-loaded database settings to DatabaseSettings.from_env()."""
    applied: dict[str, str | None] = {}
    for key, value in env.items():
        if key.startswith(SCOREBOARD_ENV_PREFIXES) and os.environ.get(key) != value:
            applied[key] = os.environ.get(key)
            os.environ[key] = value
    try:
        yield
    finally:
        for key, previous in applied.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def ingest_scoreboard_results(
    *, result_files: list[str], model_name: str, root: Path, env: dict[str, str]
) -> None:
    if not result_files:
        print("eval run: no LightEval result files found; nothing to ingest into the scoreboard")
        return
    try:
        with _scoreboard_env(env):
            recorded = asyncio.run(
                _ingest_scoreboard_results(result_files=result_files, model_name=model_name, root=root)
            )
    except Exception as error:  # noqa: BLE001 - scoreboard ingestion must not fail the eval run
        print(f"eval run: scoreboard ingestion failed (results are still on disk): {error}")
        return
    for line in recorded:
        print(f"eval run: scoreboard score recorded: {line}")


def run_eval(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> int:
    args.tasks = resolve_run_tasks(args, config)
    lighteval_plan = build_lighteval_plan(args, root=root, env=env, config=config)
    base_url = base_url_from_lighteval_command(lighteval_plan.command) or local_openai_base_url(
        config, env, args
    )

    manage_server = not getattr(args, "no_server", False) and is_local_base_url(base_url)
    infer_plan: CommandPlan | None = None
    if manage_server:
        infer_plan = build_infer_plan(
            infer_args_namespace(args, port=port_from_base_url(base_url)),
            root=root,
            env=env,
            config=config,
        )

    if args.dry_run:
        if infer_plan is not None:
            print(format_plan_for_display(infer_plan))
        print(format_plan_for_display(lighteval_plan))
        return 0

    output_dir = output_dir_from_command(lighteval_plan.command) or (root / "results/lighteval")
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    server_process: subprocess.Popen[bytes] | None = None
    server_log: Path | None = None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if manage_server and infer_plan is not None:
        if server_is_healthy(base_url):
            print(f"eval run: reusing healthy server at {base_url}")
        else:
            server_log = output_dir / "server_logs" / f"vllm_{stamp}.log"
            server_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"eval run: starting vLLM server (log: {server_log})")
            with server_log.open("wb") as log_file:
                server_process = subprocess.Popen(
                    infer_plan.command,
                    cwd=str(infer_plan.cwd) if infer_plan.cwd else None,
                    env=infer_plan.env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            timeout_s = float(
                pick(getattr(args, "server_timeout", None), DEFAULT_SERVER_TIMEOUT_S)
            )
            wait_for_server(base_url, process=server_process, log_path=server_log, timeout_s=timeout_s)
            print(f"eval run: server healthy at {base_url}")

    performance_output = output_dir / "performance" / f"performance_{stamp}.json"
    metrics_url = getattr(args, "metrics_url", None) or derive_metrics_url(base_url)
    try:
        exit_code = run_lighteval_with_performance(
            lighteval_plan.command,
            cwd=lighteval_plan.cwd,
            env=lighteval_plan.env,
            root=root,
            performance_output=performance_output,
            metrics_url=metrics_url,
            scoreboard_task_id=getattr(args, "scoreboard_task_id", None),
        )
    finally:
        if server_process is not None and not getattr(args, "keep_server", False):
            print("eval run: stopping vLLM server")
            stop_server(server_process)
        elif server_process is not None:
            print(f"eval run: leaving vLLM server running (pid {server_process.pid})")

    if exit_code == 0 and getattr(args, "scoreboard", False):
        try:
            report = json.loads(performance_output.read_text())
        except (OSError, json.JSONDecodeError):
            report = {}
        result_files = report.get("source_files", {}).get("results", [])
        ingest_scoreboard_results(
            result_files=list(result_files),
            model_name=scoreboard_model_name(args, config),
            root=root,
            env=env,
        )
    print(f"eval run: finished with exit code {exit_code}; performance report: {performance_output}")
    return exit_code
