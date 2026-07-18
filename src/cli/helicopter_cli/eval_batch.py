from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import queue
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark_catalog_defaults import (
    CATALOG_RUN_STATUS,
    CATALOG_SCOPE,
    CATALOG_SOURCE,
    CATALOG_TARGET_KIND,
)
from .commands import table
from .env import pick
from .eval_run import (
    SCOREBOARD_LOCK,
    _scoreboard_env,
    run_eval,
    scoreboard_dataset_name,
    scoreboard_model_name,
)
from .function_calling import run_function_calling_eval


DEFAULT_GPU_IDLE_MIB = 2048.0
DEFAULT_PORT_BASE = 8000

UNIT_KINDS = ("lighteval", "fc")


@dataclass
class BatchUnit:
    model: str
    kind: str  # one of UNIT_KINDS
    tasks: list[str]
    status: str = "pending"
    message: str = ""
    attempts: int = 0
    elapsed_seconds: float = 0.0
    skipped_tasks: list[str] = field(default_factory=list)
    slot_index: int | None = None
    gpu: int | None = None
    port: int | None = None
    exit_code: int | None = None
    started_at: str | None = None
    ended_at: str | None = None


@dataclass
class GpuSlot:
    index: int
    gpu: int | None
    port: int


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_as_str_list(item))
        return result
    return [str(value)]


def batch_config(config: dict[str, Any]) -> dict[str, Any]:
    value = table(config, "eval").get("batch", {})
    return value if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _stamp_from_iso(value: str) -> str:
    return value.replace("-", "").replace(":", "").replace("+00:00", "Z")


def resolve_batch_plan(
    args: Any,
    config: dict[str, Any],
    *,
    lighteval_tasks_override: list[str] | None = None,
) -> list[BatchUnit]:
    batch = batch_config(config)
    models = _as_str_list(pick(getattr(args, "models", None), batch.get("models")))
    lighteval_tasks = list(lighteval_tasks_override or _as_str_list(getattr(args, "tasks", None)))
    fc_tasks = _as_str_list(getattr(args, "fc_tasks", None))
    if not lighteval_tasks and not fc_tasks:
        # No CLI benchmark selection: fall back to the config suite as a whole.
        lighteval_tasks = _as_str_list(batch.get("tasks"))
        fc_tasks = _as_str_list(batch.get("fc_tasks"))

    if not models:
        raise SystemExit("no models given: pass --models or set [eval.batch].models in the config")
    if not lighteval_tasks and not fc_tasks:
        raise SystemExit(
            "no benchmarks given: pass --tasks/--fc-tasks or set [eval.batch].tasks / "
            "[eval.batch].fc_tasks in the config"
        )

    units: list[BatchUnit] = []
    for model in models:
        if lighteval_tasks:
            units.append(BatchUnit(model=model, kind="lighteval", tasks=list(lighteval_tasks)))
        if fc_tasks:
            units.append(BatchUnit(model=model, kind="fc", tasks=list(fc_tasks)))
    return units


def detect_idle_gpus(*, threshold_mib: float = DEFAULT_GPU_IDLE_MIB) -> list[int]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    gpus: list[int] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            used = float(parts[1])
        except ValueError:
            continue
        if used < threshold_mib:
            gpus.append(int(parts[0]))
    return gpus


def resolve_slots(args: Any, config: dict[str, Any], env: dict[str, str]) -> list[GpuSlot]:
    """Build the GPU slots the batch may run on.

    With --no-server or an explicit --base-url the endpoint is external, so
    GPU pinning and per-slot ports do not apply: one generic slot is returned.
    """
    if getattr(args, "no_server", False) or getattr(args, "base_url", None):
        return [GpuSlot(index=0, gpu=None, port=0)]

    port_base = int(pick(getattr(args, "port_base", None), batch_config(config).get("port_base"), DEFAULT_PORT_BASE))
    explicit = _as_str_list(getattr(args, "gpus", None) or batch_config(config).get("gpus"))
    if explicit:
        gpus = [int(item) for item in explicit]
    else:
        gpus = detect_idle_gpus(
            threshold_mib=float(
                pick(
                    getattr(args, "gpu_idle_max_mem", None),
                    batch_config(config).get("gpu_idle_max_mem"),
                    DEFAULT_GPU_IDLE_MIB,
                )
            )
        )
    if not gpus:
        # No visible GPU management available: single slot on the default port.
        return [GpuSlot(index=0, gpu=None, port=port_base)]
    return [GpuSlot(index=i, gpu=gpu, port=port_base + i) for i, gpu in enumerate(gpus)]


async def _query_completed_datasets(
    *,
    model_name: str,
    datasets: list[str],
    root: Path,
) -> set[str]:
    scoreboard_path = root / "src/scoreboard-server"
    if str(scoreboard_path) not in sys.path:
        sys.path.insert(0, str(scoreboard_path))

    from scoreboard_server.cores.normalize import normalize_model_name, split_dataset
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.models import Score
    from scoreboard_server.db.settings import DatabaseSettings

    await init_db(DatabaseSettings.from_env(), generate_schemas=False)
    completed: set[str] = set()
    try:
        normalized_model = normalize_model_name(model_name)
        for dataset in datasets:
            name, split = split_dataset(dataset)
            exists = await Score.filter(
                task__model__model_name=normalized_model,
                task__benchmark__benchmark_name=name,
                task__benchmark__benchmark_split=split,
                task__is_tmp=False,
            ).exists()
            if exists:
                completed.add(dataset)
    finally:
        await close_db()
    return completed


def unit_dataset_names(unit: BatchUnit) -> dict[str, str]:
    """Map each task entry of a unit to its scoreboard benchmark name."""
    if unit.kind == "lighteval":
        return {task: scoreboard_dataset_name(task) for task in unit.tasks}
    return {task: task for task in unit.tasks}


def filter_completed_units(
    units: list[BatchUnit],
    *,
    args: Any,
    config: dict[str, Any],
    env: dict[str, str],
    root: Path,
) -> None:
    """Drop tasks that already have a scoreboard score; mutates units in place."""
    for unit in units:
        mapping = unit_dataset_names(unit)
        unit_args = copy.copy(args)
        unit_args.model = unit.model
        model_name = scoreboard_model_name(unit_args, config)
        try:
            with SCOREBOARD_LOCK, _scoreboard_env(env):
                completed = asyncio.run(
                    _query_completed_datasets(
                        model_name=model_name,
                        datasets=sorted(set(mapping.values())),
                        root=root,
                    )
                )
        except Exception as error:  # noqa: BLE001 - fall back to running everything
            print(f"eval batch: skip-completed check failed ({error}); running all benchmarks")
            return
        remaining = [task for task in unit.tasks if mapping[task] not in completed]
        unit.skipped_tasks = [task for task in unit.tasks if mapping[task] in completed]
        unit.tasks = remaining
        if not remaining:
            unit.status = "skipped"
            unit.message = "all benchmarks already scored"


async def _query_catalog_lighteval_tasks(
    *,
    root: Path,
    env: dict[str, str],
    scope: str,
    fields: list[str],
    limit: int | None,
) -> list[str]:
    scoreboard_path = root / "src/scoreboard-server"
    if str(scoreboard_path) not in sys.path:
        sys.path.insert(0, str(scoreboard_path))

    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        rows = await store.list_benchmark_catalog(
            scope=scope,
            fields=fields or None,
            source=CATALOG_SOURCE,
            target_kind=CATALOG_TARGET_KIND,
            run_status=CATALOG_RUN_STATUS,
            limit=limit,
        )
    finally:
        await close_db()
    return [str(row["name"]) for row in rows]


def query_catalog_lighteval_tasks(*, args: Any, root: Path, env: dict[str, str]) -> list[str]:
    scope = str(
        pick(
            getattr(args, "benchmark_scope", None),
            CATALOG_SCOPE,
        )
    )
    fields = _as_str_list(getattr(args, "benchmark_fields", None))
    limit = getattr(args, "benchmark_limit", None)
    with SCOREBOARD_LOCK, _scoreboard_env(env):
        tasks = asyncio.run(
            _query_catalog_lighteval_tasks(
                root=root,
                env=env,
                scope=scope,
                fields=fields,
                limit=int(limit) if limit else None,
            )
        )
    if not tasks:
        raise SystemExit(f"no LightEval tasks found in benchmark_catalog for scope={scope!r}")
    return tasks


def _safe_unit_slug(unit: BatchUnit, slot: GpuSlot) -> str:
    digest = hashlib.sha1(",".join(unit.tasks).encode("utf-8")).hexdigest()[:8]
    raw = f"slot{slot.index:02d}_gpu{slot.gpu if slot.gpu is not None else 'none'}_{unit.model}_{unit.kind}_{digest}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or f"unit_{digest}"


def _unit_args(args: Any, unit: BatchUnit, slot: GpuSlot) -> Any:
    unit_args = copy.copy(args)
    unit_args.model = unit.model
    unit_args.tasks = ",".join(unit.tasks)
    unit_args.no_server = getattr(args, "no_server", False)
    unit_args.keep_server = False
    if slot.port:
        unit_args.base_url = f"http://127.0.0.1:{slot.port}/v1"
    batch_dir = getattr(args, "_batch_run_dir", None) or getattr(args, "output_dir", None)
    if batch_dir:
        unit_dir = Path(str(batch_dir)) / _safe_unit_slug(unit, slot)
        unit_args.output_dir = str(unit_dir / ("lighteval" if unit.kind == "lighteval" else "function_calling"))
    return unit_args


def _unit_env(env: dict[str, str], slot: GpuSlot) -> dict[str, str]:
    if slot.gpu is None:
        return dict(env)
    slot_env = dict(env)
    slot_env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
    return slot_env


def run_unit(
    unit: BatchUnit,
    *,
    args: Any,
    slot: GpuSlot,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
    max_retries: int,
) -> None:
    label = f"{unit.model}/{unit.kind}"
    unit_args = _unit_args(args, unit, slot)
    unit_env = _unit_env(env, slot)
    runner = run_eval if unit.kind == "lighteval" else run_function_calling_eval
    started = time.monotonic()
    unit.started_at = _utc_now()
    unit.slot_index = slot.index
    unit.gpu = slot.gpu
    unit.port = slot.port or None
    for attempt in range(1, max_retries + 2):
        unit.attempts = attempt
        gpu_note = f" gpu={slot.gpu}" if slot.gpu is not None else ""
        print(f"eval batch: [{label}] attempt {attempt}{gpu_note} tasks={unit_args.tasks}")
        try:
            exit_code = runner(unit_args, root=root, env=unit_env, config=config)
        except SystemExit as error:
            exit_code = 1
            unit.message = str(error)
        except Exception as error:  # noqa: BLE001 - one unit must not kill the batch
            exit_code = 1
            unit.message = f"{type(error).__name__}: {error}"
        unit.exit_code = exit_code
        if exit_code == 0:
            unit.status = "completed"
            unit.message = ""
            break
        unit.status = "failed"
        if not unit.message:
            unit.message = f"exit code {exit_code}"
        if attempt <= max_retries:
            print(f"eval batch: [{label}] failed ({unit.message}); retrying")
    unit.elapsed_seconds = time.monotonic() - started
    unit.ended_at = _utc_now()


def format_summary(units: list[BatchUnit]) -> str:
    lines = ["model\tkind\tstatus\tattempts\telapsed_s\ttasks\tskipped\tmessage"]
    for unit in units:
        lines.append(
            "\t".join(
                [
                    unit.model,
                    unit.kind,
                    unit.status,
                    str(unit.attempts),
                    f"{unit.elapsed_seconds:.0f}",
                    ",".join(unit.tasks) or "-",
                    ",".join(unit.skipped_tasks) or "-",
                    unit.message or "-",
                ]
            )
        )
    return "\n".join(lines)


def _unit_record(unit: BatchUnit) -> dict[str, Any]:
    return {
        "model": unit.model,
        "kind": unit.kind,
        "tasks": list(unit.tasks),
        "skipped_tasks": list(unit.skipped_tasks),
        "status": unit.status,
        "message": unit.message,
        "attempts": unit.attempts,
        "elapsed_seconds": unit.elapsed_seconds,
        "slot_index": unit.slot_index,
        "gpu": unit.gpu,
        "port": unit.port,
        "exit_code": unit.exit_code,
        "started_at": unit.started_at,
        "ended_at": unit.ended_at,
    }


def _slot_record(slot: GpuSlot) -> dict[str, Any]:
    return {"index": slot.index, "gpu": slot.gpu, "port": slot.port or None}


def _status_counts(units: list[BatchUnit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit.status] = counts.get(unit.status, 0) + 1
    return counts


def resolve_batch_report_path(args: Any, config: dict[str, Any], root: Path, *, started_at: str) -> Path | None:
    batch = batch_config(config)
    configured = pick(getattr(args, "batch_output", None), batch.get("output"))
    if configured:
        path = Path(str(configured))
    elif getattr(args, "dry_run", False):
        return None
    else:
        path = Path("results/eval_batch") / f"batch_{_stamp_from_iso(started_at)}.json"
    if not path.is_absolute():
        path = root / path
    return path


def write_batch_report(
    *,
    path: Path | None,
    units: list[BatchUnit],
    slots: list[GpuSlot],
    args: Any,
    exit_code: int,
    started_at: str,
    ended_at: str,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "started_at": started_at,
        "ended_at": ended_at,
        "exit_code": exit_code,
        "summary": {
            "units": len(units),
            "status_counts": _status_counts(units),
        },
        "selection": {
            "models": _as_str_list(getattr(args, "models", None)),
            "tasks": _as_str_list(getattr(args, "tasks", None)),
            "fc_tasks": _as_str_list(getattr(args, "fc_tasks", None)),
            "scoreboard": bool(getattr(args, "scoreboard", False)),
            "rerun": bool(getattr(args, "rerun", False)),
            "dry_run": bool(getattr(args, "dry_run", False)),
        },
        "runtime": {
            "parallel": getattr(args, "parallel", None),
            "max_retries": getattr(args, "max_retries", None),
            "no_server": bool(getattr(args, "no_server", False)),
            "base_url": getattr(args, "base_url", None),
            "port_base": getattr(args, "port_base", None),
            "wkv_mode": getattr(args, "wkv_mode", None),
            "emb_device": getattr(args, "emb_device", None),
            "tensor_parallel_size": getattr(args, "tensor_parallel_size", None),
            "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", None),
            "max_num_seqs": getattr(args, "max_num_seqs", None),
            "max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None),
            "vllm_env": getattr(args, "vllm_env", None),
        },
        "slots": [_slot_record(slot) for slot in slots],
        "units": [_unit_record(unit) for unit in units],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"eval batch: wrote report {path}")


def run_batch(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> int:
    started_at = _utc_now()
    report_path: Path | None = None
    slots = resolve_slots(args, config, env)
    report_path = resolve_batch_report_path(args, config, root, started_at=started_at)
    batch_run_dir = (
        report_path.with_suffix("")
        if report_path is not None
        else root / "results/eval_batch" / f"dry_run_{_stamp_from_iso(started_at)}"
    )
    setattr(args, "_batch_run_dir", str(batch_run_dir))

    lighteval_tasks_override = None
    if getattr(args, "tasks_from_db", False):
        if getattr(args, "tasks", None):
            raise SystemExit("--tasks-from-db cannot be combined with --tasks")
        lighteval_tasks_override = query_catalog_lighteval_tasks(args=args, root=root, env=env)
    units = resolve_batch_plan(args, config, lighteval_tasks_override=lighteval_tasks_override)

    skip_completed = bool(getattr(args, "scoreboard", False)) and not getattr(args, "rerun", False)
    if skip_completed:
        filter_completed_units(units, args=args, config=config, env=env, root=root)

    runnable = [unit for unit in units if unit.status == "pending"]
    if getattr(args, "dry_run", False):
        print(
            f"eval batch: {len(runnable)} unit(s) over {len(slots)} slot(s): "
            + "; ".join(f"{unit.model}/{unit.kind}:{','.join(unit.tasks)}" for unit in runnable)
        )
        for unit in runnable:
            run_unit(
                unit,
                args=args,
                slot=slots[0],
                root=root,
                env=env,
                config=config,
                max_retries=0,
            )
            if unit.status == "completed":
                unit.status = "dry_run"
                unit.message = "plan only"
        failed = [unit for unit in units if unit.status == "failed"]
        exit_code = 1 if failed else 0
        write_batch_report(
            path=report_path,
            units=units,
            slots=slots,
            args=args,
            exit_code=exit_code,
            started_at=started_at,
            ended_at=_utc_now(),
        )
        return exit_code

    if not runnable:
        print("eval batch: nothing to run (all benchmarks already scored)")
        print(format_summary(units))
        write_batch_report(
            path=report_path,
            units=units,
            slots=slots,
            args=args,
            exit_code=0,
            started_at=started_at,
            ended_at=_utc_now(),
        )
        return 0

    workers = max(1, min(int(getattr(args, "parallel", None) or 1), len(slots), len(runnable)))
    max_retries = max(0, int(getattr(args, "max_retries", None) or 0))
    print(
        f"eval batch: running {len(runnable)} unit(s) on {len(slots)} slot(s) "
        f"with {workers} worker(s)"
    )

    slot_queue: queue.Queue[GpuSlot] = queue.Queue()
    for slot in slots:
        slot_queue.put(slot)

    def worker(unit: BatchUnit) -> None:
        slot = slot_queue.get()
        try:
            run_unit(
                unit,
                args=args,
                slot=slot,
                root=root,
                env=env,
                config=config,
                max_retries=max_retries,
            )
        finally:
            slot_queue.put(slot)

    if workers == 1:
        for unit in runnable:
            worker(unit)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker, unit) for unit in runnable]
            for future in futures:
                future.result()

    print(format_summary(units))
    failed = [unit for unit in units if unit.status == "failed"]
    exit_code = 1 if failed else 0
    write_batch_report(
        path=report_path,
        units=units,
        slots=slots,
        args=args,
        exit_code=exit_code,
        started_at=started_at,
        ended_at=_utc_now(),
    )
    return exit_code
