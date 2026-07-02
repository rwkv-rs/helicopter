from __future__ import annotations

import asyncio
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scoreboard import _scoreboard_import_root


def export_scoreboard_task_results(
    *,
    task_id: int,
    output_path: str | Path,
    repo_root: Path,
    include_context: bool = True,
) -> dict[str, Any]:
    payload = asyncio.run(
        _load_scoreboard_task_results(
            task_id=int(task_id),
            repo_root=repo_root,
            include_context=include_context,
        )
    )
    rows = [
        export_eval_record_row(row, task_id=int(task_id), include_context=include_context)
        for row in payload["rows"]
    ]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return _json_safe(
        {
            "task_id": int(task_id),
            "output_path": str(target),
            "total": len(rows),
            "task": payload["task_bundle"]["task"],
            "benchmark": payload["task_bundle"]["benchmark"],
            "model": payload["task_bundle"]["model"],
            "scores": payload["scores"],
        }
    )


async def _load_scoreboard_task_results(
    *,
    task_id: int,
    repo_root: Path,
    include_context: bool,
) -> dict[str, Any]:
    _scoreboard_import_root(repo_root)
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        task_bundle = await store.get_task_bundle(task_id=str(task_id))
        if task_bundle is None:
            raise ValueError(f"scoreboard task not found: {task_id}")
        rows = await store.list_eval_records_for_space(
            task_id=str(task_id),
            only_wrong=False,
            include_context=include_context,
        )
        scores = await store.list_scores_rows(task_id=str(task_id))
        return {"task_bundle": task_bundle, "rows": rows, "scores": scores}
    finally:
        await close_db()


def export_eval_record_row(
    row: Mapping[str, Any],
    *,
    task_id: int,
    include_context: bool = True,
) -> dict[str, Any]:
    context = row.get("context") if isinstance(row.get("context"), Mapping) else {}
    stage = _first_stage(context)
    metadata = context.get("metadata") if isinstance(context.get("metadata"), Mapping) else None
    sampling_config = context.get("sampling_config") if isinstance(context.get("sampling_config"), Mapping) else None
    payload: dict[str, Any] = {
        "task_id": int(task_id),
        "sample_index": row.get("sample_index"),
        "repeat_index": row.get("repeat_index"),
        "pass_index": row.get("pass_index"),
        "answer": row.get("answer"),
        "reference_answer": row.get("ref_answer"),
        "ref_answer": row.get("ref_answer"),
        "is_passed": bool(row.get("is_passed")),
        "fail_reason": row.get("fail_reason"),
        "metadata": dict(metadata) if metadata is not None else None,
    }
    if include_context:
        payload.update(
            {
                "prompt": stage.get("prompt"),
                "completion": stage.get("completion"),
                "stop_reason": stage.get("stop_reason"),
                "sampling_config": dict(sampling_config) if sampling_config is not None else None,
                "context": dict(context),
            }
        )
    return {key: value for key, value in payload.items() if value is not None}


def _first_stage(context: Mapping[str, Any]) -> Mapping[str, Any]:
    stages = context.get("stages")
    if isinstance(stages, Sequence) and stages:
        first = stages[0]
        if isinstance(first, Mapping):
            return first
    return {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
