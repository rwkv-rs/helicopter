from __future__ import annotations

from typing import Any

from scoreboard_server.cores.normalize import metric_from_context, score_to_percent
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.score_history.index import ScoreHistoryResponse


async def score_history_response(
    store: ScoreboardStore,
    *,
    model: str,
    benchmark: str,
    scope: str = "official",
) -> ScoreHistoryResponse:
    rows = await store.list_score_history(
        model=model,
        dataset=benchmark,
        is_tmp=scope == "non_official",
    )
    points: list[dict[str, Any]] = []
    for row in rows:
        metric, value = metric_from_context(
            row.get("metrics") or {}, row.get("sampling_config")
        )
        percent = score_to_percent(value)
        if metric is None or percent is None:
            continue
        board = (
            "naive" if str(row.get("evaluator") or "").endswith("_naive") else "normal"
        )
        points.append(
            {
                "score_id": row.get("score_id"),
                "task_id": row.get("task_id"),
                "cot_mode": row.get("cot_mode"),
                "evaluator": row.get("evaluator"),
                "board": board,
                "percent": percent,
                "metric": metric,
                "created_at": row.get("created_at").isoformat()
                if hasattr(row.get("created_at"), "isoformat")
                else str(row.get("created_at")),
                "sampling_summary": "",
                "model": row.get("model"),
                "benchmark": row.get("dataset"),
                "visibility": row.get("visibility"),
                "eligibility": row.get("eligibility"),
                "comparable": row.get("comparable"),
                "dirty": row.get("dirty"),
                "samples": row.get("num_samples"),
            }
        )
    groups = []
    for cot_mode in ("NoCoT", "CoT"):
        bucket = [point for point in points if point["cot_mode"] == cot_mode]
        if bucket:
            groups.append({"cot_mode": cot_mode, "points": bucket})
    return {
        "scope": scope,
        "model": model,
        "benchmark": benchmark,
        "total": len(points),
        "groups": groups,
    }
