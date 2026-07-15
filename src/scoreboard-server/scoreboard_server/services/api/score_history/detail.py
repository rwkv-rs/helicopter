from __future__ import annotations

from typing import Any

from scoreboard_server.cores.normalize import metric_from_context, score_to_percent, stop_token_display
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.score_history.detail import ScoreHistoryDetailResponse


async def score_history_detail_response(store: ScoreboardStore, *, task_id: int) -> ScoreHistoryDetailResponse:
    detail = await store.get_score_history_detail(task_id=str(task_id))
    if detail is None:
        return {"found": False, "task_id": task_id}
    score = detail.get("score") or {}
    task = detail.get("task") or {}
    metric, value = metric_from_context(score.get("metrics") or {}, task.get("sampling_config"))
    context = detail.get("context") if isinstance(detail.get("context"), dict) else {}
    return {
        "found": True,
        "task_id": task_id,
        "model": score.get("model"),
        "benchmark": score.get("dataset"),
        "cot_mode": score.get("cot_mode"),
        "evaluator": task.get("evaluator"),
        "board": "naive" if str(task.get("evaluator") or "").endswith("_naive") else "normal",
        "metric": metric,
        "percent": score_to_percent(value),
        "metrics": score.get("metrics") or {},
        "sampling": _sampling_detail(task.get("sampling_config")),
        "stages": [
            {
                "prompt": str(stage.get("prompt") or ""),
                "completion": str(stage.get("completion") or ""),
                "stop_reason": stage.get("stop_reason"),
            }
            for stage in (context.get("stages") or [])
            if isinstance(stage, dict)
        ],
    }


def _sampling_detail(sampling_config: Any) -> dict[str, Any]:
    config = sampling_config if isinstance(sampling_config, dict) else {}
    nested = config.get("sampling_config") if isinstance(config.get("sampling_config"), dict) else {}
    stages = {}
    for stage_name, stage_cfg in nested.items():
        cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
        stages[str(stage_name)] = {
            "temperature": cfg.get("temperature"),
            "top_k": cfg.get("top_k"),
            "top_p": cfg.get("top_p"),
            "max_tokens": cfg.get("max_new_tokens"),
            "stop_tokens": [
                {"id": int(token_id), "token": stop_token_display(int(token_id))}
                for token_id in cfg.get("stop_tokens", [])
                if str(token_id).lstrip("-").isdigit()
            ],
            "penalties": {
                "presence_penalty": cfg.get("presence_penalty"),
                "repetition_penalty": cfg.get("repetition_penalty"),
                "penalty_decay": cfg.get("penalty_decay"),
            },
        }
    return {
        "stages": stages,
        "effective_sample_count": config.get("effective_sample_count"),
        "avg_k": config.get("avg_k"),
        "pass_ks": config.get("pass_ks"),
        "n_shot": config.get("n_shot"),
        "sample_limit": config.get("sample_limit"),
        "prompt_profile": config.get("prompt_profile"),
    }
