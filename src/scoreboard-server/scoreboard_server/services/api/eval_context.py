from __future__ import annotations

import json
from typing import Any

from scoreboard_server.cores.normalize import stop_token_display
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.eval_context import EvalContextResponse


async def eval_context_response(
    store: ScoreboardStore,
    *,
    task_id: int,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
) -> EvalContextResponse:
    context = await store.get_eval_context_for_space(
        task_id=str(task_id),
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
    )
    event: dict[str, Any] = {"view": "text", "raw_text": "", "context": None, "stop_tokens": {}, "errors": []}
    if context is None:
        event["raw_text"] = "当前样本没有 context 内容。"
        return event
    event["raw_text"] = json.dumps(context, ensure_ascii=False, indent=2) if isinstance(context, dict) else str(context)
    if isinstance(context, dict) and (isinstance(context.get("stages"), list) or isinstance(context.get("sampling_config"), dict)):
        event["view"] = "structured"
        event["context"] = context
        event["stop_tokens"] = _stop_tokens(context.get("sampling_config"))
    return event


def _stop_tokens(sampling_config: Any) -> dict[str, list[dict[str, Any]]]:
    mapping = {}
    if not isinstance(sampling_config, dict):
        return mapping
    for stage_name, stage_cfg in sampling_config.items():
        if not isinstance(stage_cfg, dict):
            continue
        tokens = []
        for token_id in stage_cfg.get("stop_tokens", []):
            try:
                tid = int(token_id)
            except (TypeError, ValueError):
                continue
            tokens.append({"id": tid, "token": stop_token_display(tid)})
        if tokens:
            mapping[str(stage_name)] = tokens
    return mapping
