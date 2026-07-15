from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


AUTO_MODEL_LABEL = "每档最新（调度策略）"
DEFAULT_TABLE_VIEW = "benchmark_detail_delta"
TABLE_VIEW_LABELS: dict[str, str] = {
    "benchmark_detail_latest": "明细（最新）",
    "field_avg_latest": "领域均分（最新）",
    "benchmark_detail_delta": "明细（上一代 vs 最新）",
    "field_avg_delta": "领域均分（上一代 vs 最新）",
}
DOMAIN_GROUPS: tuple[dict[str, Any], ...] = (
    {"key": "knowledge", "label": "Knowledge", "title": "知识类（MMLU / Multi-choice）"},
    {"key": "math", "label": "Math", "title": "数学推理（AIME / Math-500 等）"},
    {"key": "coding", "label": "Coding", "title": "代码"},
    {"key": "agent", "label": "Agent", "title": "Agent 工作流"},
    {"key": "instruction_following", "label": "Instruction Following", "title": "指令遵循（IFEval 等）"},
    {"key": "function_call", "label": "Function Call", "title": "函数调用"},
)
EVAL_PAGE_SIZE = 15

AGENT_BENCHMARK_TOKENS = (
    "agentbench",
    "apex_agent",
    "apex_agents",
    "browsecomp",
    "claweval",
    "deepsearchqa",
    "deepswe",
    "e_bench",
    "hle_with_tools",
    "hy_backend",
    "hy_companybench",
    "hy_euler",
    "hy_finmodelbench",
    "hy_math",
    "hy_skillsworld",
    "hy_swe",
    "mcp_atlas",
    "mcp_bench",
    "nl2repo",
    "prodbench",
    "skillsbench",
    "swe_bench",
    "swebench",
    "tau_bench",
    "tau2_bench",
    "tau3_bench",
    "terminal_bench",
    "terminalbench",
    "toolathlon",
    "wide_search",
    "widesearch",
    "wildclawbench",
)


def now_utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    return now_utc_naive()


def parse_nonneg_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed


def canonical_task_status(status: Any) -> str:
    mapping = {"running": "Running", "completed": "Completed", "failed": "Failed"}
    return mapping.get(str(status or "").strip().lower(), str(status or "Running"))


def canonical_completion_status(status: Any) -> str:
    mapping = {"running": "Running", "completed": "Completed", "failed": "Failed"}
    raw = str(status or "").strip().lower()
    if raw not in mapping:
        raise ValueError(f"unsupported completion status: {status!r}")
    return mapping[raw]


def canonical_cot_mode(payload: Mapping[str, Any]) -> str:
    for candidate in (
        payload.get("cot_mode"),
        (payload.get("task_details") or {}).get("cot_mode")
        if isinstance(payload.get("task_details"), dict)
        else None,
        (payload.get("sampling_config") or {}).get("cot_mode")
        if isinstance(payload.get("sampling_config"), dict)
        else None,
    ):
        raw = str(candidate or "").strip().lower().replace("-", "_")
        if raw in {"nocot", "no_cot"}:
            return "NoCoT"
        if raw == "cot":
            return "CoT"
    return "CoT" if bool(payload.get("cot", False)) else "NoCoT"


def split_dataset(dataset: str) -> tuple[str, str]:
    raw = str(dataset or "").strip()
    for suffix in ("_test", "_eval", "_val", "_dev", "_train"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)], suffix[1:]
    return raw, ""


def join_dataset(benchmark_name: str, benchmark_split: str | None) -> str:
    split = str(benchmark_split or "").strip()
    return f"{benchmark_name}_{split}" if split else benchmark_name


def normalize_model_name(model: str) -> str:
    return str(model or "").strip()


def parse_model_tags(model: str) -> tuple[str, str, str]:
    lowered = normalize_model_name(model).lower().replace("_", "-")
    arch = "unknown"
    data = "unknown"
    params = "unknown"
    arch_match = re.search(r"\brwkv\d+[a-z]*\b", lowered)
    data_match = re.search(r"\bg\d[a-z0-9]*\b", lowered)
    param_match = re.search(r"\b\d+(?:\.\d+)?b\b", lowered)
    if arch_match:
        arch = arch_match.group(0)
    if data_match:
        data = data_match.group(0)
    if param_match:
        params = param_match.group(0).replace(".", "_")
    return arch, data, params


def display_param(param: str | None) -> str:
    return (param or "?").replace("_", ".")


def sanitize_json(value: Any, *, max_depth: int = 6, _depth: int = 0) -> Any:
    if _depth > max_depth:
        return "[truncated depth]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").replace("\x00", "")
    if isinstance(value, Mapping):
        return {str(sanitize_json(k, _depth=_depth + 1)): sanitize_json(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item, _depth=_depth + 1) for item in value]
    if isinstance(value, set):
        return [sanitize_json(item, _depth=_depth + 1) for item in sorted(value, key=str)]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def json_key(value: Any) -> str:
    return json.dumps(sanitize_json(value), ensure_ascii=False, sort_keys=True, default=str)


def git_hash() -> str:
    env_sha = os.environ.get("RWKV_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def iter_stage_indices(payload: Mapping[str, Any]) -> Iterable[int]:
    found: set[int] = set()
    for key in payload:
        match = re.fullmatch(r"prompt(\d+)", str(key))
        if match:
            found.add(int(match.group(1)))
    return sorted(found)


def score_to_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if -1.0 <= value <= 1.0 else value


def numeric_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_from_context(metrics: Mapping[str, Any], sampling_config: Any = None) -> tuple[str | None, float | None]:
    configured: list[str] = []
    if isinstance(sampling_config, Mapping):
        for key in ("display_metric_key", "score_metric_key", "primary_metric_key", "metric_key"):
            value = sampling_config.get(key)
            if isinstance(value, str):
                configured.append(value)
        for avg in _as_iter(sampling_config.get("avg_k")):
            parsed = numeric_value(avg)
            if parsed and parsed > 0:
                configured.append(f"avg@{parsed:g}")
        for pass_k in _as_iter(sampling_config.get("pass_ks")):
            parsed = numeric_value(pass_k)
            if parsed and parsed > 0 and float(parsed).is_integer():
                configured.append(f"pass@{int(parsed)}")
    for key in configured:
        value = numeric_value(metrics.get(key))
        if value is not None:
            return key, value
    for prefix in ("avg@", "pass@"):
        candidates = [
            (str(key), numeric_value(value))
            for key, value in metrics.items()
            if str(key).lower().startswith(prefix) and numeric_value(value) is not None
        ]
        if candidates:
            return max(candidates, key=lambda item: _metric_k(item[0]))  # type: ignore[return-value]
    for key in ("success_rate", "accuracy", "score", "official_score", "f1"):
        value = numeric_value(metrics.get(key))
        if value is not None:
            return key, value
    return None, None


def _as_iter(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    if value is None:
        return ()
    return (value,)


def _metric_k(key: str) -> float:
    match = re.search(r"@(\d+(?:\.\d+)?)", key)
    return float(match.group(1)) if match else 0.0


def domain_for(dataset: str, evaluator: Any = None) -> str:
    token = f"{dataset} {evaluator or ''}".lower()
    normalized_token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    compact_token = re.sub(r"[^a-z0-9]+", "", token)
    if any(part in normalized_token or part in compact_token for part in AGENT_BENCHMARK_TOKENS):
        return "agent"
    if any(part in token for part in ("human_eval", "humaneval", "mbpp", "livecodebench", "code")):
        return "coding"
    if any(part in token for part in ("ifeval", "instruction")):
        return "instruction_following"
    if any(part in token for part in ("bfcl", "mcp", "tau", "browsecomp", "function", "tool")):
        return "function_call"
    if any(part in token for part in ("gsm", "math", "aime", "amc", "minerva", "olympiad", "gaokao", "svamp")):
        return "math"
    return "knowledge"


def is_naive(evaluator: Any, sampling_config: Any) -> bool:
    if str(evaluator or "").endswith("_naive"):
        return True
    if isinstance(sampling_config, str):
        try:
            sampling_config = json.loads(sampling_config)
        except (TypeError, ValueError):
            sampling_config = None
    return isinstance(sampling_config, Mapping) and sampling_config.get("prompt_profile") == "naive"


def stop_token_display(token_id: int) -> str:
    if token_id == 0:
        return "<|endoftext|>"
    if 32 <= token_id <= 126:
        return chr(token_id)
    return f"<token:{token_id}>"
