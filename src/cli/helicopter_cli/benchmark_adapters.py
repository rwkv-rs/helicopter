from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


SWE_BENCH_DATASETS: dict[str, dict[str, str]] = {
    "swe_bench": {
        "hf_dataset": "princeton-nlp/SWE-bench",
        "harness_dataset": "princeton-nlp/SWE-bench",
    },
    "swe_bench_lite": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
    "swe_bench_verified": {
        "hf_dataset": "princeton-nlp/SWE-bench_Verified",
        "harness_dataset": "princeton-nlp/SWE-bench_Verified",
    },
    "swe_bench_lite_oracle": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite_oracle",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
    "swe_bench_lite_bm25_13k": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite_bm25_13K",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
}

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_BLOCK_RE = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_JSON_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

TAU_ADAPTER_BENCHMARKS: dict[str, dict[str, str]] = {
    "tau_bench_airline": {"domain": "airline", "version": "tau_v1"},
    "tau_bench_retail": {"domain": "retail", "version": "tau_v1"},
    "tau2_bench_airline": {"domain": "airline", "version": "tau_v2"},
    "tau2_bench_retail": {"domain": "retail", "version": "tau_v2"},
    "tau2_bench_telecom": {"domain": "telecom", "version": "tau_v2"},
    "tau3_bench_airline": {"domain": "airline", "version": "tau_v3"},
    "tau3_bench_banking_knowledge": {"domain": "banking_knowledge", "version": "tau_v3"},
    "tau3_bench_mock": {"domain": "mock", "version": "tau3_light"},
    "tau3_bench_mock_long_context": {"domain": "mock", "version": "tau3_light_long_context"},
    "tau3_bench_retail": {"domain": "retail", "version": "tau_v3"},
    "tau3_bench_telecom": {"domain": "telecom", "version": "tau_v3"},
}

TAU3_LIGHT_MOCK_TASK_IDS = frozenset(
    {
        "create_task_1_with_env_assertions",
        "update_task_with_history_and_env_assertions",
        "impossible_task_1",
    }
)

MCP_BENCH_TASK_FILES: dict[str, tuple[str, ...]] = {
    "mcp_bench": (
        "mcpbench_tasks_single_runner_format.json",
        "mcpbench_tasks_multi_2server_runner_format.json",
        "mcpbench_tasks_multi_3server_runner_format.json",
    ),
    "mcp_bench_single": ("mcpbench_tasks_single_runner_format.json",),
    "mcp_bench_multi_2server": ("mcpbench_tasks_multi_2server_runner_format.json",),
    "mcp_bench_multi_3server": ("mcpbench_tasks_multi_3server_runner_format.json",),
}

MCP_BENCH_PASS_THRESHOLD = 7.0
MCP_BENCH_DEFAULT_MAX_HISTORY_CHARS = 16000
MCP_BENCH_DEFAULT_MAX_TOOL_SCHEMA_CHARS = 2000
MCP_BENCH_DEFAULT_MAX_RESULT_CHARS = 8000
MCP_BENCH_DEFAULT_MAX_ERROR_CHARS = 2000


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class SweBenchRecord:
    task_id: str
    instance_id: str
    prompt: str
    repo: str
    base_commit: str
    hints_text: str
    retrieved_context: str
    source_dataset: str
    harness_dataset_name: str
    patch: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TauAdapterRecord:
    task_id: str
    domain: str
    instruction: str
    task: dict[str, Any]
    benchmark_version: str
    index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class McpBenchTaskSpec:
    task_id: str
    task_description: str
    fuzzy_description: str = ""
    dependency_analysis: str = ""
    distraction_servers: tuple[str, ...] = ()


@dataclass(frozen=True)
class McpBenchRecord:
    task_file: str
    server_name: str
    combination_name: str
    combination_type: str
    servers: tuple[str, ...]
    task: McpBenchTaskSpec
    runtime_root: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class McpToolCall:
    full_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class McpDecision:
    final_answer: str
    tool_call: McpToolCall | None


@dataclass
class TauGenerationStage:
    prompt: str
    completion: str
    finish_reason: str
    parsed_name: str = ""
    parse_error: str = ""


@dataclass(frozen=True)
class TauExternalModelConfig:
    api_key: str
    model_name: str
    base_url: str | None = None


def load_suite(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def suite_adapter_entries(suite: Mapping[str, Any], benchmark_names: Sequence[str]) -> list[tuple[str, dict[str, Any]]]:
    raw_benchmarks = suite.get("benchmarks", {})
    if not isinstance(raw_benchmarks, Mapping):
        raise SystemExit("suite file must contain a [benchmarks] table")
    wanted = {name for name in benchmark_names if name}
    selected: list[tuple[str, dict[str, Any]]] = []
    for name, raw_entry in raw_benchmarks.items():
        if wanted and name not in wanted:
            continue
        entry = dict(raw_entry) if isinstance(raw_entry, Mapping) else {}
        if str(entry.get("status") or "") == "adapter" or entry.get("adapter"):
            selected.append((str(name), entry))
    missing = sorted(wanted - {name for name, _entry in selected})
    if missing:
        raise SystemExit(f"selected benchmarks are not configured as adapters: {', '.join(missing)}")
    if not selected:
        raise SystemExit("suite adapter selection is empty")
    return selected


def _read_json_or_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    payload = json.loads(text)
                    if isinstance(payload, Mapping):
                        rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping) and isinstance(payload.get("instances"), list):
        return [item for item in payload["instances"] if isinstance(item, Mapping)]
    raise ValueError(f"unsupported JSON source for SWE-Bench rows: {path}")


def _load_local_rows(path: Path) -> list[Mapping[str, Any]]:
    target = path.expanduser().resolve()
    if target.is_dir():
        candidates = sorted(target.glob("*.jsonl")) + sorted(target.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"no JSON/JSONL SWE-Bench source files under {target}")
        target = candidates[0]
    return _read_json_or_jsonl(target)


def load_swebench_source_rows(dataset_name: str, split: str) -> list[Mapping[str, Any]]:
    source_override = os.environ.get("HELICOPTER_SWEBENCH_SOURCE") or os.environ.get("RWKV_SKILLS_SWEBENCH_SOURCE")
    if source_override:
        return _load_local_rows(Path(source_override))
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install datasets to load SWE-Bench sources") from exc
    dataset = load_dataset(dataset_name, split=split)
    return sorted(dataset, key=lambda item: str(item.get("instance_id", "")))


def _extract_context(row: Mapping[str, Any]) -> str:
    for key in (
        "retrieved_context",
        "context",
        "text",
        "file_context",
        "oracle_context",
        "bm25_context",
        "repo_context",
    ):
        value = row.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)
    return ""


def normalize_swebench_row(
    row: Mapping[str, Any],
    *,
    source_dataset: str,
    harness_dataset: str,
) -> SweBenchRecord:
    instance_id = str(row.get("instance_id") or row.get("task_id") or row.get("id") or "").strip()
    if not instance_id:
        raise ValueError(f"SWE-Bench row missing instance_id: {row}")
    problem_statement = str(row.get("problem_statement") or row.get("prompt") or row.get("issue") or "").strip()
    metadata = {
        "instance_id": instance_id,
        "repo": str(row.get("repo") or "").strip(),
        "base_commit": str(row.get("base_commit") or "").strip(),
        "source_dataset": source_dataset,
        "harness_dataset_name": harness_dataset,
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": row.get("PASS_TO_PASS"),
        "environment_setup_commit": row.get("environment_setup_commit"),
        "difficulty": row.get("difficulty"),
    }
    return SweBenchRecord(
        task_id=instance_id,
        instance_id=instance_id,
        prompt=problem_statement,
        repo=str(metadata["repo"]),
        base_commit=str(metadata["base_commit"]),
        hints_text=str(row.get("hints_text") or "").strip(),
        retrieved_context=_extract_context(row),
        source_dataset=source_dataset,
        harness_dataset_name=harness_dataset,
        patch=str(row.get("patch") or ""),
        metadata=metadata,
    )


def load_swebench_records(benchmark_name: str, entry: Mapping[str, Any], *, split: str) -> list[SweBenchRecord]:
    spec = SWE_BENCH_DATASETS.get(benchmark_name, {})
    source_dataset = str(entry.get("hf_dataset") or spec.get("hf_dataset") or "")
    harness_dataset = str(entry.get("harness_dataset") or spec.get("harness_dataset") or "")
    if not source_dataset or not harness_dataset:
        raise ValueError(f"{benchmark_name} is missing hf_dataset/harness_dataset adapter metadata")
    rows = load_swebench_source_rows(source_dataset, split)
    return [
        normalize_swebench_row(row, source_dataset=source_dataset, harness_dataset=harness_dataset)
        for row in rows
    ]


def select_records(records: Sequence[SweBenchRecord], *, max_samples: int | None, sample_seed: int | None) -> list[tuple[int, SweBenchRecord]]:
    indexed = list(enumerate(records))
    if max_samples is None or max_samples <= 0 or max_samples >= len(indexed):
        return indexed
    if sample_seed is None:
        return indexed[:max_samples]
    rng = random.Random(sample_seed)
    return sorted(rng.sample(indexed, max_samples), key=lambda item: item[0])


def build_swebench_prompt(record: SweBenchRecord, *, max_context_chars: int | None = None, prompt_profile: str = "normal") -> str:
    if str(prompt_profile or "normal").strip().lower() == "naive":
        return f"User: {record.prompt.strip()}\n\nAssistant: <think>\n</think>\n```diff\n"
    retrieved_context = record.retrieved_context.strip()
    if max_context_chars is not None and max_context_chars > 0 and len(retrieved_context) > max_context_chars:
        retrieved_context = retrieved_context[:max_context_chars].rstrip()
    lines = [
        "User: You are resolving a real GitHub issue from SWE-bench.",
        "Return only a unified git diff patch. Do not include prose, commands, or markdown outside the patch.",
        "The patch must be applicable with git apply from the repository root.",
        "",
        f"Instance: {record.instance_id}",
    ]
    if record.repo:
        lines.append(f"Repository: {record.repo}")
    if record.base_commit:
        lines.append(f"Base commit: {record.base_commit}")
    lines.extend(["", "Issue:", record.prompt.strip()])
    if record.hints_text:
        lines.extend(["", "Hints:", record.hints_text])
    if retrieved_context:
        lines.extend(["", "Retrieved repository context:", retrieved_context])
    lines.extend(["", "Assistant: <think>\n</think>\n```diff\n"])
    return "\n".join(lines)


def extract_swebench_patch(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    matches = list(_FENCED_BLOCK_RE.finditer(cleaned))
    if matches:
        cleaned = matches[-1].group(1).strip()
    for marker in ("diff --git ", "--- a/"):
        index = cleaned.find(marker)
        if index >= 0:
            return cleaned[index:].strip()
    return ""


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[str, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1, int(max_tokens)),
        "temperature": float(temperature),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "authorization": f"Bearer {api_key or 'EMPTY'}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible endpoint HTTP {exc.code}: {detail}") from exc
    choices = data.get("choices") if isinstance(data, Mapping) else None
    if not isinstance(choices, list) or not choices:
        return "", ""
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    content = str(message.get("content") or choice.get("text") or "")
    finish_reason = str(choice.get("finish_reason") or "")
    return content, finish_reason


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {"value": str(value)}


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text.removeprefix("export ").strip()
    key, separator, value = text.partition("=")
    if not separator:
        return None
    key = key.strip()
    if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_adapter_env_file(path: str | None = None) -> str | None:
    candidates = [
        path,
        os.environ.get("HELICOPTER_ENV_FILE"),
        ".env",
    ]
    loaded_paths: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        target = Path(candidate).expanduser().resolve()
        if not target.is_file():
            continue
        for line in target.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_assignment(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)
        loaded_paths.append(str(target))
    return os.pathsep.join(loaded_paths) if loaded_paths else None


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        text = value.strip()
        if text:
            return text
    return None


def _first_value(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = value.strip()
        if text:
            return text
    return None


def _normalize_openai_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().rstrip("/")
    if not text:
        return None
    suffix = "/chat/completions"
    if text.endswith(suffix):
        text = text[: -len(suffix)].rstrip("/")
    return text or None


def resolve_tau_user_model_config(args: argparse.Namespace) -> TauExternalModelConfig:
    api_key = _first_value(
        getattr(args, "tau_user_api_key", None),
        _first_env("USER_API_KEY", "API_KEY", "OPENAI_API_KEY"),
    )
    model_name = _first_value(
        getattr(args, "tau_user_model", None),
        _first_env("USER_MODEL_NAME", "model_name", "MODEL_NAME"),
    )
    base_url = _first_value(
        getattr(args, "tau_user_base_url", None),
        _first_env("USER_BASE_URL", "OPENAI_BASE_URL", "API_BASE", "BASE_URL"),
    )
    missing = []
    if not api_key:
        missing.append("USER_API_KEY")
    if not model_name:
        missing.append("USER_MODEL_NAME")
    if missing:
        raise ValueError(
            "official TAU adapters require user simulator config in .env: "
            + ", ".join(missing)
        )
    return TauExternalModelConfig(
        api_key=str(api_key),
        model_name=str(model_name),
        base_url=_normalize_openai_base_url(base_url),
    )


def resolve_tau_judge_model_config(
    args: argparse.Namespace,
    *,
    default_model: TauExternalModelConfig,
) -> TauExternalModelConfig:
    explicit_model = _first_value(getattr(args, "tau_judge_model", None), _first_env("JUDGE_MODEL"))
    model_name = explicit_model or default_model.model_name
    api_key = _first_value(
        getattr(args, "tau_judge_api_key", None),
        _first_env("JUDGE_API_KEY"),
        default_model.api_key if explicit_model is None else None,
    )
    base_url = _first_value(
        getattr(args, "tau_judge_base_url", None),
        _first_env("JUDGE_BASE_URL"),
        default_model.base_url if explicit_model is None else None,
    )
    if not api_key:
        raise ValueError("official TAU adapters require JUDGE_API_KEY in .env or a user-model fallback")
    return TauExternalModelConfig(
        api_key=str(api_key),
        model_name=str(model_name),
        base_url=_normalize_openai_base_url(base_url),
    )


def _tau_litellm_model_name(model_config: TauExternalModelConfig) -> str:
    model_name = str(model_config.model_name or "").strip()
    if model_name.startswith("openai/"):
        return model_name.removeprefix("openai/")
    if not model_name or "/" in model_name:
        return model_name
    base_url = _normalize_openai_base_url(model_config.base_url) or ""
    if "api.deepseek.com" in base_url and model_name.startswith("deepseek-"):
        return f"deepseek/{model_name}"
    return model_name


def _tau_litellm_provider_args(model_config: TauExternalModelConfig) -> dict[str, str]:
    model_name = str(model_config.model_name or "").strip()
    base_url = _normalize_openai_base_url(model_config.base_url) or ""
    if "api.deepseek.com" in base_url:
        return {}
    if model_name.startswith("openai/") or (base_url and "/" not in model_name):
        return {"custom_llm_provider": "openai"}
    return {}


def _tau_openai_temperature(value: float) -> float:
    return max(0.001, float(value))


def _tau_timeout_args() -> dict[str, float]:
    value = _first_env("RWKV_TAU_LLM_TIMEOUT_S", "RWKV_TAU_USER_TIMEOUT_S", "RWKV_LLM_TIMEOUT_S")
    if not value:
        return {}
    try:
        parsed = float(value)
    except ValueError:
        return {}
    if parsed <= 0:
        return {}
    return {"timeout": parsed}


def _resolve_tau_bench_root(raw_root: str | None = None) -> Path:
    repo_root = _repo_root()
    candidates = [
        raw_root,
        os.environ.get("HELICOPTER_TAU2_BENCH_ROOT"),
        os.environ.get("RWKV_TAU3_BENCH_ROOT"),
        os.environ.get("TAU3_BENCH_ROOT"),
        os.environ.get("RWKV_TAU2_BENCH_ROOT"),
        os.environ.get("TAU2_BENCH_ROOT"),
        str(repo_root / "references/tau2-bench"),
        "/tmp/tau2-bench",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(str(candidate)).expanduser().resolve()
        if (root / "src" / "tau2").is_dir() and (root / "data" / "tau2").is_dir():
            return root
    checked = ", ".join(str(item) for item in candidates if item)
    raise FileNotFoundError(
        "tau2-bench reference checkout not found; set HELICOPTER_TAU2_BENCH_ROOT. "
        f"Checked: {checked}"
    )


def _ensure_tau2_runtime_path(raw_root: str | None = None, raw_data_root: str | None = None) -> Path:
    root = _resolve_tau_bench_root(raw_root)
    src_root = root / "src"
    data_root = Path(raw_data_root).expanduser().resolve() if raw_data_root else root / "data"
    if not (src_root / "tau2").is_dir():
        raise FileNotFoundError(f"missing tau2 source package under {src_root}")
    if not (data_root / "tau2").is_dir():
        raise FileNotFoundError(f"missing tau2 data package under {data_root}")
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    os.environ.setdefault("TAU2_DATA_DIR", str(data_root))
    return root


def _resolve_tau_v1_root(raw_root: str | None = None) -> Path:
    repo_root = _repo_root()
    candidates = [
        raw_root,
        os.environ.get("HELICOPTER_TAU_V1_ROOT"),
        os.environ.get("RWKV_TAU_V1_ROOT"),
        os.environ.get("TAU_V1_ROOT"),
        str(repo_root / "references/tau-bench"),
        str(repo_root / "references/tau_v1"),
        "/tmp/tau-bench",
        "/tmp/tau_v1",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(str(candidate)).expanduser().resolve()
        if (root / "tau_bench" / "envs").is_dir():
            return root
    checked = ", ".join(str(item) for item in candidates if item)
    raise FileNotFoundError(
        "tau-bench v1 runtime not found; set HELICOPTER_TAU_V1_ROOT. "
        f"Checked: {checked}"
    )


def _ensure_tau_v1_runtime_path(raw_root: str | None = None) -> Path:
    root = _resolve_tau_v1_root(raw_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _tau_reward_basis_without_nl(criteria: dict[str, Any]) -> list[str]:
    raw = criteria.get("reward_basis") or []
    values = [str(item).removeprefix("RewardType.") for item in raw]
    values = [item for item in values if item != "NL_ASSERTION"]
    if values:
        return values
    if criteria.get("env_assertions"):
        values.append("ENV_ASSERTION")
    if criteria.get("actions"):
        values.append("ACTION")
    if criteria.get("communicate_info"):
        values.append("COMMUNICATE")
    return values


def _sanitize_tau3_light_mock_task(task_payload: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    task = json.loads(json.dumps(dict(task_payload), ensure_ascii=False, default=str))
    criteria = task.get("evaluation_criteria")
    if isinstance(criteria, dict):
        criteria.pop("nl_assertions", None)
        criteria["reward_basis"] = _tau_reward_basis_without_nl(criteria)
    task_id = str(task.get("id") or f"tau3_light_mock_{index}")
    return {
        "task_id": task_id,
        "domain": "mock",
        "index": int(index),
        "instruction": _tau_task_instruction(task),
        "task": task,
        "benchmark_version": "tau3_light",
    }


def _tau_task_instruction(task: Mapping[str, Any]) -> str:
    ticket = str(task.get("ticket") or "").strip()
    if ticket:
        return ticket
    scenario = task.get("user_scenario")
    if isinstance(scenario, Mapping):
        instructions = scenario.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            return instructions.strip()
    description = task.get("description")
    if isinstance(description, Mapping):
        purpose = str(description.get("purpose") or "").strip()
        if purpose:
            return purpose
    return str(task.get("id") or "").strip()


def _long_tau_archive(*, target_line: str, label: str) -> str:
    before = [
        f"{label} archive row {idx:03d}: unrelated task inventory, no current user request."
        for idx in range(90)
    ]
    after = [
        f"{label} archive appendix {idx:03d}: historical note, ignore unless directly relevant."
        for idx in range(90)
    ]
    return "\n".join(
        [
            "Reference archive for prior support sessions.",
            *before,
            target_line,
            *after,
            "End of reference archive.",
        ]
    )


def _tau3_long_context_create_task() -> dict[str, Any]:
    return {
        "id": "mock_long_context_create_task",
        "description": {
            "purpose": "Lightweight tau3-style long-context tool-use task.",
            "notes": "The current request is buried in a long prior archive before the final user turn.",
        },
        "user_scenario": {
            "persona": "Professional and direct communicator",
            "instructions": "Create one task for the user after reading the current request.",
        },
        "ticket": "Create a task named Important Meeting for user_1.",
        "initial_state": {
            "message_history": [
                {
                    "role": "user",
                    "content": _long_tau_archive(
                        target_line="Current request evidence: user_1 needs a task titled Important Meeting.",
                        label="create",
                    ),
                    "turn_idx": 0,
                },
                {
                    "role": "assistant",
                    "content": "I have loaded the reference archive and will use only the current request.",
                    "turn_idx": 0,
                },
                {
                    "role": "user",
                    "content": "Please create a task titled Important Meeting for user_1 now.",
                    "turn_idx": 1,
                },
            ]
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "create_important_meeting",
                    "name": "create_task",
                    "arguments": {"user_id": "user_1", "title": "Important Meeting"},
                    "info": "Create the requested task.",
                }
            ],
            "env_assertions": [
                {
                    "env_type": "assistant",
                    "func_name": "assert_task_status",
                    "arguments": {"task_id": "task_2", "expected_status": "pending"},
                }
            ],
            "reward_basis": ["DB", "ENV_ASSERTION"],
        },
    }


def _tau3_long_context_update_task() -> dict[str, Any]:
    return {
        "id": "mock_long_context_update_task",
        "description": {
            "purpose": "Lightweight tau3-style long-context state update task.",
            "notes": "A previous tool call creates task_2; the current request asks the agent to update it.",
        },
        "user_scenario": {
            "persona": "Professional and direct communicator",
            "instructions": "Continue the task-management conversation and update the existing task.",
        },
        "ticket": "Mark task_2 as completed.",
        "initial_state": {
            "message_history": [
                {
                    "role": "user",
                    "content": "I need to create a task for the project review meeting.",
                    "turn_idx": 0,
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "create_task",
                            "arguments": {
                                "user_id": "user_1",
                                "title": "Project Review",
                                "description": "Review Q4 project status",
                            },
                        }
                    ],
                    "turn_idx": 0,
                },
                {
                    "role": "tool",
                    "id": "call_1",
                    "content": (
                        '{"task_id":"task_2","title":"Project Review",'
                        '"description":"Review Q4 project status","status":"pending"}'
                    ),
                    "turn_idx": 0,
                },
                {
                    "role": "assistant",
                    "content": "The Project Review task was created with ID task_2 and is pending.",
                    "turn_idx": 0,
                },
                {
                    "role": "user",
                    "content": _long_tau_archive(
                        target_line="Current request evidence: task_2 must be marked completed.",
                        label="update",
                    ),
                    "turn_idx": 1,
                },
                {
                    "role": "user",
                    "content": "Please mark task_2 completed now.",
                    "turn_idx": 2,
                },
            ]
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "complete_task_2",
                    "name": "update_task_status",
                    "arguments": {"task_id": "task_2", "status": "completed"},
                    "info": "Mark the existing task completed.",
                }
            ],
            "env_assertions": [
                {
                    "env_type": "assistant",
                    "func_name": "assert_task_status",
                    "arguments": {"task_id": "task_2", "expected_status": "completed"},
                }
            ],
            "reward_basis": ["DB", "ENV_ASSERTION"],
        },
    }


def _tau3_mock_long_context_rows() -> list[dict[str, Any]]:
    return [
        _sanitize_tau3_light_mock_task(task, index=index)
        for index, task in enumerate((_tau3_long_context_create_task(), _tau3_long_context_update_task()))
    ]


def _normalize_official_tau_task_row(task: Any, *, domain: str, version: str, index: int) -> dict[str, Any]:
    payload = _dump_model(task)
    task_id = str(getattr(task, "id", None) or payload.get("id") or index)
    return {
        "task_id": task_id,
        "domain": domain,
        "index": int(index),
        "instruction": _tau_task_instruction(payload),
        "task": payload,
        "benchmark_version": version,
    }


def _tau_v1_task_source(domain: str, split: str) -> tuple[str, str]:
    domain = str(domain).strip().lower()
    split = str(split or "test").strip().lower()
    if domain == "airline":
        mapping = {"test": ("tau_bench.envs.airline.tasks_test", "TASKS")}
    elif domain == "retail":
        mapping = {
            "test": ("tau_bench.envs.retail.tasks_test", "TASKS_TEST"),
            "train": ("tau_bench.envs.retail.tasks_train", "TASKS_TRAIN"),
            "dev": ("tau_bench.envs.retail.tasks_dev", "TASKS_DEV"),
        }
    else:
        raise ValueError(f"unsupported tau-bench v1 domain: {domain}")
    try:
        return mapping[split]
    except KeyError as exc:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"tau-bench v1 domain={domain} does not support split={split}; valid: {valid}") from exc


def _load_tau_v1_rows(*, benchmark_name: str, domain: str, split: str, root: Path) -> list[dict[str, Any]]:
    del benchmark_name, root
    module_name, attr_name = _tau_v1_task_source(domain, split)
    module = __import__(module_name, fromlist=[attr_name])
    raw_tasks = getattr(module, attr_name)
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(raw_tasks):
        payload = _dump_model(task)
        rows.append(
            {
                "task_id": str(index),
                "domain": domain,
                "index": index,
                "instruction": str(getattr(task, "instruction", "") or payload.get("instruction") or ""),
                "task": payload,
                "benchmark_version": "tau_v1",
            }
        )
    return rows


def load_tau_adapter_records(
    benchmark_name: str,
    entry: Mapping[str, Any],
    *,
    split: str,
    tau_bench_root: str | None = None,
    tau_data_root: str | None = None,
) -> list[TauAdapterRecord]:
    if benchmark_name not in TAU_ADAPTER_BENCHMARKS:
        raise ValueError(f"unsupported tau adapter benchmark: {benchmark_name}")
    adapter_spec = TAU_ADAPTER_BENCHMARKS[benchmark_name]
    domain = str(entry.get("tau_domain") or adapter_spec["domain"])
    version = str(entry.get("tau_version") or adapter_spec["version"])
    if version == "tau_v1":
        root = _ensure_tau_v1_runtime_path(tau_bench_root)
        rows = _load_tau_v1_rows(benchmark_name=benchmark_name, domain=domain, split=split, root=root)
    elif benchmark_name == "tau3_bench_mock_long_context":
        root = _ensure_tau2_runtime_path(tau_bench_root, tau_data_root)
        rows = _tau3_mock_long_context_rows()
    else:
        root = _ensure_tau2_runtime_path(tau_bench_root, tau_data_root)
        registry_module = __import__("tau2.registry", fromlist=["registry"])
        registry = getattr(registry_module, "registry")
        get_tasks = registry.get_tasks_loader(domain)
        raw_tasks = get_tasks("base" if split in {"test", "base"} else split)
        rows = []
        if benchmark_name == "tau3_bench_mock":
            for task in raw_tasks:
                payload = _dump_model(task)
                if str(payload.get("id") or "") not in TAU3_LIGHT_MOCK_TASK_IDS:
                    continue
                rows.append(_sanitize_tau3_light_mock_task(payload, index=len(rows)))
        else:
            rows = [
                _normalize_official_tau_task_row(task, domain=domain, version=version, index=index)
                for index, task in enumerate(raw_tasks)
            ]
    records = []
    for row in rows:
        records.append(
            TauAdapterRecord(
                task_id=str(row["task_id"]),
                domain=str(row["domain"]),
                instruction=str(row.get("instruction") or ""),
                task=dict(row["task"]),
                benchmark_version=str(row.get("benchmark_version") or entry.get("tau_version") or "tau3_light"),
                index=int(row.get("index") or len(records)),
                metadata={
                    "benchmark_name": benchmark_name,
                    "tau_bench_root": str(root),
                    "source": "tau-bench-v1" if version == "tau_v1" else "tau2-bench",
                },
            )
        )
    return records


def _tau_tool_schema(tool: Any) -> dict[str, Any]:
    raw_schema = getattr(tool, "openai_schema", None)
    if isinstance(raw_schema, Mapping):
        function = raw_schema.get("function")
        if isinstance(function, Mapping):
            parameters = function.get("parameters")
            properties = parameters.get("properties") if isinstance(parameters, Mapping) else {}
            return {
                "name": str(function.get("name") or getattr(tool, "name", "")),
                "description": str(function.get("description") or getattr(tool, "short_desc", "")),
                "arguments": dict(properties) if isinstance(properties, Mapping) else {},
                "required": list(parameters.get("required") or []) if isinstance(parameters, Mapping) else [],
            }
    return {
        "name": str(getattr(tool, "name", "")),
        "description": str(getattr(tool, "short_desc", "") or getattr(tool, "long_desc", "")),
        "arguments": {},
        "required": [],
    }


def build_tau_agent_prompt(
    *,
    domain_policy: str,
    tools: Sequence[Any],
    messages: Sequence[Any],
    history_max_chars: int,
    prompt_max_chars: int,
) -> str:
    tool_schemas = [_tau_tool_schema(tool) for tool in tools]
    tool_schemas.append(
        {
            "name": "respond",
            "description": "Send a natural-language message to the user. Include ###STOP### when the task is complete.",
            "arguments": {"content": {"type": "string"}},
            "required": ["content"],
        }
    )
    system = "\n".join(
        [
            "You are a TAU benchmark agent controlling tools in an official tau2 environment.",
            "Return exactly one JSON object and no prose.",
            'JSON schema: {"name":"tool_name","arguments":{...}}',
            "Use only exact listed tool names. If the task is complete, call respond with content containing ###STOP###.",
            "Do not invent IDs; IDs must come from the user, prior messages, tool outputs, or policy.",
            "",
            "Tools:",
            json.dumps(tool_schemas, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Domain policy:",
            str(domain_policy or "").strip(),
        ]
    )
    history = _render_tau_prompt_messages(messages)
    if history_max_chars > 0 and len(history) > history_max_chars:
        history = history[-history_max_chars:].lstrip()
        history = "[earlier history truncated]\n" + history
    prompt = "\n\n".join([f"System: {system}", history, "Assistant: ```json\n{"]).strip()
    if prompt_max_chars > 0 and len(prompt) > prompt_max_chars:
        overflow = len(prompt) - prompt_max_chars
        shorter_history = history[max(0, overflow + 512):].lstrip()
        if shorter_history:
            shorter_history = "[earlier history truncated]\n" + shorter_history
        prompt = "\n\n".join([f"System: {system}", shorter_history, "Assistant: ```json\n{"]).strip()
    return prompt


def _render_tau_prompt_messages(messages: Sequence[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(getattr(message, "role", "") or "").strip().lower()
        if hasattr(message, "tool_messages"):
            for tool_message in getattr(message, "tool_messages", []) or []:
                parts.append(_render_tau_tool_message(tool_message))
            continue
        if role == "user":
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                parts.append(f"User: {content}")
        elif role == "assistant":
            tool_calls = getattr(message, "tool_calls", None)
            content = str(getattr(message, "content", "") or "").strip()
            if tool_calls:
                for call in tool_calls:
                    payload = {
                        "name": str(getattr(call, "name", "") or ""),
                        "arguments": dict(getattr(call, "arguments", {}) or {}),
                    }
                    parts.append("Assistant: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            elif content:
                parts.append(f"Assistant: {content}")
        elif role == "tool":
            parts.append(_render_tau_tool_message(message))
    return "\n\n".join(parts) if parts else "User: Start the task."


def _render_tau_tool_message(message: Any) -> str:
    payload = {
        "requestor": str(getattr(message, "requestor", "assistant") or "assistant"),
        "ok": not bool(getattr(message, "error", False)),
        "content": str(getattr(message, "content", "") or ""),
    }
    return "User: Function output:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _extract_json_object_text(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    matches = list(_JSON_FENCED_BLOCK_RE.finditer(cleaned))
    if matches:
        cleaned = matches[-1].group(1).strip()
    if cleaned and not cleaned.lstrip().startswith("{"):
        candidates = ["{" + cleaned, cleaned]
    else:
        candidates = [cleaned]
    for candidate in candidates:
        start = candidate.find("{")
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(candidate[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return candidate[start:index + 1]
    raise ValueError(f"no JSON object found in tau completion: {cleaned[:200]}")


def parse_tau_agent_decision(text: str) -> tuple[str, dict[str, Any]]:
    payload = json.loads(_extract_json_object_text(text))
    if not isinstance(payload, Mapping):
        raise ValueError("tau decision must be a JSON object")
    name = str(payload.get("name") or payload.get("tool_name") or payload.get("function") or "").strip()
    arguments = payload.get("arguments")
    payload_type = str(payload.get("type") or "").strip().lower()
    if not name and payload_type in {"message", "response", "final_answer", "answer"}:
        name = "respond"
        arguments = {"content": str(payload.get("content") or payload.get("answer") or payload.get("message") or "")}
    if not isinstance(arguments, Mapping):
        arguments = payload.get("parameters")
    if not isinstance(arguments, Mapping):
        arguments = {}
    if "." in name:
        prefix, raw_name = name.split(".", 1)
        if prefix in {"assistant", "user"} and raw_name.strip():
            name = raw_name.strip()
    if name in {"final_answer", "answer"}:
        name = "respond"
        if not arguments:
            arguments = {"content": str(payload.get("answer") or "")}
        elif "content" not in arguments:
            arguments = {"content": str(arguments.get("answer") or arguments.get("message") or "")}
    if name == "respond" and "content" not in arguments:
        top_level_content = payload.get("content") or payload.get("answer") or payload.get("message")
        if top_level_content is not None:
            arguments = {**dict(arguments), "content": str(top_level_content)}
    if not name:
        raise ValueError(f"tau decision missing name: {payload}")
    return name, dict(arguments)


class EndpointTauAgent:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        tools: Sequence[Any],
        domain_policy: str,
        history_max_chars: int,
        prompt_max_chars: int,
    ) -> None:
        message_module = __import__("tau2.data_model.message", fromlist=["AssistantMessage", "ToolCall"])
        self._AssistantMessage = getattr(message_module, "AssistantMessage")
        self._ToolCall = getattr(message_module, "ToolCall")
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.tools = list(tools)
        self.domain_policy = domain_policy
        self.history_max_chars = history_max_chars
        self.prompt_max_chars = prompt_max_chars
        self.tool_names = {str(getattr(tool, "name", "") or "") for tool in self.tools}
        self.stages: list[TauGenerationStage] = []
        self.parse_errors: list[str] = []
        self.seed: int | None = None

    def get_init_state(self, message_history: list[Any] | None = None) -> list[Any]:
        return list(message_history or [])

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        content = getattr(message, "content", None)
        return isinstance(content, str) and "###STOP###" in content

    def stop(self, message: Any | None = None, state: Any | None = None) -> None:
        del message, state

    def generate_next_message(self, message: Any, state: list[Any] | None) -> tuple[Any, list[Any]]:
        history = list(state or [])
        if message is not None:
            history.append(message)
        prompt = build_tau_agent_prompt(
            domain_policy=self.domain_policy,
            tools=self.tools,
            messages=history,
            history_max_chars=self.history_max_chars,
            prompt_max_chars=self.prompt_max_chars,
        )
        completion, finish_reason = call_chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout_s=self.timeout_s,
        )
        parsed_name = ""
        parse_error = ""
        try:
            parsed_name, arguments = parse_tau_agent_decision(completion)
            assistant = self._decision_to_message(parsed_name, arguments)
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)
            self.parse_errors.append(parse_error)
            assistant = self._AssistantMessage(role="assistant", content="I cannot continue safely. ###STOP###")
        self.stages.append(
            TauGenerationStage(
                prompt=prompt,
                completion=completion,
                finish_reason=finish_reason,
                parsed_name=parsed_name,
                parse_error=parse_error,
            )
        )
        history.append(assistant)
        return assistant, history

    def _decision_to_message(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name == "respond":
            content = str(
                arguments.get("content")
                or arguments.get("answer")
                or arguments.get("message")
                or ""
            ).strip()
            if not content:
                raise ValueError("empty tau respond content")
            return self._AssistantMessage(role="assistant", content=content)
        if name not in self.tool_names:
            raise ValueError(f"unknown tau tool name: {name}")
        return self._AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                self._ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=name,
                    arguments=dict(arguments),
                    requestor="assistant",
                )
            ],
        )


class StaticStopTauUser:
    def __init__(
        self,
        *,
        initial_content: str = "",
        send_initial: bool = False,
        stop_content: str = "###STOP###",
    ) -> None:
        message_module = __import__("tau2.data_model.message", fromlist=["UserMessage"])
        self._UserMessage = getattr(message_module, "UserMessage")
        self.initial_content = str(initial_content or "").strip()
        self.send_initial = bool(send_initial and self.initial_content)
        self.stop_content = stop_content

    def get_init_state(self, message_history: list[Any] | None = None) -> Any:
        return {"messages": list(message_history or []), "sent_initial": not self.send_initial}

    def set_seed(self, seed: int) -> None:
        del seed

    def stop(self, message: Any | None = None, state: Any | None = None) -> None:
        del message, state

    def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
        del message
        if isinstance(state, dict) and not state.get("sent_initial", True):
            content = self.initial_content
            state["sent_initial"] = True
        else:
            content = self.stop_content
        user_message = self._UserMessage(role="user", content=content, cost=0.0)
        if isinstance(state, dict):
            state.setdefault("messages", []).append(user_message)
        return user_message, state


def _is_lightweight_tau_record(record: TauAdapterRecord) -> bool:
    return str(record.benchmark_version).strip().lower() in {
        "tau3_light",
        "tau3_light_long_context",
        "tau_v3_light",
    }


def _tau_environment_kwargs(domain: str) -> dict[str, Any]:
    if str(domain).strip().lower() == "banking_knowledge":
        return {"retrieval_variant": "bm25"}
    return {}


def _build_tau_user(
    *,
    task: Any,
    environment: Any,
    user_model: TauExternalModelConfig | None,
    temperature: float,
) -> Any:
    if user_model is None:
        return StaticStopTauUser()
    user_module = __import__("tau2.user.user_simulator", fromlist=["UserSimulator"])
    UserSimulator = getattr(user_module, "UserSimulator")
    try:
        user_tools = environment.get_user_tools()
    except Exception:  # noqa: BLE001
        user_tools = None
    return UserSimulator(
        tools=user_tools,
        instructions=str(getattr(task, "user_scenario", "")),
        llm=_tau_litellm_model_name(user_model),
        llm_args={
            "temperature": _tau_openai_temperature(temperature),
            "stream": False,
            "api_key": user_model.api_key,
            "api_base": user_model.base_url,
            **_tau_litellm_provider_args(user_model),
            **_tau_timeout_args(),
        },
    )


def configure_tau_judge(judge_model: TauExternalModelConfig | None) -> None:
    if judge_model is None:
        return
    model_name = _tau_litellm_model_name(judge_model)
    if not model_name or not judge_model.api_key:
        return
    llm_args: dict[str, Any] = {
        "temperature": _tau_openai_temperature(0.0),
        "stream": False,
        "api_key": judge_model.api_key,
        "response_format": {"type": "json_object"},
    }
    if judge_model.base_url:
        llm_args["api_base"] = judge_model.base_url
    llm_args.update(_tau_litellm_provider_args(judge_model))
    llm_args.update(_tau_timeout_args())
    for module_name in ("tau2.config", "tau2.evaluator.evaluator_nl_assertions"):
        try:
            module = __import__(module_name, fromlist=["DEFAULT_LLM_NL_ASSERTIONS"])
        except Exception:  # noqa: BLE001
            continue
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS", model_name)
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", dict(llm_args))


def _task_uses_nl_assertions(task: Any) -> bool:
    criteria = getattr(task, "evaluation_criteria", None)
    assertions = getattr(criteria, "nl_assertions", None)
    return bool(assertions)


def _tau_v1_user_provider(model_config: TauExternalModelConfig) -> str:
    base_url = _normalize_openai_base_url(model_config.base_url) or ""
    if "api.deepseek.com" in base_url:
        return "deepseek"
    return "openai"


def _apply_tau_v1_user_env(model_config: TauExternalModelConfig) -> None:
    os.environ["OPENAI_API_KEY"] = model_config.api_key
    if model_config.base_url:
        os.environ["OPENAI_BASE_URL"] = model_config.base_url
        os.environ["OPENAI_API_BASE"] = model_config.base_url


def _tau_v1_tool_schema(tool_info: Mapping[str, Any]) -> dict[str, Any]:
    function = tool_info.get("function")
    if not isinstance(function, Mapping):
        return {
            "name": str(tool_info.get("name") or "unknown_tool"),
            "description": str(tool_info.get("description") or ""),
            "arguments": {},
            "required": [],
        }
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else {}
    return {
        "name": str(function.get("name") or "unknown_tool"),
        "description": str(function.get("description") or ""),
        "arguments": dict(properties) if isinstance(properties, Mapping) else {},
        "required": list(parameters.get("required") or []) if isinstance(parameters, Mapping) else [],
    }


def _render_tau_v1_prompt_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role == "user":
            content = str(message.get("content") or "").strip()
            if content:
                parts.append(f"User: {content}")
        elif role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)) and tool_calls:
                rendered_calls: list[dict[str, Any]] = []
                for call in tool_calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    raw_arguments = function.get("arguments")
                    try:
                        arguments = json.loads(str(raw_arguments or "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                    rendered_calls.append(
                        {
                            "name": str(function.get("name") or ""),
                            "arguments": arguments,
                        }
                    )
                if rendered_calls:
                    parts.append("Assistant: " + json.dumps(rendered_calls[-1], ensure_ascii=False, separators=(",", ":")))
                    continue
            content = str(message.get("content") or "").strip()
            if content:
                parts.append(f"Assistant: {content}")
        elif role == "tool":
            payload = {
                "name": str(message.get("name") or "unknown_tool"),
                "content": str(message.get("content") or ""),
            }
            parts.append("Tool: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(parts) if parts else "User: Hi! How can I help you today?"


def build_tau_v1_agent_prompt(
    *,
    domain_policy: str,
    tools_info: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    history_max_chars: int,
    prompt_max_chars: int,
) -> str:
    tool_schemas = [_tau_v1_tool_schema(tool) for tool in tools_info]
    tool_schemas.append(
        {
            "name": "respond",
            "description": "Send a natural-language message to the user. Include ###STOP### when the task is complete.",
            "arguments": {"content": {"type": "string"}},
            "required": ["content"],
        }
    )
    system = "\n".join(
        [
            "You are a tau-bench customer-service agent.",
            "Return exactly one JSON object and no prose.",
            'For tool calls use {"name":"tool_name","arguments":{...}}.',
            'For user messages use {"type":"message","content":"..."} or {"name":"respond","arguments":{"content":"..."}}.',
            "Use only exact listed tool names. Do not invent IDs; ask the user or inspect tools first.",
            "",
            "Tools:",
            json.dumps(tool_schemas, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Domain policy:",
            str(domain_policy or "").strip(),
        ]
    )
    history = _render_tau_v1_prompt_messages(messages)
    if history_max_chars > 0 and len(history) > history_max_chars:
        history = "[earlier history truncated]\n" + history[-history_max_chars:].lstrip()
    prompt = "\n\n".join([f"System: {system}", history, "Assistant: ```json\n{"]).strip()
    if prompt_max_chars > 0 and len(prompt) > prompt_max_chars:
        overflow = len(prompt) - prompt_max_chars
        shorter_history = history[max(0, overflow + 512):].lstrip()
        if shorter_history:
            shorter_history = "[earlier history truncated]\n" + shorter_history
        prompt = "\n\n".join([f"System: {system}", shorter_history, "Assistant: ```json\n{"]).strip()
    return prompt


def _normalize_tau_task_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    criteria = normalized.get("evaluation_criteria")
    if isinstance(criteria, Mapping):
        normalized_criteria = dict(criteria)
        reward_basis = normalized_criteria.get("reward_basis")
        if isinstance(reward_basis, Sequence) and not isinstance(reward_basis, (str, bytes)):
            normalized_criteria["reward_basis"] = [
                str(value).removeprefix("RewardType.") for value in reward_basis
            ]
        normalized["evaluation_criteria"] = normalized_criteria
    return normalized


def _tau_task_needs_initial_user(task_payload: Mapping[str, Any]) -> bool:
    initial_state = task_payload.get("initial_state")
    history = initial_state.get("message_history") if isinstance(initial_state, Mapping) else None
    if not history:
        return True
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return True
    last = history[-1] if history else {}
    if isinstance(last, Mapping):
        return str(last.get("role") or "").strip().lower() != "user"
    return str(getattr(last, "role", "") or "").strip().lower() != "user"


def run_tau_v1_adapter(
    name: str,
    records: Sequence[TauAdapterRecord],
    *,
    selected: Sequence[tuple[int, TauAdapterRecord]],
    args: argparse.Namespace,
    run_root: Path,
    split: str,
    env_file_loaded: str | None,
    user_model: TauExternalModelConfig,
) -> dict[str, Any]:
    _ensure_tau_v1_runtime_path(args.tau_bench_root)
    _apply_tau_v1_user_env(user_model)
    envs_module = __import__("tau_bench.envs", fromlist=["get_env"])
    get_env = getattr(envs_module, "get_env")
    types_module = __import__("tau_bench.types", fromlist=["Action", "RESPOND_ACTION_NAME"])
    Action = getattr(types_module, "Action")
    respond_action_name = str(getattr(types_module, "RESPOND_ACTION_NAME"))

    benchmark_dir = run_root / name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    completions: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for ordinal, (dataset_index, record) in enumerate(selected):
        started = time.monotonic()
        stages: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        reward = 0.0
        is_passed = False
        runtime_error = ""
        termination_reason = ""
        parse_errors: list[str] = []
        details: dict[str, Any] = {}
        try:
            env = get_env(
                record.domain,
                user_strategy="llm",
                user_model=_tau_litellm_model_name(user_model),
                user_provider=_tau_v1_user_provider(user_model),
                task_split=split,
                task_index=record.index,
            )
            reset = env.reset(task_index=record.index)
            info = _dump_model(getattr(reset, "info", {}))
            messages = [
                {"role": "system", "content": str(getattr(env, "wiki", "") or "")},
                {"role": "user", "content": str(getattr(reset, "observation", "") or "")},
            ]
            for step_index in range(max(1, int(args.tau_max_steps))):
                prompt = build_tau_v1_agent_prompt(
                    domain_policy=str(getattr(env, "wiki", "") or ""),
                    tools_info=list(getattr(env, "tools_info", []) or []),
                    messages=messages,
                    history_max_chars=args.tau_history_max_chars,
                    prompt_max_chars=args.tau_prompt_max_chars,
                )
                completion, finish_reason = call_chat_completion(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model_name,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                )
                parsed_name = ""
                parse_error = ""
                try:
                    parsed_name, arguments = parse_tau_agent_decision(completion)
                except Exception as exc:  # noqa: BLE001
                    arguments = {"content": "I cannot continue safely. ###STOP###"}
                    parsed_name = "respond"
                    parse_error = str(exc)
                    parse_errors.append(parse_error)
                stages.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "finish_reason": finish_reason,
                        "parsed_name": parsed_name,
                        "parse_error": parse_error,
                        "step": step_index,
                    }
                )
                if parsed_name == "respond":
                    content = str(arguments.get("content") or "").strip()
                    step = env.step(Action(name=respond_action_name, kwargs={"content": content}))
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": str(getattr(step, "observation", "") or "")})
                else:
                    step = env.step(Action(name=parsed_name, kwargs=dict(arguments)))
                    call_id = f"call_{uuid.uuid4().hex[:12]}"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": parsed_name,
                                        "arguments": json.dumps(dict(arguments), ensure_ascii=False),
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": parsed_name,
                            "content": str(getattr(step, "observation", "") or ""),
                        }
                    )
                reward = float(getattr(step, "reward", 0.0) or 0.0)
                info = _dump_model(getattr(step, "info", {}))
                details = info
                if bool(getattr(step, "done", False)):
                    termination_reason = "done"
                    break
                if parse_error and len(parse_errors) >= max(1, int(args.tau_max_errors)):
                    termination_reason = "parse_errors"
                    break
            else:
                termination_reason = "max_steps"
            is_passed = reward >= 1.0 - 1e-6
            trace = list(messages)
        except Exception as exc:  # noqa: BLE001
            reward = 0.0
            is_passed = False
            runtime_error = f"{type(exc).__name__}: {exc}"
            termination_reason = "runtime_error"
            details = {"runtime_error": runtime_error}
            trace = list(messages)
        latency_s = time.monotonic() - started
        row = {
            "benchmark_name": name,
            "dataset_split": split,
            "sample_index": ordinal,
            "dataset_index": dataset_index,
            "task_id": record.task_id,
            "domain": record.domain,
            "benchmark_version": record.benchmark_version,
            "instruction": record.instruction,
            "model_name": args.model_name,
            "latency_s": latency_s,
            "reward": reward,
            "is_passed": is_passed,
            "termination_reason": termination_reason,
            "runtime_error": runtime_error,
            "parse_errors": parse_errors,
            "stages": stages,
            "agent_trace": trace,
            "details": details,
            "metadata": record.metadata,
        }
        completions.append(row)
        eval_rows.append(
            {
                "task_id": record.task_id,
                "sample_index": ordinal,
                "is_passed": is_passed,
                "reward": reward,
                "fail_reason": runtime_error or ("passed" if is_passed else termination_reason),
            }
        )
        print(
            f"{name}: generated {ordinal + 1}/{len(selected)} task={record.task_id} "
            f"reward={reward:.3f} passed={is_passed}"
        )

    completions_path = benchmark_dir / "completions.jsonl"
    eval_path = benchmark_dir / "eval.jsonl"
    write_jsonl(completions_path, completions)
    write_jsonl(eval_path, eval_rows)
    passed = sum(1 for item in eval_rows if item["is_passed"])
    parse_error_count = sum(1 for item in completions if item["parse_errors"])
    metrics = {
        "adapter": "tau",
        "benchmark": name,
        "samples": len(selected),
        "source_rows": len(records),
        "passed": passed,
        "success_rate": (passed / len(eval_rows)) if eval_rows else 0.0,
        "avg_reward": (sum(float(item["reward"]) for item in eval_rows) / len(eval_rows)) if eval_rows else 0.0,
        "parse_error_count": parse_error_count,
        "parse_error_rate": (parse_error_count / len(eval_rows)) if eval_rows else 0.0,
        "split": split,
        "env_file_loaded": env_file_loaded,
        "external_user_model": user_model.model_name,
        "external_judge_model": None,
        "completions_path": str(completions_path),
        "eval_path": str(eval_path),
    }
    (benchmark_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_tau_adapter(name: str, entry: Mapping[str, Any], args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    env_file_loaded = load_adapter_env_file(getattr(args, "env_file", None))
    split = str(getattr(args, "split", None) or entry.get("split") or "base")
    records = load_tau_adapter_records(
        name,
        entry,
        split=split,
        tau_bench_root=args.tau_bench_root,
        tau_data_root=args.tau_data_root,
    )
    selected = select_records(records, max_samples=args.max_samples, sample_seed=args.sample_seed)
    official_records = [record for _dataset_index, record in selected if not _is_lightweight_tau_record(record)]
    user_model: TauExternalModelConfig | None = None
    judge_model: TauExternalModelConfig | None = None
    if official_records:
        user_model = resolve_tau_user_model_config(args)
        judge_model = resolve_tau_judge_model_config(args, default_model=user_model)
        configure_tau_judge(judge_model)
    if selected and all(str(record.benchmark_version) == "tau_v1" for _index, record in selected):
        if user_model is None:
            user_model = resolve_tau_user_model_config(args)
        return run_tau_v1_adapter(
            name,
            records,
            selected=selected,
            args=args,
            run_root=run_root,
            split=split,
            env_file_loaded=env_file_loaded,
            user_model=user_model,
        )
    benchmark_dir = run_root / name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    task_module = __import__("tau2.data_model.tasks", fromlist=["Task"])
    Task = getattr(task_module, "Task")
    registry_module = __import__("tau2.registry", fromlist=["registry"])
    registry = getattr(registry_module, "registry")
    orchestrator_module = __import__("tau2.orchestrator.orchestrator", fromlist=["Orchestrator"])
    Orchestrator = getattr(orchestrator_module, "Orchestrator")
    evaluator_module = __import__("tau2.evaluator.evaluator", fromlist=["EvaluationType", "evaluate_simulation"])
    EvaluationType = getattr(evaluator_module, "EvaluationType")
    evaluate_simulation = getattr(evaluator_module, "evaluate_simulation")

    completions: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for ordinal, (dataset_index, record) in enumerate(selected):
        task = Task.model_validate(_normalize_tau_task_payload(record.task))
        env_kwargs = _tau_environment_kwargs(record.domain)
        env_constructor = registry.get_env_constructor(record.domain)
        try:
            environment = env_constructor(solo_mode=False, **env_kwargs)
        except TypeError:
            if env_kwargs:
                environment = env_constructor(solo_mode=False)
            else:
                raise
        agent = EndpointTauAgent(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            tools=environment.get_tools(),
            domain_policy=str(environment.get_policy()),
            history_max_chars=args.tau_history_max_chars,
            prompt_max_chars=args.tau_prompt_max_chars,
        )
        if _is_lightweight_tau_record(record):
            user = StaticStopTauUser(
                initial_content=record.instruction,
                send_initial=_tau_task_needs_initial_user(record.task),
            )
        else:
            user = _build_tau_user(
                task=task,
                environment=environment,
                user_model=user_model,
                temperature=args.tau_user_temperature,
            )
        orchestrator = Orchestrator(
            domain=record.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=args.tau_max_steps,
            max_errors=args.tau_max_errors,
            seed=(args.sample_seed or 0) + dataset_index,
            solo_mode=False,
            validate_communication=True,
        )
        started = time.monotonic()
        runtime_error = ""
        try:
            simulation = orchestrator.run()
            evaluation_type = (
                EvaluationType.ALL_WITH_NL_ASSERTIONS
                if _task_uses_nl_assertions(task)
                else EvaluationType.ALL
            )
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                evaluation_type=evaluation_type,
                solo_mode=False,
                domain=record.domain,
                env_kwargs=env_kwargs,
            )
            simulation.reward_info = reward_info
            reward = float(getattr(reward_info, "reward", 0.0))
            is_passed = reward >= 1.0 - 1e-6
            details = _dump_model(reward_info)
            termination_reason = str(getattr(simulation, "termination_reason", ""))
            trace = [_dump_model(message) for message in list(getattr(simulation, "messages", []) or [])]
        except Exception as exc:  # noqa: BLE001
            reward = 0.0
            is_passed = False
            details = {"runtime_error": f"{type(exc).__name__}: {exc}"}
            termination_reason = "runtime_error"
            trace = []
            runtime_error = f"{type(exc).__name__}: {exc}"
        latency_s = time.monotonic() - started
        stages = [
            {
                "prompt": stage.prompt,
                "completion": stage.completion,
                "finish_reason": stage.finish_reason,
                "parsed_name": stage.parsed_name,
                "parse_error": stage.parse_error,
            }
            for stage in agent.stages
        ]
        row = {
            "benchmark_name": name,
            "dataset_split": split,
            "sample_index": ordinal,
            "dataset_index": dataset_index,
            "task_id": record.task_id,
            "domain": record.domain,
            "benchmark_version": record.benchmark_version,
            "instruction": record.instruction,
            "model_name": args.model_name,
            "latency_s": latency_s,
            "reward": reward,
            "is_passed": is_passed,
            "termination_reason": termination_reason,
            "runtime_error": runtime_error,
            "parse_errors": list(agent.parse_errors),
            "stages": stages,
            "agent_trace": trace,
            "details": details,
            "metadata": record.metadata,
        }
        completions.append(row)
        eval_rows.append(
            {
                "task_id": record.task_id,
                "sample_index": ordinal,
                "is_passed": is_passed,
                "reward": reward,
                "fail_reason": runtime_error or ("passed" if is_passed else termination_reason),
            }
        )
        print(
            f"{name}: generated {ordinal + 1}/{len(selected)} task={record.task_id} "
            f"reward={reward:.3f} passed={is_passed}"
        )

    completions_path = benchmark_dir / "completions.jsonl"
    eval_path = benchmark_dir / "eval.jsonl"
    write_jsonl(completions_path, completions)
    write_jsonl(eval_path, eval_rows)
    passed = sum(1 for item in eval_rows if item["is_passed"])
    parse_error_count = sum(1 for item in completions if item["parse_errors"])
    metrics = {
        "adapter": "tau",
        "benchmark": name,
        "samples": len(selected),
        "source_rows": len(records),
        "passed": passed,
        "success_rate": (passed / len(eval_rows)) if eval_rows else 0.0,
        "avg_reward": (sum(float(item["reward"]) for item in eval_rows) / len(eval_rows)) if eval_rows else 0.0,
        "parse_error_count": parse_error_count,
        "parse_error_rate": (parse_error_count / len(eval_rows)) if eval_rows else 0.0,
        "split": split,
        "env_file_loaded": env_file_loaded,
        "external_user_model": user_model.model_name if user_model is not None else None,
        "external_judge_model": judge_model.model_name if judge_model is not None else None,
        "completions_path": str(completions_path),
        "eval_path": str(eval_path),
    }
    (benchmark_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _resolve_mcp_bench_root(raw_root: str | None = None) -> Path:
    repo_root = _repo_root()
    candidates = [
        raw_root,
        os.environ.get("HELICOPTER_MCP_BENCH_ROOT"),
        os.environ.get("MCP_BENCH_ROOT"),
        str(repo_root / "references/mcp-bench"),
        "/tmp/mcp-bench",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(str(candidate)).expanduser().resolve()
        if (root / "mcp_servers" / "commands.json").is_file() and (root / "tasks").is_dir():
            return root
    checked = ", ".join(str(item) for item in candidates if item)
    raise FileNotFoundError(f"MCP-Bench reference checkout not found; checked: {checked}")


def load_mcp_bench_records(name: str, entry: Mapping[str, Any], *, mcp_bench_root: str | None = None) -> list[McpBenchRecord]:
    root = _resolve_mcp_bench_root(mcp_bench_root)
    task_files = tuple(entry.get("task_files") or MCP_BENCH_TASK_FILES.get(name, ()))
    if not task_files:
        raise ValueError(f"{name} is missing MCP-Bench task file mapping")
    records: list[McpBenchRecord] = []
    for file_name in task_files:
        path = root / "tasks" / str(file_name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for group in payload.get("server_tasks", []) or []:
            if not isinstance(group, Mapping):
                continue
            servers = tuple(str(item) for item in (group.get("servers") or []) if str(item).strip())
            for task_payload in group.get("tasks", []) or []:
                if not isinstance(task_payload, Mapping):
                    continue
                task_id = str(task_payload.get("task_id") or "").strip()
                if not task_id:
                    continue
                task = McpBenchTaskSpec(
                    task_id=task_id,
                    task_description=str(task_payload.get("task_description") or ""),
                    fuzzy_description=str(task_payload.get("fuzzy_description") or ""),
                    dependency_analysis=str(task_payload.get("dependency_analysis") or ""),
                    distraction_servers=tuple(str(item) for item in (task_payload.get("distraction_servers") or [])),
                )
                records.append(
                    McpBenchRecord(
                        task_file=str(file_name),
                        server_name=str(group.get("server_name") or ""),
                        combination_name=str(group.get("combination_name") or ""),
                        combination_type=str(group.get("combination_type") or ""),
                        servers=servers,
                        task=task,
                        runtime_root=str(root),
                        metadata={
                            "source": "mcp-bench",
                            "task_file": str(file_name),
                            "server_name": str(group.get("server_name") or ""),
                            "combination_name": str(group.get("combination_name") or ""),
                            "combination_type": str(group.get("combination_type") or ""),
                            "servers": list(servers),
                            "mcp_bench_root": str(root),
                        },
                    )
                )
    return records


class McpBenchWorkerClient:
    def __init__(self, *, runtime_root: str | Path, worker_script: str | Path) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.worker_script = Path(worker_script).expanduser().resolve()
        python_bin = self.runtime_root / ".venv" / "bin" / "python"
        if not python_bin.is_file():
            raise FileNotFoundError(f"missing MCP-Bench runtime python: {python_bin}")
        if not self.worker_script.is_file():
            raise FileNotFoundError(f"missing MCP-Bench worker script: {self.worker_script}")
        self._proc = subprocess.Popen(
            [str(python_bin), str(self.worker_script), "--runtime-root", str(self.runtime_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, name="McpBenchWorkerStderr", daemon=True)
        self._stderr_thread.start()
        self._closed = False

    def open_task(self, record: McpBenchRecord) -> dict[str, Any]:
        response = self._request("open_task", {"servers": list(record.servers)})
        tools = response.get("available_tools")
        if not isinstance(tools, dict):
            raise RuntimeError("worker returned invalid available_tools payload")
        return tools

    def call_tool(self, full_tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request("call_tool", {"tool_name": full_tool_name, "arguments": dict(arguments)})
        if not isinstance(response, dict):
            raise RuntimeError("worker returned invalid tool response")
        return response

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request("evaluate", {"request": dict(request)})
        if not isinstance(response, dict):
            raise RuntimeError("worker returned invalid evaluation payload")
        return response

    def close_task(self) -> None:
        if not self._closed:
            try:
                self._request("close_task", {})
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._proc.poll() is None:
                try:
                    self._request("shutdown", {})
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._closed = True
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5.0)

    def _request(self, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed and action != "shutdown":
            raise RuntimeError("worker client already closed")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("worker pipes are unavailable")
        wire = json.dumps({"action": action, "payload": dict(payload)}, ensure_ascii=False)
        try:
            self._proc.stdin.write(wire + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._worker_failure_message("worker stdin closed")) from exc
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(self._worker_failure_message("worker stdout closed"))
            raw = line.strip()
            if not raw:
                continue
            try:
                response = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(response, Mapping):
                continue
            if not response.get("ok", False):
                raise RuntimeError(self._worker_failure_message(str(response.get("error") or "unknown worker error")))
            data = response.get("data")
            return dict(data) if isinstance(data, Mapping) else {}

    def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            self._stderr_lines.append(line.rstrip("\n"))

    def _worker_failure_message(self, message: str) -> str:
        stderr = "\n".join(self._stderr_lines)
        return f"{message}\nworker stderr:\n{stderr}" if stderr else message


def _truncate_text(text: Any, max_chars: int) -> str:
    rendered = str(text or "")
    if max_chars <= 0 or len(rendered) <= max_chars:
        return rendered
    return rendered[: max(0, max_chars - 32)].rstrip() + "\n[truncated]"


def _mcp_presented_task(record: McpBenchRecord) -> str:
    return record.task.fuzzy_description.strip() or record.task.task_description.strip()


def _render_mcp_tool_catalog(available_tools: Mapping[str, Any], *, max_schema_chars: int) -> str:
    rendered: list[dict[str, Any]] = []
    for full_name, raw_tool in sorted(available_tools.items()):
        tool = raw_tool if isinstance(raw_tool, Mapping) else {}
        schema_text = _truncate_text(json.dumps(tool.get("input_schema") or {}, ensure_ascii=False, sort_keys=True), max_schema_chars)
        rendered.append(
            {
                "name": str(full_name),
                "description": str(tool.get("description") or ""),
                "input_schema": schema_text,
            }
        )
    return json.dumps(rendered, ensure_ascii=False, indent=2)


def _build_mcp_planning_prompt(
    record: McpBenchRecord,
    available_tools: Mapping[str, Any],
    history: str,
    *,
    max_history_chars: int,
    max_tool_schema_chars: int,
) -> str:
    history_text = _truncate_text(history, max_history_chars).strip() or "No MCP tool calls have been made yet."
    return "\n\n".join(
        [
            "System: You are an MCP-Bench agent. Use the available MCP tools to solve the task.",
            "Return exactly one JSON object and no markdown or prose.",
            'To call a tool, return {"name":"Server:tool_name","arguments":{...}}.',
            'When the task is complete, return {"name":"final_answer","arguments":{"answer":"..."}}.',
            "Use only exact tool names from the catalog. Do not invent tool outputs.",
            "",
            "Tool catalog:",
            _render_mcp_tool_catalog(available_tools, max_schema_chars=max_tool_schema_chars),
            "",
            "Task:",
            _mcp_presented_task(record),
            "",
            "Tool evidence so far:",
            history_text,
            "",
            "Assistant: ```json\n{",
        ]
    )


def _parse_mcp_decision(text: str) -> McpDecision:
    payload = json.loads(_extract_json_object_text(text))
    if not isinstance(payload, Mapping):
        raise ValueError("MCP decision must be a JSON object")
    name = str(
        payload.get("name")
        or payload.get("tool_name")
        or payload.get("tool")
        or payload.get("Tool name")
        or payload.get("tool")
        or ""
    ).strip()
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = payload.get("parameters")
    if not isinstance(arguments, Mapping):
        arguments = payload.get("Tool input")
    if not isinstance(arguments, Mapping):
        arguments = payload.get("tool_input")
    if not isinstance(arguments, Mapping):
        arguments = {}
    payload_type = str(payload.get("type") or "").strip().lower()
    if name in {"final_answer", "answer", "respond"} or payload_type in {"final_answer", "answer", "message"}:
        answer = str(arguments.get("answer") or payload.get("answer") or payload.get("content") or "").strip()
        return McpDecision(final_answer=answer, tool_call=None)
    if not name:
        raise ValueError("MCP decision missing tool name")
    return McpDecision(final_answer="", tool_call=McpToolCall(full_name=name, arguments=dict(arguments)))


def _normalize_mcp_tool_call(call: McpToolCall, available_tools: Mapping[str, Any]) -> McpToolCall:
    full_name = call.full_name.strip()
    if full_name in available_tools:
        return call
    if ":" not in full_name:
        matches = sorted(name for name in available_tools if name.endswith(f":{full_name}"))
        if len(matches) == 1:
            return McpToolCall(full_name=matches[0], arguments=dict(call.arguments))
    raise ValueError(f"planned MCP tool `{full_name}` was not found")


def _append_mcp_history(history: str, round_num: int, execution: Mapping[str, Any], *, max_chars: int) -> str:
    result = _truncate_text(execution.get("result") or "", MCP_BENCH_DEFAULT_MAX_RESULT_CHARS)
    error = _truncate_text(execution.get("error") or "", MCP_BENCH_DEFAULT_MAX_ERROR_CHARS)
    params = json.dumps(execution.get("parameters") or {}, ensure_ascii=False, sort_keys=True)
    if execution.get("success"):
        line = f"Round {round_num}: {execution.get('tool')} with {params} succeeded. Result: {result}"
    else:
        line = f"Round {round_num}: {execution.get('tool')} with {params} failed. Error: {error}"
    return _truncate_text("\n".join(part for part in (history, line) if part), max_chars)


def _mcp_rule_metrics(
    execution_results: Sequence[Mapping[str, Any]],
    available_tools: Mapping[str, Any],
    *,
    planning_json_compliance: float,
) -> dict[str, Any]:
    if not execution_results:
        return {
            "input_schema_compliance": None,
            "valid_tool_name_rate": None,
            "execution_success_rate": None,
            "valid_call_failure_rate": None,
            "planning_json_compliance": planning_json_compliance,
        }
    try:
        import jsonschema
    except ModuleNotFoundError:  # pragma: no cover
        jsonschema = None  # type: ignore[assignment]
    valid_tool_calls = 0
    schema_compliant_calls = 0
    successful_executions = 0
    valid_call_failures = 0
    for result in execution_results:
        tool_name = str(result.get("tool") or "")
        success = bool(result.get("success"))
        parameters = result.get("parameters") if isinstance(result.get("parameters"), Mapping) else {}
        tool_info = available_tools.get(tool_name)
        if isinstance(tool_info, Mapping):
            valid_tool_calls += 1
            schema = tool_info.get("input_schema")
            if not isinstance(schema, Mapping) or jsonschema is None:
                schema_compliant_calls += 1
            else:
                try:
                    jsonschema.validate(dict(parameters), dict(schema))
                    schema_compliant_calls += 1
                except Exception:  # noqa: BLE001
                    pass
            if not success:
                valid_call_failures += 1
        if success:
            successful_executions += 1
    total_calls = len(execution_results)
    return {
        "input_schema_compliance": schema_compliant_calls / valid_tool_calls if valid_tool_calls else 0.0,
        "valid_tool_name_rate": valid_tool_calls / total_calls,
        "execution_success_rate": successful_executions / total_calls,
        "valid_call_failure_rate": valid_call_failures / valid_tool_calls if valid_tool_calls else 0.0,
        "planning_json_compliance": planning_json_compliance,
    }


def _resolve_mcp_judge_config(args: argparse.Namespace) -> dict[str, str] | None:
    model = _first_value(getattr(args, "mcp_judge_model", None), _first_env("MCP_JUDGE_MODEL", "JUDGE_MODEL"))
    if not model:
        return None
    api_key = _first_value(
        getattr(args, "mcp_judge_api_key", None),
        _first_env("MCP_JUDGE_API_KEY", "JUDGE_API_KEY"),
        getattr(args, "api_key", None),
    )
    base_url = _first_value(
        getattr(args, "mcp_judge_base_url", None),
        _first_env("MCP_JUDGE_BASE_URL", "JUDGE_BASE_URL"),
        getattr(args, "base_url", None),
    )
    if not api_key or not base_url:
        raise ValueError("MCP judge requires api key and base url")
    return {"model": str(model), "api_key": str(api_key), "base_url": str(base_url).rstrip("/")}


def _mcp_passed(evaluation: Mapping[str, Any]) -> bool:
    return (
        float(evaluation.get("task_completion_score") or 0.0) >= MCP_BENCH_PASS_THRESHOLD
        and float(evaluation.get("tool_selection_score") or 0.0) >= MCP_BENCH_PASS_THRESHOLD
        and float(evaluation.get("planning_effectiveness_and_efficiency_score") or 0.0) >= MCP_BENCH_PASS_THRESHOLD
    )


def run_mcp_bench_adapter(name: str, entry: Mapping[str, Any], args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    env_file_loaded = load_adapter_env_file(getattr(args, "env_file", None))
    records = load_mcp_bench_records(name, entry, mcp_bench_root=getattr(args, "mcp_bench_root", None))
    selected = select_records(records, max_samples=args.max_samples, sample_seed=args.sample_seed)
    runtime_root = Path(records[0].runtime_root).expanduser().resolve() if records else _resolve_mcp_bench_root(args.mcp_bench_root)
    worker_script = _repo_root() / "src/cli/helicopter_cli/mcp_bench_worker.py"
    judge_config = _resolve_mcp_judge_config(args)
    max_rounds = max(1, int(getattr(args, "mcp_max_rounds", None) or 6))
    max_errors = max(1, int(getattr(args, "mcp_max_errors", None) or 2))
    history_max_chars = max(1, int(getattr(args, "mcp_history_max_chars", None) or MCP_BENCH_DEFAULT_MAX_HISTORY_CHARS))
    tool_schema_max_chars = max(1, int(getattr(args, "mcp_tool_schema_max_chars", None) or MCP_BENCH_DEFAULT_MAX_TOOL_SCHEMA_CHARS))

    benchmark_dir = run_root / name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    completions: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    worker = McpBenchWorkerClient(runtime_root=runtime_root, worker_script=worker_script)
    try:
        for ordinal, (dataset_index, record) in enumerate(selected):
            started = time.monotonic()
            available_tools: dict[str, Any] = {}
            stages: list[dict[str, Any]] = []
            execution_results: list[dict[str, Any]] = []
            history = ""
            final_answer = ""
            runtime_error = ""
            termination_reason = ""
            parse_attempts = 0
            parsed_attempts = 0
            parse_errors: list[str] = []
            evaluation: dict[str, Any] = {}
            judge_error = ""
            try:
                available_tools = worker.open_task(record)
                if not available_tools:
                    raise RuntimeError("MCP-Bench worker returned no tools")
                for round_num in range(1, max_rounds + 1):
                    prompt = _build_mcp_planning_prompt(
                        record,
                        available_tools,
                        history,
                        max_history_chars=history_max_chars,
                        max_tool_schema_chars=tool_schema_max_chars,
                    )
                    completion, finish_reason = call_chat_completion(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model_name,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout_s=args.timeout_s,
                    )
                    parse_attempts += 1
                    stage: dict[str, Any] = {
                        "round": round_num,
                        "prompt": prompt,
                        "completion": completion,
                        "finish_reason": finish_reason,
                        "parse_error": "",
                        "tool": "",
                    }
                    try:
                        decision = _parse_mcp_decision(completion)
                        parsed_attempts += 1
                    except Exception as exc:  # noqa: BLE001
                        parse_error = str(exc)
                        stage["parse_error"] = parse_error
                        parse_errors.append(parse_error)
                        stages.append(stage)
                        if len(parse_errors) >= max_errors:
                            termination_reason = "parse_errors"
                            break
                        continue
                    if decision.tool_call is None:
                        final_answer = decision.final_answer
                        termination_reason = "final_answer"
                        stages.append(stage)
                        break
                    try:
                        tool_call = _normalize_mcp_tool_call(decision.tool_call, available_tools)
                        tool_response = worker.call_tool(tool_call.full_name, tool_call.arguments)
                        execution = {
                            "tool": tool_call.full_name,
                            "server": tool_call.full_name.split(":", 1)[0] if ":" in tool_call.full_name else "",
                            "parameters": dict(tool_call.arguments),
                            "round_num": round_num,
                            "success": bool(tool_response.get("success")),
                            "result": str(tool_response.get("result") or ""),
                            "error": str(tool_response.get("error") or ""),
                        }
                    except Exception as exc:  # noqa: BLE001
                        execution = {
                            "tool": decision.tool_call.full_name,
                            "server": decision.tool_call.full_name.split(":", 1)[0] if ":" in decision.tool_call.full_name else "",
                            "parameters": dict(decision.tool_call.arguments),
                            "round_num": round_num,
                            "success": False,
                            "result": "",
                            "error": str(exc),
                        }
                    execution_results.append(execution)
                    stage["tool"] = str(execution["tool"])
                    stage["execution"] = execution
                    stages.append(stage)
                    history = _append_mcp_history(history, round_num, execution, max_chars=history_max_chars)
                else:
                    termination_reason = "max_rounds"

                planning_json_compliance = (parsed_attempts / parse_attempts) if parse_attempts else 0.0
                evaluation.update(
                    _mcp_rule_metrics(
                        execution_results,
                        available_tools,
                        planning_json_compliance=planning_json_compliance,
                    )
                )
                if judge_config is not None:
                    try:
                        official = worker.evaluate(
                            {
                                "judge_config": judge_config,
                                "task": _mcp_presented_task(record),
                                "execution_results": execution_results,
                                "final_solution": final_answer,
                                "total_rounds": len(stages),
                                "available_tools": available_tools,
                                "planning_json_compliance": planning_json_compliance,
                                "accumulated_information": history,
                                "concrete_task_description": record.task.task_description,
                                "dependency_analysis": record.task.dependency_analysis,
                            }
                        )
                        evaluation.update(official)
                    except Exception as exc:  # noqa: BLE001
                        judge_error = str(exc)
                if not termination_reason:
                    termination_reason = "stopped"
            except Exception as exc:  # noqa: BLE001
                runtime_error = f"{type(exc).__name__}: {exc}"
                termination_reason = "runtime_error"
            finally:
                worker.close_task()
            is_passed = _mcp_passed(evaluation) if judge_config and not judge_error else False
            row = {
                "benchmark_name": name,
                "sample_index": ordinal,
                "dataset_index": dataset_index,
                "task_id": record.task.task_id,
                "task_file": record.task_file,
                "server_name": record.server_name,
                "servers": list(record.servers),
                "combination_type": record.combination_type,
                "task": _mcp_presented_task(record),
                "model_name": args.model_name,
                "latency_s": time.monotonic() - started,
                "available_tool_count": len(available_tools),
                "final_answer": final_answer,
                "is_passed": is_passed,
                "termination_reason": termination_reason,
                "runtime_error": runtime_error,
                "parse_errors": parse_errors,
                "judge_ran": judge_config is not None and not judge_error,
                "judge_error": judge_error,
                "evaluation": evaluation,
                "execution_results": execution_results,
                "stages": stages,
                "metadata": record.metadata,
            }
            completions.append(row)
            eval_rows.append(
                {
                    "task_id": record.task.task_id,
                    "sample_index": ordinal,
                    "is_passed": is_passed,
                    "termination_reason": termination_reason,
                    "runtime_error": runtime_error,
                    "judge_ran": row["judge_ran"],
                    "evaluation": evaluation,
                }
            )
            print(
                f"{name}: generated {ordinal + 1}/{len(selected)} task={record.task.task_id} "
                f"tools={len(available_tools)} passed={is_passed}"
            )
    finally:
        worker.close()

    completions_path = benchmark_dir / "completions.jsonl"
    eval_path = benchmark_dir / "eval.jsonl"
    write_jsonl(completions_path, completions)
    write_jsonl(eval_path, eval_rows)
    passed = sum(1 for item in eval_rows if item["is_passed"])
    judge_ran = sum(1 for item in eval_rows if item["judge_ran"])
    runtime_errors = sum(1 for item in eval_rows if item["runtime_error"])
    metrics = {
        "adapter": "mcp_bench",
        "benchmark": name,
        "samples": len(selected),
        "source_rows": len(records),
        "passed": passed,
        "success_rate": (passed / len(eval_rows)) if eval_rows else 0.0,
        "runtime_error_count": runtime_errors,
        "judge_ran": judge_ran,
        "judge_model": judge_config.get("model") if judge_config else None,
        "split": str(getattr(args, "split", None) or ""),
        "env_file_loaded": env_file_loaded,
        "mcp_bench_root": str(runtime_root),
        "completions_path": str(completions_path),
        "eval_path": str(eval_path),
    }
    (benchmark_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            file.write("\n")


def run_swebench_harness(
    *,
    predictions_path: Path,
    dataset_name: str,
    split: str,
    run_id: str,
    max_workers: int,
    timeout_s: float | None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max(1, int(max_workers))),
        "--run_id",
        run_id,
    ]
    subprocess.run(cmd, check=True, timeout=timeout_s)


def run_swebench_adapter(name: str, entry: Mapping[str, Any], args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    split = str(getattr(args, "split", None) or entry.get("split") or "test")
    records = load_swebench_records(name, entry, split=split)
    selected = select_records(records, max_samples=args.max_samples, sample_seed=args.sample_seed)
    benchmark_dir = run_root / name
    completions: list[dict[str, Any]] = []
    predictions: list[dict[str, str]] = []
    for ordinal, (dataset_index, record) in enumerate(selected):
        prompt = build_swebench_prompt(
            record,
            max_context_chars=args.swebench_max_context_chars,
            prompt_profile=args.swebench_prompt_profile,
        )
        started = time.monotonic()
        completion, finish_reason = call_chat_completion(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model_name,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        latency_s = time.monotonic() - started
        patch = extract_swebench_patch(completion)
        completions.append(
            {
                "benchmark_name": name,
                "dataset_split": split,
                "sample_index": ordinal,
                "dataset_index": dataset_index,
                "instance_id": record.instance_id,
                "prompt": prompt,
                "completion0": completion,
                "finish_reason": finish_reason,
                "latency_s": latency_s,
                "model_name": args.model_name,
                "source_dataset": record.source_dataset,
                "harness_dataset_name": record.harness_dataset_name,
                "patch_chars": len(patch),
                "metadata": record.metadata,
            }
        )
        predictions.append(
            {
                "instance_id": record.instance_id,
                "model_name_or_path": args.model_name,
                "model_patch": patch,
            }
        )
        print(f"{name}: generated {ordinal + 1}/{len(selected)} instance={record.instance_id} patch_chars={len(patch)}")

    completions_path = benchmark_dir / "completions.jsonl"
    predictions_path = benchmark_dir / "predictions.jsonl"
    write_jsonl(completions_path, completions)
    write_jsonl(predictions_path, predictions)
    nonempty = sum(1 for item in predictions if item["model_patch"].strip())
    metrics: dict[str, Any] = {
        "adapter": "swebench",
        "benchmark": name,
        "samples": len(selected),
        "source_rows": len(records),
        "predictions": len(predictions),
        "nonempty_patches": nonempty,
        "nonempty_patch_rate": (nonempty / len(predictions)) if predictions else 0.0,
        "swebench_harness_ran": 0.0,
        "source_dataset": records[0].source_dataset if records else entry.get("hf_dataset"),
        "harness_dataset_name": records[0].harness_dataset_name if records else entry.get("harness_dataset"),
        "split": split,
        "completions_path": str(completions_path),
        "predictions_path": str(predictions_path),
    }
    if args.swebench_run_harness:
        run_swebench_harness(
            predictions_path=predictions_path,
            dataset_name=str(metrics["harness_dataset_name"]),
            split=split,
            run_id=args.run_id,
            max_workers=args.swebench_harness_workers,
            timeout_s=args.swebench_harness_timeout_s,
        )
        metrics["swebench_harness_ran"] = 1.0
    metrics_path = benchmark_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite_path)
    entries = suite_adapter_entries(suite, args.benchmark or [])
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.run_id = run_id
    run_root = Path(args.output_dir).expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    for name, entry in entries:
        adapter = str(entry.get("adapter") or "")
        if adapter == "swebench":
            all_metrics.append(run_swebench_adapter(name, entry, args, run_root))
        elif adapter == "tau":
            all_metrics.append(run_tau_adapter(name, entry, args, run_root))
        elif adapter == "mcp_bench":
            all_metrics.append(run_mcp_bench_adapter(name, entry, args, run_root))
        else:
            raise SystemExit(f"unsupported adapter for {name}: {adapter}")
    summary = {
        "run_id": run_id,
        "model": args.model_name,
        "base_url": args.base_url,
        "benchmarks": [item["benchmark"] for item in all_metrics],
        "metrics": all_metrics,
        "run_root": str(run_root),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m helicopter_cli.benchmark_adapters")
    parser.add_argument("--suite-path", required=True)
    parser.add_argument("--benchmark", action="append", help="suite benchmark name; repeatable")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file")
    parser.add_argument("--run-id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--swebench-max-context-chars", type=int)
    parser.add_argument("--swebench-prompt-profile", choices=("normal", "naive"), default="normal")
    parser.add_argument("--swebench-run-harness", action="store_true")
    parser.add_argument("--swebench-harness-workers", type=int, default=1)
    parser.add_argument("--swebench-harness-timeout-s", type=float)
    parser.add_argument("--tau-bench-root")
    parser.add_argument("--tau-data-root")
    parser.add_argument("--tau-max-steps", type=int, default=12)
    parser.add_argument("--tau-max-errors", type=int, default=4)
    parser.add_argument("--tau-history-max-chars", type=int, default=16000)
    parser.add_argument("--tau-prompt-max-chars", type=int, default=24576)
    parser.add_argument("--tau-user-model")
    parser.add_argument("--tau-user-base-url")
    parser.add_argument("--tau-user-api-key")
    parser.add_argument("--tau-user-temperature", type=float, default=0.0)
    parser.add_argument("--tau-judge-model")
    parser.add_argument("--tau-judge-base-url")
    parser.add_argument("--tau-judge-api-key")
    parser.add_argument("--mcp-bench-root")
    parser.add_argument("--mcp-max-rounds", type=int, default=6)
    parser.add_argument("--mcp-max-errors", type=int, default=2)
    parser.add_argument("--mcp-history-max-chars", type=int, default=MCP_BENCH_DEFAULT_MAX_HISTORY_CHARS)
    parser.add_argument("--mcp-tool-schema-max-chars", type=int, default=MCP_BENCH_DEFAULT_MAX_TOOL_SCHEMA_CHARS)
    parser.add_argument("--mcp-judge-model")
    parser.add_argument("--mcp-judge-base-url")
    parser.add_argument("--mcp-judge-api-key")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
