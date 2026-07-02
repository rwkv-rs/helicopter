from __future__ import annotations

import ast
import asyncio
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .openai_client import chat_completion
from .sampling import apply_limit_or_sample, dataset_sample_suffix, validate_limit_or_sample
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


API_BANK_REPO_URL = "https://github.com/AlibabaResearch/DAMO-ConvAI.git"
API_BANK_CACHE_ROOT_NAME = "DAMO-ConvAI"
API_BANK_ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
    "CancelTimedSwitch": {"device_id": "name"},
    "TimedSwitch": {"device_id": "name"},
}
API_BANK_SANDBOX_CACHE: dict[str, "ApiBankSandbox"] = {}


@dataclass(frozen=True, slots=True)
class ApiBankCallResult:
    success: bool
    result: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ApiBankExpectedCall:
    name: str
    arguments: dict[str, Any]
    expected_result: Any


@dataclass(frozen=True, slots=True)
class ApiBankSample:
    sample_index: int
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_call: ApiBankExpectedCall
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ApiBankResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str

    def to_scoreboard(self) -> ScoreboardEvalResult:
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=self.answer,
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class ApiBankRunConfig:
    base_url: str
    model: str
    benchmark: str
    level: int
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    source_root: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 768
    timeout_s: float = 600.0
    scoreboard_dataset: str | None = None
    job_name: str = "function_api_bank"
    job_id: str | None = None
    runner: str = "helicopter_eval.apibank"
    cot_mode: str = "CoT"


class ApiBankSandbox:
    def __init__(self, source_root: str | Path) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self._api_classes: dict[str, type] | None = None
        self._tools: dict[str, Any] = {}
        self._init_databases: dict[str, Any] | None = None

    def api_call(self, api_name: str, arguments: Mapping[str, Any]) -> ApiBankCallResult:
        try:
            tool = self.init_tool(api_name)
            api_info = self._api_info(api_name)
            normalized_arguments = _normalize_api_bank_arguments(api_name, arguments)
            processed = {
                key: self._coerce_arg(value, api_info.get("input_parameters", {}).get(key, {}).get("type"))
                for key, value in normalized_arguments.items()
            }
            return ApiBankCallResult(True, tool.call(**processed))
        except Exception as exc:  # noqa: BLE001
            return ApiBankCallResult(False, error=str(exc))

    def check_api_call_correctness(self, api_name: str, actual: Any, expected: Any) -> bool:
        return bool(self.init_tool(api_name).check_api_call_correctness(actual, expected))

    def get_api_description(self, api_name: str) -> dict[str, Any] | None:
        try:
            info = dict(self._api_info(api_name))
        except Exception:
            return None
        info.pop("class", None)
        info.pop("init_database", None)
        return info

    def init_tool(self, api_name: str) -> Any:
        if api_name in self._tools:
            return self._tools[api_name]
        info = self._api_info(api_name)
        args: list[Any] = []
        if "init_database" in info:
            args.append(info["init_database"])
        if (
            api_name != "CheckToken"
            and "token" in info.get("input_parameters", {})
            and "CheckToken" in self._api_classes_by_name()
        ):
            args.append(self.init_tool("CheckToken"))
        tool = info["class"](*args)
        self._tools[api_name] = tool
        return tool

    def _api_info(self, api_name: str) -> dict[str, Any]:
        cls = self._api_classes_by_name().get(api_name)
        if cls is None:
            raise ValueError(f"invalid API-Bank tool name: {api_name}")
        info: dict[str, Any] = {
            "name": api_name,
            "class": cls,
            "description": getattr(cls, "description", ""),
            "input_parameters": getattr(cls, "input_parameters", {}),
            "output_parameters": getattr(cls, "output_parameters", {}),
        }
        database_name = getattr(cls, "database_name", None)
        init_databases = self._load_init_databases()
        if database_name in init_databases:
            info["init_database"] = init_databases[database_name]
        return info

    def _api_classes_by_name(self) -> dict[str, type]:
        if self._api_classes is not None:
            return self._api_classes
        classes: dict[str, type] = {}
        with _temporary_api_bank_import_path(self.source_root):
            api_base = importlib.import_module("apis.api").API
            apis_dir = self.source_root / "apis"
            for file_path in sorted(apis_dir.glob("*.py")):
                if file_path.name in {"__init__.py", "api.py", "tool_search.py"}:
                    continue
                try:
                    module = importlib.import_module(f"apis.{file_path.stem}")
                except Exception:
                    continue
                for value in vars(module).values():
                    if isinstance(value, type) and issubclass(value, api_base) and value is not api_base:
                        classes[value.__name__] = value
        self._api_classes = classes
        return classes

    def _load_init_databases(self) -> dict[str, Any]:
        if self._init_databases is not None:
            return self._init_databases
        databases: dict[str, Any] = {}
        db_dir = self.source_root / "init_database"
        if db_dir.is_dir():
            for file_path in db_dir.glob("*.json"):
                databases[file_path.stem] = json.loads(file_path.read_text(encoding="utf-8"))
        self._init_databases = databases
        return databases

    @staticmethod
    def _coerce_arg(value: Any, arg_type: Any) -> Any:
        if arg_type == "int":
            return int(value)
        if arg_type == "float":
            return float(value)
        if arg_type == "bool":
            return value if isinstance(value, bool) else str(value) == "True"
        if str(arg_type) in {"list", "list(str)"}:
            return _coerce_api_bank_list_arg(value)
        return value


def resolve_api_bank_source_root(config: ApiBankRunConfig) -> Path:
    candidates: list[Path] = []
    for raw in (
        config.source_root,
        os.getenv("API_BANK_SOURCE_ROOT"),
        os.getenv("RWKV_API_BANK_SOURCE_ROOT"),
    ):
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            Path("/home/chase/GitHub/API-Bank"),
            Path("/home/chase/GitHub/api-bank"),
            Path("/home/chase/GitHub/DAMO-ConvAI/api-bank"),
            Path("/tmp/rwkv-official-refs/DAMO-ConvAI/api-bank"),
            Path("/tmp/ref-DAMO-ConvAI/api-bank"),
        ]
    )
    for candidate in candidates:
        resolved = _normalize_api_bank_root(candidate)
        if _api_bank_required_paths(resolved):
            return resolved
    return _ensure_api_bank_cache() / "api-bank"


def _ensure_api_bank_cache() -> Path:
    cache_root = Path(os.getenv("HELICOPTER_CACHE_DIR") or "~/.cache/helicopter-eval").expanduser()
    repo_root = cache_root / API_BANK_CACHE_ROOT_NAME
    api_bank_root = repo_root / "api-bank"
    if _api_bank_required_paths(api_bank_root):
        return repo_root
    cache_root.mkdir(parents=True, exist_ok=True)
    with _api_bank_cache_lock(cache_root):
        if _api_bank_required_paths(api_bank_root):
            return repo_root
        if not repo_root.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", API_BANK_REPO_URL, str(repo_root)],
                check=True,
            )
        subprocess.run(["git", "-C", str(repo_root), "sparse-checkout", "set", "api-bank"], check=True)
        if not _api_bank_required_paths(api_bank_root):
            raise FileNotFoundError(f"API-Bank sparse checkout did not create expected files in {api_bank_root}")
    return repo_root


@contextmanager
def _api_bank_cache_lock(cache_root: Path, *, timeout_s: float = 600.0) -> Iterator[None]:
    lock_dir = cache_root / f"{API_BANK_CACHE_ROOT_NAME}.lock"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for API-Bank cache lock: {lock_dir}")
            time.sleep(0.25)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except FileNotFoundError:
            pass


def _normalize_api_bank_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name == "level-1-given-desc":
        return resolved.parent.parent
    if (resolved / "api-bank").is_dir():
        return resolved / "api-bank"
    return resolved


def _api_bank_required_paths(source_root: Path) -> bool:
    return (source_root / "lv1-lv2-samples" / "level-1-given-desc").is_dir() and (source_root / "apis").is_dir()


def _level_dir(source_root: Path) -> Path:
    return source_root / "lv1-lv2-samples" / "level-1-given-desc"


def load_samples(config: ApiBankRunConfig) -> list[ApiBankSample]:
    if config.split != "test":
        raise ValueError("APIBank only provides test split")
    if config.level not in {1, 2}:
        raise ValueError("APIBank level must be 1 or 2")
    validate_limit_or_sample(limit=config.limit, sample_size=config.sample_size)
    source_root = resolve_api_bank_source_root(config)
    sandbox = _api_bank_sandbox_for(source_root)
    rows: list[ApiBankSample] = []
    for file_path in sorted(_level_dir(source_root).glob("*.jsonl")):
        if _api_bank_level_from_name(file_path.name) != config.level:
            continue
        history = _read_jsonl(file_path)
        api_names = sorted(
            {
                str(item.get("api_name") or "").strip()
                for item in history
                if isinstance(item, Mapping) and item.get("role") == "API"
            }
        )
        tools = tuple(_api_bank_tool_schema(sandbox, name, {}) for name in api_names)
        for turn_index, item in enumerate(history):
            if not isinstance(item, Mapping) or item.get("role") != "API":
                continue
            api_name = str(item.get("api_name") or "").strip()
            if not api_name:
                continue
            arguments = item.get("param_dict") if isinstance(item.get("param_dict"), Mapping) else {}
            sample = ApiBankSample(
                sample_index=len(rows),
                task_id=f"{config.benchmark}__{file_path.stem}_{turn_index:03d}",
                instruction=_render_api_bank_history(history[:turn_index]),
                tools=tools,
                expected_call=ApiBankExpectedCall(
                    name=api_name,
                    arguments=dict(arguments),
                    expected_result=item.get("result"),
                ),
                metadata={
                    "source_format": "official_api_bank",
                    "source_path": str(file_path),
                    "source_root": str(source_root),
                    "level": config.level,
                    "turn_index": turn_index,
                    "api_name": api_name,
                },
            )
            rows.append(sample)
            if config.limit is not None and config.sample_size is None and len(rows) >= int(config.limit):
                return rows
    return apply_limit_or_sample(
        rows,
        limit=config.limit,
        sample_size=config.sample_size,
        sample_seed=config.sample_seed,
        sort_key=lambda sample: sample.sample_index,
    )


def build_prompt(sample: ApiBankSample) -> str:
    tool_catalog = json.dumps(sample.tools, ensure_ascii=False, indent=2, sort_keys=True)
    output_schema = json.dumps(
        {
            "type": "object",
            "required": ["name", "arguments"],
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are choosing the next API call for an API-Bank conversation.\n\n"
        "Tools:\n"
        f"{tool_catalog}\n\n"
        "Output JSON schema:\n"
        f"{output_schema}\n\n"
        "Return exactly one JSON object that validates against the schema. "
        "Use only listed tool names. Return no prose, no markdown, and no extra text outside the JSON value.\n"
        "API-Bank date convention: if a month/day or relative date has no explicit year and the conversation "
        "does not state today's date, use year 2023.\n\n"
        f"Conversation:\n{sample.instruction}\n\n"
        "Next API call:"
    )


def decode_tool_calls(response: str) -> list[dict[str, Any]]:
    payload = _loads_json_or_literal(_extract_json_value(response))
    calls = _coerce_tool_call_payloads(payload)
    return [{"name": str(call["name"]), "arguments": dict(call.get("arguments") or {})} for call in calls]


def evaluate_completion(
    sample: ApiBankSample,
    completion: str,
    *,
    sandbox: ApiBankSandbox | None = None,
) -> tuple[bool, str, list[dict[str, Any]]]:
    try:
        decoded_calls = decode_tool_calls(completion)
    except Exception as exc:  # noqa: BLE001
        return False, f"parse_error:{exc}", []
    if not decoded_calls:
        return False, "missing_call", decoded_calls
    actual = decoded_calls[0]
    if str(actual.get("name") or "") != sample.expected_call.name:
        return False, f"api_name_mismatch:{actual.get('name')}!={sample.expected_call.name}", decoded_calls
    arguments = actual.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, "arguments_not_object", decoded_calls

    source_root = Path(str(sample.metadata["source_root"]))
    sandbox = sandbox or _api_bank_sandbox_for(source_root)
    call_result = sandbox.api_call(sample.expected_call.name, dict(arguments))
    if not call_result.success:
        return False, f"api_execution_failed:{call_result.error}", decoded_calls
    expected_result = _normalize_api_bank_expected_result(
        sample.expected_call.name,
        sample.expected_call.expected_result,
    )
    try:
        ok = sandbox.check_api_call_correctness(
            sample.expected_call.name,
            copy.deepcopy(call_result.result),
            copy.deepcopy(expected_result),
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"check_error:{exc}", decoded_calls
    return bool(ok), "" if ok else "api_result_mismatch", decoded_calls


def generate_completion(sample: ApiBankSample, config: ApiBankRunConfig) -> ApiBankResult:
    prompt = build_prompt(sample)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    passed, fail_reason, decoded_calls = evaluate_completion(sample, completion)
    answer = json.dumps(decoded_calls, ensure_ascii=False, sort_keys=True)
    reference = json.dumps(
        [{"name": sample.expected_call.name, "arguments": sample.expected_call.arguments}],
        ensure_ascii=False,
        sort_keys=True,
    )
    return ApiBankResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=completion,
        answer=answer,
        reference_answer=reference,
        is_passed=passed,
        fail_reason=fail_reason,
    )


def evaluate_samples(samples: Sequence[ApiBankSample], config: ApiBankRunConfig) -> list[ApiBankResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: ApiBankRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    dataset += dataset_sample_suffix(sample_size=config.sample_size, sample_seed=config.sample_seed)
    return dataset


def job_id(config: ApiBankRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: ApiBankRunConfig) -> dict[str, Any]:
    return {
        "tool": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: ApiBankRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_apibank_tool_call",
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[ApiBankResult], *, config: ApiBankRunConfig, repo_root: Path) -> int:
    task_id = asyncio.run(
        write_scoreboard_results(
            [result.to_scoreboard() for result in results],
            config=ScoreboardWriteConfig(
                dataset=scoreboard_dataset_name(config),
                model=config.model,
                job_name=config.job_name,
                job_id=job_id(config),
                benchmark=config.benchmark,
                runner=config.runner,
                cot_mode=config.cot_mode,
                sampling_config=task_sampling_config(config),
                completion_sampling_config=completion_sampling_config(config),
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_apibank(config: ApiBankRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "level": config.level,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: ApiBankRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "source": "git+https://github.com/AlibabaResearch/DAMO-ConvAI.git#api-bank",
        "split": config.split,
        "level": config.level,
        "limit": config.limit,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


def _api_bank_sandbox_for(source_root: Path) -> ApiBankSandbox:
    key = str(source_root.expanduser().resolve())
    sandbox = API_BANK_SANDBOX_CACHE.get(key)
    if sandbox is None:
        sandbox = ApiBankSandbox(source_root)
        API_BANK_SANDBOX_CACHE[key] = sandbox
    return sandbox


def _api_bank_tool_schema(sandbox: ApiBankSandbox, api_name: str, fallback_args: Mapping[str, Any]) -> dict[str, Any]:
    description = sandbox.get_api_description(api_name)
    if not description:
        return {
            "name": api_name,
            "description": f"API-Bank tool {api_name}",
            "parameters": {
                "type": "object",
                "properties": {str(key): {"type": _json_type(value)} for key, value in fallback_args.items()},
                "required": [str(key) for key in fallback_args],
            },
        }
    parameters = description.get("input_parameters")
    properties: dict[str, Any] = {}
    required: list[str] = []
    if isinstance(parameters, Mapping):
        for key, spec in parameters.items():
            spec = spec if isinstance(spec, Mapping) else {}
            properties[str(key)] = {
                "type": _api_bank_json_type(spec.get("type")),
                "description": str(spec.get("description") or ""),
            }
            required.append(str(key))
    return {
        "name": api_name,
        "description": str(description.get("description") or ""),
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _api_bank_level_from_name(file_name: str) -> int | None:
    if "level-1" in file_name:
        return 1
    if "level-2" in file_name:
        return 2
    return None


def _render_api_bank_history(history: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for item in history:
        role = str(item.get("role") or "").strip()
        if role == "User":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            lines.append(f"User: {text}" if text else "User:")
        elif role == "AI":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            lines.append(f"Assistant: {text}" if text else "Assistant:")
        elif role == "API":
            args = ", ".join(
                f"{key}={_official_arg_repr(value)}" for key, value in dict(item.get("param_dict") or {}).items()
            )
            lines.append(f"API: [{item.get('api_name')}({args})] Response: {item.get('result')}")
    return "\n".join(lines).strip()


def _normalize_api_bank_arguments(api_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    aliases = API_BANK_ARGUMENT_ALIASES.get(str(api_name), {})
    return {aliases.get(str(key), str(key)): value for key, value in dict(arguments).items()}


def _normalize_api_bank_expected_result(api_name: str, expected: Any) -> Any:
    if not isinstance(expected, Mapping):
        return expected
    normalized = copy.deepcopy(dict(expected))
    input_payload = normalized.get("input")
    if isinstance(input_payload, Mapping):
        normalized["input"] = _normalize_api_bank_arguments(api_name, input_payload)
    return normalized


def _coerce_api_bank_list_arg(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return value


def _extract_json_value(response: str) -> str:
    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return text
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1] if end > start else text[start:]


def _loads_json_or_literal(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return ast.literal_eval(candidate)


def _coerce_tool_call_payloads(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [call for item in payload for call in _coerce_tool_call_payloads(item)]
    if not isinstance(payload, Mapping):
        raise ValueError("tool-call payload must be an object or array")
    if "tool_calls" in payload:
        return _coerce_tool_call_payloads(payload["tool_calls"])
    source: Mapping[str, Any] = payload
    if isinstance(payload.get("function"), Mapping):
        source = payload["function"]
    elif isinstance(payload.get("function_call"), Mapping):
        source = payload["function_call"]
    name = str(source.get("name") or "").strip()
    arguments = source.get("arguments")
    if isinstance(arguments, str):
        arguments = _loads_json_or_literal(arguments) if arguments.strip() else {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    if not name:
        raise ValueError("tool-call payload missing name")
    return [{"name": name, "arguments": dict(arguments)}]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _official_arg_repr(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    return repr(value)


def _api_bank_json_type(value: Any) -> str:
    return {
        "int": "integer",
        "float": "number",
        "str": "string",
        "list": "array",
        "list(str)": "array",
        "bool": "boolean",
    }.get(str(value), "string")


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "string"


@contextmanager
def _temporary_api_bank_import_path(source_root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    root_text = str(source_root)
    sys.path.insert(0, root_text)
    os.chdir(source_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(root_text)
        except ValueError:
            pass


__all__ = [
    "ApiBankRunConfig",
    "ApiBankSample",
    "ApiBankSandbox",
    "decode_tool_calls",
    "dry_run_summary",
    "evaluate_completion",
    "load_samples",
    "run_apibank",
]
