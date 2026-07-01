from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from .apibank import decode_tool_calls
from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


AGENTBENCH_CONTROLLER_DEFAULT_URL = "http://127.0.0.1:5020/api"
AGENTBENCH_TASK_NAMES = {
    "agentbench_db": "dbbench-std",
    "agentbench_kg": "kg-std",
}


@dataclass(frozen=True, slots=True)
class AgentBenchRecord:
    task_id: str
    task_name: str
    index: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentBenchResult:
    sample_index: int
    task_id: str
    official_index: int
    task_name: str
    prompt: str
    completion: str
    reward: float
    is_passed: bool
    fail_reason: str
    trace: tuple[dict[str, Any], ...]

    def to_scoreboard(self) -> ScoreboardEvalResult:
        answer = {
            "reward": self.reward,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "official_index": self.official_index,
            "trace": list(self.trace),
        }
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=json.dumps(answer, ensure_ascii=False, sort_keys=True),
            reference_answer="official_agentbench_controller",
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class AgentBenchRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    limit: int | None = None
    split: str = "test"
    source_path: str | None = None
    source_root: str | None = None
    controller_url: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    max_steps: int = 20
    timeout_s: float = 600.0
    controller_timeout_s: float = 120.0
    scoreboard_dataset: str | None = None
    job_name: str = "function_agentbench"
    job_id: str | None = None
    runner: str = "helicopter_eval.agentbench"
    cot_mode: str = "CoT"


class AgentBenchControllerClient:
    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = max(1.0, float(timeout_s))

    def start_sample(self, task_name: str, index: int) -> tuple[str, dict[str, Any]]:
        data, headers = self._post("start_sample", {"name": task_name, "index": int(index)})
        session_id = headers.get("session_id") or headers.get("Session-Id") or headers.get("Session-ID")
        if not session_id:
            raise RuntimeError("AgentBench controller did not return session_id header")
        return str(session_id), data

    def interact(self, session_id: str, message: Mapping[str, Any]) -> dict[str, Any]:
        data, _headers = self._post("interact", {"messages": [dict(message)]}, headers={"session_id": session_id})
        return data

    def cancel(self, session_id: str) -> None:
        try:
            self._post("cancel", {}, headers={"session_id": session_id})
        except Exception:  # noqa: BLE001
            pass

    def _post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        request_headers = {"content-type": "application/json"}
        request_headers.update(dict(headers or {}))
        request = urllib.request.Request(
            f"{self.base_url}/{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw) if raw.strip() else {}
                if not isinstance(data, dict):
                    raise RuntimeError("AgentBench controller response must be a JSON object")
                return data, response.headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AgentBench controller HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"AgentBench controller request failed: {exc.reason}") from exc


def load_agentbench_manifest_records(path: str | Path) -> list[AgentBenchRecord]:
    target = Path(path).expanduser().resolve()
    records: list[AgentBenchRecord] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(f"AgentBench manifest row {line_index} is not a JSON object")
            records.append(
                AgentBenchRecord(
                    task_id=str(payload.get("task_id") or f"agentbench_{line_index:05d}"),
                    task_name=str(payload.get("task_name") or payload.get("name") or ""),
                    index=int(payload.get("index", line_index)),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
    if not records:
        raise ValueError(f"AgentBench manifest is empty: {target}")
    return records


def load_agentbench_rows_from_source(
    path: str | Path,
    *,
    dataset_name: str,
    task_name: str,
) -> list[AgentBenchRecord]:
    target = Path(path).expanduser().resolve()
    count = _agentbench_data_count(target)
    return [
        AgentBenchRecord(
            task_id=f"{dataset_name}__{index:05d}",
            task_name=task_name,
            index=index,
            metadata={
                "source_format": "official_agentbench_controller",
                "source_path": str(target),
                "task_name": task_name,
            },
        )
        for index in range(count)
    ]


def load_samples(config: AgentBenchRunConfig) -> list[AgentBenchRecord]:
    if config.split != "test":
        raise ValueError("AgentBench only provides test split")
    if config.dataset_name not in AGENTBENCH_TASK_NAMES:
        raise ValueError(f"unknown AgentBench dataset: {config.dataset_name}")
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")

    records = _load_records(config)
    if config.limit is not None:
        records = records[: int(config.limit)]
    if not records:
        raise ValueError("AgentBench run selected zero samples")
    return records


def build_prompt(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    allow_final_answer_text: bool,
) -> str:
    tool_schemas = [_normalize_openai_tool(tool) for tool in tools]
    if allow_final_answer_text:
        tool_schemas.append(
            {
                "name": "final_answer",
                "description": "Submit the final answer as assistant text content.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            }
        )
    schema = {
        "type": "object",
        "required": ["name", "arguments"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"},
        },
    }
    kg_instruction = (
        "\nFor AgentBench KG final answers, call final_answer with the exact content `Final Answer: #id`."
        if allow_final_answer_text
        else ""
    )
    return (
        "You are solving an AgentBench interactive tool-use task.\n\n"
        "Tools:\n"
        f"{json.dumps(tool_schemas, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "Output JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        "Return exactly one JSON object that validates against the schema. "
        "Use only listed tool names. Return no prose, no markdown, and no extra text outside the JSON value."
        f"{kg_instruction}\n\n"
        "Conversation:\n"
        f"{_format_messages(messages)}\n\n"
        "Next action:"
    )


def generate_sample(
    sample_index: int,
    record: AgentBenchRecord,
    config: AgentBenchRunConfig,
    *,
    controller: AgentBenchControllerClient,
) -> AgentBenchResult:
    session_id, data = controller.start_sample(record.task_name, record.index)
    messages = _coerce_mapping_list(data.get("messages"))
    tools = _coerce_mapping_list(data.get("tools"))
    prompts: list[str] = []
    completions: list[str] = []
    trace: list[dict[str, Any]] = []
    reward = 0.0
    fail_reason = ""
    finished = False
    try:
        for round_index in range(1, max(1, int(config.max_steps)) + 1):
            prompt = build_prompt(
                messages,
                tools,
                allow_final_answer_text=_is_agentbench_kg(record),
            )
            completion = chat_completion(
                base_url=config.base_url,
                model=config.model,
                prompt=prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                timeout_s=config.timeout_s,
                response_format={"type": "json_object"},
            )
            prompts.append(prompt)
            completions.append(completion)
            try:
                decoded_calls = decode_tool_calls(completion)
                assistant_message = _assistant_message_from_calls(decoded_calls, round_index)
            except Exception as exc:  # noqa: BLE001
                fail_reason = f"parse_error:{exc}"
                trace.append({"round": round_index, "completion": completion, "parse_error": str(exc)})
                break
            messages.append(assistant_message)
            response = controller.interact(session_id, assistant_message)
            trace.append(
                {
                    "round": round_index,
                    "completion": completion,
                    "decoded_calls": decoded_calls,
                    "controller_response": response,
                }
            )
            if bool(response.get("finish")):
                reward = float(response.get("reward") or response.get("score") or 0.0)
                finished = True
                break
            messages.extend(_coerce_mapping_list(response.get("messages")))
        else:
            fail_reason = "max_steps"
    finally:
        if session_id and not finished:
            controller.cancel(session_id)

    is_passed = bool(finished and reward > 0.0)
    if finished and not is_passed and not fail_reason:
        fail_reason = "reward_zero"
    return AgentBenchResult(
        sample_index=sample_index,
        task_id=record.task_id,
        official_index=record.index,
        task_name=record.task_name,
        prompt=_join_round_artifacts(prompts),
        completion=_join_round_artifacts(completions),
        reward=reward,
        is_passed=is_passed,
        fail_reason="" if is_passed else fail_reason,
        trace=tuple(trace),
    )


def evaluate_samples(samples: Sequence[AgentBenchRecord], config: AgentBenchRunConfig) -> list[AgentBenchResult]:
    controller = AgentBenchControllerClient(_controller_url(config), timeout_s=config.controller_timeout_s)
    return [
        generate_sample(sample_index, sample, config, controller=controller)
        for sample_index, sample in enumerate(samples)
    ]


def scoreboard_dataset_name(config: AgentBenchRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: AgentBenchRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: AgentBenchRunConfig) -> dict[str, Any]:
    return {
        "tool": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
            "max_steps": config.max_steps,
        }
    }


def task_sampling_config(config: AgentBenchRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_agentbench_json_tool_call",
        "controller_url": _controller_url(config),
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[AgentBenchResult], *, config: AgentBenchRunConfig, repo_root: Path) -> int:
    rewards = [result.reward for result in results]
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
                extra_metrics={
                    "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                    "agent_error_rate": sum(1 for result in results if result.fail_reason) / len(results)
                    if results
                    else 0.0,
                },
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_agentbench(config: AgentBenchRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    avg_reward = sum(result.reward for result in results) / len(results) if results else 0.0
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "source_dataset": config.dataset_name,
        "controller_url": _controller_url(config),
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
        "avg_reward": avg_reward,
    }


def dry_run_summary(config: AgentBenchRunConfig) -> dict[str, Any]:
    samples = load_samples(config)
    source = _resolve_records_path(config)
    return {
        "benchmark": config.benchmark,
        "source": str(source) if source is not None else "official_agentbench_source",
        "split": config.split,
        "source_dataset": config.dataset_name,
        "limit": config.limit,
        "available_samples": len(samples),
        "base_url": config.base_url,
        "model": config.model,
        "controller_url": _controller_url(config),
        "controller_required": True,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


def _load_records(config: AgentBenchRunConfig) -> list[AgentBenchRecord]:
    source_path = _resolve_records_path(config)
    if source_path is None:
        raise FileNotFoundError(
            "AgentBench data not found. Set HELICOPTER_AGENTBENCH_DATA_ROOT, "
            "RWKV_AGENTBENCH_SOURCE_ROOT, or source_path to an AgentBench manifest/source file."
        )
    if _path_looks_like_manifest(source_path):
        records = load_agentbench_manifest_records(source_path)
    else:
        records = load_agentbench_rows_from_source(
            source_path,
            dataset_name=config.dataset_name,
            task_name=AGENTBENCH_TASK_NAMES[config.dataset_name],
        )
    for record in records:
        if not record.task_name:
            raise ValueError(f"AgentBench record missing task_name: {record.task_id}")
    return records


def _resolve_records_path(config: AgentBenchRunConfig) -> Path | None:
    if config.source_path:
        candidate = Path(config.source_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"AgentBench source_path does not exist: {candidate}")
    for candidate in _manifest_candidates(config):
        if candidate.is_file():
            return candidate.resolve()
    for candidate in _official_source_candidates(config):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _manifest_candidates(config: AgentBenchRunConfig) -> list[Path]:
    dataset_env = config.dataset_name.upper()
    candidates: list[Path] = []
    for key in (
        f"HELICOPTER_{dataset_env}_MANIFEST",
        f"RWKV_{dataset_env}_MANIFEST",
        f"{dataset_env}_MANIFEST",
        "HELICOPTER_AGENTBENCH_MANIFEST",
        "RWKV_AGENTBENCH_MANIFEST",
        "AGENTBENCH_MANIFEST",
    ):
        raw = os.getenv(key)
        if raw:
            candidates.append(Path(raw).expanduser())
    for raw in (
        config.source_root,
        os.getenv("HELICOPTER_AGENTBENCH_DATA_ROOT"),
        os.getenv("RWKV_AGENTBENCH_DATA_ROOT"),
        os.getenv("AGENTBENCH_DATA_ROOT"),
    ):
        if raw:
            candidates.append(Path(raw).expanduser() / config.dataset_name / "test.jsonl")
    candidates.extend(
        [
            _repo_root() / "data" / config.dataset_name / "test.jsonl",
            Path("/home/chase/GitHub/rwkv-skills/data") / config.dataset_name / "test.jsonl",
            Path("/home/chase/rwkv-skills/data") / config.dataset_name / "test.jsonl",
            Path("/tmp/rwkv-skills/data") / config.dataset_name / "test.jsonl",
        ]
    )
    return candidates


def _official_source_candidates(config: AgentBenchRunConfig) -> list[Path]:
    candidates: list[Path] = []
    for raw in (
        config.source_root,
        os.getenv("HELICOPTER_AGENTBENCH_SOURCE_ROOT"),
        os.getenv("RWKV_AGENTBENCH_SOURCE_ROOT"),
        os.getenv("AGENTBENCH_SOURCE_ROOT"),
    ):
        if raw:
            candidates.append(_agentbench_data_file(config.dataset_name, Path(raw).expanduser()))
    candidates.extend(
        [
            _agentbench_data_file(config.dataset_name, Path("/tmp/ref-AgentBench")),
            _agentbench_data_file(config.dataset_name, Path("/home/chase/GitHub/rwkv-skills/references/AgentBench")),
            _agentbench_data_file(config.dataset_name, Path("/home/chase/GitHub/AgentBench")),
            _agentbench_data_file(config.dataset_name, Path("/home/chase/AgentBench")),
        ]
    )
    return candidates


def _agentbench_data_file(dataset_name: str, source_root: Path) -> Path:
    if dataset_name == "agentbench_db":
        return source_root / "data" / "dbbench" / "standard.jsonl"
    if dataset_name == "agentbench_kg":
        return source_root / "data" / "knowledgegraph" / "std.json"
    raise ValueError(f"unknown AgentBench dataset: {dataset_name}")


def _agentbench_data_count(path: Path) -> int:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return sum(1 for line in raw.splitlines() if line.strip())
    payload = json.loads(raw)
    if isinstance(payload, list):
        return len(payload)
    raise ValueError(f"unsupported AgentBench data file format: {path}")


def _path_looks_like_manifest(path: Path) -> bool:
    if path.suffix != ".jsonl":
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            return isinstance(payload, Mapping) and {"task_id", "task_name", "index"}.issubset(payload.keys())
    return False


def _assistant_message_from_calls(decoded_calls: Sequence[Mapping[str, Any]], round_index: int) -> dict[str, Any]:
    if not decoded_calls:
        raise ValueError("missing tool call")
    first = decoded_calls[0]
    first_name = str(first.get("name") or "").strip()
    first_arguments = first.get("arguments")
    if not isinstance(first_arguments, Mapping):
        first_arguments = {}
    if first_name == "final_answer":
        return {"role": "assistant", "content": str(first_arguments.get("answer") or "")}
    tool_calls: list[dict[str, Any]] = []
    for index, call in enumerate(decoded_calls):
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        tool_calls.append(
            {
                "id": f"call_{round_index}_{index}",
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": json.dumps(dict(arguments), ensure_ascii=False),
                },
            }
        )
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _normalize_openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
    return {
        "name": str(function.get("name") or ""),
        "description": str(function.get("description") or ""),
        "parameters": dict(function.get("parameters") or {}),
    }


def _format_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role") or "message")
        content = message.get("content")
        parts: list[str] = []
        if content is not None and str(content) != "":
            parts.append(f"{role}: {content}")
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes, bytearray)):
            parts.append(f"{role} tool_calls: {json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)}")
        if not parts:
            parts.append(f"{role}: {json.dumps(dict(message), ensure_ascii=False, sort_keys=True)}")
        rendered.extend(parts)
    return "\n".join(rendered) if rendered else "(empty)"


def _coerce_mapping_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _join_round_artifacts(items: Sequence[str]) -> str:
    return "\n\n".join(f"--- round {index + 1} ---\n{item}" for index, item in enumerate(items))


def _controller_url(config: AgentBenchRunConfig) -> str:
    return str(
        config.controller_url
        or os.getenv("HELICOPTER_AGENTBENCH_CONTROLLER_URL")
        or os.getenv("AGENTBENCH_CONTROLLER_URL")
        or os.getenv("AGENTRL_CONTROLLER_URL")
        or AGENTBENCH_CONTROLLER_DEFAULT_URL
    ).rstrip("/")


def _is_agentbench_kg(record: AgentBenchRecord) -> bool:
    return "kg" in record.task_name.lower()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = [
    "AGENTBENCH_CONTROLLER_DEFAULT_URL",
    "AgentBenchControllerClient",
    "AgentBenchRecord",
    "AgentBenchRunConfig",
    "build_prompt",
    "dry_run_summary",
    "evaluate_samples",
    "load_agentbench_manifest_records",
    "load_agentbench_rows_from_source",
    "load_samples",
    "run_agentbench",
]
