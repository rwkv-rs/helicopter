from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


RESPOND_TOOL_NAME = "respond"
TAU_BENCH_DATASETS: dict[str, dict[str, str]] = {
    "tau_bench_airline": {"domain": "airline", "version": "tau_bench", "split": "test", "job_name": "function_tau_bench"},
    "tau_bench_retail": {"domain": "retail", "version": "tau_bench", "split": "test", "job_name": "function_tau_bench"},
    "tau_bench_telecom": {"domain": "telecom", "version": "tau_bench", "split": "test", "job_name": "function_tau_bench"},
    "tau2_bench_airline": {"domain": "airline", "version": "tau_v2", "split": "base", "job_name": "function_tau2_bench"},
    "tau2_bench_retail": {"domain": "retail", "version": "tau_v2", "split": "base", "job_name": "function_tau2_bench"},
    "tau2_bench_telecom": {"domain": "telecom", "version": "tau_v2", "split": "base", "job_name": "function_tau2_bench"},
    "tau3_bench_airline": {"domain": "airline", "version": "tau_v3", "split": "base", "job_name": "function_tau3_bench"},
    "tau3_bench_retail": {"domain": "retail", "version": "tau_v3", "split": "base", "job_name": "function_tau3_bench"},
    "tau3_bench_telecom": {"domain": "telecom", "version": "tau_v3", "split": "base", "job_name": "function_tau3_bench"},
    "tau3_bench_banking_knowledge": {
        "domain": "banking_knowledge",
        "version": "tau_v3",
        "split": "base",
        "job_name": "function_tau3_bench",
    },
    "tau3_bench_mock": {"domain": "mock", "version": "tau_v3_light", "split": "base", "job_name": "function_tau3_bench"},
    "tau3_bench_mock_long_context": {
        "domain": "mock",
        "version": "tau_v3_light",
        "split": "base",
        "job_name": "function_tau3_bench",
    },
}
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class TauManifestRecord:
    sample_index: int
    task_id: str
    domain: str
    instruction: str
    task: dict[str, Any]
    benchmark_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TauBenchResult:
    sample_index: int
    task_id: str
    domain: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    reward: float
    is_passed: bool
    fail_reason: str
    trace: tuple[dict[str, Any], ...]
    details: dict[str, Any]

    def to_scoreboard(self) -> ScoreboardEvalResult:
        answer = {
            "answer": self.answer,
            "reward": self.reward,
            "task_id": self.task_id,
            "domain": self.domain,
            "trace": list(self.trace),
            "details": self.details,
        }
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=json.dumps(answer, ensure_ascii=False, sort_keys=True),
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class TauBenchRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    limit: int | None = None
    split: str = "base"
    source_path: str | None = None
    source_root: str | None = None
    runtime_root: str | None = None
    data_root: str | None = None
    user_base_url: str | None = None
    user_model: str | None = None
    user_api_key: str | None = None
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: float = 600.0
    max_steps: int = 200
    max_errors: int = 10
    history_max_chars: int = 16000
    prompt_max_chars: int = 24576
    scoreboard_dataset: str | None = None
    job_name: str = "function_tau_bench"
    job_id: str | None = None
    runner: str = "helicopter_eval.tau_bench"
    cot_mode: str = "CoT"


@dataclass(frozen=True, slots=True)
class TauModelConfig:
    base_url: str
    model: str
    api_key: str | None = None


@dataclass(frozen=True, slots=True)
class TauRuntimePreflight:
    ok: bool
    vendor_root: str | None
    data_root: str | None
    error: str | None = None


class TauRuntimeBridge:
    def __init__(self, config: TauBenchRunConfig, *, domain: str) -> None:
        self.config = config
        self.domain = domain
        self.vendor_root, self.data_root = _configure_tau_paths(config)
        registry_module = importlib.import_module("tau2.registry")
        self.registry = getattr(registry_module, "registry")
        self._environment_constructor = self.registry.get_env_constructor(domain)

    def load_task(self, payload: Mapping[str, Any]) -> Any:
        tasks_module = importlib.import_module("tau2.data_model.tasks")
        Task = getattr(tasks_module, "Task")
        return Task.model_validate(_normalize_tau_task_payload(payload))

    def create_environment(self) -> Any:
        kwargs = {"retrieval_variant": "bm25"} if self.domain == "banking_knowledge" else {}
        try:
            return self._environment_constructor(solo_mode=False, **kwargs)
        except TypeError:
            if kwargs:
                raise
            return self._environment_constructor()

    def build_orchestrator(self, *, agent: Any, user: Any, environment: Any, task: Any, seed: int | None) -> Any:
        orchestrator_module = importlib.import_module("tau2.orchestrator.orchestrator")
        Orchestrator = getattr(orchestrator_module, "Orchestrator")
        return Orchestrator(
            domain=self.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max(1, int(self.config.max_steps)),
            max_errors=max(1, int(self.config.max_errors)),
            seed=seed,
            solo_mode=False,
            validate_communication=True,
        )

    def evaluate(self, *, simulation: Any, task: Any, judge_model: TauModelConfig | None) -> tuple[float, bool, dict[str, Any]]:
        if judge_model is not None:
            _configure_tau_judge(judge_model)
        evaluator_module = importlib.import_module("tau2.evaluator.evaluator")
        EvaluationType = getattr(evaluator_module, "EvaluationType")
        evaluate_simulation = getattr(evaluator_module, "evaluate_simulation")
        evaluation_type = EvaluationType.ALL_WITH_NL_ASSERTIONS if _task_uses_nl_assertions(task) else EvaluationType.ALL
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=False,
            domain=self.domain,
            env_kwargs={"retrieval_variant": "bm25"} if self.domain == "banking_knowledge" else {},
        )
        simulation.reward_info = reward_info
        details = _model_dump_safe(reward_info)
        reward = float(getattr(reward_info, "reward", 0.0) or 0.0)
        details["termination_reason"] = str(getattr(simulation, "termination_reason", ""))
        return reward, reward >= (1.0 - 1e-6), details


class HelicopterTauAgent:
    def __init__(
        self,
        *,
        config: TauBenchRunConfig,
        tools: Sequence[Any],
        domain_policy: str,
        domain: str,
    ) -> None:
        message_module = importlib.import_module("tau2.data_model.message")
        self._AssistantMessage = getattr(message_module, "AssistantMessage")
        self._ToolCall = getattr(message_module, "ToolCall")
        self._MultiToolMessage = getattr(message_module, "MultiToolMessage")
        self._ToolMessage = getattr(message_module, "ToolMessage")
        self._UserMessage = getattr(message_module, "UserMessage")
        self.config = config
        self.tools = list(tools)
        self.tools_by_name = {_tool_name(tool): tool for tool in self.tools if _tool_name(tool)}
        self.tool_names = set(self.tools_by_name)
        self.domain_policy = str(domain_policy)
        self.domain = domain
        self.seed: int | None = None
        self.turn_index = 0
        self.prompts: list[str] = []
        self.completions: list[str] = []
        self.parse_errors: list[str] = []

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)

    def get_init_state(self, message_history: list[Any] | None = None) -> list[Any]:
        return list(message_history or [])

    def stop(self, message: Any | None = None, state: list[Any] | None = None) -> None:
        del message, state

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        content = getattr(message, "content", None)
        return isinstance(content, str) and "###STOP###" in content

    def generate_next_message(self, message: Any, state: list[Any] | None) -> tuple[Any, list[Any]]:
        history = list(state or [])
        if message is not None:
            _append_tau_message(history, message, MultiToolMessage=self._MultiToolMessage)
        prompt_messages = _messages_to_prompt_messages(
            history,
            ToolMessage=self._ToolMessage,
            UserMessage=self._UserMessage,
        )
        prompt = build_tau_agent_prompt(
            self.domain_policy,
            self.tools,
            prompt_messages,
            domain=self.domain,
            history_max_chars=self.config.history_max_chars,
            prompt_max_chars=self.config.prompt_max_chars,
        )
        completion = chat_completion(
            base_url=self.config.base_url,
            model=self.config.model,
            prompt=prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            timeout_s=self.config.timeout_s,
            response_format={"type": "json_object"},
        )
        self.prompts.append(prompt)
        self.completions.append(completion)
        try:
            name, arguments = parse_tau_decision(completion)
            assistant_message = self._decision_to_message(name, arguments)
        except Exception as exc:  # noqa: BLE001
            self.parse_errors.append(str(exc))
            assistant_message = self._AssistantMessage(
                role="assistant",
                content="I am unable to continue safely. ###STOP###",
            )
        self.turn_index += 1
        history.append(assistant_message)
        return assistant_message, history

    def _decision_to_message(self, name: str, arguments: Mapping[str, Any]) -> Any:
        normalized_name = _strip_requestor_prefix(name)
        if normalized_name == RESPOND_TOOL_NAME:
            content = str(
                arguments.get("content")
                or arguments.get("answer")
                or arguments.get("message")
                or ""
            ).strip()
            if not content:
                raise ValueError("empty tau respond content")
            return self._AssistantMessage(role="assistant", content=content)
        if normalized_name not in self.tool_names:
            raise ValueError(f"unknown tau tool name: {normalized_name}")
        missing = _missing_required_tool_arguments(self.tools_by_name.get(normalized_name), arguments)
        if missing:
            return self._AssistantMessage(
                role="assistant",
                content=f"I need the required argument(s) {', '.join(missing)} before using {normalized_name}.",
            )
        return self._AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                self._ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=normalized_name,
                    arguments=dict(arguments),
                    requestor="assistant",
                )
            ],
        )


class StaticStopTauUser:
    def __init__(self, *, first_content: str = "", stop_content: str = "###STOP###") -> None:
        self.first_content = str(first_content).strip()
        self.stop_content = stop_content
        self._emitted_first_content = False
        message_module = importlib.import_module("tau2.data_model.message")
        try:
            user_base = importlib.import_module("tau2.user.user_simulator_base")
        except ModuleNotFoundError:
            user_base = importlib.import_module("tau2.user.base")
        self._UserMessage = getattr(message_module, "UserMessage")
        self._UserState = getattr(user_base, "UserState")

    def get_init_state(self, message_history: list[Any] | None = None) -> Any:
        return self._UserState(system_messages=[], messages=list(message_history or []))

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        content = getattr(message, "content", "")
        return isinstance(content, str) and "###STOP###" in content

    def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
        del message
        if self.first_content and not self._emitted_first_content:
            content = self.first_content
            self._emitted_first_content = True
        else:
            content = self.stop_content
        user_message = self._UserMessage(role="user", content=content, cost=0.0)
        state.messages.append(user_message)
        return user_message, state

    def set_seed(self, seed: int) -> None:
        del seed

    def stop(self, message: Any | None = None, state: Any | None = None) -> None:
        del message, state


def load_samples(config: TauBenchRunConfig) -> list[TauManifestRecord]:
    if config.dataset_name not in TAU_BENCH_DATASETS:
        raise ValueError(f"unknown TAU dataset: {config.dataset_name}")
    expected_split = TAU_BENCH_DATASETS[config.dataset_name]["split"]
    if config.split != expected_split:
        raise ValueError(f"{config.dataset_name} only provides {expected_split} split")
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    path = _resolve_manifest_path(config)
    if path is None:
        raise FileNotFoundError(f"no prepared TAU manifest found for {config.dataset_name}")
    rows = load_tau_manifest_records(path)
    if config.limit is not None:
        rows = rows[: int(config.limit)]
    if not rows:
        raise ValueError("TAU run selected zero samples")
    return rows


def load_tau_manifest_records(path: str | Path) -> list[TauManifestRecord]:
    target = Path(path).expanduser().resolve()
    records: list[TauManifestRecord] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(f"TAU manifest row {line_index} is not a JSON object")
            task = payload.get("task")
            if not isinstance(task, Mapping):
                raise ValueError(f"TAU manifest row {line_index} missing task object")
            records.append(
                TauManifestRecord(
                    sample_index=line_index,
                    task_id=str(payload.get("task_id") or task.get("id") or line_index),
                    domain=str(payload.get("domain") or ""),
                    instruction=str(payload.get("instruction") or task.get("ticket") or ""),
                    task=dict(task),
                    benchmark_version=str(payload.get("benchmark_version") or ""),
                    metadata={key: value for key, value in payload.items() if key != "task"},
                )
            )
    return records


def build_tau_agent_prompt(
    domain_policy: str,
    tools: Sequence[Any],
    messages: Sequence[Mapping[str, str]],
    *,
    domain: str,
    history_max_chars: int,
    prompt_max_chars: int,
) -> str:
    tool_schemas = [_normalize_tool_schema(tool) for tool in tools]
    tool_schemas.append(
        {
            "name": RESPOND_TOOL_NAME,
            "description": "Send a natural-language message to the user. Include ###STOP### when the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        }
    )
    system_sections = [
        "You are the assistant in the official tau-bench simulation.",
        "Follow the domain policy exactly.",
        "Use a real tool call when you need information or need to change state.",
        "Use respond only when sending a message to the user.",
        "When the task is complete and no more tool calls are needed, use respond and include ###STOP### in the content.",
        'Return exactly one JSON object with shape {"name":"tool_name","arguments":{...}}.',
        "Use only exact listed tool names; never invent wrapper or pseudo tools.",
    ]
    if domain == "telecom":
        system_sections.append("Telecom device actions in policy text are user instructions, not JSON tool names.")
    system_sections.extend(
        [
            "Tools:",
            json.dumps(tool_schemas, ensure_ascii=False, indent=2, sort_keys=False),
            "Policy:",
            _normalize_text(domain_policy),
        ]
    )
    transcript = _format_prompt_messages(messages, max_chars=max(0, int(history_max_chars)))
    prompt = "\n".join(
        [
            _normalize_text("\n".join(system_sections)),
            "",
            "Conversation:",
            transcript,
            "",
            "Next JSON function call:",
        ]
    )
    if len(prompt) <= prompt_max_chars:
        return prompt
    overflow = len(prompt) - int(prompt_max_chars)
    return "\n".join(
        [
            _normalize_text("\n".join(system_sections)),
            "",
            "Conversation:",
            _format_prompt_messages(messages, max_chars=max(0, int(history_max_chars) - overflow - 512)),
            "",
            "Next JSON function call:",
        ]
    )


def parse_tau_decision(text: str) -> tuple[str, dict[str, Any]]:
    cleaned = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    fenced = _extract_last_fenced_block(cleaned)
    if fenced is not None:
        cleaned = fenced.strip()
    try:
        candidate = _extract_json_value(cleaned)
        payload = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        payload = _partial_decision_payload(cleaned, cause=exc)
    coerced = _coerce_decision_payload(payload)
    return _strip_requestor_prefix(str(coerced["name"])), dict(coerced["arguments"])


def run_tau_bench(config: TauBenchRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    if _requires_user_model(samples) and _user_model_config(config) is None:
        raise ValueError(
            "TAU/Tau2/Tau3 formal scoring requires --tau-user-model and --tau-user-base-url "
            "for non-lightweight user simulation"
        )
    preflight = preflight_runtime(config, domain=samples[0].domain)
    if not preflight.ok:
        raise RuntimeError(f"TAU official runtime is unavailable: {preflight.error}")
    results = [run_tau_sample(sample, config) for sample in samples]
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def run_tau_sample(sample: TauManifestRecord, config: TauBenchRunConfig) -> TauBenchResult:
    runtime = TauRuntimeBridge(config, domain=sample.domain)
    task = runtime.load_task(sample.task)
    environment = runtime.create_environment()
    agent = HelicopterTauAgent(
        config=config,
        tools=environment.get_tools(),
        domain_policy=str(environment.get_policy()),
        domain=sample.domain,
    )
    user = _build_user(runtime=runtime, sample=sample, task=task, environment=environment, config=config)
    orchestrator = runtime.build_orchestrator(
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        seed=sample.sample_index + 1,
    )
    start = time.perf_counter()
    simulation = orchestrator.run()
    reward, is_passed, details = runtime.evaluate(
        simulation=simulation,
        task=task,
        judge_model=_judge_model_config(config),
    )
    fail_reason = "" if is_passed else str(details.get("termination_reason") or "unresolved")
    details["sample_duration_s"] = time.perf_counter() - start
    details["parse_errors"] = list(agent.parse_errors)
    trace = _trajectory_dump(list(getattr(simulation, "messages", []) or getattr(orchestrator, "trajectory", []) or []))
    return TauBenchResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        domain=sample.domain,
        prompt=agent.prompts[-1] if agent.prompts else sample.instruction,
        completion=agent.completions[-1] if agent.completions else "",
        answer=json.dumps({"reward": reward, "trace": trace}, ensure_ascii=False, sort_keys=True),
        reference_answer=f"domain={sample.domain}\ntask_id={sample.task_id}\nbenchmark_version={sample.benchmark_version}",
        reward=reward,
        is_passed=is_passed,
        fail_reason=fail_reason,
        trace=tuple(trace),
        details=details,
    )


def write_results(results: Sequence[TauBenchResult], *, config: TauBenchRunConfig, repo_root: Path) -> int:
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
                extra_metrics={"avg_reward": sum(result.reward for result in results) / len(results) if results else 0.0},
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def dry_run_summary(config: TauBenchRunConfig) -> dict[str, Any]:
    samples = load_samples(config)
    domains = sorted({sample.domain for sample in samples})
    preflight = preflight_runtime(config, domain=domains[0] if domains else TAU_BENCH_DATASETS[config.dataset_name]["domain"])
    return {
        "benchmark": config.benchmark,
        "source": str(_resolve_manifest_path(config)),
        "split": config.split,
        "source_dataset": config.dataset_name,
        "limit": config.limit,
        "available_samples": len(samples),
        "domains": domains,
        "benchmark_versions": sorted({sample.benchmark_version for sample in samples}),
        "base_url": config.base_url,
        "model": config.model,
        "user_model_required": _requires_user_model(samples),
        "user_model_configured": _user_model_config(config) is not None,
        "judge_configured": _judge_model_config(config) is not None,
        "runtime_available": preflight.ok,
        "runtime_vendor_root": preflight.vendor_root,
        "runtime_data_root": preflight.data_root,
        "runtime_error": preflight.error,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


def preflight_runtime(config: TauBenchRunConfig, *, domain: str) -> TauRuntimePreflight:
    try:
        vendor_root, data_root = _configure_tau_paths(config)
        importlib.import_module("tau2.registry")
        registry_module = importlib.import_module("tau2.registry")
        registry = getattr(registry_module, "registry")
        registry.get_env_constructor(domain)
        return TauRuntimePreflight(ok=True, vendor_root=str(vendor_root), data_root=str(data_root))
    except Exception as exc:  # noqa: BLE001
        vendor, data = _best_tau_roots(config)
        return TauRuntimePreflight(
            ok=False,
            vendor_root=str(vendor) if vendor is not None else None,
            data_root=str(data) if data is not None else None,
            error=f"{type(exc).__name__}: {exc}",
        )


def scoreboard_dataset_name(config: TauBenchRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: TauBenchRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: TauBenchRunConfig) -> dict[str, Any]:
    return {
        "decision": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
            "max_steps": config.max_steps,
        }
    }


def task_sampling_config(config: TauBenchRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_tau_official",
        "sampling_config": completion_sampling_config(config),
        "user_model": _model_metadata(_user_model_config(config)),
        "judge_model": _model_metadata(_judge_model_config(config)),
    }


def _resolve_manifest_path(config: TauBenchRunConfig) -> Path | None:
    if config.source_path:
        candidate = Path(config.source_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"TAU source_path does not exist: {candidate}")
    env_key = config.dataset_name.upper()
    for key in (
        f"HELICOPTER_{env_key}_MANIFEST",
        f"RWKV_{env_key}_MANIFEST",
        f"{env_key}_MANIFEST",
        "HELICOPTER_TAU_MANIFEST",
        "RWKV_SKILLS_TAU_SOURCE",
    ):
        raw = os.getenv(key)
        if raw:
            candidate = Path(raw).expanduser()
            if candidate.exists():
                return candidate.resolve()
    for raw in (
        config.source_root,
        os.getenv("HELICOPTER_TAU_DATA_ROOT"),
        os.getenv("RWKV_TAU_DATA_ROOT"),
        os.getenv("TAU_DATA_ROOT"),
    ):
        if raw:
            found = _manifest_from_root(Path(raw).expanduser(), config.dataset_name, config.split)
            if found is not None:
                return found
    for root in (
        _repo_root() / "data",
        Path("/home/chase/GitHub/rwkv-skills/data"),
        Path("/home/chase/rwkv-skills/data"),
        Path("/tmp/rwkv-skills/data"),
    ):
        found = _manifest_from_root(root, config.dataset_name, config.split)
        if found is not None:
            return found
    return None


def _manifest_from_root(root: Path, dataset_name: str, split: str) -> Path | None:
    dataset_dir = root / dataset_name
    candidates = [
        dataset_dir / f"{split}.jsonl",
        dataset_dir / f"{dataset_name}_{split}.jsonl",
        dataset_dir / f"{dataset_name}.jsonl",
    ]
    candidates.extend(sorted(dataset_dir.glob("*.jsonl")) if dataset_dir.is_dir() else [])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _configure_tau_paths(config: TauBenchRunConfig) -> tuple[Path, Path]:
    vendor, data = _best_tau_roots(config)
    if vendor is None:
        raise FileNotFoundError("missing tau2 runtime root; set --tau-runtime-root or TAU2_BENCH_ROOT")
    if data is None:
        raise FileNotFoundError("missing tau2 data root; set --tau-data-root or TAU2_DATA_DIR")
    vendor = vendor.resolve()
    data = data.resolve()
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    os.environ.setdefault("TAU2_DATA_DIR", str(data))
    return vendor, data


def _best_tau_roots(config: TauBenchRunConfig) -> tuple[Path | None, Path | None]:
    bench_roots = [
        config.runtime_root,
        os.getenv("HELICOPTER_TAU_RUNTIME_ROOT"),
        os.getenv("RWKV_TAU3_BENCH_ROOT"),
        os.getenv("TAU3_BENCH_ROOT"),
        os.getenv("RWKV_TAU2_BENCH_ROOT"),
        os.getenv("TAU2_BENCH_ROOT"),
        "/home/chase/GitHub/rwkv-skills/references/tau2-bench",
        "/home/chase/GitHub/rwkv-skills/src/eval/agent_bench/data/tau_v2",
        "/tmp/rwkv-official-refs/tau2-bench",
    ]
    vendor: Path | None = None
    data: Path | None = None
    for raw in bench_roots:
        if not raw:
            continue
        root = Path(raw).expanduser()
        for candidate in (root / "src", root):
            if (candidate / "tau2").exists():
                vendor = candidate
                if (root / "data" / "tau2").exists():
                    data = root / "data"
                break
        if vendor is not None:
            break
    for raw in (
        config.data_root,
        os.getenv("HELICOPTER_TAU_DATA_ROOT"),
        os.getenv("RWKV_TAU3_DATA_ROOT"),
        os.getenv("TAU3_DATA_ROOT"),
        os.getenv("RWKV_TAU2_DATA_ROOT"),
        os.getenv("TAU2_DATA_DIR"),
        data,
    ):
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if (candidate / "tau2").exists():
            data = candidate
            break
    return vendor, data


def _build_user(
    *,
    runtime: TauRuntimeBridge,
    sample: TauManifestRecord,
    task: Any,
    environment: Any,
    config: TauBenchRunConfig,
) -> Any:
    del runtime
    if _is_lightweight_tau_record(sample):
        return StaticStopTauUser(first_content=sample.instruction)
    user_model = _user_model_config(config)
    if user_model is None:
        raise ValueError("missing TAU user model config")
    user_module = importlib.import_module("tau2.user.user_simulator")
    UserSimulator = getattr(user_module, "UserSimulator")
    try:
        user_tools = environment.get_user_tools()
    except Exception:  # noqa: BLE001
        user_tools = None
    return UserSimulator(
        tools=user_tools,
        instructions=str(getattr(task, "user_scenario", "")),
        llm=_litellm_model_name(user_model),
        llm_args={
            "temperature": max(0.001, float(config.temperature)),
            "stream": False,
            "api_key": user_model.api_key,
            "api_base": user_model.base_url,
            **_litellm_provider_args(user_model),
        },
    )


def _user_model_config(config: TauBenchRunConfig) -> TauModelConfig | None:
    model = (config.user_model or os.getenv("HELICOPTER_TAU_USER_MODEL") or os.getenv("USER_MODEL_NAME") or "").strip()
    base_url = (config.user_base_url or os.getenv("HELICOPTER_TAU_USER_BASE_URL") or os.getenv("USER_BASE_URL") or "").strip()
    api_key = (config.user_api_key or os.getenv("HELICOPTER_TAU_USER_API_KEY") or os.getenv("USER_API_KEY") or "").strip()
    if not model or not base_url:
        return None
    return TauModelConfig(base_url=base_url, model=model, api_key=api_key or None)


def _judge_model_config(config: TauBenchRunConfig) -> TauModelConfig | None:
    model = (config.judge_model or os.getenv("HELICOPTER_JUDGE_MODEL") or os.getenv("JUDGE_MODEL") or "").strip()
    base_url = (config.judge_base_url or os.getenv("HELICOPTER_JUDGE_BASE_URL") or os.getenv("JUDGE_BASE_URL") or "").strip()
    api_key = (config.judge_api_key or os.getenv("HELICOPTER_JUDGE_API_KEY") or os.getenv("JUDGE_API_KEY") or "").strip()
    if not model:
        return None
    return TauModelConfig(base_url=base_url, model=model, api_key=api_key or None)


def _configure_tau_judge(judge_model: TauModelConfig) -> None:
    if not judge_model.model:
        return
    args = {
        "temperature": 0.001,
        "stream": False,
        "api_key": judge_model.api_key,
        "response_format": {"type": "json_object"},
    }
    if judge_model.base_url:
        args["api_base"] = judge_model.base_url
    args.update(_litellm_provider_args(judge_model))
    for module_name in ("tau2.config", "tau2.evaluator.evaluator_nl_assertions"):
        module = importlib.import_module(module_name)
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS", _litellm_model_name(judge_model))
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", dict(args))


def _requires_user_model(samples: Sequence[TauManifestRecord]) -> bool:
    return any(not _is_lightweight_tau_record(sample) for sample in samples)


def _is_lightweight_tau_record(sample: TauManifestRecord) -> bool:
    version = sample.benchmark_version.strip().lower()
    return version in {"tau_v3_light", "tau3_light", "tau_light"} or (
        sample.domain == "mock" and version.startswith("tau_v3_light")
    )


def _model_metadata(config: TauModelConfig | None) -> dict[str, Any]:
    return {
        "model": config.model if config else None,
        "base_url": config.base_url if config else None,
        "configured": config is not None,
    }


def _litellm_model_name(config: TauModelConfig) -> str:
    if config.model.startswith("openai/"):
        return config.model.removeprefix("openai/")
    if not config.model or "/" in config.model:
        return config.model
    if "api.deepseek.com" in config.base_url and config.model.startswith("deepseek-"):
        return f"deepseek/{config.model}"
    return config.model


def _litellm_provider_args(config: TauModelConfig) -> dict[str, str]:
    if "api.deepseek.com" in config.base_url:
        return {}
    if config.model.startswith("openai/") or (config.base_url and "/" not in config.model):
        return {"custom_llm_provider": "openai"}
    return {}


def _normalize_tau_task_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    criteria = normalized.get("evaluation_criteria")
    if isinstance(criteria, Mapping):
        normalized_criteria = dict(criteria)
        reward_basis = normalized_criteria.get("reward_basis")
        if isinstance(reward_basis, Sequence) and not isinstance(reward_basis, (str, bytes, bytearray)):
            normalized_criteria["reward_basis"] = [
                str(value).removeprefix("RewardType.") if isinstance(value, str) else value
                for value in reward_basis
            ]
        normalized["evaluation_criteria"] = normalized_criteria
    return normalized


def _task_uses_nl_assertions(task: Any) -> bool:
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria is None:
        return False
    reward_basis = getattr(criteria, "reward_basis", None)
    return any("NL_ASSERTION" in str(item) for item in list(reward_basis or []))


def _normalize_tool_schema(tool: Any) -> dict[str, Any]:
    if hasattr(tool, "openai_schema"):
        schema = getattr(tool, "openai_schema")
        if isinstance(schema, Mapping):
            return dict(schema)
    if hasattr(tool, "model_dump"):
        dumped = tool.model_dump()
        if isinstance(dumped, Mapping):
            if isinstance(dumped.get("function"), Mapping):
                return dict(dumped["function"])
            return dict(dumped)
    if isinstance(tool, Mapping):
        if isinstance(tool.get("function"), Mapping):
            return dict(tool["function"])
        return dict(tool)
    name = str(getattr(tool, "name", "") or tool.__class__.__name__)
    description = str(getattr(tool, "description", "") or "")
    parameters = getattr(tool, "parameters", None) or getattr(tool, "args_schema", None) or {}
    if hasattr(parameters, "model_json_schema"):
        parameters = parameters.model_json_schema()
    return {"name": name, "description": description, "parameters": parameters if isinstance(parameters, Mapping) else {}}


def _tool_name(tool: Any) -> str:
    schema = _normalize_tool_schema(tool)
    return str(schema.get("name") or "").strip()


def _missing_required_tool_arguments(tool: Any, arguments: Mapping[str, Any]) -> list[str]:
    schema = _normalize_tool_schema(tool) if tool is not None else {}
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        return []
    required = parameters.get("required")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes, bytearray)):
        return []
    return [str(key) for key in required if str(key) not in arguments]


def _messages_to_prompt_messages(
    history: Sequence[Any],
    *,
    ToolMessage: Any,
    UserMessage: Any,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, ToolMessage):
            content = str(getattr(item, "content", "") or "")
            messages.append({"role": "user", "content": f"Tool result: {content}"})
            continue
        role = str(getattr(item, "role", "") or "").strip().lower()
        content = str(getattr(item, "content", "") or "").strip()
        tool_calls = getattr(item, "tool_calls", None)
        if tool_calls:
            blocks = []
            for tool_call in tool_calls:
                blocks.append(
                    json.dumps(
                        {
                            "name": str(getattr(tool_call, "name", "") or ""),
                            "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            content = "\n".join(blocks)
        if not content:
            continue
        messages.append({"role": "user" if isinstance(item, UserMessage) or role == "user" else "assistant", "content": content})
    return messages


def _append_tau_message(history: list[Any], message: Any, *, MultiToolMessage: Any) -> None:
    if isinstance(message, MultiToolMessage):
        history.extend(list(getattr(message, "tool_messages", []) or []))
    else:
        history.append(message)


def _format_prompt_messages(messages: Sequence[Mapping[str, str]], *, max_chars: int) -> str:
    rendered = []
    for message in messages:
        role = str(message.get("role") or "user").capitalize()
        content = _normalize_text(str(message.get("content") or ""))
        rendered.append(f"{role}: {content}")
    text = "\n\n".join(rendered)
    if max_chars and len(text) > max_chars:
        return text[-max_chars:]
    return text


def _coerce_decision_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("tau decision payload did not contain a function call")
        payload = payload[0]
    if not isinstance(payload, Mapping):
        raise ValueError("tau decision payload must be a JSON object")
    if "tool_calls" in payload:
        calls = payload.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("tau decision tool_calls payload must contain a call")
        return _coerce_decision_payload(calls[0])
    function = payload.get("function")
    if not isinstance(function, Mapping):
        function = payload.get("function_call")
    if isinstance(function, Mapping):
        name = function.get("name") or payload.get("name")
        arguments = function.get("arguments", payload.get("arguments", {}))
    else:
        name = payload.get("name") or payload.get("action") or payload.get("tool_name") or payload.get("tool")
        arguments = payload.get("arguments", payload.get("action_input", payload.get("input", payload.get("parameters", {}))))
    name_text = str(name or "").strip()
    if not name_text:
        raise ValueError("tau decision missing name")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    if not isinstance(arguments, Mapping):
        raise ValueError("tau decision arguments must be an object")
    return {"name": name_text, "arguments": dict(arguments)}


def _partial_decision_payload(text: str, *, cause: Exception) -> dict[str, Any]:
    body = str(text or "")
    start = body.find("{")
    if start < 0:
        raise ValueError(f"tau decision missing JSON object: {body}") from cause
    body = body[start:]
    name = _raw_decode_json_field(body, "name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"tau decision missing recoverable name: {body}") from cause
    try:
        arguments = _raw_decode_json_field(body, "arguments")
    except ValueError:
        arguments = {}
    return {"name": name, "arguments": arguments or {}}


def _raw_decode_json_field(body: str, key: str) -> Any:
    match = re.search(rf'"{re.escape(key)}"\s*:', body)
    if match is None:
        if key == "arguments":
            return None
        raise ValueError(f"missing JSON field {key!r}")
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(body[match.end() :].lstrip())
    return value


def _extract_json_value(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _value, end = decoder.raw_decode(text[index:])
            return text[index : index + end]
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON value found")


def _extract_last_fenced_block(text: str) -> str | None:
    matches = list(_FENCED_BLOCK_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1)


def _strip_requestor_prefix(name: str) -> str:
    text = str(name or "").strip()
    if "." in text:
        prefix, rest = text.split(".", 1)
        if prefix in {"assistant", "user"} and rest.strip():
            return rest.strip()
    return text


def _trajectory_dump(trajectory: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in trajectory:
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                rows.append(dumped)
                continue
        rows.append(
            {
                "role": str(getattr(message, "role", "unknown")),
                "content": str(getattr(message, "content", "")),
            }
        )
    return rows


def _model_dump_safe(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        dumped = item.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(item, Mapping):
        return dict(item)
    return {"value": str(item)}


def _normalize_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", str(value or "").replace("\r\n", "\n")).strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = [
    "TAU_BENCH_DATASETS",
    "TauBenchRunConfig",
    "TauManifestRecord",
    "build_tau_agent_prompt",
    "dry_run_summary",
    "load_samples",
    "load_tau_manifest_records",
    "parse_tau_decision",
    "preflight_runtime",
    "run_tau_bench",
]
