from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .agent_format import (
    canonical_intermediate_rows,
    conversion_errors_text,
    read_json_records,
    swebench_prediction_rows,
    write_jsonl,
)
from .commands import local_openai_base_url
from .config import resolve_model_entry, table
from .eval_run import DEFAULT_SERVER_TIMEOUT_S
from .env import env_value, pick
from .paths import resolve_path


DEFAULT_AGENT_BENCHMARK_SOURCE = "benchmarks/agent_benchmarks.json"
DEFAULT_OUTPUT_DIR = "results/agent_harness"


@dataclass(frozen=True)
class HarnessProfile:
    name: str
    kind: str
    sandbox: str
    entrypoint: str
    required_tools: tuple[str, ...]
    official_source: str | None = None
    prediction_artifact: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class AgentBenchmark:
    name: str
    display_name: str
    pipeline: str
    priority: str
    run_status: str
    harness_profile: str
    official_dataset: str | None = None
    reproducibility: str | None = None
    problem_domain: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class AgentHarnessSource:
    benchmarks: tuple[AgentBenchmark, ...]
    profiles: dict[str, HarnessProfile]
    pipelines: dict[str, tuple[str, ...]]
    excluded: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PreflightRow:
    benchmark: str
    pipeline: str
    harness_profile: str
    sandbox: str
    status: str
    missing_tools: tuple[str, ...]
    run_status: str
    notes: str | None = None


def _source_path(root: Path, source: str | None) -> Path:
    text = source or DEFAULT_AGENT_BENCHMARK_SOURCE
    path = Path(text)
    return path if path.is_absolute() else root / path


def _profile_from_row(name: str, row: dict[str, Any]) -> HarnessProfile:
    return HarnessProfile(
        name=name,
        kind=str(row.get("kind") or name),
        sandbox=str(row.get("sandbox") or "external"),
        entrypoint=str(row.get("entrypoint") or ""),
        required_tools=tuple(str(item) for item in row.get("required_tools", []) or []),
        official_source=str(row["official_source"]) if row.get("official_source") else None,
        prediction_artifact=str(row["prediction_artifact"]) if row.get("prediction_artifact") else None,
        notes=str(row["notes"]) if row.get("notes") else None,
    )


def _benchmark_from_row(row: dict[str, Any]) -> AgentBenchmark:
    name = str(row.get("name") or "")
    if not name:
        raise SystemExit(f"agent benchmark row is missing name: {row!r}")
    harness_profile = str(row.get("harness_profile") or "")
    if not harness_profile:
        raise SystemExit(f"agent benchmark row is missing harness_profile: {name}")
    return AgentBenchmark(
        name=name,
        display_name=str(row.get("display_name") or name),
        pipeline=str(row.get("pipeline") or ""),
        priority=str(row.get("priority") or ""),
        run_status=str(row.get("run_status") or ""),
        harness_profile=harness_profile,
        official_dataset=str(row["official_dataset"]) if row.get("official_dataset") else None,
        reproducibility=str(row["reproducibility"]) if row.get("reproducibility") else None,
        problem_domain=str(row["problem_domain"]) if row.get("problem_domain") else None,
        notes=str(row["notes"]) if row.get("notes") else None,
    )


def load_agent_harness_source(root: Path, source: str | None) -> AgentHarnessSource:
    path = _source_path(root, source)
    if not path.exists():
        raise SystemExit(f"agent benchmark source not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must contain a JSON object")

    profiles_raw = raw.get("harness_profiles", {})
    if not isinstance(profiles_raw, dict):
        raise SystemExit(f"{path}: harness_profiles must be a JSON object")
    profiles = {str(name): _profile_from_row(str(name), dict(row)) for name, row in profiles_raw.items()}

    benchmarks = tuple(_benchmark_from_row(dict(row)) for row in raw.get("benchmarks", []) or [])
    missing_profiles = sorted({row.harness_profile for row in benchmarks if row.harness_profile not in profiles})
    if missing_profiles:
        raise SystemExit(f"{path}: missing harness profile definitions: {', '.join(missing_profiles)}")

    pipelines_raw = raw.get("pipelines", []) or []
    pipelines: dict[str, tuple[str, ...]] = {}
    for row in pipelines_raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if not name:
            continue
        pipelines[name] = tuple(str(item) for item in row.get("benchmarks", []) or [])

    excluded = tuple(dict(row) for row in raw.get("excluded", []) or [] if isinstance(row, dict))
    return AgentHarnessSource(benchmarks=benchmarks, profiles=profiles, pipelines=pipelines, excluded=excluded)


def _select_benchmarks(source: AgentHarnessSource, *, pipeline: str | None, benchmark: str | None) -> list[AgentBenchmark]:
    rows = list(source.benchmarks)
    if pipeline:
        rows = [row for row in rows if row.pipeline == pipeline]
    if benchmark:
        rows = [row for row in rows if row.name == benchmark]
        if not rows:
            raise SystemExit(f"unknown agent benchmark: {benchmark}")
    return rows


def _format_text_rows(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column) or "") for column in columns))
    return "\n".join(lines) + "\n"


def list_rows(source: AgentHarnessSource, *, pipeline: str | None, benchmark: str | None) -> list[dict[str, Any]]:
    rows = []
    for item in _select_benchmarks(source, pipeline=pipeline, benchmark=benchmark):
        profile = source.profiles[item.harness_profile]
        rows.append(
            {
                "benchmark": item.name,
                "display_name": item.display_name,
                "pipeline": item.pipeline,
                "priority": item.priority,
                "run_status": item.run_status,
                "harness_profile": item.harness_profile,
                "harness_kind": profile.kind,
                "sandbox": profile.sandbox,
                "entrypoint": profile.entrypoint,
                "official_dataset": item.official_dataset or "",
                "official_source": profile.official_source or "",
            }
        )
    return rows


def preflight_rows(source: AgentHarnessSource, *, pipeline: str | None, benchmark: str | None) -> list[PreflightRow]:
    rows = []
    for item in _select_benchmarks(source, pipeline=pipeline, benchmark=benchmark):
        profile = source.profiles[item.harness_profile]
        missing_tools = tuple(tool for tool in profile.required_tools if shutil.which(tool) is None)
        if item.reproducibility == "internal_only":
            status = "internal_only"
        elif missing_tools:
            status = "blocked"
        elif item.run_status.startswith("local_") and profile.kind == "browser_search_answer":
            status = "local_proxy_available"
        elif item.run_status.startswith("local_"):
            status = "local_proxy_available_official_harness_required"
        elif item.run_status == "external_harness_required":
            status = "external_harness_ready_to_prepare"
        else:
            status = item.run_status or "unknown"
        rows.append(
            PreflightRow(
                benchmark=item.name,
                pipeline=item.pipeline,
                harness_profile=item.harness_profile,
                sandbox=profile.sandbox,
                status=status,
                missing_tools=missing_tools,
                run_status=item.run_status,
                notes=profile.notes,
            )
        )
    return rows


def _output_dir(config: dict[str, Any], *, root: Path, env: dict[str, str], args: Any) -> Path:
    agent_config = table(config, "agent_harness")
    value = pick(
        getattr(args, "output_dir", None),
        env_value(env, "HELICOPTER_AGENT_HARNESS_OUTPUT_DIR"),
        agent_config.get("output_dir"),
        DEFAULT_OUTPUT_DIR,
    )
    return resolve_path(str(value), root=root, env=env)


def _max_workers(config: dict[str, Any], args: Any) -> int:
    agent_config = table(config, "agent_harness")
    value = pick(getattr(args, "n_concurrent", None), agent_config.get("n_concurrent"), 1)
    return max(1, int(value))


def _run_id(model_name: str, benchmark: str, args: Any) -> str:
    return str(pick(getattr(args, "run_id", None), f"{model_name}_{benchmark}")).replace("/", "_")


def _served_model_name(config: dict[str, Any], model_name: str) -> str:
    try:
        model = resolve_model_entry(config, model_name)
    except SystemExit:
        return model_name
    return str(model.get("served_model_name") or model.get("name") or model_name)


def _args_with_benchmark(args: Any, benchmark: str) -> Any:
    values = dict(vars(args))
    values["benchmark"] = benchmark
    return SimpleNamespace(**values)


def plan_for_benchmark(
    source: AgentHarnessSource,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    benchmark_name = getattr(args, "benchmark", None)
    if not benchmark_name:
        raise SystemExit("agent-harness plan requires a benchmark")
    selected = _select_benchmarks(source, pipeline=None, benchmark=benchmark_name)
    benchmark = selected[0]
    profile = source.profiles[benchmark.harness_profile]
    model_name = str(getattr(args, "model", None) or "MODEL")
    served_model_name = _served_model_name(config, model_name)
    output_dir = _output_dir(config, root=root, env=env, args=args) / benchmark.name
    prediction_path = output_dir / "predictions.jsonl"
    run_id = _run_id(served_model_name, benchmark.name, args)
    base_url = getattr(args, "base_url", None) or local_openai_base_url(config, env, args)
    max_workers = _max_workers(config, args)

    steps: list[dict[str, Any]] = []
    adapter_contract: dict[str, Any] = {
        "model": served_model_name,
        "base_url": base_url,
        "output_dir": str(output_dir),
        "prediction_artifact": profile.prediction_artifact,
    }

    if profile.kind == "swebench":
        dataset = benchmark.official_dataset or benchmark.display_name
        steps = [
            {
                "name": "generate_patch_predictions",
                "status": "adapter_required",
                "artifact": str(prediction_path),
                "schema": {
                    "instance_id": "official instance id",
                    "model_name_or_path": served_model_name,
                    "model_patch": "unified diff patch generated by the coding agent",
                },
            },
            {
                "name": "official_sandbox_eval",
                "status": "ready_after_predictions",
                "command": [
                    "python",
                    "-m",
                    "swebench.harness.run_evaluation",
                    "--dataset_name",
                    dataset,
                    "--predictions_path",
                    str(prediction_path),
                    "--max_workers",
                    str(max_workers),
                    "--run_id",
                    run_id,
                ],
            },
        ]
    elif profile.kind == "terminal_bench":
        steps = [
            {
                "name": "implement_terminal_agent_adapter",
                "status": "adapter_required",
                "contract": {
                    "base_url": base_url,
                    "model": served_model_name,
                    "task_environment": "official Terminal-Bench/Harbor Docker task",
                },
            },
            {
                "name": "official_sandbox_eval",
                "status": "ready_after_adapter",
                "entrypoint": profile.entrypoint,
                "notes": "Use the official tb/harbor runner with the Helicopter OpenAI-compatible agent adapter.",
            },
        ]
    elif profile.kind == "browser_search_answer":
        steps = [
            {
                "name": "local_answer_proxy",
                "status": "available",
                "command": [
                    "helicopter",
                    "eval",
                    "run",
                    model_name,
                    "browsecomp",
                    "--output-dir",
                    str(output_dir),
                ],
            },
            {
                "name": "browser_runtime_harness",
                "status": "separate_adapter_required",
                "notes": "The current local task does not launch or audit a browser/search runtime.",
            },
        ]
    else:
        steps = [
            {
                "name": "official_harness_adapter",
                "status": "adapter_required",
                "entrypoint": profile.entrypoint,
                "sandbox": profile.sandbox,
                "notes": profile.notes,
            }
        ]

    return {
        "benchmark": asdict(benchmark),
        "harness_profile": asdict(profile),
        "adapter_contract": adapter_contract,
        "steps": steps,
    }


def _plan_path_for(output_dir: Path, benchmark: str) -> Path:
    return output_dir / benchmark / "plan.json"


def _write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _local_proxy_command(
    *,
    benchmark: AgentBenchmark,
    output_dir: Path,
    args: Any,
    config: dict[str, Any],
    env: dict[str, str],
) -> list[str]:
    model = str(getattr(args, "model", "") or "")
    command = [
        "helicopter",
        "eval",
        "run",
        model or "MODEL",
        benchmark.name,
        "--output-dir",
        str(output_dir / benchmark.name),
    ]
    base_url = getattr(args, "base_url", None) or local_openai_base_url(config, env, args)
    if base_url:
        command.extend(["--base-url", str(base_url)])
    if getattr(args, "max_samples", None) is not None:
        command.extend(["--max-samples", str(getattr(args, "max_samples"))])
    if getattr(args, "no_server", False):
        command.append("--no-server")
    if getattr(args, "keep_server", False):
        command.append("--keep-server")
    return command


def _run_local_answer_proxy(
    *,
    benchmark: AgentBenchmark,
    output_dir: Path,
    args: Any,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> int:
    model = str(getattr(args, "model", "") or "")
    if not model:
        raise SystemExit(f"agent run {benchmark.name} requires --model")
    from .eval_run import run_eval

    run_args = SimpleNamespace(
        model=model,
        tasks=benchmark.name,
        backend="endpoint-litellm",
        model_args=None,
        lighteval_model_name=None,
        base_url=getattr(args, "base_url", None),
        provider=None,
        api_key=None,
        concurrent_requests=None,
        max_model_length=None,
        max_new_tokens=None,
        max_samples=getattr(args, "max_samples", None),
        output_dir=str(output_dir / benchmark.name),
        dataset_loading_processes=None,
        num_fewshot_seeds=None,
        custom_tasks=None,
        load_tasks_multilingual=None,
        save_details=None,
        push_to_hub=None,
        public_run=None,
        results_org=None,
        job_id=None,
        extra=None,
        performance_output=None,
        metrics_url=None,
        scoreboard_task_id=None,
        wkv_mode=None,
        emb_device=None,
        tensor_parallel_size=None,
        gpu_memory_utilization=None,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        enable_auto_tool_choice=None,
        vllm_env=None,
        no_server=bool(getattr(args, "no_server", False)),
        keep_server=bool(getattr(args, "keep_server", False)),
        server_timeout=float(getattr(args, "server_timeout", None) or DEFAULT_SERVER_TIMEOUT_S),
        scoreboard=False,
        dry_run=False,
    )
    return run_eval(run_args, root=root, env=env, config=config)


def run_agent_benchmarks(
    source: AgentHarnessSource,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
    args: Any,
) -> tuple[list[dict[str, Any]], int]:
    selected = _select_benchmarks(
        source,
        pipeline=getattr(args, "pipeline", None),
        benchmark=getattr(args, "benchmark", None),
    )
    if not selected:
        raise SystemExit("no agent benchmarks selected")

    output_dir = _output_dir(config, root=root, env=env, args=args)
    rows: list[dict[str, Any]] = []
    exit_code = 0
    for benchmark in selected:
        profile = source.profiles[benchmark.harness_profile]
        one_args = _args_with_benchmark(args, benchmark.name)
        preflight = preflight_rows(source, pipeline=None, benchmark=benchmark.name)[0]
        plan = plan_for_benchmark(source, root=root, env=env, config=config, args=one_args)
        plan_path = _plan_path_for(output_dir, benchmark.name)
        row: dict[str, Any] = {
            "benchmark": benchmark.name,
            "pipeline": benchmark.pipeline,
            "harness_profile": benchmark.harness_profile,
            "sandbox": profile.sandbox,
            "status": "unknown",
            "message": "",
            "missing_tools": ",".join(preflight.missing_tools),
            "plan_path": str(plan_path),
        }

        if benchmark.reproducibility == "internal_only":
            row["status"] = "internal_only"
            row["message"] = "internal-name placeholder; no external reproducible harness is configured"
            exit_code = 1
        elif preflight.missing_tools:
            row["status"] = "blocked"
            row["message"] = f"missing required tools: {', '.join(preflight.missing_tools)}"
            exit_code = 1
        elif profile.kind == "browser_search_answer":
            if not getattr(args, "allow_proxy", False):
                row["status"] = "blocked_proxy"
                row["message"] = "local BrowseComp answer proxy requires --allow-proxy; it is not a browser-runtime score"
                exit_code = 1
            elif not getattr(args, "model", None):
                row["status"] = "blocked"
                row["message"] = "local proxy run requires --model"
                exit_code = 1
            elif getattr(args, "dry_run", False):
                row["status"] = "dry_run"
                row["message"] = "would run local answer proxy"
                row["command"] = _local_proxy_command(
                    benchmark=benchmark,
                    output_dir=output_dir,
                    args=args,
                    config=config,
                    env=env,
                )
            else:
                rc = _run_local_answer_proxy(
                    benchmark=benchmark,
                    output_dir=output_dir,
                    args=args,
                    root=root,
                    env=env,
                    config=config,
                )
                row["status"] = "completed" if rc == 0 else "failed"
                row["message"] = "local answer proxy completed" if rc == 0 else f"local answer proxy failed: {rc}"
                exit_code = exit_code or rc
        else:
            row["status"] = "external_harness_not_implemented"
            row["message"] = "official harness adapter is not implemented in Helicopter; use the plan artifact"
            exit_code = 1

        if not getattr(args, "dry_run", False) and row["status"] not in {"completed", "dry_run"}:
            _write_plan(plan_path, plan)
        rows.append(row)
    return rows, exit_code


def convert_agent_outputs(
    source: AgentHarnessSource,
    *,
    root: Path,
    args: Any,
) -> dict[str, Any]:
    benchmark_name = getattr(args, "benchmark", None)
    if not benchmark_name:
        raise SystemExit("agent-harness convert requires a benchmark")
    input_text = getattr(args, "input", None)
    if not input_text:
        raise SystemExit("agent-harness convert requires --input")
    output_text = getattr(args, "output", None)
    if not output_text:
        raise SystemExit("agent-harness convert requires --output")

    benchmark = _select_benchmarks(source, pipeline=None, benchmark=benchmark_name)[0]
    profile = source.profiles[benchmark.harness_profile]
    target = str(getattr(args, "target", "auto") or "auto")
    if target == "auto":
        target = "swebench-predictions" if profile.kind == "swebench" else "intermediate"

    input_path = Path(str(input_text))
    if not input_path.is_absolute():
        input_path = root / input_path
    output_path = Path(str(output_text))
    if not output_path.is_absolute():
        output_path = root / output_path

    records = read_json_records(input_path)
    model = str(getattr(args, "model", None) or "")
    if target == "intermediate":
        rows, errors = canonical_intermediate_rows(records, benchmark=benchmark.name, model=model)
    elif target == "swebench-predictions":
        if profile.kind != "swebench":
            raise SystemExit(f"{benchmark.name} uses {profile.kind}; target swebench-predictions is invalid")
        if not model:
            raise SystemExit("target swebench-predictions requires --model")
        rows, errors = swebench_prediction_rows(
            records,
            model=model,
            allow_empty_patch=bool(getattr(args, "allow_empty_patch", False)),
        )
    else:
        raise SystemExit(f"unknown agent output conversion target: {target}")

    if errors and not getattr(args, "allow_invalid", False):
        raise SystemExit(conversion_errors_text(errors))
    write_jsonl(output_path, rows)
    return {
        "benchmark": benchmark.name,
        "target": target,
        "input": str(input_path),
        "output": str(output_path),
        "records": len(records),
        "written": len(rows),
        "errors": [asdict(error) for error in errors],
    }


def format_agent_harness_output(rows: list[dict[str, Any]], output_format: str) -> str:
    if output_format == "jsonl":
        return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if output_format == "summary":
        counts: dict[str, int] = {}
        for row in rows:
            key = str(row.get("status") or row.get("pipeline") or "unknown")
            counts[key] = counts.get(key, 0) + 1
        lines = [f"total\t{len(rows)}"]
        lines.extend(f"{key}\t{counts[key]}" for key in sorted(counts))
        return "\n".join(lines) + "\n"
    if rows and "message" in rows[0]:
        columns = (
            "benchmark",
            "pipeline",
            "harness_profile",
            "status",
            "missing_tools",
            "message",
            "plan_path",
        )
        return _format_text_rows(rows, columns)
    columns = (
        "benchmark",
        "pipeline",
        "harness_profile",
        "sandbox",
        "status" if rows and "status" in rows[0] else "run_status",
        "missing_tools" if rows and "missing_tools" in rows[0] else "entrypoint",
    )
    return _format_text_rows(rows, columns)


def run_agent_harness(args: Any, *, root: Path, env: dict[str, str], config: dict[str, Any]) -> int:
    source = load_agent_harness_source(root, getattr(args, "source", None))
    action = getattr(args, "agent_action", None)
    if action == "list":
        rows = list_rows(source, pipeline=getattr(args, "pipeline", None), benchmark=getattr(args, "benchmark", None))
        print(format_agent_harness_output(rows, getattr(args, "format", "text")), end="")
        return 0
    if action == "preflight":
        preflight = preflight_rows(
            source,
            pipeline=getattr(args, "pipeline", None),
            benchmark=getattr(args, "benchmark", None),
        )
        rows = [
            {
                **asdict(row),
                "missing_tools": ",".join(row.missing_tools),
            }
            for row in preflight
        ]
        print(format_agent_harness_output(rows, getattr(args, "format", "text")), end="")
        if getattr(args, "strict", False) and any(row.status == "blocked" for row in preflight):
            return 1
        return 0
    if action == "plan":
        plan = plan_for_benchmark(source, root=root, env=env, config=config, args=args)
        if getattr(args, "format", "text") == "jsonl":
            print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
            return 0
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if action == "convert":
        result = convert_agent_outputs(source, root=root, args=args)
        if getattr(args, "format", "text") == "jsonl":
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        print(
            _format_text_rows(
                [result],
                ("benchmark", "target", "records", "written", "output"),
            ),
            end="",
        )
        return 0
    if action == "run":
        rows, exit_code = run_agent_benchmarks(source, root=root, env=env, config=config, args=args)
        print(format_agent_harness_output(rows, getattr(args, "format", "text")), end="")
        return exit_code
    raise SystemExit(f"unknown agent harness action: {action}")
