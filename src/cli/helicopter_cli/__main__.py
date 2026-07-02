from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from .commands import (
    EMB_DEVICES,
    WKV_MODES,
    build_eval_infer_plan,
    build_infer_plan,
    build_takeoff_plan,
    prepend_venv_path,
)
from .config import load_config
from .env import DEFAULT_ENV_FILE, load_env
from .eval_catalog import BenchmarkSpec, RunnerSpec, job_plan_to_dict, load_rwkv_skills_catalog
from .paths import find_root
from .runner import run_command


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="TOML config path; defaults to the newest configs/local/*.toml")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="dotenv file to load first")
    parser.add_argument("--dry-run", action="store_true", help="print the command without executing it")


def benchmark_to_dict(benchmark: BenchmarkSpec) -> dict[str, Any]:
    return {
        "name": benchmark.name,
        "field": benchmark.field,
        "dataset": benchmark.dataset,
        "default_split": benchmark.default_split,
        "dataset_slug": benchmark.dataset_slug,
        "cot_modes": list(benchmark.cot_modes),
        "scheduler_jobs": list(benchmark.scheduler_jobs),
        "target_eval_attempts": benchmark.target_eval_attempts,
    }


def runner_to_dict(runner: RunnerSpec) -> dict[str, Any]:
    return {
        "name": runner.name,
        "group": runner.group,
        "scheduler_domain": runner.scheduler_domain,
        "module": runner.module,
        "is_cot": runner.is_cot,
        "fallback_dataset_slugs": list(runner.fallback_dataset_slugs),
        "extra_args": list(runner.extra_args),
        "batch_flag": runner.batch_flag,
        "probe_flag": runner.probe_flag,
    }


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def handle_eval_benchmarks(args: argparse.Namespace, **_: Any) -> int:
    catalog = load_rwkv_skills_catalog()
    benchmarks = catalog.select_benchmarks(names=args.benchmark, fields=args.field)
    if args.json:
        print_json(
            {
                "count": len(benchmarks),
                "field_counts": catalog.field_counts(),
                "benchmarks": [benchmark_to_dict(item) for item in benchmarks],
            }
        )
        return 0

    print(f"rwkv-skills benchmarks: {len(benchmarks)} selected / {len(catalog.benchmarks)} total")
    for field, count in catalog.field_counts().items():
        print(f"{field}: {count}")
    for item in benchmarks:
        jobs = ",".join(item.scheduler_jobs) if item.scheduler_jobs else "-"
        print(f"{item.name}\t{item.field}\t{item.dataset_slug}\t{jobs}")
    return 0


def handle_eval_runners(args: argparse.Namespace, **_: Any) -> int:
    catalog = load_rwkv_skills_catalog()
    runners = tuple(
        runner for runner in catalog.runners if not args.group or runner.group in set(args.group)
    )
    if args.json:
        print_json({"count": len(runners), "runners": [runner_to_dict(item) for item in runners]})
        return 0

    print(f"rwkv-skills runners: {len(runners)} selected / {len(catalog.runners)} total")
    for item in runners:
        print(f"{item.name}\t{item.group}\t{item.scheduler_domain}\t{item.module}")
    return 0


def handle_eval_plan(args: argparse.Namespace, **_: Any) -> int:
    catalog = load_rwkv_skills_catalog()
    rows = catalog.build_job_plan(names=args.benchmark, fields=args.field)
    if args.ready_only:
        rows = tuple(row for row in rows if row.status == "ready")
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    if args.json:
        print_json(
            {
                "benchmarks": len(catalog.select_benchmarks(names=args.benchmark, fields=args.field)),
                "jobs": len(rows),
                "status_counts": dict(sorted(status_counts.items())),
                "inference_defaults": asdict(catalog.inference_defaults),
                "plan": [job_plan_to_dict(row) for row in rows],
            }
        )
        return 0

    print(
        "rwkv-skills eval plan: "
        f"{len(rows)} jobs, statuses={dict(sorted(status_counts.items()))}, "
        f"model={catalog.inference_defaults.model_name}, protocol={catalog.inference_defaults.protocol}"
    )
    for row in rows:
        runner = row.runner or "-"
        module = row.module or "-"
        print(f"{row.status}\t{row.benchmark}\t{row.field}\t{row.dataset_slug}\t{runner}\t{module}")
    return 0


def handle_eval_runnable(args: argparse.Namespace, **_: Any) -> int:
    from helicopter_eval.catalog_runner import catalog_run_spec_to_dict, resolve_catalog_run_spec

    catalog = load_rwkv_skills_catalog()
    specs = tuple(
        resolve_catalog_run_spec(benchmark)
        for benchmark in catalog.select_benchmarks(names=args.benchmark, fields=args.field)
    )
    status_counts: dict[str, int] = {}
    for spec in specs:
        status_counts[spec.status] = status_counts.get(spec.status, 0) + 1
    if args.json:
        print_json(
            {
                "count": len(specs),
                "status_counts": dict(sorted(status_counts.items())),
                "runnable": [catalog_run_spec_to_dict(spec) for spec in specs],
            }
        )
        return 0

    print(f"rwkv-skills runnable catalog: {len(specs)} selected, statuses={dict(sorted(status_counts.items()))}")
    for spec in specs:
        kind = spec.kind or "-"
        print(f"{spec.status}\t{spec.benchmark}\t{spec.field}\t{spec.dataset_slug}\t{kind}\t{spec.reason}")
    return 0


def handle_eval_run(args: argparse.Namespace, *, root: Any, **_: Any) -> int:
    if args.benchmark != "gsm8k":
        raise SystemExit("only gsm8k is implemented for eval run in this stage")
    from helicopter_eval.gsm8k import Gsm8kRunConfig, dry_run_summary, run_gsm8k

    defaults = load_rwkv_skills_catalog().inference_defaults
    run_config = Gsm8kRunConfig(
        base_url=str(args.base_url or defaults.base_url),
        model=str(args.model or defaults.model_name),
        limit=args.limit,
        split=str(args.split),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        timeout_s=float(args.timeout_s),
        job_id=str(args.job_id),
    )
    payload = dry_run_summary(run_config) if args.dry_run else run_gsm8k(run_config, repo_root=root)
    print_json(payload)
    return 0


def handle_eval_run_catalog(args: argparse.Namespace, *, root: Any, **_: Any) -> int:
    from helicopter_eval.catalog_runner import (
        dry_run_catalog_spec,
        export_catalog_sample_manifest,
        resolve_catalog_run_spec,
        run_catalog_spec,
    )

    catalog = load_rwkv_skills_catalog()
    try:
        benchmark = catalog.benchmarks_by_name[str(args.benchmark)]
    except KeyError as exc:
        raise SystemExit(f"unknown benchmark: {args.benchmark}") from exc
    spec = resolve_catalog_run_spec(benchmark)
    if spec.status != "implemented":
        raise SystemExit(f"{spec.benchmark} is not runnable yet: {spec.reason}")
    defaults = catalog.inference_defaults
    kwargs = {
        "base_url": str(args.base_url or defaults.base_url),
        "model": str(args.model or defaults.model_name),
        "limit": args.limit,
        "sample_size": getattr(args, "sample_size", None),
        "sample_seed": int(getattr(args, "sample_seed", 42)),
        "longbench_source_path": getattr(args, "longbench_source_path", None),
        "longbench_infer_protocol": getattr(args, "longbench_infer_protocol", None),
        "longbench_temperature": getattr(args, "longbench_temperature", None),
        "longbench_top_p": getattr(args, "longbench_top_p", None),
        "longbench_presence_penalty": getattr(args, "longbench_presence_penalty", None),
        "longbench_frequency_penalty": getattr(args, "longbench_frequency_penalty", None),
        "longbench_seed_requests": bool(getattr(args, "longbench_seed_requests", False)),
        "longbench_stop_suffixes": tuple(getattr(args, "longbench_stop_suffix", None) or ()),
        "agentbench_controller_url": getattr(args, "agentbench_controller_url", None),
        "mcp_runtime_root": getattr(args, "mcp_runtime_root", None),
        "mcp_worker_script": getattr(args, "mcp_worker_script", None),
        "mcp_max_rounds": getattr(args, "mcp_max_rounds", None),
        "judge_base_url": getattr(args, "judge_base_url", None),
        "judge_model": getattr(args, "judge_model", None),
        "judge_api_key": getattr(args, "judge_api_key", None),
        "swebench_run_harness": getattr(args, "swebench_run_harness", False),
        "swebench_predictions_dir": getattr(args, "swebench_predictions_dir", None),
        "swebench_harness_run_id": getattr(args, "swebench_harness_run_id", None),
        "swebench_max_workers": getattr(args, "swebench_max_workers", None),
        "swebench_cache_level": getattr(args, "swebench_cache_level", None),
        "swebench_clean": getattr(args, "swebench_clean", False),
        "swebench_timeout_s": getattr(args, "swebench_timeout_s", None),
        "swebench_max_context_chars": getattr(args, "swebench_max_context_chars", None),
        "tau_runtime_root": getattr(args, "tau_runtime_root", None),
        "tau_data_root": getattr(args, "tau_data_root", None),
        "tau_user_base_url": getattr(args, "tau_user_base_url", None),
        "tau_user_model": getattr(args, "tau_user_model", None),
        "tau_user_api_key": getattr(args, "tau_user_api_key", None),
        "tau_max_steps": getattr(args, "tau_max_steps", None),
        "tau_max_errors": getattr(args, "tau_max_errors", None),
        "tau_history_max_chars": getattr(args, "tau_history_max_chars", None),
        "tau_prompt_max_chars": getattr(args, "tau_prompt_max_chars", None),
        "candidate_router_mode": getattr(args, "candidate_router_mode", None),
        "candidate_router_chunk_tools": getattr(args, "candidate_router_chunk_tools", None),
        "candidate_router_batch_size": getattr(args, "candidate_router_batch_size", None),
        "candidate_router_context_chars": getattr(args, "candidate_router_context_chars", None),
        "candidate_router_prompt_max_chars": getattr(args, "candidate_router_prompt_max_chars", None),
        "candidate_router_candidate_max_tokens": getattr(args, "candidate_router_candidate_max_tokens", None),
        "candidate_router_aggregate_max_tokens": getattr(args, "candidate_router_aggregate_max_tokens", None),
        "candidate_router_max_candidates": getattr(args, "candidate_router_max_candidates", None),
        "candidate_router_tool_schema_mode": getattr(args, "candidate_router_tool_schema_mode", None),
    }
    try:
        write_sample_manifest = getattr(args, "write_sample_manifest", None)
        if write_sample_manifest:
            payload = export_catalog_sample_manifest(
                spec,
                output_path=str(write_sample_manifest),
                base_url=kwargs["base_url"],
                model=kwargs["model"],
                limit=kwargs["limit"],
                sample_size=kwargs["sample_size"],
                sample_seed=kwargs["sample_seed"],
                longbench_source_path=kwargs["longbench_source_path"],
                longbench_infer_protocol=kwargs["longbench_infer_protocol"],
                longbench_temperature=kwargs["longbench_temperature"],
                longbench_top_p=kwargs["longbench_top_p"],
                longbench_presence_penalty=kwargs["longbench_presence_penalty"],
                longbench_frequency_penalty=kwargs["longbench_frequency_penalty"],
                longbench_seed_requests=kwargs["longbench_seed_requests"],
                longbench_stop_suffixes=kwargs["longbench_stop_suffixes"],
            )
        else:
            payload = (
                dry_run_catalog_spec(spec, **kwargs)
                if args.dry_run
                else run_catalog_spec(spec, repo_root=root, **kwargs)
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print_json(payload)
    return 0


def handle_eval_run_free_response(args: argparse.Namespace, *, root: Any, **_: Any) -> int:
    from helicopter_eval.free_response import FreeResponseRunConfig, dry_run_summary, run_free_response

    defaults = load_rwkv_skills_catalog().inference_defaults
    run_config = FreeResponseRunConfig(
        base_url=str(args.base_url or defaults.base_url),
        model=str(args.model or defaults.model_name),
        benchmark=str(args.benchmark),
        dataset_name=str(args.dataset),
        dataset_config=str(args.dataset_config) if args.dataset_config is not None else None,
        question_field=str(args.question_field),
        answer_field=str(args.answer_field),
        limit=args.limit,
        sample_size=getattr(args, "sample_size", None),
        sample_seed=int(getattr(args, "sample_seed", 42)),
        split=str(args.split),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        timeout_s=float(args.timeout_s),
        answer_marker=str(args.answer_marker) if args.answer_marker else None,
        job_name=str(args.job_name),
        job_id=str(args.job_id) if args.job_id else None,
    )
    payload = dry_run_summary(run_config) if args.dry_run else run_free_response(run_config, repo_root=root)
    print_json(payload)
    return 0


def handle_eval_run_multiple_choice(args: argparse.Namespace, *, root: Any, **_: Any) -> int:
    from helicopter_eval.multiple_choice import MultipleChoiceRunConfig, dry_run_summary, run_multiple_choice

    defaults = load_rwkv_skills_catalog().inference_defaults
    run_config = MultipleChoiceRunConfig(
        base_url=str(args.base_url or defaults.base_url),
        model=str(args.model or defaults.model_name),
        benchmark=str(args.benchmark),
        dataset_name=str(args.dataset),
        dataset_config=str(args.dataset_config) if args.dataset_config is not None else None,
        question_field=str(args.question_field),
        choices_field=str(args.choices_field),
        answer_field=str(args.answer_field),
        limit=args.limit,
        sample_size=getattr(args, "sample_size", None),
        sample_seed=int(getattr(args, "sample_seed", 42)),
        split=str(args.split),
        choice_fields=tuple(args.choice_field or ()),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        timeout_s=float(args.timeout_s),
        choice_labels=str(args.choice_labels),
        job_name=str(args.job_name),
        job_id=str(args.job_id) if args.job_id else None,
    )
    payload = dry_run_summary(run_config) if args.dry_run else run_multiple_choice(run_config, repo_root=root)
    print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helicopter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="start vLLM for an RWKV model")
    add_common_options(infer)
    infer.add_argument("model", help="model alias from configs")
    infer.add_argument("--wkv-mode", choices=WKV_MODES)
    infer.add_argument("--emb-device", choices=EMB_DEVICES)
    infer.add_argument("--host")
    infer.add_argument("--port")
    infer.add_argument("--served-model-name")
    infer.add_argument("--tensor-parallel-size", type=int)
    infer.add_argument("--gpu-memory-utilization", type=float)
    infer.add_argument("--max-model-len", type=int)
    infer.add_argument("--max-num-seqs", type=int)
    infer.add_argument("--max-num-batched-tokens", type=int)
    infer.add_argument("--enable-auto-tool-choice", action="store_true", default=None)
    infer.set_defaults(plan_builder=build_infer_plan)

    takeoff = subparsers.add_parser("takeoff", help="start verl training for an RWKV model")
    add_common_options(takeoff)
    takeoff.add_argument("model", help="model alias from configs")
    takeoff.add_argument("algorithm", choices=("grpo",))
    takeoff.add_argument("--dataset", required=True, help="dataset alias from configs")
    takeoff.add_argument("--num-nodes", type=int)
    takeoff.add_argument("--num-devices", type=int)
    takeoff.add_argument("--wkv-mode", choices=WKV_MODES)
    takeoff.add_argument("--emb-device", choices=EMB_DEVICES)
    takeoff.add_argument("--override", action="append", help="extra Hydra override passed to verl")
    takeoff.set_defaults(plan_builder=build_takeoff_plan)

    eval_parser = subparsers.add_parser("eval", help="rwkv-skills-compatible evaluation controls")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_benchmarks = eval_subparsers.add_parser("benchmarks", help="list rwkv-skills benchmark catalog")
    eval_benchmarks.add_argument("benchmark", nargs="*", help="benchmark names or aliases; default: all")
    eval_benchmarks.add_argument("--field", action="append", help="filter by benchmark field")
    eval_benchmarks.add_argument("--json", action="store_true")
    eval_benchmarks.set_defaults(handler=handle_eval_benchmarks)

    eval_runners = eval_subparsers.add_parser("runners", help="list rwkv-skills runner catalog")
    eval_runners.add_argument("--group", action="append", help="filter by runner group")
    eval_runners.add_argument("--json", action="store_true")
    eval_runners.set_defaults(handler=handle_eval_runners)

    eval_plan = eval_subparsers.add_parser("plan", help="build the rwkv-skills benchmark/job matrix")
    eval_plan.add_argument("benchmark", nargs="*", default=("all",), help="benchmark names or aliases; default: all")
    eval_plan.add_argument("--field", action="append", help="filter by benchmark field")
    eval_plan.add_argument("--ready-only", action="store_true", help="only show runnable registry rows")
    eval_plan.add_argument("--json", action="store_true")
    eval_plan.set_defaults(handler=handle_eval_plan)

    eval_runnable = eval_subparsers.add_parser("runnable", help="show catalog benchmarks implemented in helicopter")
    eval_runnable.add_argument("benchmark", nargs="*", default=("all",), help="benchmark names or aliases; default: all")
    eval_runnable.add_argument("--field", action="append", help="filter by benchmark field")
    eval_runnable.add_argument("--json", action="store_true")
    eval_runnable.set_defaults(handler=handle_eval_runnable)

    eval_run = eval_subparsers.add_parser("run", help="run a implemented benchmark and write scoreboard DB rows")
    add_common_options(eval_run)
    eval_run.add_argument("benchmark", choices=("gsm8k",))
    eval_run.add_argument("--base-url", help="OpenAI-compatible vLLM base URL")
    eval_run.add_argument("--model", help="served model name")
    eval_run.add_argument("--limit", type=int)
    eval_run.add_argument("--split", default="test")
    eval_run.add_argument("--temperature", type=float, default=0.0)
    eval_run.add_argument("--top-p", type=float, default=1.0)
    eval_run.add_argument("--max-tokens", type=int, default=512)
    eval_run.add_argument("--timeout-s", type=float, default=600.0)
    eval_run.add_argument("--job-id", default="helicopter-gsm8k")
    eval_run.set_defaults(handler=handle_eval_run)

    eval_run_catalog = eval_subparsers.add_parser(
        "run-catalog",
        help="run an implemented rwkv-skills catalog benchmark",
    )
    add_common_options(eval_run_catalog)
    eval_run_catalog.add_argument("benchmark", help="rwkv-skills benchmark name")
    eval_run_catalog.add_argument("--base-url", help="OpenAI-compatible vLLM base URL")
    eval_run_catalog.add_argument("--model", help="served model name")
    eval_run_catalog.add_argument("--limit", type=int)
    eval_run_catalog.add_argument("--sample-size", type=int, help="randomly sample N free-response rows")
    eval_run_catalog.add_argument("--sample-seed", type=int, default=42, help="seed for --sample-size")
    eval_run_catalog.add_argument("--longbench-source-path", help="LongBench JSONL manifest or local source root")
    eval_run_catalog.add_argument("--write-sample-manifest", help="write the resolved LongBench sample manifest and exit")
    eval_run_catalog.add_argument(
        "--longbench-infer-protocol",
        choices=("chat", "completions"),
        help="LongBench infer request protocol",
    )
    eval_run_catalog.add_argument("--longbench-temperature", type=float, help="LongBench generation temperature")
    eval_run_catalog.add_argument("--longbench-top-p", type=float, help="LongBench generation top_p")
    eval_run_catalog.add_argument("--longbench-presence-penalty", type=float, help="LongBench presence penalty")
    eval_run_catalog.add_argument("--longbench-frequency-penalty", type=float, help="LongBench frequency penalty")
    eval_run_catalog.add_argument(
        "--longbench-seed-requests",
        action="store_true",
        help="send rwkv-skills-compatible per-sample seeds for LongBench",
    )
    eval_run_catalog.add_argument(
        "--longbench-stop-suffix",
        action="append",
        help="LongBench stop suffix; repeat to pass multiple suffixes",
    )
    eval_run_catalog.add_argument("--agentbench-controller-url", help="AgentBench controller API URL")
    eval_run_catalog.add_argument("--mcp-runtime-root", help="MCP-Bench official runtime root")
    eval_run_catalog.add_argument("--mcp-worker-script", help="MCP-Bench worker script path")
    eval_run_catalog.add_argument("--mcp-max-rounds", type=int, help="maximum MCP planning rounds")
    eval_run_catalog.add_argument("--judge-base-url", help="OpenAI-compatible judge base URL")
    eval_run_catalog.add_argument("--judge-model", help="judge model name")
    eval_run_catalog.add_argument("--judge-api-key", help="judge API key")
    eval_run_catalog.add_argument("--swebench-run-harness", action="store_true", help="run official SWE-Bench harness")
    eval_run_catalog.add_argument("--swebench-predictions-dir", help="directory for SWE-Bench predictions JSONL")
    eval_run_catalog.add_argument("--swebench-harness-run-id", help="SWE-Bench harness run id")
    eval_run_catalog.add_argument("--swebench-max-workers", type=int, help="SWE-Bench harness worker count")
    eval_run_catalog.add_argument("--swebench-cache-level", help="SWE-Bench harness cache level")
    eval_run_catalog.add_argument("--swebench-clean", action="store_true", help="pass --clean True to SWE-Bench harness")
    eval_run_catalog.add_argument("--swebench-timeout-s", type=float, help="SWE-Bench harness timeout in seconds")
    eval_run_catalog.add_argument("--swebench-max-context-chars", type=int, help="truncate retrieved SWE context")
    eval_run_catalog.add_argument("--tau-runtime-root", help="TAU/Tau2/Tau3 official runtime root")
    eval_run_catalog.add_argument("--tau-data-root", help="TAU/Tau2/Tau3 official data root")
    eval_run_catalog.add_argument("--tau-user-base-url", help="OpenAI-compatible user simulator base URL")
    eval_run_catalog.add_argument("--tau-user-model", help="user simulator model name")
    eval_run_catalog.add_argument("--tau-user-api-key", help="user simulator API key")
    eval_run_catalog.add_argument("--tau-max-steps", type=int, help="maximum TAU simulation steps")
    eval_run_catalog.add_argument("--tau-max-errors", type=int, help="maximum TAU tool/protocol errors")
    eval_run_catalog.add_argument("--tau-history-max-chars", type=int, help="TAU transcript budget")
    eval_run_catalog.add_argument("--tau-prompt-max-chars", type=int, help="TAU prompt budget")
    eval_run_catalog.add_argument("--candidate-router-mode", choices=("off", "parallel"), help="BFCL v3 candidate router mode")
    eval_run_catalog.add_argument("--candidate-router-chunk-tools", type=int, help="BFCL v3 candidate-router tools per shard")
    eval_run_catalog.add_argument("--candidate-router-batch-size", type=int, help="BFCL v3 candidate-router request fanout")
    eval_run_catalog.add_argument("--candidate-router-context-chars", type=int, help="BFCL v3 candidate-router history budget")
    eval_run_catalog.add_argument("--candidate-router-prompt-max-chars", type=int, help="BFCL v3 candidate-router prompt budget")
    eval_run_catalog.add_argument(
        "--candidate-router-candidate-max-tokens",
        type=int,
        help="BFCL v3 candidate-router per-shard generation cap",
    )
    eval_run_catalog.add_argument(
        "--candidate-router-aggregate-max-tokens",
        type=int,
        help="BFCL v3 candidate-router aggregation generation cap",
    )
    eval_run_catalog.add_argument("--candidate-router-max-candidates", type=int, help="BFCL v3 aggregation candidate cap")
    eval_run_catalog.add_argument(
        "--candidate-router-tool-schema-mode",
        choices=("minimal", "compact", "full"),
        help="BFCL v3 tool schema verbosity inside candidate-router prompts",
    )
    eval_run_catalog.set_defaults(handler=handle_eval_run_catalog)

    eval_run_free_response = eval_subparsers.add_parser(
        "run-free-response",
        help="run a HF free-response benchmark and write scoreboard DB rows",
    )
    add_common_options(eval_run_free_response)
    eval_run_free_response.add_argument("benchmark", help="scoreboard benchmark name")
    eval_run_free_response.add_argument("--dataset", required=True, help="HF dataset path")
    eval_run_free_response.add_argument("--dataset-config", help="HF dataset config/name")
    eval_run_free_response.add_argument("--question-field", default="question")
    eval_run_free_response.add_argument("--answer-field", default="answer")
    eval_run_free_response.add_argument("--answer-marker", default="####")
    eval_run_free_response.add_argument("--base-url", help="OpenAI-compatible vLLM base URL")
    eval_run_free_response.add_argument("--model", help="served model name")
    eval_run_free_response.add_argument("--limit", type=int)
    eval_run_free_response.add_argument("--sample-size", type=int, help="randomly sample N rows")
    eval_run_free_response.add_argument("--sample-seed", type=int, default=42, help="seed for --sample-size")
    eval_run_free_response.add_argument("--split", default="test")
    eval_run_free_response.add_argument("--temperature", type=float, default=0.0)
    eval_run_free_response.add_argument("--top-p", type=float, default=1.0)
    eval_run_free_response.add_argument("--max-tokens", type=int, default=512)
    eval_run_free_response.add_argument("--timeout-s", type=float, default=600.0)
    eval_run_free_response.add_argument("--job-name", default="free_response_judge")
    eval_run_free_response.add_argument("--job-id")
    eval_run_free_response.set_defaults(handler=handle_eval_run_free_response)

    eval_run_multiple_choice = eval_subparsers.add_parser(
        "run-multiple-choice",
        help="run a HF multiple-choice benchmark and write scoreboard DB rows",
    )
    add_common_options(eval_run_multiple_choice)
    eval_run_multiple_choice.add_argument("benchmark", help="scoreboard benchmark name")
    eval_run_multiple_choice.add_argument("--dataset", required=True, help="HF dataset path")
    eval_run_multiple_choice.add_argument("--dataset-config", help="HF dataset config/name")
    eval_run_multiple_choice.add_argument("--question-field", default="question")
    eval_run_multiple_choice.add_argument("--choices-field", default="choices")
    eval_run_multiple_choice.add_argument("--choice-field", action="append", help="repeat for A/B/C/D style option columns")
    eval_run_multiple_choice.add_argument("--answer-field", default="answer")
    eval_run_multiple_choice.add_argument("--choice-labels", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    eval_run_multiple_choice.add_argument("--base-url", help="OpenAI-compatible vLLM base URL")
    eval_run_multiple_choice.add_argument("--model", help="served model name")
    eval_run_multiple_choice.add_argument("--limit", type=int)
    eval_run_multiple_choice.add_argument("--sample-size", type=int, help="randomly sample N rows")
    eval_run_multiple_choice.add_argument("--sample-seed", type=int, default=42, help="seed for --sample-size")
    eval_run_multiple_choice.add_argument("--split", default="test")
    eval_run_multiple_choice.add_argument("--temperature", type=float, default=0.0)
    eval_run_multiple_choice.add_argument("--top-p", type=float, default=1.0)
    eval_run_multiple_choice.add_argument("--max-tokens", type=int, default=32)
    eval_run_multiple_choice.add_argument("--timeout-s", type=float, default=600.0)
    eval_run_multiple_choice.add_argument("--job-name", default="multi_choice_plain")
    eval_run_multiple_choice.add_argument("--job-id")
    eval_run_multiple_choice.set_defaults(handler=handle_eval_run_multiple_choice)

    eval_infer = eval_subparsers.add_parser("infer", help="start the default local 0.4B vLLM-RWKV service")
    add_common_options(eval_infer)
    eval_infer.add_argument("--model-path", help="RWKV checkpoint path")
    eval_infer.add_argument("--served-model-name", help="served OpenAI-compatible model name")
    eval_infer.add_argument("--host")
    eval_infer.add_argument("--port", type=int)
    eval_infer.add_argument("--wkv-mode", choices=WKV_MODES)
    eval_infer.add_argument("--emb-device", choices=EMB_DEVICES)
    eval_infer.add_argument("--tensor-parallel-size", type=int)
    eval_infer.add_argument("--gpu-memory-utilization", type=float)
    eval_infer.add_argument("--max-model-len", type=int)
    eval_infer.add_argument("--max-num-seqs", type=int)
    eval_infer.add_argument("--max-num-batched-tokens", type=int)
    eval_infer.set_defaults(plan_builder=build_eval_infer_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = find_root()
    env, _ = load_env(root, getattr(args, "env_file", DEFAULT_ENV_FILE))
    config, _ = load_config(root, getattr(args, "config", None))
    prepend_venv_path(env, root, config)

    if hasattr(args, "handler"):
        return int(args.handler(args, root=root, env=env, config=config))

    plan = args.plan_builder(args, root=root, env=env, config=config)
    return run_command(
        plan.command,
        cwd=plan.cwd,
        env=plan.env,
        shown_env=plan.shown_env,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
