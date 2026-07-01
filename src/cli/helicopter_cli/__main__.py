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
