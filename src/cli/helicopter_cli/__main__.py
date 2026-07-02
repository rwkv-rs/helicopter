from __future__ import annotations

import argparse

from .commands import (
    EMB_DEVICES,
    LIGHTEVAL_BACKENDS,
    WKV_MODES,
    build_infer_plan,
    build_lighteval_export_plan,
    build_lighteval_plan,
    build_lighteval_tasks_plan,
    build_takeoff_plan,
    prepend_venv_path,
)
from .config import load_config
from .env import DEFAULT_ENV_FILE, load_env
from .paths import find_root
from .runner import run_command


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="TOML config path; defaults to the newest configs/local/*.toml")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="dotenv file to load first")
    parser.add_argument("--dry-run", action="store_true", help="print the command without executing it")


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
    infer.add_argument("--vllm-env", action="append", help="explicit VLLM_* environment override, e.g. VLLM_WSL2_ENABLE_PIN_MEMORY=1")
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

    eval_parser = subparsers.add_parser("eval", help="run model evaluations")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    lighteval = eval_subparsers.add_parser("lighteval", help="run Hugging Face LightEval")
    add_common_options(lighteval)
    lighteval.add_argument("model", help="model alias from configs")
    lighteval.add_argument("tasks", help="LightEval task string, e.g. 'gsm8k' or 'gsm8k|0'")
    lighteval.add_argument("--backend", choices=LIGHTEVAL_BACKENDS, default="endpoint-litellm")
    lighteval.add_argument("--model-args", help="raw LightEval model args string or YAML config path")
    lighteval.add_argument("--lighteval-model-name", help="model name passed to LightEval/LiteLLM")
    lighteval.add_argument("--base-url", help="OpenAI-compatible endpoint base URL")
    lighteval.add_argument("--provider", help="LiteLLM provider prefix; defaults to openai")
    lighteval.add_argument("--api-key", help="API key passed through OPENAI_API_KEY")
    lighteval.add_argument("--concurrent-requests", type=int)
    lighteval.add_argument("--max-model-length", type=int)
    lighteval.add_argument("--max-samples", type=int)
    lighteval.add_argument("--output-dir")
    lighteval.add_argument("--dataset-loading-processes", type=int)
    lighteval.add_argument("--num-fewshot-seeds", type=int)
    lighteval.add_argument("--custom-tasks", help="custom LightEval task Python file")
    lighteval.add_argument("--load-tasks-multilingual", action="store_true", default=None)
    lighteval.add_argument("--save-details", dest="save_details", action="store_true", default=None)
    lighteval.add_argument("--no-save-details", dest="save_details", action="store_false")
    lighteval.add_argument("--push-to-hub", action="store_true", default=None)
    lighteval.add_argument("--public-run", action="store_true", default=None)
    lighteval.add_argument("--results-org")
    lighteval.add_argument("--job-id", type=int)
    lighteval.add_argument("--extra", action="append", help="extra argument passed to LightEval")
    lighteval.set_defaults(plan_builder=build_lighteval_plan)

    lighteval_tasks = eval_subparsers.add_parser("lighteval-tasks", help="list or inspect LightEval tasks")
    add_common_options(lighteval_tasks)
    lighteval_tasks.add_argument("task_action", choices=("list", "dump", "inspect", "export", "coverage"))
    lighteval_tasks.add_argument("tasks", nargs="?", help="task id for inspect")
    lighteval_tasks.add_argument("--custom-tasks", help="custom LightEval task Python file")
    lighteval_tasks.add_argument("--load-tasks-multilingual", action="store_true", default=None)
    lighteval_tasks.add_argument("--num-samples", type=int)
    lighteval_tasks.add_argument("--show-config", action="store_true", default=None)
    lighteval_tasks.add_argument("--output", help="output file for export; defaults to stdout")
    lighteval_tasks.add_argument("--format", choices=("text", "jsonl", "summary", "tasks"), default="text")
    lighteval_tasks.add_argument("--contains", action="append", help="case-insensitive task-name filter for export")
    lighteval_tasks.add_argument("--limit", type=int, help="maximum number of exported tasks")
    lighteval_tasks.add_argument("--include-supersets", action="store_true", default=None)
    lighteval_tasks.add_argument("--source", help="benchmark list for coverage; supports text/json/jsonl or rwkv-skills registry Python")
    lighteval_tasks.add_argument(
        "--source-format",
        choices=("auto", "text", "json", "jsonl", "rwkv-skills-registry"),
        default="auto",
    )
    lighteval_tasks.add_argument("--candidate-limit", type=int, default=5)
    lighteval_tasks.set_defaults(plan_builder=build_lighteval_tasks_plan)

    lighteval_export = eval_subparsers.add_parser("lighteval-export", help="export LightEval details parquet to per-sample records")
    add_common_options(lighteval_export)
    lighteval_export.add_argument("details", nargs="+", help="LightEval details parquet file or directory")
    lighteval_export.add_argument("--output", help="output file; defaults to stdout")
    lighteval_export.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    lighteval_export.set_defaults(plan_builder=build_lighteval_export_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = find_root()
    env, _ = load_env(root, args.env_file)
    config, _ = load_config(root, args.config)
    prepend_venv_path(env, root, config)

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
