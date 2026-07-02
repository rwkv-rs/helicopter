from __future__ import annotations

import argparse

from .commands import (
    EMB_DEVICES,
    LIGHTEVAL_BACKENDS,
    WKV_MODES,
    build_infer_plan,
    build_lighteval_export_plan,
    build_lighteval_plan,
    build_lighteval_suite_plan,
    build_lighteval_tasks_plan,
    build_suite_adapter_plan,
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

    suite = eval_subparsers.add_parser("lighteval-suite", help="run a configured LightEval task suite")
    add_common_options(suite)
    suite.add_argument("model", help="model alias from configs")
    suite.add_argument("suite", help="suite alias from [lighteval_suites], e.g. rwkv_skills")
    suite.add_argument("--mapped-only", action="store_true", help="run only benchmarks with direct LightEval tasks")
    suite.add_argument("--field", action="append", help="limit to a suite field; repeatable")
    suite.add_argument("--benchmark", action="append", help="limit to a benchmark name; repeatable")
    suite.add_argument("--model-args", help="raw LightEval model args string or YAML config path")
    suite.add_argument("--lighteval-model-name", help="model name passed to LightEval/LiteLLM")
    suite.add_argument("--base-url", help="OpenAI-compatible endpoint base URL")
    suite.add_argument("--provider", help="LiteLLM provider prefix; defaults to openai")
    suite.add_argument("--api-key", help="API key passed through OPENAI_API_KEY")
    suite.add_argument("--concurrent-requests", type=int)
    suite.add_argument("--max-model-length", type=int)
    suite.add_argument("--max-samples", type=int)
    suite.add_argument("--output-dir")
    suite.add_argument("--dataset-loading-processes", type=int)
    suite.add_argument("--num-fewshot-seeds", type=int)
    suite.add_argument("--custom-tasks", help="custom LightEval task Python file")
    suite.add_argument("--load-tasks-multilingual", action="store_true", default=None)
    suite.add_argument("--save-details", dest="save_details", action="store_true", default=None)
    suite.add_argument("--no-save-details", dest="save_details", action="store_false")
    suite.add_argument("--push-to-hub", action="store_true", default=None)
    suite.add_argument("--public-run", action="store_true", default=None)
    suite.add_argument("--results-org")
    suite.add_argument("--job-id", type=int)
    suite.add_argument("--extra", action="append", help="extra argument passed to LightEval")
    suite.set_defaults(plan_builder=build_lighteval_suite_plan)

    suite_adapter = eval_subparsers.add_parser("suite-adapter", help="run adapter-backed suite benchmarks")
    add_common_options(suite_adapter)
    suite_adapter.add_argument("model", help="model alias from configs")
    suite_adapter.add_argument("suite", help="suite alias from [lighteval_suites], e.g. rwkv_skills")
    suite_adapter.add_argument("--field", action="append", help="limit to a suite field; repeatable")
    suite_adapter.add_argument("--benchmark", action="append", help="limit to an adapter benchmark name; repeatable")
    suite_adapter.add_argument("--lighteval-model-name", help="model name passed to the OpenAI-compatible endpoint")
    suite_adapter.add_argument("--adapter-model-name", help="adapter output model_name_or_path; defaults to served model name")
    suite_adapter.add_argument("--base-url", help="OpenAI-compatible endpoint base URL")
    suite_adapter.add_argument("--api-key", help="API key passed through OPENAI_API_KEY")
    suite_adapter.add_argument("--output-dir")
    suite_adapter.add_argument("--run-id")
    suite_adapter.add_argument("--split", default=None)
    suite_adapter.add_argument("--max-samples", type=int)
    suite_adapter.add_argument("--sample-seed", type=int)
    suite_adapter.add_argument("--max-tokens", type=int)
    suite_adapter.add_argument("--temperature", type=float)
    suite_adapter.add_argument("--timeout-s", type=float)
    suite_adapter.add_argument("--swebench-max-context-chars", type=int)
    suite_adapter.add_argument("--swebench-prompt-profile", choices=("normal", "naive"))
    suite_adapter.add_argument("--swebench-run-harness", action="store_true", default=None)
    suite_adapter.add_argument("--swebench-harness-workers", type=int)
    suite_adapter.add_argument("--swebench-harness-timeout-s", type=float)
    suite_adapter.add_argument("--tau-bench-root")
    suite_adapter.add_argument("--tau-data-root")
    suite_adapter.add_argument("--tau-max-steps", type=int)
    suite_adapter.add_argument("--tau-max-errors", type=int)
    suite_adapter.add_argument("--tau-history-max-chars", type=int)
    suite_adapter.add_argument("--tau-prompt-max-chars", type=int)
    suite_adapter.add_argument("--tau-user-model")
    suite_adapter.add_argument("--tau-user-base-url")
    suite_adapter.add_argument("--tau-user-api-key")
    suite_adapter.add_argument("--tau-user-temperature", type=float)
    suite_adapter.add_argument("--tau-judge-model")
    suite_adapter.add_argument("--tau-judge-base-url")
    suite_adapter.add_argument("--tau-judge-api-key")
    suite_adapter.add_argument("--mcp-bench-root")
    suite_adapter.add_argument("--mcp-max-rounds", type=int)
    suite_adapter.add_argument("--mcp-max-errors", type=int)
    suite_adapter.add_argument("--mcp-history-max-chars", type=int)
    suite_adapter.add_argument("--mcp-tool-schema-max-chars", type=int)
    suite_adapter.add_argument("--mcp-judge-model")
    suite_adapter.add_argument("--mcp-judge-base-url")
    suite_adapter.add_argument("--mcp-judge-api-key")
    suite_adapter.set_defaults(plan_builder=build_suite_adapter_plan)

    lighteval_tasks = eval_subparsers.add_parser("lighteval-tasks", help="list or inspect LightEval tasks")
    add_common_options(lighteval_tasks)
    lighteval_tasks.add_argument("task_action", choices=("list", "dump", "inspect"))
    lighteval_tasks.add_argument("tasks", nargs="?", help="task id for inspect")
    lighteval_tasks.add_argument("--custom-tasks", help="custom LightEval task Python file")
    lighteval_tasks.add_argument("--load-tasks-multilingual", action="store_true", default=None)
    lighteval_tasks.add_argument("--num-samples", type=int)
    lighteval_tasks.add_argument("--show-config", action="store_true", default=None)
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
