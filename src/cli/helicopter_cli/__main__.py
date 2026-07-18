from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .agent_harness import DEFAULT_AGENT_BENCHMARK_SOURCE, run_agent_harness
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
from .eval_batch import run_batch
from .eval_run import DEFAULT_SERVER_TIMEOUT_S, run_eval
from .function_calling import FC_TASKS, run_function_calling_eval
from .paths import find_root
from .performance import (
    base_url_from_lighteval_command,
    derive_metrics_url,
    output_dir_from_command,
    run_completions_performance,
    run_lighteval_with_performance,
)
from .runner import run_command


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="TOML config path; defaults to the newest configs/local/*.toml")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="dotenv file to load first")
    parser.add_argument("--dry-run", action="store_true", help="print the command without executing it")


def add_lighteval_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=LIGHTEVAL_BACKENDS, default="endpoint-litellm")
    parser.add_argument("--model-args", help="raw LightEval model args string or YAML config path")
    parser.add_argument("--lighteval-model-name", help="model name passed to LightEval/LiteLLM")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint base URL")
    parser.add_argument("--provider", help="LiteLLM provider prefix; defaults to openai")
    parser.add_argument("--api-key", help="API key passed through OPENAI_API_KEY")
    parser.add_argument("--concurrent-requests", type=int)
    parser.add_argument("--max-model-length", type=int)
    parser.add_argument("--max-new-tokens", type=int, help="cap generated tokens through LightEval generation_parameters")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset-loading-processes", type=int)
    parser.add_argument("--num-fewshot-seeds", type=int)
    parser.add_argument("--custom-tasks", help="custom LightEval task Python file")
    parser.add_argument("--load-tasks-multilingual", action="store_true", default=None)
    parser.add_argument("--save-details", dest="save_details", action="store_true", default=None)
    parser.add_argument("--no-save-details", dest="save_details", action="store_false")
    parser.add_argument("--push-to-hub", action="store_true", default=None)
    parser.add_argument("--public-run", action="store_true", default=None)
    parser.add_argument("--results-org")
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--extra", action="append", help="extra argument passed to LightEval")
    parser.add_argument("--performance-output", help="write run performance metrics JSON here")
    parser.add_argument("--metrics-url", help="Prometheus metrics URL for token throughput; defaults to <base-url without /v1>/metrics")
    parser.add_argument("--scoreboard-task-id", help="merge performance metrics into this scoreboard task score")


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

    eval_run = eval_subparsers.add_parser(
        "run",
        help="one-shot evaluation: start vLLM, run LightEval, stop vLLM",
    )
    add_common_options(eval_run)
    eval_run.add_argument("model", help="model alias from configs")
    eval_run.add_argument(
        "tasks",
        nargs="?",
        help="LightEval task string; defaults to [lighteval].tasks from the config",
    )
    add_lighteval_run_options(eval_run)
    eval_run.add_argument("--wkv-mode", choices=WKV_MODES)
    eval_run.add_argument("--emb-device", choices=EMB_DEVICES)
    eval_run.add_argument("--tensor-parallel-size", type=int)
    eval_run.add_argument("--gpu-memory-utilization", type=float)
    eval_run.add_argument("--max-num-seqs", type=int)
    eval_run.add_argument("--max-num-batched-tokens", type=int)
    eval_run.add_argument("--enable-auto-tool-choice", action="store_true", default=None)
    eval_run.add_argument(
        "--vllm-env",
        action="append",
        help="explicit VLLM_* environment override for the managed server",
    )
    eval_run.add_argument(
        "--no-server",
        action="store_true",
        help="never start vLLM; assume the endpoint is already serving",
    )
    eval_run.add_argument(
        "--keep-server",
        action="store_true",
        help="leave the managed vLLM server running after the evaluation",
    )
    eval_run.add_argument(
        "--server-timeout",
        type=float,
        default=DEFAULT_SERVER_TIMEOUT_S,
        help="seconds to wait for the managed vLLM server to become healthy",
    )
    eval_run.add_argument(
        "--scoreboard",
        action="store_true",
        help="record per-task scores into the scoreboard database after the run",
    )
    eval_run.set_defaults(plan_builder=None)

    eval_perf = eval_subparsers.add_parser(
        "perf",
        help="raw OpenAI completions performance probe",
    )
    add_common_options(eval_perf)
    eval_perf.add_argument("model", help="model alias from configs")
    eval_perf.add_argument("--base-url", help="OpenAI-compatible endpoint base URL")
    eval_perf.add_argument("--api-key", help="API key passed to the endpoint")
    eval_perf.add_argument("--served-model-name", help="model name sent in the OpenAI request")
    eval_perf.add_argument("--profile", choices=("prefill", "decode"), default="decode")
    eval_perf.add_argument("--prompt-tokens", type=int, help="synthetic prompt token target")
    eval_perf.add_argument("--output-tokens", type=int, help="max generated tokens per request")
    eval_perf.add_argument("--requests", type=int, help="number of requests to issue")
    eval_perf.add_argument("--concurrency", type=int, help="maximum in-flight requests")
    eval_perf.add_argument("--request-rate", type=float, help="optional request launch rate in requests/second")
    eval_perf.add_argument("--timeout", type=float, help="per-request timeout in seconds")
    eval_perf.add_argument("--ignore-eos", action="store_true", default=None, help="ask vLLM to continue until max_tokens")
    eval_perf.add_argument("--output", help="write performance report JSON here")
    eval_perf.set_defaults(plan_builder=None)

    eval_batch = eval_subparsers.add_parser(
        "batch",
        help="batch scheduler: sweep models x benchmarks across idle GPUs",
    )
    add_common_options(eval_batch)
    add_lighteval_run_options(eval_batch)
    eval_batch.add_argument("--models", action="append", help="model alias (repeat or comma-separate); defaults to [eval.batch].models")
    eval_batch.add_argument("--tasks", action="append", help="LightEval task string (repeat or comma-separate); defaults to [eval.batch].tasks")
    eval_batch.add_argument("--tasks-from-db", action="store_true", help="load LightEval tasks from the scoreboard benchmark_catalog table")
    eval_batch.add_argument("--benchmark-scope", help="benchmark_catalog scope used with --tasks-from-db")
    eval_batch.add_argument("--benchmark-fields", action="append", help="benchmark_catalog field filter used with --tasks-from-db")
    eval_batch.add_argument("--benchmark-limit", type=int, help="maximum number of DB benchmark tasks to load")
    eval_batch.add_argument("--fc-tasks", action="append", help="native function-calling task id (repeat or comma-separate); defaults to [eval.batch].fc_tasks")
    eval_batch.add_argument("--gpus", help="comma-separated GPU indexes; defaults to idle-GPU auto-detection")
    eval_batch.add_argument("--gpu-idle-max-mem", type=float, help="MiB of used memory below which a GPU counts as idle")
    eval_batch.add_argument("--parallel", type=int, default=1, help="number of units to run concurrently (capped by available GPU slots)")
    eval_batch.add_argument("--max-retries", type=int, default=0, help="retries per failed unit")
    eval_batch.add_argument("--port-base", type=int, help="first port for managed vLLM servers; slot i uses port-base + i")
    eval_batch.add_argument("--batch-output", help="write the batch run report JSON here; defaults to results/eval_batch for real runs")
    eval_batch.add_argument("--rerun", action="store_true", help="run benchmarks even when the scoreboard already has a score")
    eval_batch.add_argument("--wkv-mode", choices=WKV_MODES)
    eval_batch.add_argument("--emb-device", choices=EMB_DEVICES)
    eval_batch.add_argument("--tensor-parallel-size", type=int)
    eval_batch.add_argument("--gpu-memory-utilization", type=float)
    eval_batch.add_argument("--max-num-seqs", type=int)
    eval_batch.add_argument("--max-num-batched-tokens", type=int)
    eval_batch.add_argument("--enable-auto-tool-choice", action="store_true", default=None)
    eval_batch.add_argument("--vllm-env", action="append", help="explicit VLLM_* environment override for managed servers")
    eval_batch.add_argument("--no-server", action="store_true", help="never start vLLM; assume the endpoint is already serving")
    eval_batch.add_argument(
        "--server-timeout",
        type=float,
        default=DEFAULT_SERVER_TIMEOUT_S,
        help="seconds to wait for each managed vLLM server to become healthy",
    )
    eval_batch.add_argument(
        "--scoreboard",
        action="store_true",
        help="record scores into the scoreboard database and skip already-scored benchmarks",
    )
    eval_batch.set_defaults(plan_builder=None)

    fc = eval_subparsers.add_parser(
        "function-calling",
        aliases=("fc",),
        help="run native OpenAI tool_calls function-calling benchmarks",
    )
    add_common_options(fc)
    fc.add_argument("model", help="model alias from configs")
    fc.add_argument(
        "tasks",
        nargs="?",
        default="all",
        help=f"comma-separated FC task ids or all; known: {', '.join(FC_TASKS)}",
    )
    fc.add_argument("--base-url", help="OpenAI-compatible endpoint base URL")
    fc.add_argument("--max-samples", type=int)
    fc.add_argument("--output-dir")
    fc.add_argument(
        "--no-server",
        action="store_true",
        help="never start vLLM; assume the endpoint is already serving",
    )
    fc.add_argument(
        "--keep-server",
        action="store_true",
        help="leave the managed vLLM server running after the evaluation",
    )
    fc.add_argument(
        "--scoreboard",
        action="store_true",
        help="record native tool_calls scores into the scoreboard database after the run",
    )
    fc.set_defaults(plan_builder=None)

    lighteval = eval_subparsers.add_parser("lighteval", help="run Hugging Face LightEval")
    add_common_options(lighteval)
    lighteval.add_argument("model", help="model alias from configs")
    lighteval.add_argument("tasks", help="LightEval task string, e.g. 'gsm8k' or 'gsm8k|0'")
    add_lighteval_run_options(lighteval)
    lighteval.set_defaults(plan_builder=build_lighteval_plan)

    lighteval_tasks = eval_subparsers.add_parser("lighteval-tasks", help="list or inspect LightEval tasks")
    add_common_options(lighteval_tasks)
    lighteval_tasks.add_argument("task_action", choices=("list", "dump", "inspect", "export", "coverage", "judges"))
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

    agent_harness = eval_subparsers.add_parser(
        "agent-harness",
        aliases=("agent",),
        help="inspect and prepare external agent benchmark harnesses",
    )
    add_common_options(agent_harness)
    agent_harness.add_argument("agent_action", choices=("list", "preflight", "plan", "convert", "run"))
    agent_harness.add_argument("benchmark", nargs="?", help="benchmark id for list/preflight filtering or plan")
    agent_harness.add_argument("--source", default=DEFAULT_AGENT_BENCHMARK_SOURCE)
    agent_harness.add_argument("--pipeline")
    agent_harness.add_argument("--format", choices=("text", "jsonl", "summary"), default="text")
    agent_harness.add_argument("--model", help="model alias used when generating a harness plan")
    agent_harness.add_argument("--base-url", help="OpenAI-compatible endpoint base URL for adapter planning")
    agent_harness.add_argument("--output-dir", help="output directory for generated prediction or trace artifacts")
    agent_harness.add_argument("--n-concurrent", type=int, help="official harness worker count where supported")
    agent_harness.add_argument("--max-samples", type=int, help="sample cap for implemented local proxy runs")
    agent_harness.add_argument("--no-server", action="store_true", help="reuse an existing OpenAI-compatible endpoint")
    agent_harness.add_argument("--keep-server", action="store_true", help="leave a managed local proxy server running")
    agent_harness.add_argument(
        "--server-timeout",
        type=float,
        default=DEFAULT_SERVER_TIMEOUT_S,
        help="seconds to wait for a managed local proxy server to become healthy",
    )
    agent_harness.add_argument(
        "--allow-proxy",
        action="store_true",
        help="allow non-official local proxy runs such as BrowseComp answer-only scoring",
    )
    agent_harness.add_argument("--run-id", help="official harness run id")
    agent_harness.add_argument("--input", help="RWKV/Helicopter agent output JSON or JSONL for convert")
    agent_harness.add_argument("--output", help="converted official sandbox artifact path")
    agent_harness.add_argument(
        "--target",
        choices=("auto", "intermediate", "swebench-predictions"),
        default="auto",
        help="conversion target for agent-harness convert",
    )
    agent_harness.add_argument("--allow-empty-patch", action="store_true", help="write empty SWE-bench model_patch rows")
    agent_harness.add_argument("--allow-invalid", action="store_true", help="write valid converted rows even if some rows fail")
    agent_harness.add_argument("--strict", action="store_true", help="preflight exits nonzero if required tools are missing")
    agent_harness.set_defaults(plan_builder=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = find_root()
    env, _ = load_env(root, args.env_file)
    config, _ = load_config(root, args.config)
    prepend_venv_path(env, root, config)

    if getattr(args, "eval_command", None) == "run":
        return run_eval(args, root=root, env=env, config=config)
    if getattr(args, "eval_command", None) == "perf":
        return run_completions_performance(args, root=root, env=env, config=config)
    if getattr(args, "eval_command", None) == "batch":
        return run_batch(args, root=root, env=env, config=config)
    if getattr(args, "eval_command", None) in {"function-calling", "fc"}:
        return run_function_calling_eval(args, root=root, env=env, config=config)
    if getattr(args, "eval_command", None) in {"agent-harness", "agent"}:
        return run_agent_harness(args, root=root, env=env, config=config)

    plan = args.plan_builder(args, root=root, env=env, config=config)
    if getattr(args, "eval_command", None) == "lighteval" and not args.dry_run:
        output_dir = output_dir_from_command(plan.command) or (root / "results/lighteval")
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        performance_output = (
            root / args.performance_output
            if getattr(args, "performance_output", None) and not str(args.performance_output).startswith("/")
            else args.performance_output
        )
        if performance_output is None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            performance_output = output_dir / "performance" / f"performance_{stamp}.json"
        metrics_url = getattr(args, "metrics_url", None) or derive_metrics_url(
            base_url_from_lighteval_command(plan.command)
        )
        return run_lighteval_with_performance(
            plan.command,
            cwd=plan.cwd,
            env=plan.env,
            root=root,
            performance_output=Path(performance_output),
            metrics_url=metrics_url,
            scoreboard_task_id=getattr(args, "scoreboard_task_id", None),
        )
    return run_command(
        plan.command,
        cwd=plan.cwd,
        env=plan.env,
        shown_env=plan.shown_env,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
