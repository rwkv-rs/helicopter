from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .commands import (
    WKV_MODES,
    build_infer_plan,
    build_takeoff_plan,
    prepend_venv_path,
)
from .config import load_config
from .env import DEFAULT_ENV_FILE, load_env
from .paths import find_root
from .runner import run_command


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="TOML config path")
    parser.add_argument(
        "--env-file", default=DEFAULT_ENV_FILE, help="dotenv file loaded before config"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print without execution"
    )


def _add_infer(subparsers: Any) -> None:
    infer = subparsers.add_parser("infer", help="start vLLM for an RWKV model")
    add_common_options(infer)
    infer.add_argument("model", help="model alias from config")
    infer.add_argument("--wkv-mode", choices=WKV_MODES)
    infer.add_argument("--host")
    infer.add_argument("--port")
    infer.add_argument("--served-model-name")
    infer.add_argument("--tensor-parallel-size", type=int)
    infer.add_argument("--gpu-memory-utilization", type=float)
    infer.add_argument("--max-num-seqs", type=int)
    infer.add_argument("--max-num-batched-tokens", type=int)
    infer.add_argument("--enable-auto-tool-choice", action="store_true", default=None)
    infer.add_argument("--vllm-env", action="append")
    infer.set_defaults(plan_builder=build_infer_plan)


def _add_takeoff(subparsers: Any) -> None:
    takeoff = subparsers.add_parser(
        "takeoff", help="start verl training for an RWKV model"
    )
    add_common_options(takeoff)
    takeoff.add_argument("model", help="model alias from config")
    takeoff.add_argument("algorithm", choices=("grpo",))
    takeoff.add_argument("--dataset", required=True, help="dataset alias from config")
    takeoff.add_argument("--num-nodes", type=int)
    takeoff.add_argument("--num-devices", type=int)
    takeoff.add_argument("--wkv-mode", choices=WKV_MODES)
    takeoff.add_argument("--override", action="append")
    takeoff.set_defaults(plan_builder=build_takeoff_plan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helicopter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_infer(subparsers)
    _add_takeoff(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
