from __future__ import annotations

import argparse

from .commands import (
    EMB_DEVICES,
    WKV_MODES,
    build_infer_plan,
    build_takeoff_plan,
    prepend_venv_path,
)
from .config import load_config
from .env import DEFAULT_ENV_FILE, load_env
from .paths import find_root
from .runner import run_command


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file", default=DEFAULT_ENV_FILE, help="dotenv file to load first"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the command without executing it"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helicopter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="start vLLM for an RWKV model")
    add_runtime_options(infer)
    infer.add_argument(
        "--config",
        help="serving TOML; defaults to configs/example.toml",
    )
    infer.add_argument("model", help="model alias from configs")
    infer.add_argument("--wkv-mode", choices=WKV_MODES)
    infer.add_argument("--emb-device", choices=EMB_DEVICES)
    infer.add_argument(
        "--allow-fp16-accumulation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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

    takeoff = subparsers.add_parser(
        "takeoff", help="launch a Verl-owned MaxRL config"
    )
    add_runtime_options(takeoff)
    takeoff.add_argument("--config", required=True, help="Verl-owned MaxRL TOML")
    takeoff.add_argument(
        "--override", action="append", help="operational override validated by Verl"
    )
    takeoff.set_defaults(plan_builder=build_takeoff_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = find_root()
    env, _ = load_env(root, args.env_file)
    prepend_venv_path(env, root)

    if args.command == "takeoff":
        plan = args.plan_builder(args, root=root, env=env)
    else:
        config, _ = load_config(root, args.config)
        plan = args.plan_builder(args, root=root, env=env, config=config)
    return run_command(
        plan.command,
        cwd=plan.cwd,
        env=plan.env,
        shown_env=plan.shown_env,
        dry_run=args.dry_run and args.command != "takeoff",
    )


if __name__ == "__main__":
    raise SystemExit(main())
