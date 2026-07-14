from __future__ import annotations

import argparse

from .commands import (
    EMB_DEVICES,
    WKV_MODES,
    ANY2RWKV_ACTIONS,
    ANY2RWKV_PRECISIONS,
    build_any2rwkv_plan,
    build_infer_plan,
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

    any2rwkv = subparsers.add_parser("any2rwkv", help="fetch, verify, convert, distill, quantize, or evaluate Qwen3.5 text backbones")
    add_common_options(any2rwkv)
    any2rwkv.add_argument("action", choices=ANY2RWKV_ACTIONS)
    any2rwkv.add_argument("--source", required=True, help="read-only HF checkpoint directory, or frozen source manifest for fetch/verify")
    any2rwkv.add_argument("--output", required=True, help="independent run output directory, or frozen source destination for fetch/verify")
    any2rwkv.add_argument("--precision", choices=ANY2RWKV_PRECISIONS)
    any2rwkv.add_argument("--rwkv-hf-sha")
    any2rwkv.add_argument("--rwkv-lm-sha")
    any2rwkv.add_argument("--contract", help="frozen contract.lock.json")
    any2rwkv.add_argument("--calibration-manifest")
    any2rwkv.add_argument("--dataset-manifest")
    any2rwkv.add_argument("--training-config")
    any2rwkv.add_argument("--resume")
    any2rwkv.add_argument("--kernel-oracle", help="JSON from the managed native RWKV7 kernel validation")
    any2rwkv.add_argument("--teacher")
    any2rwkv.add_argument("--evaluation-manifest")
    any2rwkv.add_argument("--p0-evidence")
    any2rwkv.add_argument("--migration-baselines")
    any2rwkv.add_argument("--ruler-scores")
    any2rwkv.add_argument("--downstream-scores")
    any2rwkv.add_argument("--scale-gate", help="accepted real-proxy run directory required before 397B fetch")
    any2rwkv.add_argument("--run-id")
    any2rwkv.add_argument("--allow-proxy-layers", action="store_true", help="permit a non-60-layer real proxy; never marks it final")
    any2rwkv.set_defaults(plan_builder=build_any2rwkv_plan)

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
