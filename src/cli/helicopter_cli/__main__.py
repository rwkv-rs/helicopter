from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat

from .commands import (
    EMB_DEVICES,
    WKV_MODES,
    build_infer_plan,
    build_takeoff_plan,
    prepend_venv_path,
)
from .config import load_config
from .env import DEFAULT_ENV_FILE, find_env_path, load_env
from .paths import find_root
from .runner import run_command


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", help="TOML config path; defaults to the newest configs/local/*.toml"
    )
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

    takeoff = subparsers.add_parser(
        "takeoff", help="start verl training for an RWKV model"
    )
    add_common_options(takeoff)
    takeoff.add_argument("model", help="model alias from configs")
    takeoff.add_argument("algorithm", choices=("grpo",))
    takeoff.add_argument("--dataset", required=True, help="dataset alias from configs")
    takeoff.add_argument("--num-nodes", type=int)
    takeoff.add_argument("--num-devices", type=int)
    takeoff.add_argument("--wkv-mode", choices=WKV_MODES)
    takeoff.add_argument("--emb-device", choices=EMB_DEVICES)
    takeoff.add_argument(
        "--override", action="append", help="extra Hydra override passed to verl"
    )
    takeoff.set_defaults(plan_builder=build_takeoff_plan)

    evaluate = subparsers.add_parser(
        "eval",
        help="run configured LightEval benchmarks and publish them",
        description=(
            "Run every configured LightEval selector for each weight in fp16 "
            "and fp32io16, then publish one complete Scoreboard campaign."
        ),
        epilog=(
            "The TOML contains schema_version = 1, an optional prompt_template "
            "(bot, assistant, or function_calling), weights relative to "
            "WEIGHT_PATH, and direct LightEval task or superset selectors in "
            "benchmarks. Missing selectors are skipped. Exit 0 means every "
            "resolved task was stored, the campaign was finalized, and its "
            "local LightEval results were removed."
        ),
    )
    evaluate.add_argument(
        "--config",
        required=True,
        help=(
            "TOML containing schema_version, optional prompt_template, weights, "
            "and benchmarks"
        ),
    )
    evaluate.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="private dotenv file; defaults to .env.local",
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="validate, resolve selectors, and print a redacted plan",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = find_root()
    if args.command == "eval":
        env_path = find_env_path(
            root,
            args.env_file,
            use_fallbacks=False,
        )
        if env_path is not None:
            try:
                env_status = env_path.lstat()
            except OSError as error:
                parser.error(f"cannot inspect eval private environment file: {error}")
            if (
                not stat.S_ISREG(env_status.st_mode)
                or stat.S_IMODE(env_status.st_mode) != 0o600
                or env_status.st_uid != os.geteuid()
            ):
                parser.error(
                    "eval private environment file must be owned by the current "
                    "user, have mode 0600, and be a regular non-symlink file: "
                    f"{env_path}"
                )
        try:
            env, _ = load_env(
                root,
                args.env_file,
                use_fallbacks=False,
                require_private=True,
            )
        except (OSError, UnicodeError) as error:
            parser.error(f"cannot securely read eval private environment file: {error}")
        from helicopter_lighteval.evaluate import run as run_evaluation

        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        return run_evaluation(
            config_path=config_path,
            env=env,
            dry_run=args.dry_run,
        )
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
