from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Any

from .commands import (
    WKV_MODES,
    build_infer_plan,
    build_takeoff_plan,
    prepend_venv_path,
    vllm_rwkv_revision,
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
    infer.add_argument("--serve-evaluation", action="store_true")
    infer.add_argument("--checkpoint-sha256")
    infer.add_argument("--tokenizer-revision")
    infer.add_argument("--chat-template-revision")
    infer.add_argument("--server-revision")
    infer.add_argument("--precision")
    infer.add_argument("--gemm-policy")
    infer.add_argument("--launch-contract")
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


def _add_eval(subparsers: Any) -> None:
    evaluation = subparsers.add_parser("eval", help="run a signed LightEval task")
    eval_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    run = eval_commands.add_parser("run", help="evaluate one canonical LightEval task")
    add_common_options(run)
    run.add_argument("model", help="served model name")
    run.add_argument(
        "task",
        help=(
            "canonical LightEval identity, e.g. "
            "lighteval/math/gsm8k@0, lighteval/math/aime24@2, "
            "lighteval/knowledge/mmlu-pro@0, or "
            "lighteval/instruction-following/ifeval@0.1"
        ),
    )
    run.add_argument("--endpoint-url", help="OpenAI base URL including /v1")
    run.add_argument("--checkpoint-sha256")
    run.add_argument("--tokenizer-revision")
    run.add_argument("--chat-template-revision")
    run.add_argument("--server-revision")
    run.add_argument("--wkv-mode")
    run.add_argument("--precision")
    run.add_argument("--gemm-policy")
    run.add_argument("--launch-contract")
    run.add_argument("--output-root", type=Path)
    run.add_argument("--cot-mode", choices=("none", "cot"))
    run.add_argument("--math-repair-strategy", choices=("A", "B", "C"))
    run.add_argument("--max-concurrent-requests", type=int)
    run.add_argument("--request-timeout-seconds", type=float)
    run.add_argument("--max-samples", type=int)
    run.add_argument("--generation-limit", type=int)
    run.add_argument("--allow-non-comparable", action="store_true", default=None)
    run.add_argument("--scoreboard-url")
    run.add_argument("--scoreboard-token-env", default="SCOREBOARD_TOKEN")
    run.add_argument("--endpoint-api-key-env", default="OPENAI_API_KEY")
    publish = eval_commands.add_parser(
        "publish", help="retry scoreboard publication from one committed manifest"
    )
    add_common_options(publish)
    publish.add_argument("manifest", type=Path)
    publish.add_argument("--scoreboard-url", required=True)
    publish.add_argument("--scoreboard-token-env", default="SCOREBOARD_TOKEN")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helicopter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_infer(subparsers)
    _add_takeoff(subparsers)
    _add_eval(subparsers)
    return parser


def _run_evaluation(
    args: argparse.Namespace,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> int:
    # Keep the base CLI independent from LightEval and its optional
    # dependencies.  This import is intentionally inside the eval command.
    from helicopter_lighteval.evaluation import EvaluationRequest, run_evaluation

    scoreboard_token = env.get(args.scoreboard_token_env) or os.environ.get(
        args.scoreboard_token_env
    )
    endpoint_api_key = env.get(args.endpoint_api_key_env) or os.environ.get(
        args.endpoint_api_key_env
    )
    file_values = config.get("eval", {})
    if not isinstance(file_values, dict):
        raise ValueError("[eval] config must be a TOML table")

    def resolved(name: str, default: Any = None) -> Any:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            return cli_value
        return file_values.get(name, default)

    def required(name: str) -> Any:
        value = resolved(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"evaluation config field is required: {name}")
        return value

    output_root_value = Path(resolved("output_root", "src/eval/lighteval/results"))
    output_root = (
        output_root_value
        if output_root_value.is_absolute()
        else root / output_root_value
    )
    scoreboard_url = resolved("scoreboard_url")
    server_revision = vllm_rwkv_revision(config, root=root, env=env)
    configured_server_revision = resolved("server_revision")
    if (
        configured_server_revision is not None
        and str(configured_server_revision) != server_revision
    ):
        raise ValueError(
            "evaluation server revision does not match the vllm-rwkv submodule: "
            f"configured {configured_server_revision}, actual {server_revision}"
        )
    config_values = {
        name: resolved(name)
        for name in (
            "endpoint_url",
            "checkpoint_sha256",
            "tokenizer_revision",
            "chat_template_revision",
            "server_revision",
            "wkv_mode",
            "precision",
            "gemm_policy",
            "launch_contract",
            "output_root",
            "cot_mode",
            "math_repair_strategy",
            "max_concurrent_requests",
            "request_timeout_seconds",
            "max_samples",
            "generation_limit",
            "allow_non_comparable",
            "scoreboard_url",
        )
    }
    config_values["server_revision"] = server_revision
    product_revision, product_dirty = _git_identity(root)
    request = EvaluationRequest(
        model=args.model,
        task=args.task,
        output_root=output_root,
        endpoint_url=str(required("endpoint_url")),
        checkpoint_sha256=str(required("checkpoint_sha256")),
        tokenizer_revision=str(required("tokenizer_revision")),
        chat_template_revision=str(required("chat_template_revision")),
        server_revision=server_revision,
        wkv_mode=str(required("wkv_mode")),
        precision=str(required("precision")),
        gemm_policy=str(required("gemm_policy")),
        launch_contract=str(required("launch_contract")),
        cot_mode=str(resolved("cot_mode", "none")),
        math_repair_strategy=str(resolved("math_repair_strategy", "A")),
        max_concurrent_requests=int(resolved("max_concurrent_requests", 16)),
        request_timeout_seconds=float(resolved("request_timeout_seconds", 3600.0)),
        generation_limit=resolved("generation_limit"),
        max_samples=resolved("max_samples"),
        scoreboard_url=scoreboard_url,
        scoreboard_token=scoreboard_token,
        endpoint_api_key=endpoint_api_key,
        allow_non_comparable=bool(resolved("allow_non_comparable", False)),
        config_digest=_evaluation_config_digest(config_values),
        product_revision=product_revision,
        product_dirty=product_dirty,
    )
    if args.dry_run:
        print(
            f"eval task={request.task} model={request.model} output_root={request.output_root}"
        )
        return 0
    outcome = run_evaluation(request)
    print(
        f"eval run={outcome.run_id} status={outcome.run_status} manifest={outcome.manifest_path}"
    )
    if outcome.publication_task_id is not None:
        print(f"scoreboard publication task={outcome.publication_task_id}")
    if outcome.publication_error:
        print(f"scoreboard publication failed: {outcome.publication_error}")
    return 0 if outcome.is_success else 1


def _git_identity(root: Path) -> tuple[str, bool]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot determine evaluator product revision") from error
    revision = result.stdout.strip()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("evaluator product revision is not a full Git commit SHA")
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot determine evaluator worktree state") from error
    return revision, bool(status.stdout.strip())


def _retry_evaluation_publication(
    args: argparse.Namespace, *, root: Path, env: dict[str, str]
) -> int:
    from helicopter_lighteval.scoreboard import retry_publication

    token = env.get(args.scoreboard_token_env) or os.environ.get(
        args.scoreboard_token_env
    )
    if not token:
        raise ValueError(
            f"scoreboard token environment variable is required: {args.scoreboard_token_env}"
        )
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    if args.dry_run:
        print(f"eval publish manifest={manifest} scoreboard={args.scoreboard_url}")
        return 0
    outcome = retry_publication(
        manifest_path=manifest,
        scoreboard_url=args.scoreboard_url,
        bearer_token=token,
    )
    print(
        f"eval publish run={manifest.parent.name} status={outcome.status} "
        f"task={outcome.task_id} retry={outcome.retry_identity}"
    )
    if outcome.error:
        print(f"scoreboard publication failed: {outcome.error}")
    return 0 if outcome.status == "published" else 1


def _evaluation_config_digest(values: dict[str, Any]) -> str:
    import hashlib
    import json

    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = find_root()
    env, _ = load_env(root, args.env_file)
    config, _ = load_config(root, args.config)
    prepend_venv_path(env, root, config)
    if args.command == "eval":
        if args.eval_command == "publish":
            return _retry_evaluation_publication(args, root=root, env=env)
        return _run_evaluation(args, root=root, env=env, config=config)
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
