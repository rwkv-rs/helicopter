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
    infer.add_argument("--max-model-len", type=int)
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
    run = eval_commands.add_parser(
        "run", help="evaluate one canonical task against one snapshot"
    )
    add_common_options(run)
    run.add_argument("model", help="served model name")
    run.add_argument(
        "task", help="canonical registry identity, e.g. lighteval/math/gsm8k@0"
    )
    run.add_argument("--snapshot", type=Path)
    run.add_argument("--snapshot-manifest", type=Path)
    run.add_argument("--snapshot-sha256")
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
    config_path: Path,
) -> int:
    from lighteval_runner.application import run_evaluation
    from lighteval_runner.config import resolve_evaluation_config
    from lighteval_runner.contracts import EvaluationRequest

    scoreboard_token = env.get(args.scoreboard_token_env) or os.environ.get(
        args.scoreboard_token_env
    )
    endpoint_api_key = env.get(args.endpoint_api_key_env) or os.environ.get(
        args.endpoint_api_key_env
    )
    allowed = frozenset(
        {
            "snapshot",
            "snapshot_manifest",
            "snapshot_sha256",
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
            "max_samples",
            "generation_limit",
            "allow_non_comparable",
            "scoreboard_url",
            "scoreboard_token",
            "endpoint_api_key",
        }
    )
    file_values = config.get("eval", {})
    if not isinstance(file_values, dict):
        raise ValueError("[eval] config must be a TOML table")
    cli_values = {
        name: getattr(args, name)
        for name in allowed - {"scoreboard_token", "endpoint_api_key"}
        if hasattr(args, name)
    }
    resolved = resolve_evaluation_config(
        allowed_fields=allowed,
        secret_fields=frozenset({"scoreboard_token", "endpoint_api_key"}),
        defaults={
            "output_root": Path("results/lighteval"),
            "cot_mode": "none",
            "math_repair_strategy": "A",
            "allow_non_comparable": False,
        },
        file_values=file_values,
        environment_values={
            "scoreboard_token": scoreboard_token,
            "endpoint_api_key": endpoint_api_key,
        },
        cli_values=cli_values,
        config_path=config_path,
    )

    def required(name: str) -> Any:
        value = resolved.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"evaluation config field is required: {name}")
        return value

    def optional_path_value(name: str) -> Path | None:
        value = resolved.get(name)
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else root / path

    output_root_value = Path(resolved.get("output_root"))
    output_root = (
        output_root_value
        if output_root_value.is_absolute()
        else root / output_root_value
    )
    scoreboard_url = resolved.get("scoreboard_url")
    product_revision, product_dirty = _git_identity(root)
    request = EvaluationRequest(
        model=args.model,
        task=args.task,
        output_root=output_root,
        snapshot_path=optional_path_value("snapshot"),
        snapshot_manifest_path=optional_path_value("snapshot_manifest"),
        snapshot_sha256=(
            str(resolved.get("snapshot_sha256"))
            if resolved.get("snapshot_sha256") is not None
            else None
        ),
        endpoint_url=str(required("endpoint_url")),
        checkpoint_sha256=str(required("checkpoint_sha256")),
        tokenizer_revision=str(required("tokenizer_revision")),
        chat_template_revision=str(required("chat_template_revision")),
        expected_server_revision=str(required("server_revision")),
        wkv_mode=str(required("wkv_mode")),
        precision=str(required("precision")),
        gemm_policy=str(required("gemm_policy")),
        launch_contract=str(required("launch_contract")),
        cot_mode=str(resolved.get("cot_mode")),
        math_repair_strategy=str(resolved.get("math_repair_strategy")),
        generation_limit_override=resolved.get("generation_limit"),
        generation_limit_override_source=resolved.provenance("generation_limit"),
        max_samples=resolved.get("max_samples"),
        publish_to_scoreboard=scoreboard_url is not None,
        scoreboard_url=scoreboard_url,
        scoreboard_token=resolved.get("scoreboard_token"),
        endpoint_api_key=resolved.get("endpoint_api_key"),
        allow_non_comparable=bool(resolved.get("allow_non_comparable")),
        config_digest=resolved.identity_digest(),
        config_evidence=resolved.redacted_payload(),
        product_revision=product_revision,
        product_dirty=product_dirty,
    )
    if args.dry_run:
        print(
            f"eval task={request.task} model={request.model} snapshot={request.snapshot_path}"
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
    from lighteval_runner.application import retry_scoreboard_publication

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
    outcome = retry_scoreboard_publication(
        manifest_path=manifest,
        scoreboard_url=args.scoreboard_url,
        scoreboard_token=token,
    )
    print(
        f"eval publish run={outcome.run_id} status={outcome.publication_status} "
        f"task={outcome.publication_task_id} retry={outcome.publication_retry_identity}"
    )
    if outcome.publication_error:
        print(f"scoreboard publication failed: {outcome.publication_error}")
    return 0 if outcome.is_success else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = find_root()
    env, _ = load_env(root, args.env_file)
    config, config_path = load_config(root, args.config)
    prepend_venv_path(env, root, config)
    if args.command == "eval":
        if args.eval_command == "publish":
            return _retry_evaluation_publication(args, root=root, env=env)
        return _run_evaluation(
            args, root=root, env=env, config=config, config_path=config_path
        )
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
