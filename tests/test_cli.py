from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def test_base_cli_import_does_not_load_eval_dependencies() -> None:
    for name in tuple(sys.modules):
        if name == "lighteval" or name.startswith(("lighteval.", "lighteval_runner")):
            sys.modules.pop(name)
    module = importlib.import_module("helicopter_cli.__main__")
    module.build_parser()
    assert "lighteval" not in sys.modules
    assert "lighteval_runner" not in sys.modules


def test_eval_cli_exposes_only_typed_canonical_run_contract() -> None:
    from helicopter_cli.__main__ import build_parser

    args = build_parser().parse_args(
        [
            "eval",
            "run",
            "rwkv-test",
            "lighteval/math/gsm8k@0",
            "--snapshot",
            "gsm8k.jsonl",
            "--snapshot-manifest",
            "gsm8k.jsonl.manifest.json",
            "--snapshot-sha256",
            "a" * 64,
            "--endpoint-url",
            "http://127.0.0.1:8000/v1",
            "--checkpoint-sha256",
            "b" * 64,
            "--tokenizer-revision",
            "tok-v1",
            "--chat-template-revision",
            "chat-v1",
            "--server-revision",
            "server-v1",
            "--wkv-mode",
            "fp32io16",
            "--precision",
            "fp16-io-fp32-state",
            "--gemm-policy",
            "fp32-accumulation",
            "--launch-contract",
            "launch-v1",
        ]
    )
    assert args.task == "lighteval/math/gsm8k@0"
    assert args.math_repair_strategy is None  # resolved default retains provenance
    assert not hasattr(args, "model_args")
    assert not hasattr(args, "custom_tasks")


@pytest.mark.parametrize(
    "obsolete",
    ["function-calling", "agent-harness", "lighteval-tasks", "lighteval-export"],
)
def test_deleted_eval_facades_are_not_cli_commands(obsolete: str) -> None:
    from helicopter_cli.__main__ import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", obsolete])


def test_infer_evaluation_mode_builds_one_strict_provider_attestation() -> None:
    from helicopter_cli.__main__ import build_parser
    from helicopter_cli.commands import build_infer_plan

    args = build_parser().parse_args(
        ["infer", "model", "--dry-run", "--serve-evaluation", "--wkv-mode", "fp32io16"]
    )
    config = {
        "models": {
            "model": {
                "path": "/nonexistent/model.pth",
                "served_model_name": "rwkv-test",
                "sha256": "b" * 64,
                "infer": {
                    "tokenizer_revision": "tok-v1",
                    "chat_template_revision": "chat-v1",
                    "server_revision": "server-v1",
                    "precision": "fp16-io-fp32-state",
                    "gemm_policy": "fp32-accumulation",
                    "launch_contract": "launch-v1",
                },
            }
        }
    }
    plan = build_infer_plan(args, root=Path("/repo"), env={}, config=config)
    index = plan.command.index("--helicopter-attestation-json")
    contract = json.loads(plan.command[index + 1])
    assert contract["model"]["checkpoint_sha256"] == "b" * 64
    assert contract["provider"]["wkv_mode"] == "fp32io16"
    assert plan.shown_env == {"VLLM_RWKV7_WKV_MODE": "fp32io16"}
    assert set(contract["capabilities"]) == {
        "openai-chat",
        "output-token-ids",
        "terminal-reason",
        "prompt-evidence",
    }


def test_eval_publish_retries_only_from_a_committed_manifest() -> None:
    from helicopter_cli.__main__ import build_parser

    args = build_parser().parse_args(
        [
            "eval",
            "publish",
            "results/run-1/manifest.json",
            "--scoreboard-url",
            "https://scoreboard.example",
            "--dry-run",
        ]
    )
    assert args.eval_command == "publish"
    assert args.manifest == Path("results/run-1/manifest.json")
    assert args.scoreboard_token_env == "SCOREBOARD_TOKEN"
