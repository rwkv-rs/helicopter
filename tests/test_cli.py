from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def test_base_cli_import_does_not_load_eval_dependencies() -> None:
    for name in tuple(sys.modules):
        if name == "lighteval" or name.startswith(
            ("lighteval.", "helicopter_lighteval", "lighteval_runner")
        ):
            sys.modules.pop(name)
    module = importlib.import_module("helicopter_cli.__main__")
    module.build_parser()
    assert "lighteval" not in sys.modules
    assert "helicopter_lighteval" not in sys.modules
    assert "lighteval_runner" not in sys.modules


def test_eval_cli_exposes_only_typed_canonical_run_contract() -> None:
    from helicopter_cli.__main__ import build_parser

    args = build_parser().parse_args(
        [
            "eval",
            "run",
            "rwkv-test",
            "lighteval/math/gsm8k@0",
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
    assert args.math_repair_strategy is None
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


def test_infer_evaluation_mode_builds_one_strict_provider_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helicopter_cli.__main__ import build_parser
    from helicopter_cli.commands import build_infer_plan

    server_revision = "a" * 40
    monkeypatch.setattr(
        "helicopter_cli.commands.vllm_rwkv_revision",
        lambda config, *, root, env: server_revision,
    )
    args = build_parser().parse_args(
        ["infer", "model", "--dry-run", "--serve-evaluation", "--wkv-mode", "fp32io16"]
    )
    config = {
        "models": {
            "model": {
                "path": "/nonexistent/rwkv7-g1h-7.2b-20260710-ctx10240.pth",
                "served_model_name": "rwkv-test",
                "sha256": "b" * 64,
                "infer": {
                    "tokenizer_revision": "tok-v1",
                    "chat_template_revision": "chat-v1",
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
    assert contract["provider"]["server_revision"] == server_revision
    assert plan.shown_env == {"VLLM_RWKV7_WKV_MODE": "fp32io16"}
    assert set(contract["capabilities"]) == {
        "openai-chat",
        "output-token-ids",
        "terminal-reason",
        "prompt-evidence",
    }
    context_index = plan.command.index("--max-model-len")
    assert plan.command[context_index + 1] == "10240"
    assert plan.command.count("--enforce-eager") == 1


def test_infer_evaluation_rejects_configured_server_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helicopter_cli.__main__ import build_parser
    from helicopter_cli.commands import build_infer_plan

    monkeypatch.setattr(
        "helicopter_cli.commands.vllm_rwkv_revision",
        lambda config, *, root, env: "a" * 40,
    )
    args = build_parser().parse_args(
        ["infer", "model", "--dry-run", "--serve-evaluation", "--wkv-mode", "fp32io16"]
    )
    config = {
        "models": {
            "model": {
                "path": "/nonexistent/rwkv7-g1h-7.2b-20260710-ctx10240.pth",
                "served_model_name": "rwkv-test",
                "sha256": "b" * 64,
                "infer": {
                    "tokenizer_revision": "tok-v1",
                    "chat_template_revision": "chat-v1",
                    "server_revision": "b" * 40,
                    "precision": "fp16-io-fp32-state",
                    "gemm_policy": "fp32-accumulation",
                    "launch_contract": "launch-v1",
                },
            }
        }
    }
    with pytest.raises(SystemExit, match="server revision.*submodule"):
        build_infer_plan(args, root=Path("/repo"), env={}, config=config)


def test_infer_context_length_is_derived_from_checkpoint_filename() -> None:
    from helicopter_cli.__main__ import build_parser
    from helicopter_cli.commands import build_infer_plan

    args = build_parser().parse_args(["infer", "model", "--dry-run"])
    config = {
        "models": {
            "model": {"path": "/parent-ctx4096/rwkv7-g1g-1.5b-20260526-ctx8192.pth"}
        }
    }
    plan = build_infer_plan(args, root=Path("/repo"), env={}, config=config)
    context_index = plan.command.index("--max-model-len")
    assert plan.command[context_index + 1] == "8192"
    assert plan.command.count("--max-model-len") == 1
    assert "--enforce-eager" not in plan.command


@pytest.mark.parametrize(
    "filename",
    [
        "rwkv7-g1g-1.5b-20260526.pth",
        "rwkv7-g1g-1.5b-20260526-ctx0.pth",
        "rwkv7-g1g-1.5b-20260526-ctx08192.pth",
        "rwkv7-g1g-1.5b-20260526-ctx8k.pth",
        "rwkv7-g1g-1.5b-20260526-CTX8192.pth",
        "rwkv7-g1g-1.5b-ctx4096-20260526-ctx8192.pth",
        "rwkv7-g1g-1.5b-CTX4096-20260526-ctx8192.pth",
    ],
)
def test_infer_rejects_invalid_or_ambiguous_checkpoint_context(filename: str) -> None:
    from helicopter_cli.__main__ import build_parser
    from helicopter_cli.commands import build_infer_plan

    args = build_parser().parse_args(["infer", "model", "--dry-run"])
    config = {"models": {"model": {"path": f"/weights/{filename}"}}}
    with pytest.raises(SystemExit, match="checkpoint filename.*ctx"):
        build_infer_plan(args, root=Path("/repo"), env={}, config=config)


def test_infer_rejects_legacy_max_model_len_config() -> None:
    from helicopter_cli.__main__ import build_parser
    from helicopter_cli.commands import build_infer_plan

    args = build_parser().parse_args(["infer", "model", "--dry-run"])
    config = {
        "infer": {"max_model_len": 8192},
        "models": {"model": {"path": "/weights/rwkv7-g1g-1.5b-20260526-ctx8192.pth"}},
    }
    with pytest.raises(SystemExit, match="max_model_len.*checkpoint filename"):
        build_infer_plan(args, root=Path("/repo"), env={}, config=config)


def test_infer_cli_no_longer_accepts_max_model_len_override() -> None:
    from helicopter_cli.__main__ import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["infer", "model", "--dry-run", "--max-model-len", "4096"]
        )


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
