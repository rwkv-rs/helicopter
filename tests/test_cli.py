from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_has_no_evaluation_lifecycle() -> None:
    from helicopter_cli.__main__ import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["infer", "model", "--serve-evaluation"])


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
