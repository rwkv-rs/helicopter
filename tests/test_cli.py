from __future__ import annotations

import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_cli_has_no_evaluation_lifecycle(self) -> None:
        from helicopter_cli.__main__ import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(["eval"])
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["infer", "model", "--serve-evaluation"])

    def test_infer_context_length_is_derived_from_checkpoint_filename(self) -> None:
        from helicopter_cli.__main__ import build_parser
        from helicopter_cli.commands import build_infer_plan

        args = build_parser().parse_args(["infer", "model", "--dry-run"])
        config = {
            "models": {
                "model": {
                    "path": "/parent-ctx4096/rwkv7-g1g-1.5b-20260526-ctx8192.pth"
                }
            }
        }
        plan = build_infer_plan(args, root=Path("/repo"), env={}, config=config)
        context_index = plan.command.index("--max-model-len")
        self.assertEqual(plan.command[context_index + 1], "8192")
        self.assertEqual(plan.command.count("--max-model-len"), 1)
        self.assertNotIn("--enforce-eager", plan.command)

    def test_infer_rejects_invalid_or_ambiguous_checkpoint_context(self) -> None:
        from helicopter_cli.__main__ import build_parser
        from helicopter_cli.commands import build_infer_plan

        filenames = (
            "rwkv7-g1g-1.5b-20260526.pth",
            "rwkv7-g1g-1.5b-20260526-ctx0.pth",
            "rwkv7-g1g-1.5b-20260526-ctx08192.pth",
            "rwkv7-g1g-1.5b-20260526-ctx8k.pth",
            "rwkv7-g1g-1.5b-20260526-CTX8192.pth",
            "rwkv7-g1g-1.5b-ctx4096-20260526-ctx8192.pth",
            "rwkv7-g1g-1.5b-CTX4096-20260526-ctx8192.pth",
        )
        args = build_parser().parse_args(["infer", "model", "--dry-run"])
        for filename in filenames:
            with self.subTest(filename=filename):
                config = {"models": {"model": {"path": f"/weights/{filename}"}}}
                with self.assertRaisesRegex(SystemExit, "checkpoint filename.*ctx"):
                    build_infer_plan(args, root=Path("/repo"), env={}, config=config)

    def test_infer_rejects_legacy_max_model_len_config(self) -> None:
        from helicopter_cli.__main__ import build_parser
        from helicopter_cli.commands import build_infer_plan

        args = build_parser().parse_args(["infer", "model", "--dry-run"])
        config = {
            "infer": {"max_model_len": 8192},
            "models": {
                "model": {
                    "path": "/weights/rwkv7-g1g-1.5b-20260526-ctx8192.pth"
                }
            },
        }
        with self.assertRaisesRegex(
            SystemExit, "max_model_len.*checkpoint filename"
        ):
            build_infer_plan(args, root=Path("/repo"), env={}, config=config)

    def test_infer_cli_no_longer_accepts_max_model_len_override(self) -> None:
        from helicopter_cli.__main__ import build_parser

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["infer", "model", "--dry-run", "--max-model-len", "4096"]
            )
