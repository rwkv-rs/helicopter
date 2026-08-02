from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from any2rwkv.core.tiny_pipeline import run_tiny_pipeline
from any2rwkv.errors import ContractError


class TinyPipelineTests(unittest.TestCase):
    def test_full_two_layer_interrupt_resume_export_and_fresh_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "run"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_id": "Qwen/Qwen3.5-2B",
                        "model_type": "qwen3_5_text",
                        "architectures": ["Qwen3_5ForCausalLM"],
                        "num_hidden_layers": 2,
                        "layer_types": ["linear_attention", "full_attention"],
                        "hidden_size": 2048,
                        "num_attention_heads": 8,
                        "num_key_value_heads": 2,
                        "head_dim": 256,
                        "linear_num_key_heads": 16,
                        "linear_num_value_heads": 16,
                        "linear_key_head_dim": 128,
                        "linear_value_head_dim": 128,
                    }
                ),
                encoding="utf-8",
            )
            interrupted = run_tiny_pipeline(
                source, output, interrupt_after_optimizer_steps=2
            )
            self.assertEqual(interrupted, {"status": "interrupted", "stage": "train"})
            complete = run_tiny_pipeline(source, output)
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(run_tiny_pipeline(source, output), complete)
            for stage in ("inspect", "split", "train", "ledger", "export", "verify"):
                self.assertTrue((output / stage / "stage.json").is_file())
            ledger = json.loads((output / "ledger" / "artifact.json").read_text())
            self.assertEqual(ledger["source_coverage"], 1.0)
            self.assertEqual(ledger["target_coverage"], 1.0)
            config = json.loads(
                (output / "export" / "checkpoint" / "config.json").read_text()
            )
            self.assertEqual(config["model_type"], "rwkv7")
            self.assertNotIn("auto_map", config)
            uninterrupted = run_tiny_pipeline(source, root / "uninterrupted")
            self.assertEqual(uninterrupted["status"], "complete")
            resumed_train = json.loads((output / "train" / "artifact.json").read_text())
            uninterrupted_train = json.loads(
                (root / "uninterrupted" / "train" / "artifact.json").read_text()
            )
            self.assertEqual(resumed_train, uninterrupted_train)
            expected_path = output / "export" / "expected.pt"
            expected_path.write_bytes(expected_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ContractError, "observable artifact changed"):
                run_tiny_pipeline(source, output)

    def test_unknown_or_partial_stage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            output = root / "run"
            (output / "unknown").mkdir(parents=True)
            with self.assertRaisesRegex(ContractError, "unknown stage"):
                run_tiny_pipeline(source, output)
            (output / "unknown").rmdir()
            (output / "inspect").mkdir()
            with self.assertRaisesRegex(ContractError, "partial"):
                run_tiny_pipeline(source, output)


if __name__ == "__main__":
    unittest.main()
