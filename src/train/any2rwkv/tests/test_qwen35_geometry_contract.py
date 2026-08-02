from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from any2rwkv.adapters.qwen35.geometry_contract import (
    GQAHeadStateMapping,
    canonical_gqa_state_mappings,
    validate_qwen35_geometry,
)
from any2rwkv.adapters.qwen35.source_adapter import Qwen35SourceAdapter
from any2rwkv.errors import ContractError


def _config(model_id: str, *, gdn_heads: int, query_heads: int) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "hidden_size": gdn_heads * 128,
        "num_attention_heads": query_heads,
        "num_key_value_heads": max(1, query_heads // 4),
        "head_dim": 256,
        "linear_num_key_heads": gdn_heads,
        "linear_num_value_heads": gdn_heads,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }


class Qwen35GeometryContractTests(unittest.TestCase):
    def test_metadata_inspection_preserves_2b_and_397b_geometry_without_weights(
        self,
    ) -> None:
        cases = (
            ("Qwen/Qwen3.5-2B", 16, 8),
            ("Qwen/Qwen3.5-397B-A17B", 64, 32),
        )
        for model_id, gdn_heads, query_heads in cases:
            with (
                self.subTest(model_id=model_id),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                (root / "config.json").write_text(
                    json.dumps(
                        _config(model_id, gdn_heads=gdn_heads, query_heads=query_heads)
                    ),
                    encoding="utf-8",
                )
                inspection = Qwen35SourceAdapter().inspect_checkpoint(
                    root, require_final_layout=False
                )
                self.assertEqual(inspection.metadata["gdn_num_heads"], gdn_heads)
                self.assertEqual(inspection.metadata["gdn_head_size"], 128)
                self.assertEqual(inspection.metadata["gqa_head_size"], 256)
                self.assertEqual(
                    len(inspection.metadata["gqa_state_mappings"]), query_heads
                )
                self.assertFalse(any(root.glob("*.safetensors")))

    def test_every_gqa_query_head_declares_both_observable_state_inputs(self) -> None:
        config = _config("Qwen/Qwen3.5-2B", gdn_heads=16, query_heads=8)
        contract = validate_qwen35_geometry(config)
        query = torch.arange(8, dtype=torch.float32)
        matrix_state = torch.arange(8, dtype=torch.float32) + 1
        prefix_value_mean = torch.arange(8, dtype=torch.float32) / 10
        observable_output = query * matrix_state + prefix_value_mean

        reconstructed = torch.empty_like(observable_output)
        for mapping in contract.gqa_state_mappings:
            self.assertIn("matrix_state", mapping.matrix_state)
            self.assertIn("prefix_value_mean_state", mapping.prefix_value_mean_state)
            head = mapping.query_head
            reconstructed[head] = (
                query[head] * matrix_state[head] + prefix_value_mean[head]
            )
        torch.testing.assert_close(reconstructed, observable_output, rtol=0, atol=0)

    def test_unknown_or_drifted_geometry_and_mapping_coverage_fail_closed(self) -> None:
        base = _config("Qwen/Qwen3.5-2B", gdn_heads=16, query_heads=8)
        mutations = (
            ("unknown Qwen3.5", {"model_id": "Qwen/unknown"}),
            ("GDN geometry", {"linear_value_head_dim": 256}),
            ("GDN geometry", {"linear_num_value_heads": 8}),
            ("GQA", {"head_dim": 128, "num_attention_heads": 16}),
            ("GQA", {"head_dim": 64, "num_attention_heads": 32}),
        )
        for message, updates in mutations:
            changed = copy.deepcopy(base)
            changed.update(updates)
            with (
                self.subTest(updates=updates),
                self.assertRaisesRegex(ContractError, message),
            ):
                validate_qwen35_geometry(changed)

        mappings = canonical_gqa_state_mappings(8)
        with self.assertRaisesRegex(ContractError, "coverage"):
            validate_qwen35_geometry(base, gqa_state_mappings=mappings[:-1])
        missing_state = list(mappings)
        missing_state[3] = GQAHeadStateMapping(3, "", "prefix.3")
        with self.assertRaisesRegex(ContractError, "requires matrix-state"):
            validate_qwen35_geometry(base, gqa_state_mappings=missing_state)


if __name__ == "__main__":
    unittest.main()
