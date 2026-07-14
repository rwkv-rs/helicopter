from __future__ import annotations

import unittest

from any2rwkv.target import (
    build_zero_step_ledger,
    canonical_text_name,
    is_sequence_mixer,
    rwkv7_mixer_specs,
)


class TargetMappingTests(unittest.TestCase):
    def test_multimodal_language_model_prefix_maps_to_text_causal_lm(self) -> None:
        self.assertEqual(
            canonical_text_name("model.language_model.layers.0.mlp.down_proj.weight"),
            "model.layers.0.mlp.down_proj.weight",
        )

    def test_target_specs_use_native_six_signal_rwkv7_parameterization(self) -> None:
        names = {spec.name for spec in rwkv7_mixer_specs(1, hidden_size=64)}
        for required in (
            "model.layers.1.attn.r_proj.weight",
            "model.layers.1.attn.w_lora.lora.2.bias",
            "model.layers.1.attn.k_proj.weight",
            "model.layers.1.attn.v_proj.weight",
            "model.layers.1.attn.a_lora.lora.2.weight",
            "model.layers.1.attn.g_lora.lora.2.weight",
        ):
            self.assertIn(required, names)

    def test_mtp_attention_is_preserved_not_converted_as_backbone(self) -> None:
        backbone = "model.language_model.layers.0.self_attn.q_proj.weight"
        mtp = "model.language_model.mtp.layers.0.self_attn.q_proj.weight"
        self.assertTrue(is_sequence_mixer(backbone))
        self.assertFalse(is_sequence_mixer(mtp))
        ledger, _, targets = build_zero_step_ledger(
            (backbone, mtp), layer_count=1, hidden_size=64, source_shard_hashes=("abc",)
        )
        self.assertEqual(ledger.sources[mtp].disposition, "preserved")
        self.assertIn("mtp.layers.0.self_attn.q_proj.weight", targets)

    def test_zero_step_ledger_preserves_shell_and_excludes_vision_explicitly(self) -> None:
        source = (
            "model.embed_tokens.weight",
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "visual.patch_embed.weight",
        )
        ledger, specs, targets = build_zero_step_ledger(
            source, layer_count=1, hidden_size=64, source_shard_hashes=("abc",)
        )
        self.assertEqual(ledger.sources["visual.patch_embed.weight"].disposition, "intentionally-unmapped")
        self.assertEqual(ledger.targets["model.embed_tokens.weight"].provenance, "copied")
        self.assertTrue(specs)
        self.assertNotIn("visual.patch_embed.weight", targets)


if __name__ == "__main__":
    unittest.main()
