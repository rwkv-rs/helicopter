from __future__ import annotations

import unittest

import torch

from any2rwkv.configuration_any2rwkv import Any2RWKV7Config
from any2rwkv.contract import build_target_config
from any2rwkv.fixture import tiny_qwen35_config
from any2rwkv.mixer import apply_partial_rope
from any2rwkv.modeling_any2rwkv import Any2RWKV7ForCausalLM


class ModelingTests(unittest.TestCase):
    def config(self) -> Any2RWKV7Config:
        source = tiny_qwen35_config(layers=4, moe=False)
        source["mtp_num_hidden_layers"] = 0
        target = build_target_config(source, require_final_layers=False)
        return Any2RWKV7Config(**target)

    def test_rope_boundary_is_position_dependent_and_shape_preserving(self) -> None:
        value = torch.randn(2, 4, 16)
        rotated = apply_partial_rope(value, torch.tensor([0, 7]), rotary_dim=8, theta=10_000.0)
        torch.testing.assert_close(rotated[0], value[0])
        self.assertEqual(rotated.shape, value.shape)
        self.assertFalse(torch.equal(rotated[1, :, :8], value[1, :, :8]))
        torch.testing.assert_close(rotated[1, :, 8:], value[1, :, 8:])

    def test_prefill_and_token_decode_are_equivalent(self) -> None:
        torch.manual_seed(7)
        model = Any2RWKV7ForCausalLM(self.config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            full = model(input_ids, use_cache=True).logits
            cache = None
            pieces = []
            for index in range(input_ids.shape[1]):
                output = model(input_ids[:, index : index + 1], past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                pieces.append(output.logits)
        torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=1e-5, atol=1e-5)

    def test_left_padding_and_generation_inputs_preserve_positions(self) -> None:
        torch.manual_seed(11)
        model = Any2RWKV7ForCausalLM(self.config()).eval()
        padded = torch.tensor([[0, 0, 1, 2, 3]])
        mask = torch.tensor([[0, 0, 1, 1, 1]])
        plain = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            padded_logits = model(padded, attention_mask=mask).logits[:, -3:]
            plain_logits = model(plain).logits
        torch.testing.assert_close(padded_logits, plain_logits, rtol=1e-5, atol=1e-5)

        embeds = model.get_input_embeddings()(plain)
        prepared = model.prepare_inputs_for_generation(
            None, inputs_embeds=embeds, attention_mask=torch.ones_like(plain)
        )
        self.assertIn("inputs_embeds", prepared)
        self.assertNotIn("input_ids", prepared)

    def test_recurrent_cache_batch_contract_and_explicit_crop_limit(self) -> None:
        model = Any2RWKV7ForCausalLM(self.config()).eval()
        cache = model(torch.tensor([[1, 2]]), use_cache=True).past_key_values
        self.assertEqual(cache.get_seq_length(), 2)
        cache.batch_repeat_interleave(2)
        self.assertEqual(cache.max_batch_size, 2)
        cache.batch_select_indices(torch.tensor([1]))
        self.assertEqual(cache.max_batch_size, 1)
        with self.assertRaisesRegex(NotImplementedError, "cannot be cropped"):
            cache.crop(1)
        cache.crop(0)
        self.assertEqual(cache.get_seq_length(), 0)

    def test_only_the_60_backbone_slots_are_rwkv7_not_mtp(self) -> None:
        config = self.config()
        model = Any2RWKV7ForCausalLM(config)
        self.assertEqual(len(model.model.layers), 4)
        self.assertTrue(all(hasattr(layer, "attn") for layer in model.model.layers))
        self.assertFalse(hasattr(model, "mtp"))


if __name__ == "__main__":
    unittest.main()
