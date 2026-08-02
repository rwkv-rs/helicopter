from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from any2rwkv.configuration_any2rwkv import (
    AnyToRWKVConfig,
    AnyToRWKVHybridConfig,
)
from any2rwkv.contract import build_target_config
from any2rwkv.fixture import tiny_qwen35_config
from any2rwkv.kernel import Rwkv7OperatorAdapter
from any2rwkv.mixer import apply_partial_rope
from any2rwkv.modeling_any2rwkv import (
    AnyToRWKVForCausalLM,
    AnyToRWKVHybridForCausalLM,
    AnyToRWKVPreservedGDN,
)


class ModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.operator_calls = 0

        def recurrent(
            r,
            w,
            k,
            v,
            a,
            b,
            *,
            initial_state,
            output_final_state,
            cu_seqlens=None,
            state_indices=None,
            mode,
        ):
            self.operator_calls += 1
            self.assertTrue(output_final_state)
            self.assertIsNone(cu_seqlens)
            self.assertIsNone(state_indices)
            self.assertEqual(mode, "fp32io16")
            state = initial_state
            outputs = []
            for token in range(r.shape[1]):
                projection = torch.einsum("bhk,bhkv->bhv", a[:, token].float(), state)
                state = (
                    w[:, token].float().exp().unsqueeze(-1) * state
                    + b[:, token].float().unsqueeze(-1) * projection.unsqueeze(-2)
                    + k[:, token].float().unsqueeze(-1)
                    * v[:, token].float().unsqueeze(-2)
                )
                outputs.append(
                    torch.einsum("bhk,bhkv->bhv", r[:, token].float(), state)
                )
            return torch.stack(outputs, dim=1).to(r.dtype), state

        adapter = Rwkv7OperatorAdapter(
            recurrent,
            lambda: "flash_rwkv",
            head_size=16,
            require_flash=True,
        )
        self.loader = patch(
            "any2rwkv.modeling_any2rwkv.load_rwkv7_operator_adapter",
            return_value=adapter,
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def config(self) -> AnyToRWKVConfig:
        source = tiny_qwen35_config(layers=4, moe=False)
        source["mtp_num_hidden_layers"] = 0
        target = build_target_config(source, require_final_layers=False)
        return AnyToRWKVConfig(**target)

    def test_rope_boundary_is_position_dependent_and_shape_preserving(self) -> None:
        value = torch.randn(2, 4, 16)
        rotated = apply_partial_rope(
            value, torch.tensor([0, 7]), rotary_dim=8, theta=10_000.0
        )
        torch.testing.assert_close(rotated[0], value[0])
        self.assertEqual(rotated.shape, value.shape)
        self.assertFalse(torch.equal(rotated[1, :, :8], value[1, :, :8]))
        torch.testing.assert_close(rotated[1, :, 8:], value[1, :, 8:])

    def test_prefill_and_token_decode_are_equivalent(self) -> None:
        torch.manual_seed(7)
        model = AnyToRWKVForCausalLM(self.config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            full = model(input_ids, use_cache=True).logits
            cache = None
            pieces = []
            for index in range(input_ids.shape[1]):
                output = model(
                    input_ids[:, index : index + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                pieces.append(output.logits)
        torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=1e-5, atol=1e-5)
        self.assertGreater(self.operator_calls, 0)

    def test_product_training_loss_backward_preserves_recurrent_cache_storage(
        self,
    ) -> None:
        torch.manual_seed(9)
        model = AnyToRWKVForCausalLM(self.config()).train()
        input_ids = torch.tensor([[1, 2, 3, 4]])

        output = model(input_ids, labels=input_ids, use_cache=False)
        self.assertIsNotNone(output.loss)
        assert output.loss is not None
        output.loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_preserved_gdn_matches_qwen_source_and_cached_continuation(
        self,
    ) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5GatedDeltaNet,
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )

        source_config = tiny_qwen35_config(layers=1, moe=False)
        source_config["mtp_num_hidden_layers"] = 0
        target_payload = build_target_config(
            source_config,
            converted_layers=0,
            require_final_layers=False,
        )
        target_config = AnyToRWKVHybridConfig(**target_payload)
        source = Qwen3_5GatedDeltaNet(SimpleNamespace(**source_config), 0).eval()
        source.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        source.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        preserved = AnyToRWKVPreservedGDN(target_config).eval()
        preserved.load_state_dict(source.state_dict(), strict=True)
        torch.manual_seed(13)
        hidden = torch.randn(2, 5, target_config.hidden_size)

        with torch.no_grad():
            expected = source(hidden)
            actual = preserved.forward_sequence(hidden)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

        model = AnyToRWKVHybridForCausalLM(target_config).eval()
        model.model.layers[0].linear_attn.load_state_dict(
            source.state_dict(), strict=True
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            complete = model(input_ids, use_cache=True).logits
            first = model(input_ids[:, :2], use_cache=True)
            resumed_output = model(
                input_ids[:, 2:],
                past_key_values=first.past_key_values,
                use_cache=True,
            )
        torch.testing.assert_close(
            torch.cat((first.logits, resumed_output.logits), dim=1),
            complete,
            rtol=2e-5,
            atol=2e-5,
        )

        cache = resumed_output.past_key_values
        cache.reset()
        self.assertEqual(cache.histories[0][0].shape, (1, 0, target_config.hidden_size))
        self.assertEqual(cache.history_positions[0][0].shape, (1, 0))
        with torch.no_grad():
            after_reset = model(
                input_ids,
                past_key_values=cache,
                use_cache=True,
            )
        torch.testing.assert_close(after_reset.logits, complete, rtol=2e-5, atol=2e-5)

        cache = after_reset.past_key_values
        cache.crop(0)
        self.assertEqual(cache.histories[0][0].shape, (1, 0, target_config.hidden_size))
        self.assertEqual(cache.history_positions[0][0].shape, (1, 0))
        with torch.no_grad():
            after_crop = model(
                input_ids,
                past_key_values=cache,
                use_cache=True,
            )
        torch.testing.assert_close(after_crop.logits, complete, rtol=2e-5, atol=2e-5)

    def test_product_forward_fails_closed_when_recurrent_contract_is_missing(
        self,
    ) -> None:
        model = AnyToRWKVForCausalLM(self.config()).eval()
        with (
            patch(
                "any2rwkv.modeling_any2rwkv.load_rwkv7_operator_adapter",
                side_effect=RuntimeError("public recurrent missing"),
            ),
            self.assertRaisesRegex(RuntimeError, "public recurrent missing"),
        ):
            model(torch.tensor([[1, 2]]))

    def test_left_padding_and_generation_inputs_preserve_positions(self) -> None:
        torch.manual_seed(11)
        model = AnyToRWKVForCausalLM(self.config()).eval()
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
        model = AnyToRWKVForCausalLM(self.config()).eval()
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
        model = AnyToRWKVForCausalLM(config)
        self.assertEqual(len(model.model.layers), 4)
        self.assertTrue(all(hasattr(layer, "attn") for layer in model.model.layers))
        self.assertFalse(hasattr(model, "mtp"))


if __name__ == "__main__":
    unittest.main()
