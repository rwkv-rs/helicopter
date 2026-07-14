from __future__ import annotations

import unittest

import torch

from any2rwkv.migration import (
    build_group_map,
    fit_headwise_teacher_trace,
    gdn_to_rwkv7_dynamics,
    kv_expand,
    kv_repeat,
    validate_attention_trace_contract,
    verify_gdn_mapping,
)
from any2rwkv.recurrent import chunked_rwkv7_scan, native_decay_from_logit, native_decay_logit, reset_state, rwkv7_scan


class RecurrentTests(unittest.TestCase):
    def signals(self, *, length: int = 17, dtype: torch.dtype = torch.float64):
        generator = torch.Generator().manual_seed(20260714)
        shape = (2, length, 3, 4)
        r = torch.randn(shape, generator=generator, dtype=dtype)
        logit = torch.randn(shape, generator=generator, dtype=dtype)
        decay = native_decay_from_logit(logit)
        k = torch.randn(shape, generator=generator, dtype=dtype) * 0.1
        v = torch.randn(shape, generator=generator, dtype=dtype) * 0.1
        a = torch.randn(shape, generator=generator, dtype=dtype) * 0.1
        b = torch.randn(shape, generator=generator, dtype=dtype) * 0.1
        state = torch.randn(2, 3, 4, 4, generator=generator, dtype=dtype) * 0.1
        return state, r, decay, k, v, a, b

    def test_full_chunked_and_decode_are_equivalent(self) -> None:
        values = self.signals()
        full, full_state = rwkv7_scan(*values)
        for chunk in (1, 2, 7, 16, 31):
            output, state = chunked_rwkv7_scan(*values, chunk_size=chunk)
            torch.testing.assert_close(output, full, rtol=0, atol=1e-12)
            torch.testing.assert_close(state, full_state, rtol=0, atol=1e-12)

    def test_zero_reset_and_native_decay_inverse(self) -> None:
        self.assertEqual(tuple(reset_state(2, 3, 4).shape), (2, 3, 4, 4))
        logits = torch.linspace(-8, 8, 64, dtype=torch.float64)
        torch.testing.assert_close(native_decay_logit(native_decay_from_logit(logits)), logits, rtol=1e-11, atol=1e-11)

    def test_conditionally_algebraic_gdn_matches_native_recurrence(self) -> None:
        generator = torch.Generator().manual_seed(7)
        shape = (2, 19, 3, 4)
        query = torch.randn(shape, generator=generator, dtype=torch.float64)
        key = torch.nn.functional.normalize(torch.randn(shape, generator=generator, dtype=torch.float64), dim=-1)
        value = torch.randn(shape, generator=generator, dtype=torch.float64)
        beta = torch.sigmoid(torch.randn((2, 19, 3, 1), generator=generator, dtype=torch.float64))
        decay = native_decay_from_logit(
            torch.randn(shape[:-1] + (1,), generator=generator, dtype=torch.float64)
        )
        state = torch.randn((2, 3, 4, 4), generator=generator, dtype=torch.float64)
        metrics = verify_gdn_mapping(state, decay, beta, query, key, value)
        self.assertLess(metrics["output_max_abs"], 1e-11)
        self.assertLess(metrics["state_max_abs"], 1e-11)
        bad_decay = torch.full(shape, 0.9, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "decay"):
            gdn_to_rwkv7_dynamics(bad_decay, beta, query, key, value)

    def test_gqa_mapping_and_baselines_are_explicit(self) -> None:
        mapping = build_group_map(8, 2)
        self.assertEqual(mapping.query_to_kv, (0, 0, 0, 0, 1, 1, 1, 1))
        weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        repeated = kv_repeat(weight, num_query_heads=8, num_kv_heads=2)
        expanded = kv_expand(weight, num_query_heads=8, num_kv_heads=2)
        self.assertEqual(repeated.shape, expanded.shape)
        self.assertFalse(torch.equal(repeated, expanded))

    def test_attention_fit_reports_every_head_and_gqa_group(self) -> None:
        generator = torch.Generator().manual_seed(9)
        inputs = torch.randn(64, 8, 4, generator=generator, dtype=torch.float64)
        weights = torch.randn(8, 4, 3, generator=generator, dtype=torch.float64)
        outputs = torch.einsum("thd,hdo->tho", inputs, weights)
        fits, rows = fit_headwise_teacher_trace(inputs, outputs, num_kv_heads=2)
        self.assertEqual(len(fits), 8)
        self.assertEqual([row.group for row in rows], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertTrue(all(row.normalized_mse < 1e-8 for row in rows))

    def test_attention_trace_contract_requires_position_mask_and_multiple_contexts(self) -> None:
        report = validate_attention_trace_contract(
            context_lengths=torch.tensor([4, 8, 16]),
            position_ids=torch.arange(16).repeat(3, 1),
            attention_mask=torch.ones(3, 16, dtype=torch.bool),
        )
        self.assertEqual(report["contexts"], [4, 8, 16])
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_attention_trace_contract(
                context_lengths=torch.tensor([8, 8, 8]),
                position_ids=torch.arange(8).repeat(3, 1),
                attention_mask=torch.ones(3, 8, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
