from __future__ import annotations

import unittest

import torch

from any2rwkv.migration import (
    build_group_map,
    fit_headwise_teacher_trace,
    fit_teacher_trace_no_bias,
    gdn_reference_scan,
    gdn_to_rwkv7_dynamics,
    kv_expand,
    kv_repeat,
    native_decay_fit_targets,
    solve_teacher_trace_normal_equations,
    teacher_trace_normal_equations,
    TraceNormalEquations,
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

    def test_decay_fit_targets_expand_heads_and_report_unreachable_values(self) -> None:
        reachable = native_decay_from_logit(torch.tensor([[-2.0, 2.0]]))
        targets = native_decay_fit_targets(reachable, target_channels=8)
        self.assertEqual(tuple(targets.logits.shape), (1, 8))
        torch.testing.assert_close(
            native_decay_from_logit(targets.logits),
            targets.expanded_source_decay,
        )
        self.assertEqual(targets.unreachable_fraction, 0.0)

        unreachable = native_decay_fit_targets(
            torch.tensor([[0.1, 0.9]]), target_channels=4
        )
        self.assertEqual(unreachable.unreachable_fraction, 0.5)
        self.assertTrue(torch.isfinite(unreachable.logits).all())

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
        metrics = verify_gdn_mapping(decay, beta, query, key, value)
        self.assertLess(metrics["output_max_abs"], 1e-11)
        bad_decay = torch.full(shape, 0.9, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "decay"):
            gdn_to_rwkv7_dynamics(bad_decay, beta, query, key, value)

    def test_exact_signal_mapping_accepts_full_gdn_decay_domain(self) -> None:
        query = torch.tensor([[[[2.0, -1.0]]]], dtype=torch.float64)
        key = torch.nn.functional.normalize(
            torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float64), dim=-1
        )
        value = torch.tensor([[[[5.0, -2.0]]]], dtype=torch.float64)
        beta = torch.tensor([[[[0.4]]]], dtype=torch.float64)
        for scalar_decay in (0.25, 1.0):
            decay = torch.tensor([[[[scalar_decay]]]], dtype=torch.float64)
            r, mapped_decay, mapped_key, mapped_value, a, b = (
                gdn_to_rwkv7_dynamics(decay, beta, query, key, value)
            )
            torch.testing.assert_close(
                r,
                torch.nn.functional.normalize(query, dim=-1) / 2**0.5,
            )
            self.assertFalse(torch.equal(r, query / 2**0.5))
            torch.testing.assert_close(mapped_decay, decay.expand_as(key))
            torch.testing.assert_close(mapped_key, key)
            torch.testing.assert_close(mapped_value, beta * value)
            torch.testing.assert_close(a, -key)
            torch.testing.assert_close(b, beta * decay * key)

    def test_native_gdn_factorization_keeps_write_key_and_folds_decay_into_erase(self) -> None:
        generator = torch.Generator().manual_seed(71)
        shape = (1, 7, 2, 4)
        query = torch.randn(shape, generator=generator, dtype=torch.float64)
        key = torch.nn.functional.normalize(
            torch.randn(shape, generator=generator, dtype=torch.float64), dim=-1
        )
        value = torch.randn(shape, generator=generator, dtype=torch.float64)
        beta = torch.sigmoid(
            torch.randn(shape[:-1] + (1,), generator=generator, dtype=torch.float64)
        )
        decay = native_decay_from_logit(
            torch.randn(shape[:-1] + (1,), generator=generator, dtype=torch.float64)
        )
        state = torch.zeros(1, 2, 4, 4, dtype=torch.float64)
        normalized_query = torch.nn.functional.normalize(query, dim=-1)
        source_output, _ = gdn_reference_scan(
            state, decay, beta, normalized_query, key, value
        )

        r = normalized_query / 2.0
        expanded_decay = decay.expand_as(key)
        expanded_beta = beta.expand_as(key)
        legacy_output, _ = rwkv7_scan(
            state,
            r,
            expanded_decay,
            key * expanded_beta,
            value,
            -key,
            key * expanded_beta,
        )
        exact_output, _ = rwkv7_scan(
            state,
            r,
            expanded_decay,
            key,
            value * beta,
            -key,
            key * beta * decay,
        )

        torch.testing.assert_close(exact_output, source_output, rtol=0, atol=1e-12)
        self.assertGreater(
            float((legacy_output - source_output).abs().max()),
            1e-6,
            "k_a=1 and erase=beta must not be accepted as exact when decay<1",
        )

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

    def test_bias_free_trace_fit_matches_the_projection_it_can_install(self) -> None:
        generator = torch.Generator().manual_seed(19)
        inputs = torch.randn(96, 7, generator=generator, dtype=torch.float64)
        weight = torch.randn(5, 7, generator=generator, dtype=torch.float64)
        outputs = inputs @ weight.T
        fit = fit_teacher_trace_no_bias(inputs, outputs, ridge=1e-8)
        torch.testing.assert_close(fit.weight.double(), weight, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(fit.bias, torch.zeros_like(fit.bias))
        # FP32 normal-equation metrics are reconstructed from additive sums;
        # near-zero residuals therefore retain small cancellation error.
        self.assertLess(fit.normalized_mse, 1e-6)
        self.assertGreater(fit.cosine, 0.999999)

        left = teacher_trace_normal_equations(inputs[:41], outputs[:41])
        right = teacher_trace_normal_equations(inputs[41:], outputs[41:])
        merged = TraceNormalEquations(
            left.gram + right.gram,
            left.rhs + right.rhs,
            left.target_squared_sum + right.target_squared_sum,
            left.tokens + right.tokens,
        )
        distributed_fit = solve_teacher_trace_normal_equations(merged, ridge=1e-8)
        torch.testing.assert_close(distributed_fit.weight, fit.weight)
        self.assertLess(
            abs(distributed_fit.normalized_mse - fit.normalized_mse), 1e-6
        )

    def test_eight_shard_augmented_statistics_recover_decay_up_and_bias(self) -> None:
        generator = torch.Generator().manual_seed(23)
        features = torch.randn(128, 4, generator=generator, dtype=torch.float64)
        weight = torch.randn(6, 4, generator=generator, dtype=torch.float64)
        bias = torch.randn(6, generator=generator, dtype=torch.float64)
        outputs = features @ weight.T + bias
        design = torch.cat(
            (features, torch.ones(128, 1, dtype=features.dtype)), dim=1
        )
        shards = [
            teacher_trace_normal_equations(design[index::8], outputs[index::8])
            for index in range(8)
        ]
        merged = TraceNormalEquations(
            sum((row.gram for row in shards), torch.zeros_like(shards[0].gram)),
            sum((row.rhs for row in shards), torch.zeros_like(shards[0].rhs)),
            sum(
                (row.target_squared_sum for row in shards),
                torch.zeros_like(shards[0].target_squared_sum),
            ),
            sum(row.tokens for row in shards),
        )
        fit = solve_teacher_trace_normal_equations(merged, ridge=1e-8)
        torch.testing.assert_close(
            fit.weight[:, :-1].double(), weight, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            fit.weight[:, -1].double(), bias, rtol=1e-5, atol=1e-5
        )

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
