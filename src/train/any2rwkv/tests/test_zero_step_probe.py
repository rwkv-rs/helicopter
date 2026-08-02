from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from any2rwkv.configuration_any2rwkv import Any2RWKV7Config
from any2rwkv.contract import build_target_config
from any2rwkv.fixture import tiny_qwen35_config
from any2rwkv.kernel import Rwkv7OperatorAdapter
from any2rwkv.mixer import ProjectionBoundaryRWKV7Attention, apply_partial_rope
from any2rwkv.recipes.qwen35_to_rwkv7.gqa_zero_step import (
    GQANativeFitConfig,
    GQANativeFitTrace,
    _AffineSufficientStatistics,
    _fit_native_projection,
    _native_parameter_set,
    _select_bias_free_statistics,
    _solve_affine_statistics,
    estimate_gqa_native_streamed_peak_bytes,
    fit_gqa_native_zero_step,
    validate_gqa_native_fit_trace,
)
from any2rwkv.recurrent import rwkv7_step
from any2rwkv.zero_step_probe import (
    RWKV7_MINIMUM_DECAY,
    TwoStateProjection,
    affine_state_rollout,
    causal_attention,
    fit_bias_free_projection,
    fit_low_rank_projection,
    fit_streamed_affine_model,
    four_state_block_output,
    iter_streamed_native_two_state_steps,
    logit_taylor_hazards,
    materialize_native_projection,
    native_signal_output_rollout,
    native_signal_rollout,
    native_two_state_rollout,
    observable_query_bases,
    probability_tangent_parameters,
    probability_taylor_hazards,
    qwen35_l2_normalize,
    rollout_hazards,
    rope_aligned_two_state_bases,
    rope_aligned_two_state_bases_streamed,
    select_bias_free_projection,
    streamed_hazard_attention,
    two_state_outputs,
    two_state_projection,
    verify_gdn_mapping,
)


def reference_rwkv7_operator(
    r,
    log_decay,
    k,
    v,
    a,
    b,
    *,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    state_indices=None,
    mode,
):
    assert output_final_state is True
    assert cu_seqlens is None
    assert cu_seqlens_cpu is None
    assert state_indices is None
    assert mode == "fp32io16"
    _batch, tokens, _heads, _head_dim = r.shape
    outputs = []
    current = initial_state
    for index in range(tokens):
        rt, wt, kt, vt, at, bt = (
            value[:, index] for value in (r, log_decay, k, v, a, b)
        )
        projection = torch.einsum("bhk,bhkv->bhv", at.float(), current)
        current = (
            wt.float().exp().unsqueeze(-1) * current
            + bt.float().unsqueeze(-1) * projection.unsqueeze(-2)
            + kt.float().unsqueeze(-1) * vt.float().unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", rt.float(), current))
    return torch.stack(outputs, dim=1).to(r.dtype), current


class _ThreadCollective:
    """Lock-step tensor collectives for deterministic CPU shard tests."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.condition = threading.Condition()
        self.steps = [0] * world_size
        self.pending = {}
        self.results = {}

    def reduce(self, rank: int, value: torch.Tensor, operation: str):
        with self.condition:
            step = self.steps[rank]
            self.steps[rank] += 1
            pending = self.pending.setdefault(
                step,
                {"operation": operation, "values": {}},
            )
            if pending["operation"] != operation:
                raise AssertionError("distributed collective order diverged")
            pending["values"][rank] = value.detach().clone()
            if len(pending["values"]) == self.world_size:
                ordered = [
                    pending["values"][index]
                    for index in range(self.world_size)
                ]
                if operation == "sum":
                    result = torch.stack(ordered).sum(dim=0)
                elif operation == "max":
                    result = torch.stack(ordered).amax(dim=0)
                else:
                    raise AssertionError(f"unknown operation: {operation}")
                self.results[step] = result
                self.condition.notify_all()
            ready = self.condition.wait_for(
                lambda: step in self.results,
                timeout=20,
            )
            if not ready:
                raise AssertionError(
                    f"distributed collective {step} did not converge"
                )
            return self.results[step].clone()


def test_exact_hazard_rollout_recovers_causal_attention() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    value = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)

    exact = causal_attention(query, key, value)
    recurrent = rollout_hazards(exact.hazards, value)

    torch.testing.assert_close(recurrent, exact.output, rtol=1e-5, atol=1e-6)


def test_streamed_hazard_attention_matches_dense_without_square_grid() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(3, 7, 2, 8, generator=generator)
    key = torch.randn(3, 7, 2, 8, generator=generator)
    value = torch.randn(3, 7, 2, 8, generator=generator)
    center = query[:2].mean(dim=(0, 1))

    dense = causal_attention(query, key, value)
    dense_bounded_hazard = logit_taylor_hazards(query, key, center)
    streamed = streamed_hazard_attention(
        query,
        key,
        value,
        center,
        development_row_start=2,
        supervised_token_start=1,
    )

    torch.testing.assert_close(
        streamed.exact_output,
        dense.output,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        streamed.bounded_output,
        rollout_hazards(dense_bounded_hazard, value),
        rtol=1e-5,
        atol=1e-6,
    )
    assert streamed.exact_rollout_metrics["nmse"] < 1e-12
    assert (
        0
        <= streamed.bounded_hazard_metrics[
            "outside_unit_interval_fraction"
        ]
        <= 1
    )


def test_affine_oracle_replays_frozen_closure_without_final_refit() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(4, 6, 2, 8, generator=generator)
    grouped_key = torch.randn(4, 6, 1, 8, generator=generator)
    grouped_value = torch.randn(4, 6, 1, 8, generator=generator)
    center = query[:2].mean(dim=(0, 1, 2), keepdim=False).unsqueeze(0)

    fitted = affine_state_rollout(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=2,
        fit_rank_one_closure=True,
    )
    replayed = affine_state_rollout(
        query[2:],
        grouped_key[2:],
        grouped_value[2:],
        center,
        calibration_batches=1,
        fit_rank_one_closure=False,
        fixed_closure_scale=fitted.closure_scale,
    )

    torch.testing.assert_close(replayed.output, fitted.output[2:])
    torch.testing.assert_close(replayed.states, fitted.states[2:])
    torch.testing.assert_close(
        replayed.closure_scale,
        fitted.closure_scale,
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="cannot fit and replay"):
        affine_state_rollout(
            query[2:],
            grouped_key[2:],
            grouped_value[2:],
            center,
            calibration_batches=1,
            fit_rank_one_closure=True,
            fixed_closure_scale=fitted.closure_scale,
        )


def test_streamed_affine_closure_matches_dense_without_time_major_state() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(4, 7, 4, 8, generator=generator)
    grouped_key = torch.randn(4, 7, 2, 8, generator=generator)
    grouped_value = torch.randn(4, 7, 2, 8, generator=generator)
    center = torch.stack(
        (
            query[:3, :, :2].mean(dim=(0, 1, 2)),
            query[:3, :, 2:].mean(dim=(0, 1, 2)),
        )
    )

    dense = affine_state_rollout(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=3,
        fit_rank_one_closure=True,
    )
    streamed = fit_streamed_affine_model(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=3,
        row_chunk_size=1,
    )

    torch.testing.assert_close(
        streamed.closure_scale,
        dense.closure_scale,
        rtol=1e-5,
        atol=1e-6,
    )
    assert streamed.peak_state_elements == 3 * 2 * 8 * 8
    assert streamed.peak_state_elements < dense.states.numel()


def test_streamed_rope_basis_matches_dense_candidate_selection() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(4, 6, 4, 8, generator=generator)
    grouped_key = torch.randn(4, 6, 2, 8, generator=generator)
    grouped_value = torch.randn(4, 6, 2, 8, generator=generator)
    center = torch.stack(
        (
            query[:3, :, :2].mean(dim=(0, 1, 2)),
            query[:3, :, 2:].mean(dim=(0, 1, 2)),
        )
    )
    dense_affine = affine_state_rollout(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=3,
        fit_rank_one_closure=True,
    )
    streamed_affine = fit_streamed_affine_model(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=3,
        row_chunk_size=2,
    )

    dense = rope_aligned_two_state_bases(
        dense_affine.states,
        dense_affine.centered_query,
        calibration_batches=3,
        native_dim=4,
        rotary_dim=2,
        steps=0,
    )
    streamed = rope_aligned_two_state_bases_streamed(
        query,
        grouped_key,
        grouped_value,
        streamed_affine,
        native_dim=4,
        rotary_dim=2,
        steps=0,
    )

    dense_projectors = dense @ dense.transpose(-1, -2)
    streamed_projectors = streamed @ streamed.transpose(-1, -2)
    torch.testing.assert_close(
        streamed_projectors,
        dense_projectors,
        rtol=2e-4,
        atol=2e-5,
    )


def test_streamed_rope_basis_matches_dense_optimizer_across_time_blocks() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(3, 18, 2, 8, generator=generator)
    grouped_key = torch.randn(3, 18, 1, 8, generator=generator)
    grouped_value = torch.randn(3, 18, 1, 8, generator=generator)
    center = query[:2].mean(dim=(0, 1, 2)).unsqueeze(0)
    dense_affine = affine_state_rollout(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=2,
        fit_rank_one_closure=True,
    )
    streamed_affine = fit_streamed_affine_model(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=2,
        row_chunk_size=1,
    )

    dense = rope_aligned_two_state_bases(
        dense_affine.states,
        dense_affine.centered_query,
        calibration_batches=2,
        native_dim=4,
        rotary_dim=2,
        steps=3,
        learning_rate=0.02,
    )
    streamed = rope_aligned_two_state_bases_streamed(
        query,
        grouped_key,
        grouped_value,
        streamed_affine,
        native_dim=4,
        rotary_dim=2,
        steps=3,
        learning_rate=0.02,
    )

    torch.testing.assert_close(
        streamed @ streamed.transpose(-1, -2),
        dense @ dense.transpose(-1, -2),
        rtol=3e-4,
        atol=3e-5,
    )


def test_streamed_native_steps_match_dense_two_state_rollout() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(3, 5, 2, 8, generator=generator)
    grouped_key = torch.randn(3, 5, 1, 8, generator=generator)
    grouped_value = torch.randn(3, 5, 1, 8, generator=generator)
    center = query[:2].mean(dim=(0, 1, 2)).unsqueeze(0)
    dense_affine = affine_state_rollout(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=2,
        fit_rank_one_closure=True,
    )
    model = fit_streamed_affine_model(
        query,
        grouped_key,
        grouped_value,
        center,
        calibration_batches=2,
        row_chunk_size=1,
    )
    basis = rope_aligned_two_state_bases(
        dense_affine.states,
        dense_affine.centered_query,
        calibration_batches=2,
        native_dim=4,
        rotary_dim=2,
        steps=0,
    )
    projection = two_state_projection(
        dense_affine.states,
        dense_affine.bias,
        query,
        basis,
        dc_indices=(3, 3),
        query_center=center.expand(2, -1),
    )
    from any2rwkv.zero_step_probe import probability_tangent_parameters

    probability, slope = probability_tangent_parameters(grouped_key, center)
    dense_native = native_two_state_rollout(
        projection,
        probability,
        slope,
    )
    collected = {
        "affine_output": torch.empty_like(dense_affine.output),
        "compressed_output": torch.empty_like(projection.output),
        "read": torch.empty_like(projection.read.flatten(2, 3)),
        "requested_decay": torch.empty_like(dense_native.requested_decay),
        "decay": torch.empty_like(dense_native.decay),
        "erase": torch.empty_like(dense_native.erase),
        "key": torch.empty_like(dense_native.key),
        "value": torch.empty_like(dense_native.value),
        "free_running_output": torch.empty_like(
            dense_native.free_running_output
        ),
    }
    for step in iter_streamed_native_two_state_steps(
        query,
        grouped_key,
        grouped_value,
        model,
        basis,
        native_dim=4,
        dc_indices=(3, 3),
    ):
        rows = slice(step.row_start, step.row_stop)
        for name, output in collected.items():
            output[rows, step.time_index] = getattr(step, name)

    expected = {
        "affine_output": dense_affine.output,
        "compressed_output": projection.output,
        "read": projection.read.flatten(2, 3),
        "requested_decay": dense_native.requested_decay,
        "decay": dense_native.decay,
        "erase": dense_native.erase,
        "key": dense_native.key,
        "value": dense_native.value,
        "free_running_output": dense_native.free_running_output,
    }
    for name, value in collected.items():
        torch.testing.assert_close(
            value,
            expected[name],
            rtol=2e-4,
            atol=2e-5,
        )


def test_streamed_peak_has_no_time_major_operator_or_square_hazard_term() -> None:
    short = estimate_gqa_native_streamed_peak_bytes(
        fit_rows=32,
        context_length=128,
        hidden_size=2048,
        query_heads=8,
        key_value_heads=2,
        source_head_dim=256,
        native_head_dim=128,
        row_chunk_size=1,
    )
    long = estimate_gqa_native_streamed_peak_bytes(
        fit_rows=32,
        context_length=256,
        hidden_size=2048,
        query_heads=8,
        key_value_heads=2,
        source_head_dim=256,
        native_head_dim=128,
        row_chunk_size=1,
    )

    assert short["affine_closure_bytes"] == long["affine_closure_bytes"]
    assert (
        long["affine_vector_workspace_bytes"]
        == 2 * short["affine_vector_workspace_bytes"]
    )
    assert (
        short["streamed_native_workspace_bytes"]
        == long["streamed_native_workspace_bytes"]
    )
    assert short["observable_gram_bytes"] == long["observable_gram_bytes"]
    assert long["fp32_trace_bytes"] == 2 * short["fp32_trace_bytes"]
    assert (
        long["hazard_output_bytes"]
        == 2 * short["hazard_output_bytes"]
    )
    wider_row_chunk = estimate_gqa_native_streamed_peak_bytes(
        fit_rows=32,
        context_length=128,
        hidden_size=2048,
        query_heads=8,
        key_value_heads=2,
        source_head_dim=256,
        native_head_dim=128,
        row_chunk_size=4,
    )
    assert (
        wider_row_chunk["streamed_native_workspace_bytes"]
        == 4 * short["streamed_native_workspace_bytes"]
    )
    for name in short:
        if name in {
            "estimated_peak_bytes",
            "streamed_native_workspace_bytes",
        }:
            continue
        assert wider_row_chunk[name] == short[name]


def test_streamed_normal_equations_match_dense_ridge_selection() -> None:
    generator = torch.Generator().manual_seed(20260725)
    source = torch.randn(6, 5, 7, generator=generator)
    target = torch.randn(6, 5, 9, generator=generator)
    prior = torch.randn(9, 7, generator=generator) * 0.05
    ridges = (1e-3, 1e-2, 1e-1)
    dense = select_bias_free_projection(
        source,
        target,
        calibration_batches=4,
        ridges=ridges,
        prior_weight=prior,
    )
    statistics = {
        name: _AffineSufficientStatistics.zeros(
            7,
            9,
            device=source.device,
        )
        for name in ("fit", "selection", "calibration")
    }
    statistics["fit"].add(source[:2], target[:2])
    statistics["selection"].add(source[2:4], target[2:4])
    statistics["calibration"].add(source[:4], target[:4])
    streamed = _select_bias_free_statistics(
        fit=statistics["fit"],
        selection=statistics["selection"],
        calibration=statistics["calibration"],
        ridges=ridges,
        prior_weight=prior,
    )

    assert streamed.ridge == dense.ridge
    torch.testing.assert_close(
        streamed.projection.weight,
        dense.projection.weight,
        rtol=2e-4,
        atol=2e-5,
    )
    for ridge in ridges:
        assert streamed.selection_nmse[ridge] == pytest.approx(
            dense.selection_nmse[ridge],
            rel=2e-4,
            abs=2e-6,
        )


def test_streamed_affine_statistics_are_offset_and_chunk_stable() -> None:
    generator = torch.Generator().manual_seed(20260727)
    source = (
        10_000
        + torch.randn(8, 16, 32, generator=generator) * 0.1
    )
    true_weight = torch.randn(12, 32, generator=generator)
    target = (source - 10_000) @ true_weight.T + 3
    predictions = []
    for row_chunk_size in (8, 2, 1):
        statistics = _AffineSufficientStatistics.zeros(
            32,
            12,
            device=source.device,
        )
        for row_start in range(0, source.shape[0], row_chunk_size):
            row_stop = min(
                source.shape[0],
                row_start + row_chunk_size,
            )
            statistics.add(
                source[row_start:row_stop],
                target[row_start:row_stop],
            )
        weight, bias = _solve_affine_statistics(
            statistics,
            ridge=1e-2,
        )
        prediction = source @ weight.T + bias
        predictions.append(prediction)
        assert float((prediction - target).square().mean()) < 1e-4

    torch.testing.assert_close(predictions[0], predictions[1])
    torch.testing.assert_close(predictions[0], predictions[2])


def test_streamed_affine_statistics_merge_matches_fresh_accumulation() -> None:
    generator = torch.Generator().manual_seed(20260727)
    source = torch.randn(7, 5, generator=generator) + 1_000
    target = torch.randn(7, 3, generator=generator) - 2_000
    left = _AffineSufficientStatistics.zeros(5, 3, device=source.device)
    right = _AffineSufficientStatistics.zeros(5, 3, device=source.device)
    expected = _AffineSufficientStatistics.zeros(5, 3, device=source.device)
    left.add(source[:3], target[:3])
    right.add(source[3:], target[3:])
    expected.add(source, target)

    left.merge_target_(right)
    left.source.merge_(right.source)

    torch.testing.assert_close(left.count, expected.count)
    torch.testing.assert_close(left.sum_source, expected.sum_source)
    torch.testing.assert_close(left.source_gram, expected.source_gram)
    torch.testing.assert_close(left.target_count, expected.target_count)
    torch.testing.assert_close(
        left.target_sum_source,
        expected.target_sum_source,
    )
    torch.testing.assert_close(left.sum_target, expected.sum_target)
    torch.testing.assert_close(
        left.source_target,
        expected.source_target,
    )
    torch.testing.assert_close(left.target_square, expected.target_square)


def test_probability_and_logit_tangents_agree_at_zero() -> None:
    generator = torch.Generator().manual_seed(20260725)
    key = torch.randn(2, 6, 3, 8, generator=generator)
    zero_query = torch.zeros_like(key)
    center = torch.zeros(3, 8)

    probability = probability_taylor_hazards(zero_query, key)
    logit = logit_taylor_hazards(zero_query, key, center)

    torch.testing.assert_close(probability, logit, rtol=1e-5, atol=1e-6)


def test_four_state_blocks_recover_full_operator() -> None:
    generator = torch.Generator().manual_seed(20260725)
    state = torch.randn(2, 5, 16, 16, generator=generator)
    query = torch.randn(2, 5, 16, generator=generator)

    direct = torch.einsum("...od,...d->...o", state, query)
    blocked = four_state_block_output(state, query)

    torch.testing.assert_close(blocked, direct, rtol=1e-5, atol=1e-6)


def test_gdn_activation_oracle_maps_exactly_to_rwkv7() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    key = qwen35_l2_normalize(
        torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    )
    value = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    decay = torch.rand(2, 7, 3, 1, generator=generator, dtype=torch.float64)
    beta = torch.rand(2, 7, 3, 1, generator=generator, dtype=torch.float64)

    metrics = verify_gdn_mapping(decay, beta, query, key, value)

    assert metrics["output_max_abs"] < 1e-12
    assert metrics["output_relative_l2"] < 1e-12


def test_two_states_fit_one_dc_and_half_minus_one_query_features() -> None:
    generator = torch.Generator().manual_seed(20260725)
    states = torch.randn(2, 5, 1, 8, 8, generator=generator)
    query = torch.randn(2, 5, 1, 8, generator=generator)
    bias = torch.randn(2, 5, 1, 8, generator=generator)
    basis, _ = torch.linalg.qr(
        torch.randn(8, 3, generator=generator),
        mode="reduced",
    )

    _, sketched = two_state_outputs(states, query, basis.unsqueeze(0))
    read = torch.cat(
        (
            torch.ones(2, 5, 1, 1),
            torch.einsum("dr,bthd->bthr", basis, query),
        ),
        dim=-1,
    )
    compressed_state = torch.cat(
        (
            bias.unsqueeze(-1),
            torch.einsum("btgod,dr->btgor", states, basis),
        ),
        dim=-1,
    )
    top, bottom = compressed_state.split(4, dim=-2)
    explicit = torch.cat(
        (
            torch.einsum("bthod,bthd->btho", top, read),
            torch.einsum("bthod,bthd->btho", bottom, read),
        ),
        dim=-1,
    )

    torch.testing.assert_close(
        explicit,
        sketched + bias,
        rtol=1e-5,
        atol=1e-6,
    )


def test_observable_basis_optimizes_future_read_instead_of_query_variance() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(2, 12, 1, 4, generator=generator)
    query[..., 0] *= 20
    state = torch.zeros(2, 12, 1, 4, 4)
    state[..., 0, 3] = 1

    basis = observable_query_bases(
        state,
        query,
        calibration_batches=2,
        rank=1,
        steps=160,
        learning_rate=0.05,
    )

    selected_direction = basis[0, :, 0].abs()
    assert int(selected_direction.argmax()) == 3
    assert float(selected_direction[3]) > 0.99


def test_observable_basis_validates_strict_state_budget() -> None:
    state = torch.zeros(1, 2, 1, 8, 8)
    query = torch.zeros(1, 2, 1, 8)

    with pytest.raises(ValueError, match="rank must fit the input dimension"):
        observable_query_bases(
            state,
            query,
            calibration_batches=1,
            rank=9,
        )


def test_two_state_projection_materializes_native_state_and_read_shapes() -> None:
    generator = torch.Generator().manual_seed(20260725)
    states = torch.randn(2, 5, 1, 8, 8, generator=generator)
    bias = torch.randn(2, 5, 1, 8, generator=generator)
    query = torch.randn(2, 5, 1, 8, generator=generator)
    basis, _ = torch.linalg.qr(
        torch.randn(8, 3, generator=generator),
        mode="reduced",
    )

    projection = two_state_projection(
        states,
        bias,
        query,
        basis.unsqueeze(0),
    )

    assert projection.states.shape == (2, 5, 1, 2, 4, 4)
    assert projection.read.shape == (2, 5, 1, 2, 4)
    assert projection.output.shape == (2, 5, 1, 8)
    full, matrix_only = two_state_outputs(
        states,
        query,
        basis.unsqueeze(0),
    )
    torch.testing.assert_close(projection.output, matrix_only + bias)
    assert full.shape == projection.output.shape


def test_two_state_projection_uses_independent_bases_and_dc_channels() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(2, 5, 1, 8, generator=generator)
    states = torch.zeros(2, 5, 1, 8, 8)
    states[..., :4, :3] = torch.randn(
        2,
        5,
        1,
        4,
        3,
        generator=generator,
    )
    states[..., 4:, 4:7] = torch.randn(
        2,
        5,
        1,
        4,
        3,
        generator=generator,
    )
    bias = torch.randn(2, 5, 1, 8, generator=generator)
    bases = torch.zeros(1, 2, 8, 3)
    bases[0, 0, :3] = torch.eye(3)
    bases[0, 1, 4:7] = torch.eye(3)

    projection = two_state_projection(
        states,
        bias,
        query,
        bases,
        dc_indices=(3, 3),
    )
    exact = torch.einsum(
        "btgod,btd->btgo",
        states,
        query[:, :, 0],
    ) + bias

    torch.testing.assert_close(projection.output, exact)
    torch.testing.assert_close(projection.read[..., 3], torch.ones(2, 5, 1, 2))
    assert projection.query_basis.shape == (1, 2, 8, 3)


def test_two_state_projection_dc_matches_centered_observable_objective() -> None:
    generator = torch.Generator().manual_seed(20260725)
    states = torch.randn(2, 4, 1, 8, 8, generator=generator)
    bias = torch.randn(2, 4, 1, 8, generator=generator)
    query_center = torch.tensor(
        [[3.0, -2.0, 1.0, 4.0, -3.0, 2.0, 5.0, -1.0]]
    )
    centered_query = torch.randn(2, 4, 1, 8, generator=generator)
    query = centered_query + query_center.view(1, 1, 1, 8)
    bases = torch.zeros(1, 2, 8, 3)
    bases[0, 0, :3] = torch.eye(3)
    bases[0, 1, 4:7] = torch.eye(3)

    projection = two_state_projection(
        states,
        bias,
        query,
        bases,
        dc_indices=(3, 3),
        query_center=query_center,
    )
    expected_halves = []
    for state_index in range(2):
        start = state_index * 4
        stop = start + 4
        basis = bases[0, state_index]
        projected_centered_query = (
            centered_query[:, :, 0] @ basis
        ) @ basis.T
        expected_halves.append(
            torch.einsum(
                "btod,btd->bto",
                states[:, :, 0, start:stop],
                projected_centered_query,
            )
            + bias[:, :, 0, start:stop]
        )
    expected = torch.cat(expected_halves, dim=-1).unsqueeze(2)

    torch.testing.assert_close(projection.output, expected)


def test_rope_aligned_two_state_bases_respect_rotary_reachability() -> None:
    generator = torch.Generator().manual_seed(20260725)
    states = torch.randn(2, 5, 1, 8, 8, generator=generator)
    query = torch.randn(2, 5, 1, 8, generator=generator)

    bases = rope_aligned_two_state_bases(
        states,
        query,
        calibration_batches=2,
        native_dim=4,
        rotary_dim=2,
        steps=0,
    )

    assert bases.shape == (1, 2, 8, 3)
    torch.testing.assert_close(bases[0, 0, :2, :2], torch.eye(2))
    torch.testing.assert_close(bases[0, 0, 2:, :2], torch.zeros(6, 2))
    torch.testing.assert_close(bases[0, 1, :2], torch.zeros(2, 3))
    torch.testing.assert_close(
        bases[0, 0].T @ bases[0, 0],
        torch.eye(3),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        bases[0, 1].T @ bases[0, 1],
        torch.eye(3),
        rtol=1e-5,
        atol=1e-6,
    )


def test_logit_taylor_hazard_is_bounded_without_clipping() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(2, 7, 3, 8, generator=generator) * 100
    key = torch.randn(2, 7, 3, 8, generator=generator)
    center = query.mean(dim=(0, 1))

    hazard = logit_taylor_hazards(query, key, center)

    assert bool((hazard >= 0).all())
    assert bool((hazard <= 1).all())


def test_bias_free_projection_recovers_reachable_native_read() -> None:
    generator = torch.Generator().manual_seed(20260725)
    source = torch.randn(3, 7, 5, generator=generator)
    weight = torch.randn(8, 5, generator=generator)
    target = (source @ weight.T).reshape(3, 7, 2, 4)

    fitted = fit_bias_free_projection(
        source,
        target,
        calibration_batches=2,
        ridge=0,
    )

    torch.testing.assert_close(fitted.prediction, target, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(fitted.weight, weight, rtol=1e-4, atol=1e-5)


def test_bias_free_projection_exposes_unreachable_constant_channel() -> None:
    source = torch.zeros(2, 4, 3)
    target = torch.ones(2, 4, 1)

    fitted = fit_bias_free_projection(
        source,
        target,
        calibration_batches=1,
        ridge=1e-3,
    )

    torch.testing.assert_close(fitted.prediction, torch.zeros_like(target))


def test_bias_free_projection_preserves_unidentified_prior_directions() -> None:
    source = torch.zeros(2, 4, 3)
    prior = torch.tensor([[1.0, -2.0, 3.0]])
    target = torch.zeros(2, 4, 1)

    fitted = fit_bias_free_projection(
        source,
        target,
        calibration_batches=1,
        ridge=1e-3,
        prior_weight=prior,
    )

    torch.testing.assert_close(fitted.weight, prior)


def test_bias_free_projection_uses_stable_dual_solution_when_underdetermined() -> None:
    generator = torch.Generator().manual_seed(20260725)
    source = torch.randn(4, 3, 32, generator=generator)
    target = torch.randn(4, 3, 8, generator=generator)

    selected = select_bias_free_projection(
        source,
        target,
        calibration_batches=2,
        ridges=(1e-3, 1e-1, 10.0),
    )

    assert selected.ridge in {1e-3, 1e-1, 10.0}
    assert set(selected.selection_nmse) == {1e-3, 1e-1, 10.0}
    assert bool(torch.isfinite(selected.projection.weight).all())
    assert bool(torch.isfinite(selected.projection.prediction).all())
    assert selected.absolute_ridge == pytest.approx(
        selected.ridge * selected.ridge_scale
    )


def test_low_rank_projection_materializes_native_affine_parameters() -> None:
    generator = torch.Generator().manual_seed(20260725)
    source = torch.randn(4, 7, 6, generator=generator)
    down = torch.randn(3, 6, generator=generator)
    up = torch.randn(8, 3, generator=generator)
    bias = torch.randn(8, generator=generator)
    target = source @ down.T @ up.T + bias

    fitted = fit_low_rank_projection(
        source,
        target,
        calibration_batches=3,
        rank=3,
        ridge=1e-6,
    )

    assert fitted.down_weight.shape == (3, 6)
    assert fitted.up_weight.shape == (8, 3)
    assert fitted.bias.shape == (8,)
    torch.testing.assert_close(
        fitted.prediction,
        target,
        rtol=2e-3,
        atol=2e-3,
    )


def test_low_rank_projection_can_match_bias_free_native_gate_contract() -> None:
    generator = torch.Generator().manual_seed(20260725)
    source = torch.randn(3, 8, 5, generator=generator)
    target = torch.randn(3, 8, 7, generator=generator)

    fitted = fit_low_rank_projection(
        source,
        target,
        calibration_batches=2,
        rank=4,
        ridge=1e-3,
        hidden_activation="sigmoid",
        output_bias=False,
    )

    assert fitted.output_bias is False
    torch.testing.assert_close(fitted.bias, torch.zeros_like(fitted.bias))
    features = torch.sigmoid(source @ fitted.down_weight.T)
    torch.testing.assert_close(
        fitted.prediction,
        features @ fitted.up_weight.T,
    )


def test_native_signal_rollout_replays_exact_rwkv7_signals() -> None:
    generator = torch.Generator().manual_seed(20260725)
    read = torch.randn(2, 5, 3, 4, generator=generator)
    key = torch.randn(2, 5, 3, 4, generator=generator)
    value = torch.randn(2, 5, 3, 4, generator=generator)
    erase = torch.rand(2, 5, 3, 4, generator=generator)
    decay = torch.rand(2, 5, 3, 4, generator=generator) * 0.4 + 0.55

    actual = native_signal_rollout(read, decay, key, value, erase)
    state = torch.zeros(2, 3, 4, 4)
    expected_outputs = []
    expected_states = []
    for index in range(read.shape[1]):
        normalized_key = torch.nn.functional.normalize(key[:, index], dim=-1)
        output, state = rwkv7_step(
            state,
            read[:, index],
            decay[:, index],
            key[:, index],
            value[:, index],
            -normalized_key,
            normalized_key * erase[:, index],
        )
        expected_outputs.append(output)
        expected_states.append(state)

    torch.testing.assert_close(
        actual.output,
        torch.stack(expected_outputs, dim=1),
    )
    torch.testing.assert_close(
        native_signal_output_rollout(read, decay, key, value, erase),
        actual.output,
    )
    torch.testing.assert_close(
        actual.states,
        torch.stack(expected_states, dim=1),
    )


def test_native_projection_materialization_is_complete_atomic_and_hash_bound() -> None:
    source = tiny_qwen35_config(layers=2, moe=False)
    source["mtp_num_hidden_layers"] = 0
    config = Any2RWKV7Config(
        **build_target_config(source, require_final_layers=False)
    )
    mixer = ProjectionBoundaryRWKV7Attention(
        config,
        1,
        source_used_rope=False,
        rotary_dim=0,
        rope_theta=10_000.0,
    ).to(torch.bfloat16)
    before = {
        name: parameter.detach().clone()
        for name, parameter in mixer.named_parameters()
    }
    candidate = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in mixer.named_parameters()
    }
    candidate["k_k"].fill_(1)
    candidate["g_norm.weight"].fill_(1)
    candidate["v_lora.lora.2.bias"].fill_(
        torch.logit(
            torch.tensor(torch.finfo(torch.bfloat16).eps**2)
        )
    )

    incomplete = dict(candidate)
    incomplete.pop("r_proj.weight")
    with pytest.raises(ValueError, match="cover every module parameter"):
        materialize_native_projection(mixer, incomplete)
    for name, parameter in mixer.named_parameters():
        torch.testing.assert_close(parameter, before[name])

    malformed = dict(candidate)
    malformed["r_proj.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        materialize_native_projection(mixer, malformed)
    for name, parameter in mixer.named_parameters():
        torch.testing.assert_close(parameter, before[name])

    installed = materialize_native_projection(mixer, candidate)
    assert len(installed.aggregate_sha256) == 64
    assert {item.name for item in installed.tensors} == set(candidate)
    for name, parameter in mixer.named_parameters():
        torch.testing.assert_close(
            parameter,
            candidate[name].to(torch.bfloat16),
            rtol=0,
            atol=0,
        )

    repeated = materialize_native_projection(mixer, candidate)
    assert repeated.aggregate_sha256 == installed.aggregate_sha256
    assert repeated.tensors == installed.tensors


def test_materialized_nonfirst_native_projection_runs_real_bf16_sequence() -> None:
    source = tiny_qwen35_config(layers=2, moe=False)
    source["mtp_num_hidden_layers"] = 0
    config = Any2RWKV7Config(
        **build_target_config(source, require_final_layers=False)
    )
    mixer = ProjectionBoundaryRWKV7Attention(
        config,
        1,
        source_used_rope=False,
        rotary_dim=0,
        rope_theta=10_000.0,
    ).to(torch.bfloat16)
    candidate = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in mixer.named_parameters()
    }
    recurrent_width = config.attention_hidden_size
    candidate["r_proj.weight"].copy_(
        torch.eye(recurrent_width, config.hidden_size)
    )
    candidate["k_proj.weight"].copy_(
        torch.eye(recurrent_width, config.hidden_size)
    )
    candidate["v_proj.weight"].copy_(
        torch.eye(recurrent_width, config.hidden_size)
    )
    candidate["o_proj.weight"].copy_(
        torch.eye(config.hidden_size, recurrent_width)
    )
    candidate["k_k"].fill_(1)
    candidate["g_norm.weight"].fill_(1)
    candidate["g_lora.lora.2.weight"][:, :recurrent_width].copy_(
        torch.eye(recurrent_width) * 2
    )
    candidate["v_lora.lora.2.bias"].fill_(
        torch.logit(
            torch.tensor(torch.finfo(torch.bfloat16).eps**2)
        )
    )
    materialize_native_projection(mixer, candidate)

    generator = torch.Generator().manual_seed(20260725)
    values = torch.randn(
        2,
        16,
        config.hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    positions = torch.arange(16).view(1, -1).expand(2, -1)
    output, _, final_state, signals = mixer.forward_sequence(
        values,
        positions=positions,
        kernel=Rwkv7OperatorAdapter(
            reference_rwkv7_operator,
            lambda: "fla",
            head_size=config.head_dim,
            require_flash=False,
        ),
        v_first=torch.zeros(
            2,
            16,
            recurrent_width,
            dtype=torch.bfloat16,
        ),
    )

    assert output.shape == values.shape
    assert final_state.shape == (
        2,
        config.num_heads,
        config.head_dim,
        config.head_dim,
    )
    assert output.dtype == torch.bfloat16
    assert bool(torch.isfinite(output.float()).all())
    assert bool(torch.isfinite(final_state).all())
    assert float(signals["gate"].detach().abs().sum()) > 0


def test_gqa_native_fit_materializes_exact_module_shapes_and_runs_bf16() -> None:
    source = tiny_qwen35_config(layers=2, moe=False)
    source.update(
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "mtp_num_hidden_layers": 0,
        }
    )
    config = Any2RWKV7Config(
        **build_target_config(source, require_final_layers=False)
    )
    mixer = ProjectionBoundaryRWKV7Attention(
        config,
        1,
        source_used_rope=True,
        rotary_dim=2,
        rope_theta=10_000.0,
        rope_num_heads=2,
        rope_head_dim=8,
    ).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(20260725)
    batch, tokens = 5, 6
    mixer_input = torch.randn(
        batch,
        tokens,
        config.hidden_size,
        generator=generator,
    )
    query_weight = torch.randn(
        2 * 8 * 2,
        config.hidden_size,
        generator=generator,
    ) * 0.1
    key_weight = torch.randn(
        8,
        config.hidden_size,
        generator=generator,
    ) * 0.1
    value_weight = torch.randn(
        8,
        config.hidden_size,
        generator=generator,
    ) * 0.1
    output_weight = torch.randn(
        config.hidden_size,
        2 * 8,
        generator=generator,
    ) * 0.1
    packed_query = (mixer_input @ query_weight.T).reshape(
        batch,
        tokens,
        2,
        16,
    )
    query_pre_rope, gate_logits = packed_query.chunk(2, dim=-1)
    grouped_key_pre_rope = (mixer_input @ key_weight.T).reshape(
        batch,
        tokens,
        1,
        8,
    )
    grouped_value = (mixer_input @ value_weight.T).reshape(
        batch,
        tokens,
        1,
        8,
    )
    positions = torch.arange(tokens).view(1, -1).expand(batch, -1)
    query = apply_partial_rope(
        query_pre_rope,
        positions,
        rotary_dim=2,
        theta=10_000.0,
    )
    grouped_key = apply_partial_rope(
        grouped_key_pre_rope,
        positions,
        rotary_dim=2,
        theta=10_000.0,
    )
    key = grouped_key.repeat_interleave(2, dim=2)
    value = grouped_value.repeat_interleave(2, dim=2)
    gate = torch.sigmoid(gate_logits)
    exact_attention = causal_attention(query, key, value).output
    mixer_output = (exact_attention * gate).flatten(2) @ output_weight.T

    trace = GQANativeFitTrace(
        mixer_input=mixer_input,
        query=query,
        key=key,
        value=value,
        grouped_key=grouped_key,
        grouped_value=grouped_value,
        gate=gate,
        mixer_output=mixer_output,
        query_weight=query_weight,
        key_weight=key_weight,
        value_weight=value_weight,
        output_weight=output_weight,
        output_bias=None,
    )
    fit_config = GQANativeFitConfig(
        calibration_rows=3,
        positions=positions,
        source_head_dim=8,
        rotary_dim=2,
        rope_theta=10_000.0,
        supervised_token_start=2,
        observable_fit_steps=2,
    )
    fitted = fit_gqa_native_zero_step(
        mixer,
        trace,
        fit_config,
    )
    non_finite_output = mixer_output.clone()
    non_finite_output[0, 0, 0] = torch.nan
    with pytest.raises(
        ValueError,
        match="contains non-finite values",
    ):
        validate_gqa_native_fit_trace(
            mixer,
            GQANativeFitTrace(
                mixer_input=mixer_input,
                query=query,
                key=key,
                value=value,
                grouped_key=grouped_key,
                grouped_value=grouped_value,
                gate=gate,
                mixer_output=non_finite_output,
                query_weight=query_weight,
                key_weight=key_weight,
                value_weight=value_weight,
                output_weight=output_weight,
                output_bias=None,
            ),
            fit_config,
        )
    non_finite_query_weight = query_weight.clone()
    non_finite_query_weight[0, 0] = torch.inf
    with pytest.raises(
        ValueError,
        match="contains non-finite values",
    ):
        validate_gqa_native_fit_trace(
            mixer,
            GQANativeFitTrace(
                mixer_input=mixer_input,
                query=query,
                key=key,
                value=value,
                grouped_key=grouped_key,
                grouped_value=grouped_value,
                gate=gate,
                mixer_output=mixer_output,
                query_weight=non_finite_query_weight,
                key_weight=key_weight,
                value_weight=value_weight,
                output_weight=output_weight,
                output_bias=None,
            ),
            fit_config,
        )

    group_center = query[:3].mean(dim=(0, 1, 2), keepdim=False).unsqueeze(0)
    affine = fit_streamed_affine_model(
        query,
        grouped_key,
        grouped_value,
        group_center,
        calibration_batches=3,
    )
    basis = rope_aligned_two_state_bases_streamed(
        query,
        grouped_key,
        grouped_value,
        affine,
        native_dim=mixer.head_dim,
        rotary_dim=2,
        steps=2,
        learning_rate=fit_config.observable_fit_learning_rate,
    )
    dense_affine = affine_state_rollout(
        query,
        grouped_key,
        grouped_value,
        group_center,
        calibration_batches=3,
        fit_rank_one_closure=False,
        fixed_closure_scale=affine.closure_scale,
    )
    dense_projection = two_state_projection(
        dense_affine.states,
        dense_affine.bias,
        query,
        basis,
        dc_indices=(mixer.head_dim - 1, mixer.head_dim - 1),
        query_center=group_center.repeat_interleave(2, dim=0),
    )
    probability, slope = probability_tangent_parameters(
        grouped_key,
        group_center,
    )
    dense_transition = native_two_state_rollout(
        dense_projection,
        probability,
        slope,
    )
    dense_fit = _fit_native_projection(
        module=mixer,
        trace=trace,
        target_read=dense_projection.read.flatten(2, 3),
        native_transition=dense_transition,
        config=fit_config,
        native_head_dim=mixer.head_dim,
        query_basis=basis,
        dc_indices=(mixer.head_dim - 1, mixer.head_dim - 1),
    )
    dense_parameters = _native_parameter_set(
        mixer,
        dense_fit,
        source_value_weight=value_weight,
    )

    assert set(fitted.parameters) == {
        name for name, _ in mixer.named_parameters()
    }
    factorized_prefixes = ("w_lora.", "a_lora.", "g_lora.")
    for name, parameter in mixer.named_parameters():
        assert fitted.parameters[name].shape == parameter.shape
        assert bool(torch.isfinite(fitted.parameters[name]).all())
        if name.startswith(factorized_prefixes):
            continue
        torch.testing.assert_close(
            fitted.parameters[name],
            dense_parameters[name],
            rtol=3e-3,
            atol=3e-4,
            msg=name,
        )

    assert fitted.report["native_parameter_projection"][
        "ridge_multiplier"
    ] == dense_fit.report["ridge_multiplier"]
    assert fitted.report["native_parameter_projection"][
        "absolute_ridge"
    ] == pytest.approx(
        dense_fit.report["absolute_ridge"],
        rel=3e-3,
        abs=3e-7,
    )
    for metric_name in (
        "pre_post_rope_read_projection",
        "decay_signal",
        "erase_signal",
        "write_key_signal",
        "write_value_signal",
        "gate_signal",
        "complete_free_running_mixer",
    ):
        assert fitted.report["native_parameter_projection"][
            metric_name
        ] == pytest.approx(
            dense_fit.report[metric_name],
            rel=1e-2,
            abs=3e-5,
        )
    assert fitted.report["observable_state_compression"]["budget"] == {
        "states_per_source_head": 2,
        "dc_per_state": 1,
        "query_features_per_state": 3,
        "shared_input_subspace": False,
    }
    assert fitted.report["streaming_workspace"] == {
        "time_major_operator_state": False,
        "square_hazard_grid": False,
        "row_chunk_size": 1,
        "affine_peak_state_elements": 3 * 1 * 8 * 8,
    }
    installed = materialize_native_projection(mixer, fitted.parameters)
    assert len(installed.aggregate_sha256) == 64
    values = mixer_input.to(torch.bfloat16)
    output, _, final_state, _ = mixer.forward_sequence(
        values,
        positions=positions,
        kernel=Rwkv7OperatorAdapter(
            reference_rwkv7_operator,
            lambda: "fla",
            head_size=config.head_dim,
            require_flash=False,
        ),
        v_first=torch.zeros(
            batch,
            tokens,
            config.attention_hidden_size,
            dtype=torch.bfloat16,
        ),
    )
    assert output.shape == values.shape
    assert final_state.shape == (
        batch,
        config.num_heads,
        config.head_dim,
        config.head_dim,
    )
    assert bool(torch.isfinite(output.float()).all())
    assert bool(torch.isfinite(final_state).all())


def test_gqa_native_fit_two_uneven_shards_matches_unsharded() -> None:
    source = tiny_qwen35_config(layers=2, moe=False)
    source.update(
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "mtp_num_hidden_layers": 0,
        }
    )
    target_config = Any2RWKV7Config(
        **build_target_config(source, require_final_layers=False)
    )
    mixer = ProjectionBoundaryRWKV7Attention(
        target_config,
        1,
        source_used_rope=True,
        rotary_dim=2,
        rope_theta=10_000.0,
        rope_num_heads=2,
        rope_head_dim=8,
    ).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(20260726)
    batch, tokens = 5, 5
    mixer_input = torch.randn(
        batch,
        tokens,
        target_config.hidden_size,
        generator=generator,
    )
    query_weight = torch.randn(
        32,
        target_config.hidden_size,
        generator=generator,
    ) * 0.1
    key_weight = torch.randn(
        8,
        target_config.hidden_size,
        generator=generator,
    ) * 0.1
    value_weight = torch.randn(
        8,
        target_config.hidden_size,
        generator=generator,
    ) * 0.1
    output_weight = torch.randn(
        target_config.hidden_size,
        16,
        generator=generator,
    ) * 0.1
    packed_query = (mixer_input @ query_weight.T).reshape(
        batch,
        tokens,
        2,
        16,
    )
    query_pre_rope, gate_logits = packed_query.chunk(2, dim=-1)
    grouped_key_pre_rope = (mixer_input @ key_weight.T).reshape(
        batch,
        tokens,
        1,
        8,
    )
    grouped_value = (mixer_input @ value_weight.T).reshape(
        batch,
        tokens,
        1,
        8,
    )
    positions = torch.arange(tokens).view(1, -1).expand(batch, -1)
    query = apply_partial_rope(
        query_pre_rope,
        positions,
        rotary_dim=2,
        theta=10_000.0,
    )
    grouped_key = apply_partial_rope(
        grouped_key_pre_rope,
        positions,
        rotary_dim=2,
        theta=10_000.0,
    )
    key = grouped_key.repeat_interleave(2, dim=2)
    value = grouped_value.repeat_interleave(2, dim=2)
    gate = torch.sigmoid(gate_logits)
    mixer_output = (
        causal_attention(query, key, value).output * gate
    ).flatten(2) @ output_weight.T
    trace = GQANativeFitTrace(
        mixer_input=mixer_input,
        query=query,
        key=key,
        value=value,
        grouped_key=grouped_key,
        grouped_value=grouped_value,
        gate=gate,
        mixer_output=mixer_output,
        query_weight=query_weight,
        key_weight=key_weight,
        value_weight=value_weight,
        output_weight=output_weight,
        output_bias=None,
    )
    unsharded = fit_gqa_native_zero_step(
        mixer,
        trace,
        GQANativeFitConfig(
            calibration_rows=2,
            positions=positions,
            source_head_dim=8,
            rotary_dim=2,
            rope_theta=10_000.0,
            supervised_token_start=1,
            observable_fit_steps=1,
        ),
    )

    collective = _ThreadCollective(world_size=2)

    def fit_shard(rank: int):
        row_indices = tuple(range(rank, batch, 2))
        index = torch.tensor(row_indices, dtype=torch.long)

        def take(value):
            return value.index_select(0, index)

        shard_trace = GQANativeFitTrace(
            mixer_input=take(trace.mixer_input),
            query=take(trace.query),
            key=take(trace.key),
            value=take(trace.value),
            grouped_key=take(trace.grouped_key),
            grouped_value=take(trace.grouped_value),
            gate=take(trace.gate),
            mixer_output=take(trace.mixer_output),
            query_weight=trace.query_weight,
            key_weight=trace.key_weight,
            value_weight=trace.value_weight,
            output_weight=trace.output_weight,
            output_bias=None,
        )
        return fit_gqa_native_zero_step(
            mixer,
            shard_trace,
            GQANativeFitConfig(
                calibration_rows=2,
                positions=take(positions),
                source_head_dim=8,
                rotary_dim=2,
                rope_theta=10_000.0,
                supervised_token_start=1,
                observable_fit_steps=1,
                global_row_indices=row_indices,
                reduce_sum=lambda value: collective.reduce(
                    rank,
                    value,
                    "sum",
                ),
                reduce_max=lambda value: collective.reduce(
                    rank,
                    value,
                    "max",
                ),
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(fit_shard, rank) for rank in range(2)]
        sharded = [future.result(timeout=30) for future in futures]

    assert collective.steps[0] == collective.steps[1]
    factorized_prefixes = ("w_lora.", "a_lora.", "g_lora.")
    for name in unsharded.parameters:
        torch.testing.assert_close(
            sharded[0].parameters[name],
            sharded[1].parameters[name],
            rtol=1e-5,
            atol=1e-6,
            msg=name,
        )
        if name.startswith(factorized_prefixes):
            continue
        torch.testing.assert_close(
            sharded[0].parameters[name],
            unsharded.parameters[name],
            rtol=5e-3,
            atol=5e-4,
            msg=name,
        )

    def factorized_output(parameters, prefix: str, activation):
        hidden = activation(
            mixer_input @ parameters[f"{prefix}.lora.0.weight"].T
        )
        output = hidden @ parameters[f"{prefix}.lora.2.weight"].T
        return output + parameters.get(f"{prefix}.lora.2.bias", 0)

    for prefix, activation in (
        ("w_lora", torch.tanh),
        ("a_lora", lambda value: value),
        ("g_lora", torch.sigmoid),
    ):
        torch.testing.assert_close(
            factorized_output(
                sharded[0].parameters,
                prefix,
                activation,
            ),
            factorized_output(
                unsharded.parameters,
                prefix,
                activation,
            ),
            rtol=1e-2,
            atol=1e-3,
            msg=prefix,
        )
    for metric_name in (
        "bounded_hazard_surrogate",
        "observable_state_compression",
        "native_transition",
        "native_parameter_projection",
    ):
        assert sharded[0].report[metric_name] == sharded[1].report[
            metric_name
        ]


def test_native_two_state_rollout_recovers_reachable_no_erase_sequence() -> None:
    generator = torch.Generator().manual_seed(20260725)
    batch, tokens, source_heads, native_dim = 2, 6, 1, 4
    source_dim = native_dim * 2
    basis = torch.eye(source_dim)[:, : native_dim - 1].unsqueeze(0)
    probability = torch.rand(
        batch,
        tokens,
        1,
        generator=generator,
    ) * 0.2
    probability[:, 0] = 1
    slope = torch.zeros(batch, tokens, 1, source_dim)
    slope[..., : native_dim - 1] = torch.randn(
        batch,
        tokens,
        1,
        native_dim - 1,
        generator=generator,
    ) * 0.1
    compressed_slope = torch.einsum(
        "hfr,bthf->bthr",
        basis,
        slope,
    )
    key = torch.cat((probability.unsqueeze(-1), compressed_slope), dim=-1)
    key = key.repeat_interleave(2, dim=2)
    decay = (1 - probability).clamp_min(
        RWKV7_MINIMUM_DECAY
    ).repeat_interleave(2, dim=2)
    value = torch.randn(
        batch,
        tokens,
        source_heads * 2,
        native_dim,
        generator=generator,
    )
    native_states = torch.empty(
        batch,
        tokens,
        source_heads * 2,
        native_dim,
        native_dim,
    )
    previous = torch.zeros_like(native_states[:, 0])
    for index in range(tokens):
        previous = (
            previous * decay[:, index, :, None, None]
            + torch.einsum(
                "bho,bhd->bhod",
                value[:, index],
                key[:, index],
            )
        )
        native_states[:, index] = previous
    read = torch.randn(
        batch,
        tokens,
        source_heads,
        native_dim,
        generator=generator,
    )
    target_output = torch.einsum(
        "bthod,bthd->btho",
        native_states,
        read.repeat_interleave(2, dim=2),
    ).reshape(batch, tokens, source_heads, source_dim)
    projection = TwoStateProjection(
        states=native_states.unflatten(2, (source_heads, 2)),
        read=read.unsqueeze(3).expand(-1, -1, -1, 2, -1),
        output=target_output,
        query_basis=basis.unsqueeze(1).expand(-1, 2, -1, -1),
        dc_indices=(0, 0),
    )

    rollout = native_two_state_rollout(
        projection,
        probability,
        slope,
    )

    torch.testing.assert_close(
        rollout.teacher_forced_states,
        projection.states,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        rollout.free_running_states,
        projection.states,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        rollout.free_running_output,
        projection.output,
        rtol=1e-4,
        atol=1e-5,
    )
    assert float(rollout.erase.max()) < 1e-4
    assert float(rollout.requested_decay[:, 0].max()) == 0
    assert float(rollout.decay[:, 0].min()) == pytest.approx(
        RWKV7_MINIMUM_DECAY
    )
