from __future__ import annotations

import torch
import pytest

from any2rwkv.zero_step_probe import (
    RWKV7_MINIMUM_DECAY,
    TwoStateProjection,
    causal_attention,
    fit_bias_free_projection,
    four_state_block_output,
    logit_taylor_hazards,
    native_two_state_rollout,
    observable_query_bases,
    probability_taylor_hazards,
    qwen35_l2_normalize,
    rope_aligned_two_state_bases,
    rollout_hazards,
    select_bias_free_projection,
    two_state_projection,
    two_state_outputs,
    verify_gdn_mapping,
)


def test_exact_hazard_rollout_recovers_causal_attention() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    value = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)

    exact = causal_attention(query, key, value)
    recurrent = rollout_hazards(exact.hazards, value)

    torch.testing.assert_close(recurrent, exact.output, rtol=1e-5, atol=1e-6)


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
