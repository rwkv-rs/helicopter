from __future__ import annotations

import torch
import pytest

from any2rwkv.configuration_any2rwkv import Any2RWKV7Config
from any2rwkv.contract import build_target_config
from any2rwkv.fixture import tiny_qwen35_config
from any2rwkv.kernel import NativeRwkv7Kernel
from any2rwkv.mixer import ProjectionBoundaryRWKV7Attention
from any2rwkv.mixer import apply_partial_rope
from any2rwkv.recipes.qwen35_to_rwkv7.gqa_zero_step import (
    GQANativeFitConfig,
    GQANativeFitTrace,
    fit_gqa_native_zero_step,
)
from any2rwkv.recurrent import rwkv7_step
from any2rwkv.zero_step_probe import (
    RWKV7_MINIMUM_DECAY,
    TwoStateProjection,
    affine_state_rollout,
    causal_attention,
    fit_bias_free_projection,
    fit_low_rank_projection,
    four_state_block_output,
    logit_taylor_hazards,
    materialize_native_projection,
    native_signal_rollout,
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


def reference_native_kernel(state, r, w, k, v, a, b):
    batch, tokens, channels = r.shape
    heads = state.shape[1]
    head_dim = channels // heads
    outputs = []
    current = state
    for index in range(tokens):
        rt, wt, kt, vt, at, bt = (
            value[:, index].view(batch, heads, head_dim)
            for value in (r, w, k, v, a, b)
        )
        decay = torch.exp(-0.6065306597 * torch.sigmoid(wt.float()))
        current = (
            current * decay.unsqueeze(-2)
            + (current @ at.float().unsqueeze(-1)) @ bt.float().unsqueeze(-2)
            + vt.float().unsqueeze(-1) @ kt.float().unsqueeze(-2)
        )
        outputs.append((current @ rt.float().unsqueeze(-1)).squeeze(-1))
    return (
        torch.stack(outputs, dim=1)
        .reshape(batch, tokens, channels)
        .to(r.dtype),
        current,
    )


def test_exact_hazard_rollout_recovers_causal_attention() -> None:
    generator = torch.Generator().manual_seed(20260725)
    query = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)
    value = torch.randn(2, 7, 3, 8, generator=generator, dtype=torch.float64)

    exact = causal_attention(query, key, value)
    recurrent = rollout_hazards(exact.hazards, value)

    torch.testing.assert_close(recurrent, exact.output, rtol=1e-5, atol=1e-6)


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
        kernel=NativeRwkv7Kernel(
            reference_native_kernel,
            head_size=config.head_dim,
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

    fitted = fit_gqa_native_zero_step(
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
            query_weight=query_weight,
            key_weight=key_weight,
            value_weight=value_weight,
            output_weight=output_weight,
            output_bias=None,
        ),
        GQANativeFitConfig(
            calibration_rows=3,
            positions=positions,
            source_head_dim=8,
            rotary_dim=2,
            rope_theta=10_000.0,
            observable_fit_steps=2,
        ),
    )

    assert set(fitted.parameters) == {
        name for name, _ in mixer.named_parameters()
    }
    for name, parameter in mixer.named_parameters():
        assert fitted.parameters[name].shape == parameter.shape
        assert bool(torch.isfinite(fitted.parameters[name]).all())
    assert fitted.report["observable_state_compression"]["budget"] == {
        "states_per_source_head": 2,
        "dc_per_state": 1,
        "query_features_per_state": 3,
        "shared_input_subspace": False,
    }
    installed = materialize_native_projection(mixer, fitted.parameters)
    assert len(installed.aggregate_sha256) == 64
    values = mixer_input.to(torch.bfloat16)
    output, _, final_state, _ = mixer.forward_sequence(
        values,
        positions=positions,
        kernel=NativeRwkv7Kernel(
            reference_native_kernel,
            head_size=config.head_dim,
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
