from __future__ import annotations

import torch

from any2rwkv.zero_step_probe import (
    causal_attention,
    four_state_block_output,
    verify_gdn_mapping,
    logit_taylor_hazards,
    probability_taylor_hazards,
    qwen35_l2_normalize,
    rollout_hazards,
    two_state_outputs,
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
