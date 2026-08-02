from __future__ import annotations

import pytest
import torch

from any2rwkv.kernel import Rwkv7OperatorAdapter


def _explicit_test_recurrent_adapter(head_size: int) -> Rwkv7OperatorAdapter:
    """Build the test-only PyTorch oracle for product recurrent call sites."""

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
        assert output_final_state is True
        assert cu_seqlens is None
        assert state_indices is None
        assert mode == "fp32io16"
        state = initial_state
        outputs = []
        for token in range(r.shape[1]):
            projection = torch.einsum("bhk,bhkv->bhv", a[:, token].float(), state)
            state = (
                w[:, token].float().exp().unsqueeze(-1) * state
                + b[:, token].float().unsqueeze(-1) * projection.unsqueeze(-2)
                + k[:, token].float().unsqueeze(-1) * v[:, token].float().unsqueeze(-2)
            )
            outputs.append(torch.einsum("bhk,bhkv->bhv", r[:, token].float(), state))
        return torch.stack(outputs, dim=1).to(r.dtype), state

    return Rwkv7OperatorAdapter(
        recurrent,
        lambda: "flash_rwkv",
        head_size=head_size,
        require_flash=True,
    )


@pytest.fixture(autouse=True)
def explicit_test_only_recurrent_operator(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    """Never let CPU tests exercise a hidden product fallback."""
    monkeypatch.setattr(
        "any2rwkv.hybrid.load_rwkv7_operator_adapter",
        _explicit_test_recurrent_adapter,
    )
    monkeypatch.setattr(
        "any2rwkv.modeling_any2rwkv.load_rwkv7_operator_adapter",
        _explicit_test_recurrent_adapter,
    )
    if request.node.name == (
        "test_fully_recurrent_global_corrective_runs_reverse_sweep_and_exports"
    ):
        monkeypatch.setattr(
            "any2rwkv.preflight.require_rwkv7_runtime",
            lambda: {"schema_version": 1, "test_only_recurrent_stub": True},
        )
