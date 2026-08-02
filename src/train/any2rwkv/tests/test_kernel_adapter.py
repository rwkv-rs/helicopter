from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from any2rwkv import kernel as kernel_module
from any2rwkv.configuration_any2rwkv import AnyToRWKVConfig
from any2rwkv.contract import build_target_config
from any2rwkv.errors import ContractError
from any2rwkv.fixture import tiny_qwen35_config
from any2rwkv.kernel import Rwkv7OperatorAdapter
from any2rwkv.mixer import ProjectionBoundaryRWKV7Attention, apply_partial_rope


class KernelAdapterTests(unittest.TestCase):
    @staticmethod
    def reference_operation(
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
        state_indices=None,
        mode,
    ):
        assert output_final_state is True
        assert cu_seqlens is None
        assert state_indices is None
        assert mode == "fp32io16"
        _batch, tokens, _heads, _size = r.shape
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

    def test_adapter_forwards_exact_six_signal_contract(self) -> None:
        seen = []

        def operation(*signals, initial_state, **kwargs):
            seen.append((initial_state.shape, len(signals), kwargs))
            return signals[3], initial_state

        adapter = Rwkv7OperatorAdapter(
            operation,
            lambda: "fla",
            head_size=64,
            require_flash=False,
        )
        state = torch.zeros(2, 2, 64, 64, dtype=torch.float32)
        signals = [torch.zeros(2, 16, 2, 64, dtype=torch.bfloat16) for _ in range(6)]
        output, final = adapter(*signals, initial_state=state)
        self.assertEqual(seen[0][0:2], (state.shape, 6))
        self.assertEqual(seen[0][2]["mode"], "fp32io16")
        self.assertEqual(output.shape, signals[0].shape)
        self.assertIs(final, state)
        self.assertEqual(adapter.last_provider, "fla")

    def test_loader_consumes_only_public_pinned_rwkv_rs_contract(self) -> None:
        def operation(
            r,
            w,
            k,
            v,
            a,
            b,
            scale=1.0,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            state_indices=None,
            mode="fp32io16",
            **kwargs,
        ):
            del r, w, k, a, b, scale, output_final_state, cu_seqlens
            del state_indices, mode, kwargs
            return v, initial_state

        module = SimpleNamespace(
            recurrent_rwkv7=operation,
            get_last_rwkv7_provider=lambda: "flash_rwkv",
        )
        kernel_module.load_rwkv7_operator_adapter.cache_clear()
        with (
            patch(
                "any2rwkv.preflight.require_rwkv7_runtime",
                return_value={},
            ),
            patch.object(
                kernel_module, "_require_exact_vcs_distribution"
            ) as provenance,
            patch.object(kernel_module.importlib, "import_module", return_value=module),
        ):
            loaded = kernel_module.load_rwkv7_operator_adapter(128)
            self.assertIsInstance(loaded, Rwkv7OperatorAdapter)
            self.assertEqual(loaded.head_size, 128)
            self.assertTrue(loaded.require_flash)
        self.assertEqual(provenance.call_count, 2)
        kernel_module.load_rwkv7_operator_adapter.cache_clear()

    def test_adapter_rejects_wrong_layout_state_and_provider(self) -> None:
        adapter = Rwkv7OperatorAdapter(
            lambda *signals, initial_state, **_kwargs: (signals[0], initial_state),
            lambda: None,
            head_size=64,
            require_flash=False,
        )
        state = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
        signals = [torch.zeros(1, 15, 1, 64, dtype=torch.bfloat16) for _ in range(6)]
        with self.assertRaisesRegex(ContractError, "provider"):
            adapter(*signals, initial_state=state)
        with self.assertRaisesRegex(ContractError, "float32"):
            adapter(*signals, initial_state=state.to(torch.bfloat16))

        flash_required = Rwkv7OperatorAdapter(
            lambda *values, initial_state, **_kwargs: (values[0], initial_state),
            lambda: "fla",
            head_size=64,
            require_flash=True,
        )
        with self.assertRaisesRegex(ContractError, "FlashRWKV operator is required"):
            flash_required(*signals, initial_state=state)

    def test_public_operator_forward_backward_and_fixed_state_continuation(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(20260802)
        shape = (2, 6, 2, 4)
        signals = [torch.randn(shape, generator=generator) for _ in range(6)]
        signals[1] = -signals[1].abs()
        initial_state = torch.randn(2, 2, 4, 4, generator=generator)
        kernel = Rwkv7OperatorAdapter(
            self.reference_operation,
            lambda: "fla",
            head_size=4,
            require_flash=False,
        )

        with torch.no_grad():
            complete, complete_state = kernel(
                *signals,
                initial_state=initial_state,
            )
            first, first_state = kernel(
                *(signal[:, :3] for signal in signals),
                initial_state=initial_state,
            )
            second, resumed_state = kernel(
                *(signal[:, 3:] for signal in signals),
                initial_state=first_state,
            )
        torch.testing.assert_close(torch.cat((first, second), dim=1), complete)
        torch.testing.assert_close(resumed_state, complete_state)

        differentiable_signals = [signal.clone().requires_grad_() for signal in signals]
        differentiable_state = initial_state.clone().requires_grad_()
        output, final_state = kernel(
            *differentiable_signals,
            initial_state=differentiable_state,
        )
        (output.square().mean() + final_state.square().mean()).backward()
        for value in (*differentiable_signals, differentiable_state):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_provenance_gate_rejects_registry_and_local_editable_installs(
        self,
    ) -> None:
        registry_distribution = SimpleNamespace(read_text=lambda _name: None)
        local_distribution = SimpleNamespace(
            read_text=lambda _name: (
                '{"url":"file:///tmp/fla-rwkv","dir_info":{"editable":true}}'
            )
        )
        for distribution in (registry_distribution, local_distribution):
            with (
                self.subTest(distribution=distribution),
                patch.object(
                    kernel_module.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
                self.assertRaisesRegex(ContractError, "pinned rwkv-rs|mismatch"),
            ):
                kernel_module._require_exact_vcs_distribution(
                    "flash-linear-attention",
                    expected_url=kernel_module.FLA_RWKV7_SOURCE_URL,
                    expected_revision=kernel_module.FLA_RWKV7_REVISION,
                )

    def test_provenance_gate_accepts_canonical_lowercase_pep610_url(self) -> None:
        distribution = SimpleNamespace(
            read_text=lambda _name: json.dumps(
                {
                    "url": "https://github.com/rwkv-rs/fla-rwkv",
                    "vcs_info": {
                        "vcs": "git",
                        "requested_revision": kernel_module.FLA_RWKV7_REVISION,
                        "commit_id": kernel_module.FLA_RWKV7_REVISION,
                    },
                }
            )
        )
        with patch.object(
            kernel_module.importlib.metadata,
            "distribution",
            return_value=distribution,
        ):
            kernel_module._require_exact_vcs_distribution(
                "flash-linear-attention",
                expected_url=kernel_module.FLA_RWKV7_SOURCE_URL,
                expected_revision=kernel_module.FLA_RWKV7_REVISION,
            )

    def test_sequence_kernel_path_matches_token_recurrence(self) -> None:
        source = tiny_qwen35_config(layers=1, moe=False)
        source["mtp_num_hidden_layers"] = 0
        config = AnyToRWKVConfig(
            **build_target_config(source, require_final_layers=False)
        )
        mixer = ProjectionBoundaryRWKV7Attention(
            config,
            0,
            source_used_rope=False,
            rotary_dim=0,
            rope_theta=10_000.0,
        ).to(torch.bfloat16)
        torch.manual_seed(17)
        values = torch.randn(1, 16, 64, dtype=torch.bfloat16)
        positions = torch.arange(16).view(1, -1)
        kernel = Rwkv7OperatorAdapter(
            self.reference_operation,
            lambda: "fla",
            head_size=config.head_dim,
            require_flash=False,
        )
        sequence, v_first, final_state, signals = mixer.forward_sequence(
            values, positions=positions, kernel=kernel
        )
        self.assertIn("decay", signals)
        torch.testing.assert_close(
            signals["decay"],
            torch.exp(-0.606531 * torch.sigmoid(signals["w"].float())),
            rtol=0,
            atol=0,
        )
        state = torch.zeros(
            1,
            config.num_heads,
            config.head_dim,
            config.head_dim,
            dtype=torch.float32,
        )
        previous = torch.zeros(1, 64, dtype=torch.bfloat16)
        token_outputs = []
        token_v_rows = []
        token_v_first = torch.zeros_like(previous)
        for index in range(16):
            output, previous, state, token_v_first, _ = mixer.forward_reference(
                values[:, index],
                previous,
                token_v_first,
                state,
                positions=positions[:, index],
            )
            token_outputs.append(output)
            token_v_rows.append(token_v_first)
        torch.testing.assert_close(
            sequence, torch.stack(token_outputs, dim=1), rtol=0.04, atol=0.04
        )
        torch.testing.assert_close(final_state, state, rtol=0.04, atol=0.04)
        torch.testing.assert_close(
            v_first, torch.stack(token_v_rows, dim=1), rtol=0, atol=0
        )

    def test_sequence_kernel_supports_expanded_recurrent_width(self) -> None:
        source = tiny_qwen35_config(layers=1, moe=False)
        source.update(
            {
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "mtp_num_hidden_layers": 0,
            }
        )
        config = AnyToRWKVConfig(
            **build_target_config(source, require_final_layers=False)
        )
        self.assertEqual(config.hidden_size, 32)
        self.assertEqual(config.attention_hidden_size, 64)
        mixer = ProjectionBoundaryRWKV7Attention(
            config,
            0,
            source_used_rope=False,
            rotary_dim=0,
            rope_theta=10_000.0,
        ).to(torch.bfloat16)
        values = torch.randn(1, 16, 32, dtype=torch.bfloat16)
        positions = torch.arange(16).view(1, -1)
        sequence, v_first, final_state, signals = mixer.forward_sequence(
            values,
            positions=positions,
            kernel=Rwkv7OperatorAdapter(
                self.reference_operation,
                lambda: "fla",
                head_size=config.head_dim,
                require_flash=False,
            ),
        )
        self.assertEqual(sequence.shape, (1, 16, 32))
        self.assertEqual(v_first.shape, (1, 16, 64))
        self.assertEqual(final_state.shape, (1, 4, 16, 16))
        self.assertEqual(signals["projected_v"].shape, (1, 16, 64))

    def test_source_rope_layout_is_preserved_before_rwkv_repartition(self) -> None:
        source = tiny_qwen35_config(layers=1, moe=False)
        source.update(
            {
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
            }
        )
        config = AnyToRWKVConfig(
            **build_target_config(source, require_final_layers=False)
        )
        mixer = ProjectionBoundaryRWKV7Attention(
            config,
            0,
            source_used_rope=True,
            rotary_dim=4,
            rope_theta=10_000.0,
            rope_num_heads=2,
            rope_head_dim=8,
        )
        value = torch.arange(16, dtype=torch.float32).view(1, 16)
        positions = torch.tensor([3])

        actual = mixer._apply_source_rope(value, positions)
        expected = apply_partial_rope(
            value.view(1, 2, 8),
            positions,
            rotary_dim=4,
            theta=10_000.0,
        ).reshape_as(value)
        incorrectly_repartitioned = apply_partial_rope(
            value.view(1, 4, 4),
            positions,
            rotary_dim=4,
            theta=10_000.0,
        ).reshape_as(value)

        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, incorrectly_repartitioned))

    def test_group_norm_affine_decomposition_matches_pytorch(self) -> None:
        """The exposed fit basis must preserve RWKV7's original GroupNorm result."""
        torch.manual_seed(29)
        batch, tokens, heads, head_dim = 2, 5, 2, 64
        hidden = heads * head_dim
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.bfloat16, 2e-2)):
            with self.subTest(dtype=dtype):
                recurrent = torch.randn(batch, tokens, hidden, dtype=dtype)
                weight = torch.randn(hidden, dtype=dtype)
                bias = torch.randn(hidden, dtype=dtype)
                flattened = recurrent.reshape(batch * tokens, hidden)

                expected = F.group_norm(
                    flattened,
                    num_groups=heads,
                    weight=weight,
                    bias=bias,
                    eps=head_dim * 1e-5,
                ).view(batch, tokens, hidden)
                norm_base = F.group_norm(
                    flattened,
                    num_groups=heads,
                    weight=None,
                    bias=None,
                    eps=head_dim * 1e-5,
                ).view(batch, tokens, hidden)
                decomposed = norm_base * weight.reshape(1, 1, hidden) + bias.reshape(
                    1, 1, hidden
                )

                torch.testing.assert_close(
                    decomposed, expected, rtol=tolerance, atol=tolerance
                )


if __name__ == "__main__":
    unittest.main()
