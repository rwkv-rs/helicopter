from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from any2rwkv.errors import ContractError
from any2rwkv.configuration_any2rwkv import Any2RWKV7Config
from any2rwkv.contract import build_target_config
from any2rwkv.fixture import tiny_qwen35_config
from any2rwkv.kernel import NativeRwkv7Kernel
from any2rwkv import kernel as kernel_module
from any2rwkv.mixer import ProjectionBoundaryRWKV7Attention, apply_partial_rope


class KernelAdapterTests(unittest.TestCase):
    @staticmethod
    def reference_operation(state, r, w, k, v, a, b):
        batch, tokens, channels = r.shape
        heads = state.shape[1]
        size = channels // heads
        outputs = []
        current = state
        for index in range(tokens):
            values = [value[:, index].view(batch, heads, size) for value in (r, w, k, v, a, b)]
            rt, wt, kt, vt, at, bt = values
            decay = torch.exp(-0.6065306597 * torch.sigmoid(wt.float()))
            current = (
                current * decay.unsqueeze(-2)
                + (current @ at.float().unsqueeze(-1)) @ bt.float().unsqueeze(-2)
                + vt.float().unsqueeze(-1) @ kt.float().unsqueeze(-2)
            )
            outputs.append((current @ rt.float().unsqueeze(-1)).squeeze(-1))
        return torch.stack(outputs, dim=1).reshape(batch, tokens, channels).to(r.dtype), current

    def test_adapter_forwards_exact_six_signal_contract(self) -> None:
        seen = []

        def operation(state, *signals):
            seen.append((state.shape, len(signals)))
            return signals[3], state

        adapter = NativeRwkv7Kernel(operation, head_size=64)
        state = torch.zeros(2, 2, 64, 64, dtype=torch.float32)
        signals = [torch.zeros(2, 16, 128, dtype=torch.bfloat16) for _ in range(6)]
        output, final = adapter(state, *signals)
        self.assertEqual(seen, [(state.shape, 6)])
        self.assertEqual(output.shape, signals[0].shape)
        self.assertIs(final, state)

    def test_loader_resolves_relative_cuda_sources_from_pinned_checkout(self) -> None:
        checkout = Path(kernel_module.__file__).resolve().parents[4] / "src/train/rwkv-lm"
        seen_head_sizes = []

        module = SimpleNamespace(
            __file__=str(checkout / "src/infctx_kernel.py"),
            load_statepassing_kernel=lambda head_size: (
                seen_head_sizes.append(head_size)
                or (lambda state, *signals: (signals[3], state))
            )
        )
        loader = SimpleNamespace(exec_module=lambda candidate: None)
        spec = SimpleNamespace(loader=loader)

        kernel_module.load_rwkv_lm_kernel.cache_clear()
        with (
            patch.object(
                kernel_module.importlib.util,
                "spec_from_file_location",
                return_value=spec,
            ),
            patch.object(
                kernel_module.importlib.util,
                "module_from_spec",
                return_value=module,
            ),
        ):
            previous = Path.cwd()
            loaded = kernel_module.load_rwkv_lm_kernel(128)
            self.assertIsInstance(loaded, NativeRwkv7Kernel)
            self.assertEqual(loaded.head_size, 128)
            self.assertEqual(Path.cwd(), previous)
        self.assertEqual(seen_head_sizes, [128])
        kernel_module.load_rwkv_lm_kernel.cache_clear()

    def test_adapter_rejects_wrong_dtype_tail_and_state(self) -> None:
        adapter = NativeRwkv7Kernel(
            lambda state, *signals: (signals[0], state), head_size=64
        )
        state = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
        signals = [torch.zeros(1, 15, 64, dtype=torch.bfloat16) for _ in range(6)]
        with self.assertRaisesRegex(ContractError, "tokens"):
            adapter(state, *signals)
        signals = [torch.zeros(1, 16, 64, dtype=torch.float16) for _ in range(6)]
        with self.assertRaisesRegex(ContractError, "bfloat16"):
            adapter(state, *signals)

    def test_sequence_kernel_path_matches_token_recurrence(self) -> None:
        source = tiny_qwen35_config(layers=1, moe=False)
        source["mtp_num_hidden_layers"] = 0
        config = Any2RWKV7Config(
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
        kernel = NativeRwkv7Kernel(
            self.reference_operation, head_size=config.head_dim
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
            output, previous, state, token_v_first, _ = mixer(
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
        torch.testing.assert_close(v_first, torch.stack(token_v_rows, dim=1), rtol=0, atol=0)

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
        config = Any2RWKV7Config(
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
            kernel=NativeRwkv7Kernel(
                self.reference_operation, head_size=config.head_dim
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
        config = Any2RWKV7Config(
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
                decomposed = (
                    norm_base * weight.reshape(1, 1, hidden)
                    + bias.reshape(1, 1, hidden)
                )

                torch.testing.assert_close(
                    decomposed, expected, rtol=tolerance, atol=tolerance
                )


if __name__ == "__main__":
    unittest.main()
