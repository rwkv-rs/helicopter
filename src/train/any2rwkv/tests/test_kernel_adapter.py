from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from any2rwkv.errors import ContractError
from any2rwkv.configuration_any2rwkv import Any2RWKV7Config
from any2rwkv.contract import build_target_config
from any2rwkv.fixture import tiny_qwen35_config
from any2rwkv.kernel import NativeRwkv7Kernel
from any2rwkv import kernel as kernel_module
from any2rwkv.mixer import ProjectionBoundaryRWKV7Attention


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

        adapter = NativeRwkv7Kernel(operation)
        state = torch.zeros(2, 2, 64, 64, dtype=torch.float32)
        signals = [torch.zeros(2, 16, 128, dtype=torch.bfloat16) for _ in range(6)]
        output, final = adapter(state, *signals)
        self.assertEqual(seen, [(state.shape, 6)])
        self.assertEqual(output.shape, signals[0].shape)
        self.assertIs(final, state)

    def test_loader_resolves_relative_cuda_sources_from_pinned_checkout(self) -> None:
        checkout = Path(kernel_module.__file__).resolve().parents[4] / "src/train/rwkv-lm"
        seen_cwd = []

        def fake_import(name):
            self.assertEqual(name, "src.model")
            seen_cwd.append(Path.cwd())
            return SimpleNamespace(
                __file__=str(checkout / "src/model.py"),
                RWKV7_STATEPASSING_CLAMPW_CUDA=lambda *values: values,
            )

        kernel_module.load_rwkv_lm_kernel.cache_clear()
        with patch.dict("sys.modules", {"src.model": None}):
            # ``None`` represents no usable cached module for this test.
            import sys

            sys.modules.pop("src.model", None)
            with patch.object(kernel_module.importlib, "import_module", side_effect=fake_import):
                previous = Path.cwd()
                loaded = kernel_module.load_rwkv_lm_kernel()
                self.assertIsInstance(loaded, NativeRwkv7Kernel)
                self.assertEqual(Path.cwd(), previous)
        self.assertEqual(seen_cwd, [checkout])
        kernel_module.load_rwkv_lm_kernel.cache_clear()

    def test_adapter_rejects_wrong_dtype_tail_and_state(self) -> None:
        adapter = NativeRwkv7Kernel(lambda state, *signals: (signals[0], state))
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
        kernel = NativeRwkv7Kernel(self.reference_operation)
        sequence, v_first, final_state, _ = mixer.forward_sequence(
            values, positions=positions, kernel=kernel
        )
        state = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
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


if __name__ == "__main__":
    unittest.main()
