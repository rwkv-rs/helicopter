from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import Tensor, nn

from any2rwkv.configuration_any2rwkv import AnyToRWKVConfig
from any2rwkv.contract import build_target_config
from any2rwkv.fixture import tiny_qwen35_config
from any2rwkv.hybrid import HybridModelPatcher
from any2rwkv.kernel import Rwkv7OperatorAdapter
from any2rwkv.mixer import ProjectionBoundaryRWKV7Attention


class SourceAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.projection = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states: Tensor, **kwargs):
        return self.projection(hidden_states), None


class SourceLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = SourceAttention(hidden)
        self.mlp = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states: Tensor, **kwargs):
        mixed, _ = self.self_attn(hidden_states, **kwargs)
        return (hidden_states + mixed + self.mlp(hidden_states),)


class SourceModel(nn.Module):
    def __init__(self, layers: int, hidden: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(SourceLayer(hidden) for _ in range(layers))

    def forward(self, hidden_states: Tensor, **kwargs):
        hidden = hidden_states
        for layer in self.model.layers:
            hidden = layer(hidden, **kwargs)[0]
        return SimpleNamespace(logits=hidden)


class HybridCheckpointTests(unittest.TestCase):
    def test_nonzero_active_layer_gets_detached_frozen_layer0_v_first(self) -> None:
        source = tiny_qwen35_config(layers=3, moe=False)
        source["mtp_num_hidden_layers"] = 0
        config = AnyToRWKVConfig(
            **build_target_config(source, require_final_layers=False)
        )
        teacher = SourceModel(3, config.hidden_size)
        mixers = [
            ProjectionBoundaryRWKV7Attention(
                config,
                layer,
                source_used_rope=False,
                rotary_dim=0,
                rope_theta=10_000.0,
            )
            for layer in range(3)
        ]
        patcher = HybridModelPatcher(teacher, mixers)
        patcher.configure(active_layer=1, converted_prefix=0)
        adapter = Rwkv7OperatorAdapter(
            lambda *signals, initial_state, **_kwargs: (signals[3], initial_state),
            lambda: "flash_rwkv",
            head_size=config.head_dim,
            require_flash=True,
        )
        with patch("any2rwkv.hybrid.load_rwkv7_operator_adapter", return_value=adapter):
            output = teacher(
                torch.randn(2, 4, config.hidden_size),
                attention_mask=torch.ones(2, 4),
                position_ids=torch.arange(4).view(1, -1).expand(2, -1),
            ).logits
        self.assertIsNotNone(patcher.context.v_first)
        self.assertFalse(patcher.context.v_first.requires_grad)
        output.square().mean().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in mixers[0].parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in mixers[1].parameters())
        )
        patcher.restore()


if __name__ == "__main__":
    unittest.main()
