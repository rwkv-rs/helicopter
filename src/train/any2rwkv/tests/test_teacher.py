from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from any2rwkv.teacher import TeacherRunner


class FakeLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear_attn = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return hidden_states + self.linear_attn(hidden_states)


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer(8), FakeLayer(8)])
        self.head = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden_states=hidden, attention_mask=attention_mask)
        return SimpleNamespace(logits=self.head(hidden))


class TeacherTraceTests(unittest.TestCase):
    def test_capture_spills_each_layer_and_hashes_metadata(self) -> None:
        runner = TeacherRunner(FakeTeacher(), source_hash="s", config_hash="c", data_hash="d")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = runner.capture_to_disk(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.ones(1, 3, dtype=torch.bool),
                position_ids=torch.arange(3).view(1, 3),
                layer_types=["linear_attention", "linear_attention"],
                trace_dir=root,
            )
            self.assertEqual([path.name for path in paths], ["layer-00.pt", "layer-01.pt"])
            self.assertTrue((root / "logits.pt").is_file())
            self.assertTrue((root / "layer-00.sha256").read_text().strip())
            trace = torch.load(paths[0], weights_only=False)
            self.assertEqual(trace["metadata"]["source_hash"], "s")


if __name__ == "__main__":
    unittest.main()
