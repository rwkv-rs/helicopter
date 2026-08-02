from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from any2rwkv.core import (
    LayerMajorResumeContract,
    activate_layer_major_training,
    assert_layer_major_isolation,
    canonical_digest,
)
from any2rwkv.errors import ContractError


def _tensor_digest(value: torch.Tensor) -> str:
    return canonical_digest(value.detach().tolist())


class TinyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


class LayerMajorContractTests(unittest.TestCase):
    def _run(self, *, interrupt_after: int | None = None, saved=None):
        torch.manual_seed(7)
        teacher = TinyTeacher()
        layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])
        output_adapter = nn.Linear(2, 1, bias=False).requires_grad_(False)
        rows = torch.tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.25]])
        with torch.no_grad():
            targets = teacher(rows).detach()
        cache = rows.clone()
        layer_index = 0
        step = 0
        trace: list[float] = []
        optimizer = None

        if saved is not None:
            teacher.load_state_dict(saved["teacher"])
            layers.load_state_dict(saved["layers"])
            output_adapter.load_state_dict(saved["adapter"])
            cache = saved["cache"].clone()
            contract = LayerMajorResumeContract.from_dict(
                saved["contract"],
                expected_digests={
                    "data_sha256": _tensor_digest(rows),
                    "current_cache_sha256": _tensor_digest(cache),
                },
            )
            layer_index, step = contract.layer_index, contract.optimizer_step
            trace = list(saved["trace"])

        while layer_index < len(layers):
            teacher_before = copy.deepcopy(teacher.state_dict())
            inactive_before = {
                (index, name): value.detach().clone()
                for index, layer in enumerate(layers)
                if index != layer_index
                for name, value in layer.state_dict().items()
            }
            active = activate_layer_major_training(
                teacher=teacher, layers=layers, layer_index=layer_index
            )
            optimizer = torch.optim.SGD(active, lr=0.05)
            if saved is not None and saved.get("optimizer") is not None:
                optimizer.load_state_dict(saved["optimizer"])
                saved = None
            while step < 3:
                optimizer.zero_grad(set_to_none=True)
                prediction = output_adapter(layers[layer_index](cache))
                loss = torch.nn.functional.mse_loss(prediction, targets)
                loss.backward()
                assert_layer_major_isolation(
                    teacher=teacher, layers=layers, layer_index=layer_index
                )
                optimizer.step()
                for name, value in teacher.state_dict().items():
                    torch.testing.assert_close(
                        value, teacher_before[name], rtol=0, atol=0
                    )
                for index, layer in enumerate(layers):
                    if index != layer_index:
                        for name, value in layer.state_dict().items():
                            torch.testing.assert_close(
                                value, inactive_before[(index, name)], rtol=0, atol=0
                            )
                trace.append(float(loss.detach()))
                step += 1
                if interrupt_after is not None and len(trace) == interrupt_after:
                    contract = LayerMajorResumeContract(
                        schema_version=1,
                        layer_index=layer_index,
                        optimizer_step=step,
                        data_cursor=step % len(rows),
                        data_sha256=_tensor_digest(rows),
                        weight_sha256=canonical_digest(
                            {
                                name: value.detach().tolist()
                                for name, value in layers.state_dict().items()
                            }
                        ),
                        trace_sha256=canonical_digest(trace),
                        solver_sha256=canonical_digest(optimizer.state_dict()),
                        state_sha256=canonical_digest(
                            {"layer": layer_index, "step": step}
                        ),
                        current_cache_sha256=_tensor_digest(cache),
                        next_cache_sha256=_tensor_digest(layers[layer_index](cache)),
                    )
                    return {
                        "teacher": copy.deepcopy(teacher.state_dict()),
                        "layers": copy.deepcopy(layers.state_dict()),
                        "adapter": copy.deepcopy(output_adapter.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "cache": cache.clone(),
                        "trace": trace,
                        "contract": contract.to_dict(),
                    }
            with torch.no_grad():
                cache = layers[layer_index](cache).detach()
            layer_index += 1
            step = 0
        return layers.state_dict(), cache, trace, teacher

    def test_interrupted_resume_matches_uninterrupted_cpu_trajectory(self) -> None:
        expected_layers, expected_cache, expected_trace, _ = self._run()
        checkpoint = self._run(interrupt_after=2)
        actual_layers, actual_cache, actual_trace, teacher = self._run(saved=checkpoint)

        self.assertEqual(actual_trace, expected_trace)
        torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
        for name in expected_layers:
            torch.testing.assert_close(
                actual_layers[name], expected_layers[name], rtol=0, atol=0
            )
        self.assertFalse(teacher.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in teacher.parameters())
        )

    def test_resume_contract_rejects_missing_unknown_or_changed_bindings(self) -> None:
        digest = "a" * 64
        payload = LayerMajorResumeContract(
            1, 0, 1, 1, digest, digest, digest, digest, digest, digest, digest
        ).to_dict()
        for mutation in (
            lambda row: row.pop("trace_sha256"),
            lambda row: row.update(extra=1),
        ):
            changed = dict(payload)
            mutation(changed)
            with self.assertRaisesRegex(ContractError, "missing or unknown"):
                LayerMajorResumeContract.from_dict(changed)
        with self.assertRaisesRegex(ContractError, "differs from current input"):
            LayerMajorResumeContract.from_dict(
                payload, expected_digests={"data_sha256": "b" * 64}
            )
        for field in (
            "data_sha256",
            "weight_sha256",
            "trace_sha256",
            "solver_sha256",
            "state_sha256",
            "current_cache_sha256",
            "next_cache_sha256",
        ):
            with self.assertRaisesRegex(ContractError, field):
                LayerMajorResumeContract.from_dict(
                    payload, expected_digests={field: "b" * 64}
                )


if __name__ == "__main__":
    unittest.main()
