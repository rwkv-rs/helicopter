from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .migration import gdn_reference_scan, gdn_to_rwkv7_dynamics
from .recurrent import chunked_rwkv7_scan, native_decay_from_logit, rwkv7_scan


@dataclass(frozen=True)
class OracleTolerance:
    output_relative_l2: float = 1e-12
    output_max_abs: float = 1e-12
    state_relative_l2: float = 1e-12
    gradient_relative_l2: float = 1e-11
    gradient_cosine: float = 0.999999999999
    finite_difference_relative_error: float = 1e-6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(left: Tensor, right: Tensor) -> float:
    value = torch.linalg.vector_norm(left - right) / torch.clamp(
        torch.linalg.vector_norm(left), min=1e-30
    )
    return float(value.detach())


def cosine(left: Tensor, right: Tensor) -> float:
    value = torch.nn.functional.cosine_similarity(
        left.flatten(), right.flatten(), dim=0
    )
    return float(value.detach())


def _clone_grad(values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(value.detach().clone().requires_grad_(True) for value in values)


def _loss(output: Tensor, state: Tensor) -> Tensor:
    return output.square().mean() + state.square().mean() * 0.1


def run_gdn_oracle_case(*, seed: int, length: int, chunk_size: int, random_state: bool) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    shape = (1, length, 2, 4)
    query = torch.randn(shape, generator=generator, dtype=torch.float64) * 0.2
    key = torch.nn.functional.normalize(torch.randn(shape, generator=generator, dtype=torch.float64), dim=-1)
    value = torch.randn(shape, generator=generator, dtype=torch.float64) * 0.2
    beta = torch.sigmoid(torch.randn((1, length, 2, 1), generator=generator, dtype=torch.float64))
    decay = native_decay_from_logit(
        torch.randn(shape[:-1] + (1,), generator=generator, dtype=torch.float64)
    )
    state = (
        torch.randn((1, 2, 4, 4), generator=generator, dtype=torch.float64) * 0.1
        if random_state
        else torch.zeros((1, 2, 4, 4), dtype=torch.float64)
    )
    source_values = _clone_grad((state, decay, beta, query, key, value))
    target_values = _clone_grad((state, decay, beta, query, key, value))
    source_output, source_state = gdn_reference_scan(*source_values)
    target_state0, target_decay, target_beta, target_query, target_key, target_value = target_values
    r, mapped_decay, k, v, a, b = gdn_to_rwkv7_dynamics(
        target_decay, target_beta, target_query, target_key, target_value
    )
    target_output, target_state = chunked_rwkv7_scan(
        target_state0, r, mapped_decay, k, v, a, b, chunk_size=chunk_size
    )
    source_grad = torch.autograd.grad(_loss(source_output, source_state), source_values)
    target_grad = torch.autograd.grad(_loss(target_output, target_state), target_values)
    gradient_rows = [
        {
            "name": name,
            "relative_l2": relative_l2(left, right),
            "cosine": cosine(left, right),
            "source_l2": float(torch.linalg.vector_norm(left).detach()),
            "target_l2": float(torch.linalg.vector_norm(right).detach()),
        }
        for name, left, right in zip(
            ("state", "decay", "beta", "query", "key", "value"),
            source_grad,
            target_grad,
            strict=True,
        )
    ]
    full_output, full_state = rwkv7_scan(target_state0.detach(), r.detach(), mapped_decay.detach(), k.detach(), v.detach(), a.detach(), b.detach())
    direction = torch.randn(query.shape, generator=generator, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    epsilon = 1e-6

    def directional_loss(offset: float) -> Tensor:
        moved_query = query + direction * offset
        moved_r, moved_decay, moved_k, moved_v, moved_a, moved_b = gdn_to_rwkv7_dynamics(
            decay, beta, moved_query, key, value
        )
        output, final = rwkv7_scan(state, moved_r, moved_decay, moved_k, moved_v, moved_a, moved_b)
        return _loss(output, final)

    finite = (directional_loss(epsilon) - directional_loss(-epsilon)) / (2 * epsilon)
    query_for_grad = query.detach().clone().requires_grad_(True)
    rr, dd, kk, vv, aa, bb = gdn_to_rwkv7_dynamics(decay, beta, query_for_grad, key, value)
    oo, ss = rwkv7_scan(state, rr, dd, kk, vv, aa, bb)
    analytic = torch.autograd.grad(_loss(oo, ss), query_for_grad)[0].mul(direction).sum()
    finite_error = float(
        ((finite - analytic).abs() / torch.clamp(analytic.abs(), min=1e-30)).detach()
    )
    return {
        "seed": seed,
        "length": length,
        "chunk_size": chunk_size,
        "initial_state": "random" if random_state else "zero",
        "output_relative_l2": relative_l2(source_output, target_output),
        "output_max_abs": float(
            (source_output - target_output).abs().max().detach()
        ),
        "state_relative_l2": relative_l2(source_state, target_state),
        "chunk_output_relative_l2": relative_l2(full_output, target_output.detach()),
        "chunk_state_relative_l2": relative_l2(full_state, target_state.detach()),
        "gradients": gradient_rows,
        "finite_difference_relative_error": finite_error,
    }


def run_gdn_oracle(*, seed: int = 20260714, tolerance: OracleTolerance = OracleTolerance()) -> dict[str, object]:
    lengths = (1, 2, 15, 16, 17, 31, 32, 65)
    chunks = (1, 2, 7, 16)
    cases = [
        run_gdn_oracle_case(
            seed=seed + index,
            length=length,
            chunk_size=min(chunk, length),
            random_state=bool(index % 2),
        )
        for index, (length, chunk) in enumerate((length, chunk) for length in lengths for chunk in chunks)
    ]
    failures: list[str] = []
    for index, case in enumerate(cases):
        checks = (
            (case["output_relative_l2"] <= tolerance.output_relative_l2, "output_relative_l2"),
            (case["output_max_abs"] <= tolerance.output_max_abs, "output_max_abs"),
            (case["state_relative_l2"] <= tolerance.state_relative_l2, "state_relative_l2"),
            (case["finite_difference_relative_error"] <= tolerance.finite_difference_relative_error, "finite_difference"),
        )
        failures.extend(f"case[{index}]:{name}" for passed, name in checks if not passed)
        for gradient in case["gradients"]:
            both_zero = max(gradient["source_l2"], gradient["target_l2"]) <= 1e-14
            if not both_zero and (
                gradient["relative_l2"] > tolerance.gradient_relative_l2
                or gradient["cosine"] < tolerance.gradient_cosine
            ):
                failures.append(f"case[{index}]:gradient:{gradient['name']}")
    return {
        "schema_version": 1,
        "fixture_count": len(cases),
        "seed": seed,
        "tolerance": tolerance.__dict__,
        "passed": not failures,
        "failures": failures,
        "cases": cases,
    }
