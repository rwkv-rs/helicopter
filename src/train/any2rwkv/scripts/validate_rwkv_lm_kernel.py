#!/usr/bin/env python3
"""Managed GPU validation for the canonical rwkv-lm state-passing kernel.

Run through helicopter-dev remote run. Required RWKV_* settings are
intentionally supplied by the managed command contract; the product loader
resolves the pinned rwkv-lm checkout without relying on caller cwd.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from any2rwkv.artifacts import file_sha256, git_sha
from any2rwkv.kernel import load_rwkv_lm_kernel
from any2rwkv.recurrent import native_decay_from_logit, rwkv7_scan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head_size = int(os.environ.get("RWKV_HEAD_SIZE", "0"))
    if head_size <= 0:
        raise SystemExit("RWKV_HEAD_SIZE must be positive")
    required = {
        "RWKV_JIT_ON": "0",
        "RWKV_MY_TESTING": "x070",
        "RWKV_KERNEL": "",
        "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
        "RWKV_TRAIN_TYPE": "infctx",
        "RWKV_FLOAT_MODE": "bf16",
        "WKV_MODE": "fp32io16",
    }
    mismatch = {key: (os.environ.get(key), value) for key, value in required.items() if os.environ.get(key) != value}
    if mismatch:
        raise SystemExit(f"managed RWKV kernel environment mismatch: {mismatch}")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {1, 8}:
        raise SystemExit(f"kernel validation requires world_size 1 or 8, got {world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
    product_root = Path(__file__).resolve().parents[4]
    rwkv_lm_checkout = product_root / "src/train/rwkv-lm"
    rwkv_hf_checkout = product_root / "src/train/rwkv-hf"
    bound_sources = {
        "src/train/rwkv-lm/src/infctx_kernel.py": rwkv_lm_checkout
        / "src/infctx_kernel.py",
        "src/train/rwkv-lm/cuda/rwkv7_statepassing_clampw.cu": rwkv_lm_checkout
        / "cuda/rwkv7_statepassing_clampw.cu",
        "src/train/rwkv-lm/cuda/rwkv7_statepassing_pybind.cpp": rwkv_lm_checkout
        / "cuda/rwkv7_statepassing_pybind.cpp",
        "src/train/any2rwkv/any2rwkv/kernel.py": product_root
        / "src/train/any2rwkv/any2rwkv/kernel.py",
        "src/train/any2rwkv/any2rwkv/recurrent.py": product_root
        / "src/train/any2rwkv/any2rwkv/recurrent.py",
        "src/train/any2rwkv/scripts/validate_rwkv_lm_kernel.py": Path(__file__).resolve(),
    }
    missing_bound_sources = [
        name for name, path in bound_sources.items() if not path.is_file()
    ]
    if missing_bound_sources:
        raise SystemExit(f"kernel evidence source files are missing: {missing_bound_sources}")
    source_identity = {
        "product_commit": git_sha(product_root),
        "rwkv_lm_commit": git_sha(rwkv_lm_checkout),
        "rwkv_hf_commit": git_sha(rwkv_hf_checkout),
        "source_sha256": {
            name: file_sha256(path) for name, path in bound_sources.items()
        },
    }
    kernel = load_rwkv_lm_kernel(head_size)
    generator = torch.Generator(device=device).manual_seed(20260714 + rank)
    heads = 2
    channels = heads * head_size

    def bf16_signal(scale: float) -> torch.Tensor:
        return (
            torch.randn(1, 32, channels, generator=generator, device=device)
            .mul_(scale)
            .to(torch.bfloat16)
            .requires_grad_(True)
        )

    r, w, k, v = (bf16_signal(scale) for scale in (0.1, 0.5, 0.1, 0.1))
    direction = torch.nn.functional.normalize(
        torch.randn(
            1, 32, heads, head_size, generator=generator, device=device
        ),
        dim=-1,
    ).reshape(1, 32, channels).to(torch.bfloat16)
    gate = torch.sigmoid(
        torch.randn(
            1, 32, channels, generator=generator, device=device
        ).mul_(0.5)
    ).to(torch.bfloat16)
    erase = (-direction).contiguous().requires_grad_(True)
    write = (direction * gate).contiguous().requires_grad_(True)
    signals = [r, w, k, v, erase, write]
    state = (
        torch.randn(
            1,
            heads,
            head_size,
            head_size,
            generator=generator,
            device=device,
        )
        .mul_(0.01)
        .requires_grad_(True)
    )
    full_output, full_state = kernel(state, *signals)
    first_output, middle_state = kernel(state, *(value[:, :16] for value in signals))
    second_output, final_state = kernel(middle_state, *(value[:, 16:] for value in signals))
    chunk_output = torch.cat((first_output, second_output), dim=1)
    output_max_abs = float((full_output - chunk_output).detach().abs().max())
    state_max_abs = float((full_state - final_state).detach().abs().max())
    loss = full_output.float().square().mean() + full_state.square().mean() * 0.01
    gradients = torch.autograd.grad(loss, (state, *signals))
    reference_inputs = [
        value.detach().float().view(1, 32, heads, head_size).requires_grad_(True)
        for value in signals
    ]
    reference_state = state.detach().clone().requires_grad_(True)
    reference_output, reference_final_state = rwkv7_scan(
        reference_state,
        reference_inputs[0],
        native_decay_from_logit(reference_inputs[1]),
        reference_inputs[2],
        reference_inputs[3],
        reference_inputs[4],
        reference_inputs[5],
    )
    reference_loss = (
        reference_output.square().mean()
        + reference_final_state.square().mean() * 0.01
    )
    reference_gradients_unshaped = torch.autograd.grad(
        reference_loss, (reference_state, *reference_inputs)
    )
    reference_gradients = tuple(
        reference.reshape_as(candidate)
        for candidate, reference in zip(
            gradients, reference_gradients_unshaped, strict=True
        )
    )
    flat_reference_output = reference_output.reshape_as(full_output)

    def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
        difference = (candidate.detach().float() - reference.detach().float()).norm()
        return float(difference / reference.detach().float().norm().clamp_min(1e-12))

    output_relative_l2 = relative_l2(full_output, flat_reference_output)
    reference_output_max_abs = float(
        (full_output.detach().float() - flat_reference_output.detach()).abs().max()
    )
    state_relative_l2 = relative_l2(full_state, reference_final_state)
    gradient_relative_l2 = [
        relative_l2(candidate, reference)
        for candidate, reference in zip(gradients, reference_gradients, strict=True)
    ]
    gradient_cosine = [
        float(
            torch.nn.functional.cosine_similarity(
                candidate.detach().float().flatten(),
                reference.detach().float().flatten(),
                dim=0,
            )
        )
        for candidate, reference in zip(gradients, reference_gradients, strict=True)
    ]
    passed = (
        output_max_abs <= 0.02
        and state_max_abs <= 0.003
        and output_relative_l2 <= 2e-3
        and reference_output_max_abs <= 2e-2
        and state_relative_l2 <= 3e-3
        and max(gradient_relative_l2) <= 5e-2
        and min(gradient_cosine) >= 0.999
        and all(torch.isfinite(gradient).all() for gradient in gradients)
    )
    result = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "kernel": "rwkv-lm/RWKV7_STATEPASSING_CLAMPW_CUDA",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "head_size": head_size,
        "heads": heads,
        "signal_domain": "native decay plus normalized rank-1 erase/write",
        "state_dtype": str(state.dtype),
        "signal_dtype": str(signals[0].dtype),
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
        "reference_output_relative_l2": output_relative_l2,
        "reference_output_max_abs": reference_output_max_abs,
        "reference_state_relative_l2": state_relative_l2,
        "gradient_relative_l2": gradient_relative_l2,
        "gradient_cosine": gradient_cosine,
        "gradient_finite": [bool(torch.isfinite(gradient).all()) for gradient in gradients],
        "gpu": torch.cuda.get_device_name(device),
        "source_identity": source_identity,
    }
    rank_results = [result]
    if world_size > 1:
        gathered: list[dict[str, object] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, result)
        rank_results = [row for row in gathered if row is not None]
    aggregate = dict(rank_results[0])
    aggregate.update(
        {
            "status": (
                "pass"
                if len(rank_results) == world_size
                and all(row["status"] == "pass" for row in rank_results)
                else "fail"
            ),
            "rank": 0,
            "local_rank": 0,
            "world_size": world_size,
            "gpu": sorted({str(row["gpu"]) for row in rank_results}),
            "output_max_abs": max(float(row["output_max_abs"]) for row in rank_results),
            "state_max_abs": max(float(row["state_max_abs"]) for row in rank_results),
            "reference_output_relative_l2": max(
                float(row["reference_output_relative_l2"]) for row in rank_results
            ),
            "reference_output_max_abs": max(
                float(row["reference_output_max_abs"]) for row in rank_results
            ),
            "reference_state_relative_l2": max(
                float(row["reference_state_relative_l2"]) for row in rank_results
            ),
            "gradient_relative_l2": [
                max(float(row["gradient_relative_l2"][index]) for row in rank_results)
                for index in range(len(result["gradient_relative_l2"]))
            ],
            "gradient_cosine": [
                min(float(row["gradient_cosine"][index]) for row in rank_results)
                for index in range(len(result["gradient_cosine"]))
            ],
            "gradient_finite": [
                all(bool(row["gradient_finite"][index]) for row in rank_results)
                for index in range(len(result["gradient_finite"]))
            ],
            "rank_results": rank_results,
        }
    )
    output = Path(args.output)
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print(json.dumps(aggregate, sort_keys=True))
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0 if aggregate["status"] == "pass" and all(aggregate["gradient_finite"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
