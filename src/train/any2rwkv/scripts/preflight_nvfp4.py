#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json

import torch


def main() -> int:
    failures: list[str] = []
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    if capability < (12, 0):
        failures.append(f"NVFP4 requires Blackwell-class CUDA capability, found {capability}")
    try:
        import modelopt.torch.quantization  # noqa: F401
        modelopt_version = importlib.metadata.version("nvidia-modelopt")
    except Exception as error:
        modelopt_version = None
        failures.append(f"ModelOpt unavailable: {error}")
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
            ModelOptNvFp4FusedMoE,
            ModelOptNvFp4LinearMethod,
        )

        vllm_support = {
            "dense": ModelOptNvFp4LinearMethod.__name__,
            "moe": ModelOptNvFp4FusedMoE.__name__,
            "config": ModelOptNvFp4Config.__name__,
        }
    except Exception as error:
        vllm_support = None
        failures.append(f"vLLM ModelOpt NVFP4 loader unavailable: {error}")
    result = {
        "status": "pass" if not failures else "fail",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "modelopt_version": modelopt_version,
        "vllm_support": vllm_support,
        "policy": {
            "dense_and_moe_matrix_weights": "eligible",
            "rwkv7_projection_weights": "eligible-as-linear",
            "recurrent_dynamic_signals": "not-serialized",
            "recurrent_state": "fp32",
            "norm_embedding_lm_head": "preserve-by-default",
        },
        "failures": failures,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
