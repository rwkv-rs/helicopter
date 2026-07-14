from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .artifacts import file_sha256


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_preflight(product_root: Path) -> dict[str, Any]:
    rwkv_hf = importlib.import_module("rwkv7_hf")
    modelopt_available = importlib.util.find_spec("modelopt") is not None
    vllm_available = importlib.util.find_spec("vllm") is not None
    transformers_version = _distribution_version("transformers")
    modelopt_hf_transformers_supported = bool(
        transformers_version
        and int(transformers_version.split(".", 1)[0]) < 5
    )
    kernel_source = product_root / "src/train/rwkv-lm/src/model.py"
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    nvfp4_capable = bool(
        modelopt_available
        and devices
        and all(device["compute_capability"][0] >= 12 for device in devices)
    )
    return {
        "schema_version": 1,
        "host": platform.node(),
        "python": {"version": sys.version, "executable": sys.executable},
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "cuda_devices": devices,
        "rwkv_hf": {
            "import_name": "rwkv7_hf",
            "module_file": str(Path(rwkv_hf.__file__).resolve()),
            "distribution_version": _distribution_version("rwkv-hf-adapter"),
        },
        "rwkv_lm": {
            "kernel_source": str(kernel_source),
            "kernel_source_sha256": (
                file_sha256(kernel_source) if kernel_source.is_file() else None
            ),
            "loader": "any2rwkv.kernel.load_rwkv_lm_kernel",
        },
        "vllm": {
            "available": vllm_available,
            "distribution_version": _distribution_version("vllm"),
            "loader_architectures": [
                "Any2RWKV7ForCausalLM",
                "Any2RWKVProxyForCausalLM",
            ],
        },
        "nvfp4": {
            "modelopt_available": modelopt_available,
            "modelopt_version": _distribution_version("nvidia-modelopt"),
            "hardware_capable": nvfp4_capable,
            "transformers_version": transformers_version,
            "modelopt_hf_declared_transformers_range": ">=4.53,<5.0",
            "modelopt_hf_transformers_supported": modelopt_hf_transformers_supported,
            "compatibility_status": (
                "supported"
                if modelopt_hf_transformers_supported
                else "requires-runtime-validation-with-transformers-5"
            ),
            "policy": "matrix weights only; FP32 recurrent state and mixed-precision dynamic/Norm/embedding/LM head",
        },
        "passed": bool(
            torch.cuda.is_available()
            and Path(rwkv_hf.__file__).is_file()
            and kernel_source.is_file()
            and vllm_available
        ),
    }
