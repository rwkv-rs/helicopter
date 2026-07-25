from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .artifacts import file_sha256, git_sha


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_preflight(
    product_root: Path,
    *,
    expected_rwkv_hf_sha: str,
    expected_rwkv_lm_sha: str,
) -> dict[str, Any]:
    rwkv_hf = importlib.import_module("rwkv7_hf")
    transformers_version = _distribution_version("transformers")
    rwkv_hf_checkout = (product_root / "src/train/rwkv-hf").resolve()
    rwkv_lm_checkout = (product_root / "src/train/rwkv-lm").resolve()
    rwkv_hf_module = Path(rwkv_hf.__file__).resolve()
    rwkv_hf_commit = git_sha(rwkv_hf_checkout)
    rwkv_lm_commit = git_sha(rwkv_lm_checkout)
    kernel_loader = rwkv_lm_checkout / "src/infctx_kernel.py"
    kernel_source = (
        rwkv_lm_checkout / "cuda/rwkv7_statepassing_clampw.cu"
    )
    kernel_binding = (
        rwkv_lm_checkout / "cuda/rwkv7_statepassing_pybind.cpp"
    )
    rwkv_hf_module_in_checkout = rwkv_hf_module.is_relative_to(rwkv_hf_checkout)
    rwkv_hf_commit_matches = rwkv_hf_commit == expected_rwkv_hf_sha
    rwkv_lm_commit_matches = rwkv_lm_commit == expected_rwkv_lm_sha
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
            "module_file": str(rwkv_hf_module),
            "module_in_checkout": rwkv_hf_module_in_checkout,
            "distribution_version": _distribution_version("rwkv-hf-adapter"),
            "checkout_commit": rwkv_hf_commit,
            "expected_commit": expected_rwkv_hf_sha,
            "commit_matches": rwkv_hf_commit_matches,
        },
        "rwkv_lm": {
            "checkout_commit": rwkv_lm_commit,
            "expected_commit": expected_rwkv_lm_sha,
            "commit_matches": rwkv_lm_commit_matches,
            "kernel_loader": str(kernel_loader),
            "kernel_loader_sha256": (
                file_sha256(kernel_loader) if kernel_loader.is_file() else None
            ),
            "kernel_source": str(kernel_source),
            "kernel_source_sha256": (
                file_sha256(kernel_source) if kernel_source.is_file() else None
            ),
            "kernel_binding": str(kernel_binding),
            "kernel_binding_sha256": (
                file_sha256(kernel_binding) if kernel_binding.is_file() else None
            ),
            "loader": "any2rwkv.kernel.load_rwkv_lm_kernel",
        },
        "transformers": {
            "transformers_version": transformers_version,
            "loader": "AutoModelForCausalLM.from_pretrained",
            "trust_remote_code": True,
        },
        "passed": bool(
            torch.cuda.is_available()
            and rwkv_hf_module.is_file()
            and rwkv_hf_module_in_checkout
            and rwkv_hf_commit_matches
            and rwkv_lm_commit_matches
            and kernel_loader.is_file()
            and kernel_source.is_file()
            and kernel_binding.is_file()
            and transformers_version is not None
        ),
    }
