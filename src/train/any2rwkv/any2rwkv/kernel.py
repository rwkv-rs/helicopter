from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import importlib
import os
import threading

import torch
from torch import Tensor

from .errors import ContractError


_RWKV_LM_IMPORT_LOCK = threading.Lock()


class NativeRwkv7Kernel:
    """Single adapter boundary for rwkv-lm's state-passing CUDA contract."""

    def __init__(
        self,
        operation: Callable[..., tuple[Tensor, Tensor]],
        *,
        head_size: int,
        chunk_size: int = 16,
    ) -> None:
        self.operation = operation
        self.head_size = head_size
        self.chunk_size = chunk_size

    def __call__(self, state: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        if state.dtype != torch.float32:
            raise ContractError(f"native RWKV7 state must be float32, got {state.dtype}")
        vectors = (r, w, k, v, a, b)
        if any(value.dtype != torch.bfloat16 for value in vectors):
            raise ContractError("native RWKV7 r/w/k/v/a/b must all be bfloat16")
        if any(value.shape != r.shape for value in vectors):
            raise ContractError("native RWKV7 six signal tensors must have identical [B,T,C] shapes")
        batch, tokens, channels = r.shape
        if channels % self.head_size or tokens % self.chunk_size:
            raise ContractError(
                f"native RWKV7 requires channels%{self.head_size}=0 and tokens%{self.chunk_size}=0; "
                f"got channels={channels} tokens={tokens}"
            )
        expected_state = (batch, channels // self.head_size, self.head_size, self.head_size)
        if tuple(state.shape) != expected_state:
            raise ContractError(f"native RWKV7 state shape must be {expected_state}, got {tuple(state.shape)}")
        return self.operation(state.contiguous(), *(value.contiguous() for value in vectors))


@lru_cache(maxsize=4)
def load_rwkv_lm_kernel(head_size: int) -> NativeRwkv7Kernel:
    """Load the pinned rwkv-lm kernel from this product checkout only."""
    configured_head_size = int(os.environ.get("RWKV_HEAD_SIZE", "0"))
    expected_head_size = int(head_size)
    if expected_head_size <= 0:
        raise ContractError("RWKV_HEAD_SIZE must select a positive native head size")
    if configured_head_size not in {0, expected_head_size}:
        raise ContractError(
            "requested RWKV7 head size differs from RWKV_HEAD_SIZE: "
            f"requested={expected_head_size} configured={configured_head_size}"
        )
    product_root = Path(__file__).resolve().parents[4]
    checkout = product_root / "src/train/rwkv-lm"
    loader_file = checkout / "src/infctx_kernel.py"
    if not loader_file.is_file():
        raise ContractError(f"pinned rwkv-lm kernel source is missing: {loader_file}")
    module_name = f"_any2rwkv_rwkv_lm_infctx_kernel_n{expected_head_size}"
    with _RWKV_LM_IMPORT_LOCK:
        spec = importlib.util.spec_from_file_location(module_name, loader_file)
        if spec is None or spec.loader is None:
            raise ContractError(
                f"could not create pinned RWKV7 loader spec: {loader_file}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    resolved = Path(str(getattr(module, "__file__", ""))).resolve()
    if resolved != loader_file.resolve():
        raise ContractError(
            f"loaded RWKV7 kernel from unexpected checkout: {resolved}"
        )
    operation = module.load_statepassing_kernel(expected_head_size)
    return NativeRwkv7Kernel(operation, head_size=expected_head_size)
