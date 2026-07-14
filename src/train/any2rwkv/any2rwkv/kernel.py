from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import importlib
import os
import sys
import threading

import torch
from torch import Tensor

from .errors import ContractError


_RWKV_LM_IMPORT_LOCK = threading.Lock()


class NativeRwkv7Kernel:
    """Single adapter boundary for rwkv-lm's state-passing CUDA contract."""

    def __init__(self, operation: Callable[..., tuple[Tensor, Tensor]], *, head_size: int = 64, chunk_size: int = 16) -> None:
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


@lru_cache(maxsize=1)
def load_rwkv_lm_kernel() -> NativeRwkv7Kernel:
    """Load the pinned rwkv-lm kernel from this product checkout only."""
    product_root = Path(__file__).resolve().parents[4]
    checkout = product_root / "src/train/rwkv-lm"
    model_file = checkout / "src/model.py"
    if not model_file.is_file():
        raise ContractError(f"pinned rwkv-lm kernel source is missing: {model_file}")
    existing = sys.modules.get("src.model")
    if existing is not None:
        resolved = Path(str(getattr(existing, "__file__", ""))).resolve()
        if resolved != model_file.resolve():
            raise ContractError(
                f"Python module src.model already resolves outside pinned rwkv-lm: {resolved}"
            )
        operation = existing.RWKV7_STATEPASSING_CLAMPW_CUDA
    else:
        # RWKV-LM's pinned model module passes relative ``cuda/...`` source
        # paths to torch's extension loader.  Resolve those paths from the
        # checkout without leaking a changed process cwd after import.
        with _RWKV_LM_IMPORT_LOCK:
            previous_cwd = Path.cwd()
            sys.path.insert(0, str(checkout))
            try:
                os.chdir(checkout)
                module = importlib.import_module("src.model")
            finally:
                os.chdir(previous_cwd)
                sys.path.remove(str(checkout))
        resolved = Path(str(module.__file__)).resolve()
        if resolved != model_file.resolve():
            raise ContractError(
                f"loaded RWKV7 kernel from unexpected checkout: {resolved}"
            )
        operation = module.RWKV7_STATEPASSING_CLAMPW_CUDA
    return NativeRwkv7Kernel(operation)
