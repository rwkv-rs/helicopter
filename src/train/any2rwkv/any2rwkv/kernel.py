from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
from collections.abc import Callable
from functools import lru_cache

import torch
from torch import Tensor

from .errors import ContractError
from .provenance import github_repository_matches

FLA_RWKV7_REVISION = "a4a8aa98df6ec5322f194a80ec57363dd045adfc"
FLA_RWKV7_SOURCE_URL = "https://github.com/rwkv-rs/fla-rwkv.git"
FLA_RWKV7_REQUIREMENT = (
    "flash-linear-attention[flash-rwkv] @ git+"
    f"{FLA_RWKV7_SOURCE_URL}@{FLA_RWKV7_REVISION}"
)
FLASH_RWKV_REVISION = "866aafd2eed146b0eda1ce03444009ae030f89e3"
FLASH_RWKV_SOURCE_URL = "https://github.com/rwkv-rs/FlashRWKV.git"
_INJECTABLE_PROVIDERS = frozenset({"fla", "flash_rwkv"})
_RECURRENT_RWKV7_PARAMETERS = frozenset(
    {
        "r",
        "w",
        "k",
        "v",
        "a",
        "b",
        "scale",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "state_indices",
        "mode",
    }
)


def _require_exact_vcs_distribution(
    name: str,
    *,
    expected_url: str,
    expected_revision: str,
) -> None:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ContractError(
            f"required distribution is not installed: {name}"
        ) from error
    raw_direct_url = distribution.read_text("direct_url.json")
    if not raw_direct_url:
        raise ContractError(
            f"{name} must be installed from the pinned rwkv-rs VCS source, not a registry"
        )
    try:
        direct_url = json.loads(raw_direct_url)
    except json.JSONDecodeError as error:
        raise ContractError(f"{name} has invalid PEP 610 direct_url.json") from error
    vcs_info = direct_url.get("vcs_info", {})
    actual_url = direct_url.get("url")
    actual = (
        actual_url,
        vcs_info.get("vcs"),
        vcs_info.get("requested_revision"),
        vcs_info.get("commit_id"),
    )
    expected = (expected_url, "git", expected_revision, expected_revision)
    exact_metadata = actual[1:] == expected[1:]
    if not github_repository_matches(actual_url, expected_url) or not exact_metadata:
        raise ContractError(
            f"{name} VCS provenance mismatch: expected={expected!r} actual={actual!r}"
        )


class Rwkv7OperatorAdapter:
    """Pinned rwkv-rs operator boundary for Any-to-RWKV conversion."""

    def __init__(
        self,
        operation: Callable[..., tuple[Tensor, Tensor]],
        provider: Callable[[], str | None],
        *,
        head_size: int,
        require_flash: bool = True,
    ) -> None:
        if head_size <= 0:
            raise ContractError("RWKV7 operator head_size must be positive")
        self.operation = operation
        self.provider = provider
        self.head_size = int(head_size)
        self.require_flash = bool(require_flash)
        self.last_provider: str | None = None

    def __call__(
        self,
        r: Tensor,
        log_decay: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        *,
        initial_state: Tensor,
        cu_seqlens: Tensor | None = None,
        state_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        vectors = (r, log_decay, k, v, a, b)
        if any(value.ndim != 4 for value in vectors):
            raise ContractError("RWKV7 operator signals must use [B,T,H,K] layout")
        if any(value.shape != r.shape for value in vectors):
            raise ContractError(
                "RWKV7 operator six signal tensors must have identical shapes"
            )
        batch, _tokens, heads, head_size = r.shape
        if head_size != self.head_size:
            raise ContractError(
                f"RWKV7 operator signal head size must be {self.head_size}, got {head_size}"
            )
        if initial_state.dtype != torch.float32:
            raise ContractError(
                f"RWKV7 operator state must be float32, got {initial_state.dtype}"
            )
        if initial_state.ndim != 4 or tuple(initial_state.shape[1:]) != (
            heads,
            head_size,
            head_size,
        ):
            raise ContractError(
                "RWKV7 operator initial state must use [N,H,K,V] layout matching signals"
            )
        if (
            cu_seqlens is None
            and state_indices is None
            and initial_state.shape[0] != batch
        ):
            raise ContractError(
                "fixed-batch RWKV7 operator state rows must equal the batch size"
            )

        try:
            result = self.operation(
                *(value.contiguous() for value in vectors),
                initial_state=initial_state.contiguous(),
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                mode="fp32io16",
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise ContractError(f"RWKV7 operator execution failed: {error}") from error
        selected_provider = self.provider()
        self.last_provider = selected_provider
        if selected_provider not in _INJECTABLE_PROVIDERS:
            raise ContractError(
                "RWKV7 operator adapter did not report an accepted provider: "
                f"{selected_provider!r}"
            )
        if self.require_flash and selected_provider != "flash_rwkv":
            raise ContractError(
                "FlashRWKV operator is required; selected provider was "
                f"{selected_provider!r}"
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise ContractError("RWKV7 operator must return (output, final_state)")
        output, final_state = result
        if not isinstance(output, Tensor) or output.shape != r.shape:
            raise ContractError("RWKV7 operator returned an invalid output")
        if (
            not isinstance(final_state, Tensor)
            or final_state.shape != initial_state.shape
        ):
            raise ContractError("RWKV7 operator returned an invalid final state")
        return output, final_state


@lru_cache(maxsize=4)
def load_rwkv7_operator_adapter(
    head_size: int,
) -> Rwkv7OperatorAdapter:
    """Load the pinned rwkv-rs operator chain and require FlashRWKV."""
    from .preflight import require_rwkv7_runtime

    require_rwkv7_runtime()
    _require_exact_vcs_distribution(
        "flash-linear-attention",
        expected_url=FLA_RWKV7_SOURCE_URL,
        expected_revision=FLA_RWKV7_REVISION,
    )
    _require_exact_vcs_distribution(
        "flash-rwkv",
        expected_url=FLASH_RWKV_SOURCE_URL,
        expected_revision=FLASH_RWKV_REVISION,
    )
    try:
        rwkv7 = importlib.import_module("fla.ops.rwkv7")
    except ImportError as error:
        raise ContractError(
            f"RWKV7 operator runtime is unavailable; install {FLA_RWKV7_REQUIREMENT}"
        ) from error
    operation = getattr(rwkv7, "recurrent_rwkv7", None)
    provider = getattr(rwkv7, "get_last_rwkv7_provider", None)
    if not callable(operation) or not callable(provider):
        raise ContractError(
            "RWKV7 operator distribution must expose recurrent_rwkv7 and "
            "get_last_rwkv7_provider"
        )
    try:
        parameters = inspect.signature(operation).parameters
    except (TypeError, ValueError) as error:
        raise ContractError(
            "RWKV7 recurrent operator has no inspectable public signature"
        ) from error
    missing = sorted(_RECURRENT_RWKV7_PARAMETERS - parameters.keys())
    if missing:
        raise ContractError(
            "RWKV7 recurrent operator public signature is incompatible; "
            f"missing={missing}"
        )
    return Rwkv7OperatorAdapter(
        operation,
        provider,
        head_size=head_size,
        require_flash=True,
    )
