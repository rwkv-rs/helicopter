from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from torch import Tensor, nn

from ..errors import ContractError

_DIGEST_FIELDS = (
    "data_sha256",
    "weight_sha256",
    "trace_sha256",
    "solver_sha256",
    "state_sha256",
    "current_cache_sha256",
    "next_cache_sha256",
)


@dataclass(frozen=True)
class LayerMajorResumeContract:
    """Serializable, fail-closed cursor for a rolling layer-input cache."""

    schema_version: int
    layer_index: int
    optimizer_step: int
    data_cursor: int
    data_sha256: str
    weight_sha256: str
    trace_sha256: str
    solver_sha256: str
    state_sha256: str
    current_cache_sha256: str
    next_cache_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ContractError("layer-major resume schema version must be 1")
        if any(
            type(value) is not int
            for value in (self.layer_index, self.optimizer_step, self.data_cursor)
        ):
            raise ContractError("layer-major resume cursor values must be integers")
        if min(self.layer_index, self.optimizer_step, self.data_cursor) < 0:
            raise ContractError("layer-major resume cursor values must be non-negative")
        for field in _DIGEST_FIELDS:
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value.lower() != value
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ContractError(f"{field} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        expected_digests: Mapping[str, str] | None = None,
    ) -> LayerMajorResumeContract:
        expected_fields = {field.name for field in cls.__dataclass_fields__.values()}
        if set(payload) != expected_fields:
            raise ContractError("layer-major resume fields are missing or unknown")
        try:
            contract = cls(**payload)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ContractError("layer-major resume field types are invalid") from error
        for field, expected in (expected_digests or {}).items():
            if field not in _DIGEST_FIELDS:
                raise ContractError(
                    f"unknown layer-major resume digest binding: {field}"
                )
            if getattr(contract, field) != expected:
                raise ContractError(
                    f"layer-major resume {field} differs from current input"
                )
        return contract


def canonical_digest(payload: object) -> str:
    """Hash JSON-compatible contract inputs without relying on object identity."""

    def encode(value: object) -> object:
        if isinstance(value, Tensor):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": value.detach().cpu().tolist(),
            }
        raise TypeError(f"unsupported digest input: {type(value).__name__}")

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=encode,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def activate_layer_major_training(
    *, teacher: nn.Module, layers: Sequence[nn.Module], layer_index: int
) -> tuple[nn.Parameter, ...]:
    """Freeze the teacher and every student layer except the active layer."""

    if not 0 <= layer_index < len(layers):
        raise ContractError(f"active layer out of range: {layer_index}")
    teacher.eval().requires_grad_(False)
    active: list[nn.Parameter] = []
    seen: set[int] = set()
    for index, layer in enumerate(layers):
        for parameter in layer.parameters():
            identity = id(parameter)
            if identity in seen:
                raise ContractError(
                    "student layers share a parameter across ownership boundaries"
                )
            seen.add(identity)
            parameter.requires_grad_(index == layer_index)
            parameter.grad = None
            if index == layer_index:
                active.append(parameter)
    if not active:
        raise ContractError("active layer has no trainable parameters")
    return tuple(active)


def assert_layer_major_isolation(
    *, teacher: nn.Module, layers: Sequence[nn.Module], layer_index: int
) -> None:
    if teacher.training or any(
        parameter.requires_grad for parameter in teacher.parameters()
    ):
        raise ContractError("teacher must remain frozen in eval mode")
    for index, layer in enumerate(layers):
        for parameter in layer.parameters():
            if parameter.requires_grad != (index == layer_index):
                raise ContractError("student trainability escaped the active layer")
            if index != layer_index and parameter.grad is not None:
                raise ContractError("inactive student layer accumulated a gradient")
