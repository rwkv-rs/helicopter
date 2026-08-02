from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...contract import _text_config
from ...errors import ContractError

_MODEL_GEOMETRY = {
    "Qwen/Qwen3.5-2B": (16, 128),
    "Qwen/Qwen3.5-397B-A17B": (64, 128),
}


@dataclass(frozen=True)
class GQAHeadStateMapping:
    query_head: int
    matrix_state: str
    prefix_value_mean_state: str


@dataclass(frozen=True)
class Qwen35GeometryContract:
    model_id: str
    gdn_num_heads: int
    gdn_head_size: int
    gqa_num_query_heads: int
    gqa_head_size: int
    gqa_state_mappings: tuple[GQAHeadStateMapping, ...]


def model_id_from_config(config: Mapping[str, object]) -> str:
    candidates = (
        config.get("model_id"),
        config.get("_name_or_path"),
        config.get("name_or_path"),
    )
    model_ids = {str(value) for value in candidates if isinstance(value, str) and value}
    if len(model_ids) > 1:
        raise ContractError(
            "Qwen3.5 source must declare exactly one canonical model ID"
        )
    if model_ids:
        model_id = model_ids.pop()
        if model_id not in _MODEL_GEOMETRY:
            raise ContractError(f"unknown Qwen3.5 text model ID: {model_id!r}")
        return model_id
    text = _text_config(config)
    observed = (
        int(text.get("linear_num_value_heads", 0)),
        int(text.get("linear_value_head_dim", 0)),
    )
    matches = tuple(
        model_id
        for model_id, geometry in _MODEL_GEOMETRY.items()
        if geometry == observed
    )
    if len(matches) != 1:
        raise ContractError(
            "Qwen3.5 source without a model ID must have uniquely supported "
            "source-native GDN geometry"
        )
    return matches[0]


def canonical_gqa_state_mappings(
    num_query_heads: int,
) -> tuple[GQAHeadStateMapping, ...]:
    if num_query_heads <= 0:
        raise ContractError("GQA query-head count must be positive")
    return tuple(
        GQAHeadStateMapping(
            query_head=index,
            matrix_state=f"gqa.query_heads.{index}.matrix_state",
            prefix_value_mean_state=(
                f"gqa.query_heads.{index}.prefix_value_mean_state"
            ),
        )
        for index in range(num_query_heads)
    )


def validate_qwen35_geometry(
    config: Mapping[str, object],
    *,
    gqa_state_mappings: Sequence[GQAHeadStateMapping] | None = None,
) -> Qwen35GeometryContract:
    model_id = model_id_from_config(config)
    expected_gdn_heads, expected_gdn_head_size = _MODEL_GEOMETRY[model_id]
    text = _text_config(config)
    gdn_heads = int(text.get("linear_num_value_heads", 0))
    gdn_key_heads = int(text.get("linear_num_key_heads", 0))
    gdn_head_size = int(text.get("linear_value_head_dim", 0))
    gdn_key_head_size = int(text.get("linear_key_head_dim", 0))
    if (
        gdn_heads != expected_gdn_heads
        or gdn_key_heads != expected_gdn_heads
        or gdn_head_size != expected_gdn_head_size
        or gdn_key_head_size != expected_gdn_head_size
    ):
        raise ContractError(
            f"{model_id} GDN geometry drifted; expected "
            f"{expected_gdn_heads}x{expected_gdn_head_size} source-native state"
        )

    query_heads = int(text.get("num_attention_heads", 0))
    gqa_head_size = int(text.get("head_dim", 0))
    if gqa_head_size != 256 or query_heads * gqa_head_size != gdn_heads * gdn_head_size:
        raise ContractError(
            "Qwen3.5 GQA must preserve 256-wide query heads aligned to GDN width"
        )
    mappings = tuple(
        gqa_state_mappings
        if gqa_state_mappings is not None
        else canonical_gqa_state_mappings(query_heads)
    )
    by_head = {entry.query_head: entry for entry in mappings}
    if len(by_head) != len(mappings) or set(by_head) != set(range(query_heads)):
        raise ContractError(
            "GQA source query-head mapping coverage is incomplete or duplicated"
        )
    if any(
        not entry.matrix_state or not entry.prefix_value_mean_state
        for entry in mappings
    ):
        raise ContractError(
            "every GQA query head requires matrix-state and prefix-value-mean mappings"
        )
    return Qwen35GeometryContract(
        model_id=model_id,
        gdn_num_heads=gdn_heads,
        gdn_head_size=gdn_head_size,
        gqa_num_query_heads=query_heads,
        gqa_head_size=gqa_head_size,
        gqa_state_mappings=mappings,
    )
