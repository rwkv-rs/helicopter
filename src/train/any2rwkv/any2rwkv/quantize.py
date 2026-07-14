from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class QuantizationDecision:
    tensor: str
    quantized: bool
    source_dtype: str
    target_dtype: str
    reason: str


KEEP_HIGH_PRECISION = ("embedding", "embeddings", "lm_head", "norm", "state", "time_decay", "r_k")


def nvfp4_policy(tensors: Iterable[tuple[str, str, int]]) -> list[QuantizationDecision]:
    decisions: list[QuantizationDecision] = []
    for name, dtype, ndim in tensors:
        protected = next((token for token in KEEP_HIGH_PRECISION if token in name.lower()), None)
        if protected:
            decisions.append(QuantizationDecision(name, False, dtype, dtype, f"mixed-precision policy protects {protected}"))
        elif ndim == 3 and "experts." in name.lower():
            decisions.append(
                QuantizationDecision(
                    name,
                    True,
                    dtype,
                    "nvfp4",
                    "eligible fused MoE matrix collection",
                )
            )
        elif ndim != 2:
            decisions.append(QuantizationDecision(name, False, dtype, dtype, "NVFP4 profile only quantizes matrix weights"))
        else:
            decisions.append(QuantizationDecision(name, True, dtype, "nvfp4", "eligible dense/MoE matrix weight"))
    return decisions


def nvfp4_quality_gate(*, ppl_increase: float, mean_kl: float, ruler_ratio: float, downstream_ratio: float) -> bool:
    return ppl_increase <= 0.05 and mean_kl <= 0.05 and ruler_ratio >= 0.98 and downstream_ratio >= 0.98


def nvfp4_performance_gate(*, throughput_gain: float, ttft_p95_regression: float, tpot_p95_regression: float, memory_reduction: float) -> bool:
    return throughput_gain >= 0.15 and ttft_p95_regression <= 0.05 and tpot_p95_regression <= 0.05 and memory_reduction >= 0.10
