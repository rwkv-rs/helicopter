from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import checkpoint_sha256, write_json
from .quantize import nvfp4_performance_gate, nvfp4_quality_gate


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def validate_nvfp4_acceptance(
    *,
    bf16_checkpoint: Path,
    nvfp4_checkpoint: Path,
    bf16_quality_path: Path,
    nvfp4_quality_path: Path,
    p0_evidence_path: Path,
    service_evidence_path: Path,
    roundtrip_path: Path,
    performance_path: Path | None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the independent NVFP4 quality and optional performance delivery gates."""
    bf16_sha = checkpoint_sha256(bf16_checkpoint)
    nvfp4_sha = checkpoint_sha256(nvfp4_checkpoint)
    bf16_quality = _read(bf16_quality_path)
    nvfp4_quality = _read(nvfp4_quality_path)
    p0 = _read(p0_evidence_path)
    service = _read(service_evidence_path)
    roundtrip = _read(roundtrip_path)
    failures: list[str] = []
    if bf16_quality.get("gates", {}).get("P2", {}).get("passed") is not True:
        failures.append("bf16:P2-not-passed")
    binding = nvfp4_quality.get("binding", {})
    if binding.get("teacher_sha256") != bf16_sha:
        failures.append("quality:teacher-is-not-bf16-checkpoint")
    if binding.get("student_sha256") != nvfp4_sha:
        failures.append("quality:student-is-not-nvfp4-checkpoint")
    metrics = nvfp4_quality.get("metrics", {}).get("quality_metrics", {})
    required_metrics = (
        "ppl_ratio",
        "mean_token_kl",
        "ruler_ci_lower_ratio",
        "downstream_ci_lower_ratio",
    )
    if any(metrics.get(name) is None for name in required_metrics):
        failures.append("quality:missing-required-metric")
        quality_passed = False
    else:
        quality_passed = nvfp4_quality_gate(
            ppl_increase=float(metrics["ppl_ratio"]) - 1.0,
            mean_kl=float(metrics["mean_token_kl"]),
            ruler_ratio=float(metrics["ruler_ci_lower_ratio"]),
            downstream_ratio=float(metrics["downstream_ci_lower_ratio"]),
        )
        if not quality_passed:
            failures.append("quality:threshold-failed")
    evidence = p0.get("evidence")
    if p0.get("student_sha256") != nvfp4_sha or not isinstance(evidence, dict) or not evidence:
        failures.append("p0:checkpoint-binding-or-evidence-missing")
    elif any(not isinstance(row, dict) or row.get("passed") is not True for row in evidence.values()):
        failures.append("p0:evidence-failed")
    if (
        service.get("model_sha256") != nvfp4_sha
        or service.get("passed") is not True
        or service.get("model_impl") != "transformers"
        or service.get("loader_contract")
        != "generic-transformers-backend-not-pure-rwkv"
    ):
        failures.append("serving:checkpoint-binding-or-acceptance-failed")
    loading = roundtrip.get("loading_info")
    if (
        Path(str(roundtrip.get("checkpoint", ""))).resolve() != nvfp4_checkpoint.resolve()
        or not isinstance(loading, dict)
        or any(loading.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        or int(roundtrip.get("prompt_count", 0)) != 32
        or int(roundtrip.get("new_tokens", 0)) != 128
    ):
        failures.append("roundtrip:strict-reload-or-generation-failed")
    performance = None
    performance_passed = False
    if performance_path is not None:
        performance = _read(performance_path)
        if performance.get("bf16_sha256") != bf16_sha or performance.get("nvfp4_sha256") != nvfp4_sha:
            failures.append("performance:checkpoint-binding-failed")
        else:
            performance_passed = nvfp4_performance_gate(
                throughput_gain=float(performance["decode_throughput_gain"]),
                ttft_p95_regression=float(performance["ttft_p95_regression"]),
                tpot_p95_regression=float(performance["tpot_p95_regression"]),
                memory_reduction=float(performance["peak_memory_reduction"]),
            )
    quality_compatible = not failures and quality_passed
    result = {
        "schema_version": 1,
        "bf16_sha256": bf16_sha,
        "nvfp4_sha256": nvfp4_sha,
        "quality_compatible": quality_compatible,
        "performance_deliverable": quality_compatible and performance_path is not None and performance_passed,
        "label": (
            "nvfp4-performance-deliverable"
            if quality_compatible and performance_path is not None and performance_passed
            else "nvfp4-quality-compatible"
            if quality_compatible
            else "experimental-nvfp4"
        ),
        "failures": failures,
        "quality_metrics": {name: metrics.get(name) for name in required_metrics},
        "performance": performance,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
