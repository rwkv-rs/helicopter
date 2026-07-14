from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_RUN_FILES = (
    "contract.lock.json",
    "mapping.json",
    "mapping-coverage.json",
    "kernel-oracle.json",
    "loss-bridge.json",
    "active-layer-trace.jsonl",
    "roundtrip-manifest.json",
    "resume-parity.json",
    "migration-baselines.json",
    "quality.json",
    "p0-evidence.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256(path: Path) -> str:
    """Hash the immutable HF config and weight payload as one checkpoint identity."""
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and (
            candidate.name in {"config.json", "model.safetensors.index.json"}
            or candidate.suffix == ".safetensors"
        )
    )
    if not files:
        raise ValueError(f"checkpoint has no config/weights: {path}")
    for candidate in files:
        digest.update(candidate.name.encode())
        digest.update(file_sha256(candidate).encode())
    return digest.hexdigest()


def default_contract_lock(product_root: Path | None = None) -> dict[str, Any]:
    lock = {
        "schema_version": 1,
        "change": "qwen35-rwkv7-conversion",
        "scope": "qwen3.5-text-only-to-native-rwkv7",
        "layers": 60,
        "canonical": {
            "equation": "S_t = S_{t-1} A_t + B_t",
            "native_update": "S= S*diag(decay) + (S*a)b^T + v*k^T; y=S*r",
            "state_orientation": "batch,head,value,key",
            "native_head_size": 64,
            "kernel": "rwkv-lm/RWKV7_STATEPASSING_CLAMPW_CUDA",
            "gdn_condition": "Qwen3.5 head-scalar decay and normalized key with matching state/head geometry",
            "gdn_mapping": "w=d; a=-k; b=(d*beta)k; v'=beta*v; k'=k; r=q/sqrt(Dk)",
        },
        "oracle": {
            "reference": "any2rwkv.recurrent.rwkv7_scan",
            "source_reference": "vllm/tests/kernels/mamba/cpu/test_cpu_gdn_ops.py::ref_gated_delta_rule",
            "fixture_count": 32,
            "seed": 20260714,
            "lengths": [1, 2, 15, 16, 17, 31, 32, 65],
            "chunks": [1, 2, 7, 16, 31],
            "tolerances": {
                "fp64_output_relative_l2": 1e-12,
                "fp64_output_max_abs": 1e-12,
                "fp64_state_relative_l2": 1e-12,
                "gradient_relative_l2": 1e-11,
                "gradient_cosine": 0.999999999999,
                "finite_difference_relative_error": 1e-6,
            },
        },
        "burn_in": {"seed": 20260714, "reset": "document", "cold_and_warmed": True},
        "corrective_sweeps": {"order": "59..0", "min_sweeps": 1, "max_sweeps": 3, "min_delta": 0.001},
        "bootstrap": {"samples": 10000, "seed": 20260714, "method": "paired-percentile", "confidence": 0.95},
        "serving": {"warmups": 20, "requests": 100, "logprob_max_abs": 0.001, "memory_drift": "max(2%,256MiB)"},
    }
    if product_root is not None:
        references = {
            "rwkv7_fp64": product_root / "src/train/any2rwkv/any2rwkv/recurrent.py",
            "gdn_mapping": product_root / "src/train/any2rwkv/any2rwkv/migration.py",
            "oracle_fixture": product_root / "src/train/any2rwkv/any2rwkv/oracle.py",
            "qwen35_gdn_cpu": product_root / "src/infer/vllm-rwkv/tests/kernels/mamba/cpu/test_cpu_gdn_ops.py",
        }
        missing = [str(path) for path in references.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"canonical reference files are missing: {missing}")
        lock["oracle"]["reference_files"] = {
            name: {"path": str(path.relative_to(product_root)), "sha256": file_sha256(path)}
            for name, path in references.items()
        }
    return lock


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def git_sha(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        manifest = path / ".helicopter-dev/source-revisions.json"
        try:
            revision = json.loads(manifest.read_text(encoding="utf-8"))["product_commit"]
        except (OSError, KeyError, json.JSONDecodeError) as manifest_error:
            raise RuntimeError(
                f"cannot resolve product commit for {path}: {error}; "
                f"managed revision manifest unavailable: {manifest_error}"
            ) from error
        if not isinstance(revision, str) or len(revision) != 40:
            raise RuntimeError(f"invalid managed product commit: {revision!r}")
        return revision


def initialize_run(
    output: Path,
    *,
    run_id: str,
    source: dict[str, Any],
    precision: str,
    command: list[str],
    product_root: Path,
    rwkv_hf_sha: str,
    rwkv_lm_sha: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    lock = default_contract_lock(product_root)
    write_json(output / "contract.lock.json", lock)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "workspace": "feat-any2rwkv",
        "change": "qwen35-rwkv7-conversion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initialized",
        "source": source,
        "product_commit": git_sha(product_root),
        "submodules": {"rwkv-hf": rwkv_hf_sha, "rwkv-lm": rwkv_lm_sha},
        "precision": precision,
        "wkv_mode": "fp32io16" if precision != "nvfp4" else "fp16",
        "state_dtype": "fp32",
        "io_dtype": "bf16/fp16",
        "command": command,
        "host": platform.node(),
        "platform": platform.platform(),
        "contract_sha256": sha256_json(lock),
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def verify_run_bundle(output: Path) -> list[str]:
    missing = [name for name in ("metadata.json", *REQUIRED_RUN_FILES) if not (output / name).is_file()]
    if missing:
        raise ValueError(f"incomplete run artifact bundle: {missing}")
    json_names = [
        name
        for name in ("metadata.json", *REQUIRED_RUN_FILES)
        if name.endswith(".json")
    ]
    payloads: dict[str, Any] = {}
    for name in json_names:
        try:
            payload = json.loads((output / name).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"run artifact is not valid JSON: {name}") from error
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"run artifact is empty or not an object: {name}")
        payloads[name] = payload
    quality = payloads["quality.json"]
    if quality.get("gates", {}).get("P0", {}).get("passed") is not True:
        raise ValueError("run bundle quality.json does not contain an accepted P0 gate")
    p0 = payloads["p0-evidence.json"]
    student_sha = p0.get("student_sha256")
    if not isinstance(student_sha, str) or len(student_sha) != 64:
        raise ValueError("run bundle P0 manifest lacks a student checkpoint SHA-256")
    if payloads["migration-baselines.json"].get("student_sha256") != student_sha:
        raise ValueError("migration baselines and P0 evidence bind different students")
    for name in ("kernel-oracle.json", "loss-bridge.json", "resume-parity.json"):
        if (
            payloads[name].get("passed") is not True
            or payloads[name].get("student_sha256") != student_sha
        ):
            raise ValueError(f"run artifact is not accepted or student-bound: {name}")
    trace_path = output / "active-layer-trace.jsonl"
    rows = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("active-layer-trace.jsonl is empty")
    for line in rows:
        if not isinstance(json.loads(line), dict):
            raise ValueError("active-layer-trace.jsonl contains a non-object row")
    return [name for name in ("metadata.json", *REQUIRED_RUN_FILES)]


def verify_scale_gate(output: Path) -> dict[str, str]:
    """Require the accepted real-proxy evidence before fetching the 397B source."""
    verify_run_bundle(output)
    quality_path = output / "quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    for gate in ("P0", "migration", "P1"):
        if quality.get("gates", {}).get(gate, {}).get("passed") is not True:
            raise ValueError(f"397B scale gate requires accepted {gate} evidence")
    p0_path = output / "p0-evidence.json"
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    student_sha = str(p0.get("student_sha256", ""))
    service_path = output / "vllm-service.json"
    if not service_path.is_file():
        raise ValueError("397B scale gate requires vllm-service.json")
    service = json.loads(service_path.read_text(encoding="utf-8"))
    if (
        service.get("schema_version") != 1
        or service.get("passed") is not True
        or service.get("model_sha256") != student_sha
        or service.get("warmups") != 20
        or service.get("requests") != 100
    ):
        raise ValueError("397B scale gate requires accepted student-bound BF16 serving evidence")
    smoke_path = output / "smoke-rubric.json"
    if not smoke_path.is_file():
        raise ValueError("397B scale gate requires smoke-rubric.json")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("student_sha256") != student_sha or smoke.get("passed") is not True:
        raise ValueError("397B scale gate smoke rubric is not accepted or student-bound")
    return {
        "student_sha256": student_sha,
        "quality_sha256": file_sha256(quality_path),
        "p0_sha256": file_sha256(p0_path),
        "service_sha256": file_sha256(service_path),
        "smoke_rubric_sha256": file_sha256(smoke_path),
    }
