#!/usr/bin/env python3
"""Run one strict-sync training command with fail-closed provenance and GPU telemetry."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHANGE_ID = "strict-on-policy-maxrl-scheduling"
EXPECTED_GPU_INDICES = tuple(range(8))
IDLE_UTILIZATION_THRESHOLD_PERCENT = 10.0
GPU_SAMPLE_INTERVAL_MS = 100
MISSING_SAMPLE_DETECTION_MS = 250
MAX_GPU_SAMPLE_GAP_MS = 1000
MIN_GPU_SAMPLE_COVERAGE = 0.99
CORRECTNESS_MIN_ROLLOUT_IS_ESS = 0.30
CORRECTNESS_MAX_ROLLOUT_IS_WEIGHT = 2.0
FORMAL_WARMUP_STEPS = 2
FORMAL_TIMED_STEPS = 5
FORMAL_PERFORMANCE_PHASES = {
    "baseline",
    "topology",
    "training-capacity",
    "rollout-capacity",
    "global-batch",
    "global-batch-quality",
    "optimization",
}
DIAGNOSTIC_PHASES = {"nsys"}
VALID_RUN_PHASES = {"correctness", *FORMAL_PERFORMANCE_PHASES, *DIAGNOSTIC_PHASES}
PROCESS_STOP_TIMEOUT_SECONDS = 10
NSYS_REQUIRED_MARKERS = {"gen", "update_actor", "step"}


def telemetry_cpu_affinity_plan() -> tuple[int, tuple[int, ...]]:
    """Reserve one allowed CPU for telemetry and return the child CPU set."""

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("strict GPU telemetry requires Linux CPU affinity support")
    available = tuple(sorted(os.sched_getaffinity(0)))
    if len(available) < 2:
        raise RuntimeError("strict GPU telemetry requires at least two allowed CPUs")
    return available[-1], available[:-1]


def set_current_process_affinity(cpus: tuple[int, ...]) -> None:
    os.sched_setaffinity(0, set(cpus))


def validate_run_phase(value: str) -> str:
    if value not in VALID_RUN_PHASES:
        raise RuntimeError(
            f"HELICOPTER_RUN_PHASE must be one of {sorted(VALID_RUN_PHASES)}, got {value!r}"
        )
    return value


def resolve_run_dir(root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"REMOTE_RUN_LOG_DIR must stay within the workspace root: {resolved}") from exc
    return resolved


def strict_child_environment(base: dict[str, str], run_dir: Path) -> dict[str, str]:
    child_env = dict(base)
    child_env["REMOTE_RUN_LOG_DIR"] = str(run_dir)
    child_env["VERL_FILE_LOGGER_PATH"] = str(run_dir / "metrics.jsonl")
    child_env["VERL_POLICY_IDENTITY_LOG_PATH"] = str(run_dir / "policy_identity.jsonl")
    return child_env


def stop_process(process: subprocess.Popen[Any] | None, *, process_group: bool = False) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if process_group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            if process_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    if len(expected_sha256) != 64:
        raise RuntimeError(f"{label} SHA-256 must contain 64 hexadecimal characters")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual}


def verify_dataset_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"dataset manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("dataset manifest must contain a non-empty files list")
    verified = []
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) < {"path", "sha256"}:
            raise RuntimeError(f"dataset manifest entry {index} must contain path and sha256")
        verified.append(verify_file(Path(item["path"]), item["sha256"], label=f"dataset file {index}"))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "files": verified,
    }


def parse_visible_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be a comma-separated integer list, got {value!r}") from exc
    if devices != EXPECTED_GPU_INDICES:
        raise RuntimeError(f"strict training requires CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, got {value!r}")
    return devices


def command_output(*command: str) -> str:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def source_metadata(root: Path, source_revisions: dict[str, Any] | None) -> dict[str, Any]:
    if not (root / ".git").exists():
        if not isinstance(source_revisions, dict) or not source_revisions.get("product_commit"):
            raise RuntimeError(
                "remote source revision manifest is missing; the synchronized workspace intentionally has no .git"
            )
        submodules = source_revisions.get("submodules", {})
        if not isinstance(submodules, dict):
            raise RuntimeError("remote source revision manifest has an invalid submodules mapping")
        return {
            "commit": source_revisions["product_commit"],
            "branch": None,
            "dirty": None,
            "status": [],
            "patch_sha256": None,
            "submodules": [f"{revision} {path}" for path, revision in sorted(submodules.items())],
            "git_metadata_available": False,
            "source": "synchronized revision manifest",
        }

    status = command_output("git", "-C", str(root), "status", "--short", "--ignore-submodules=none")
    patch = command_output("git", "-C", str(root), "diff", "--binary", "--no-ext-diff") if status else ""
    submodules = command_output("git", "-C", str(root), "submodule", "status", "--recursive")
    return {
        "commit": command_output("git", "-C", str(root), "rev-parse", "HEAD"),
        "branch": command_output("git", "-C", str(root), "branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest() if patch else None,
        "submodules": submodules.splitlines(),
        "git_metadata_available": True,
        "source": "git checkout",
    }


def environment_metadata() -> dict[str, Any]:
    import torch

    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver_and_gpus": command_output(
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ).splitlines(),
        "nvcc": command_output("nvcc", "--version"),
    }


def parse_gpu_samples(path: Path) -> dict[str, Any]:
    per_gpu: dict[int, list[dict[str, Any]]] = {index: [] for index in EXPECTED_GPU_INDICES}
    sequences: dict[int, set[int]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    sequence = int(row["sequence"])
                    index = int(row["gpu_index"])
                    parsed = {
                        "sequence": sequence,
                        "monotonic_ns": int(row["monotonic_ns"]),
                        "utilization_gpu_percent": float(row["utilization_gpu_percent"]),
                        "power_watts": float(row["power_watts"]),
                        "memory_used_mib": float(row["memory_used_mib"]),
                    }
                    per_gpu[index].append(parsed)
                    sequences.setdefault(sequence, set()).add(index)
                except (ValueError, KeyError):
                    continue

    summary = {}
    missing_gpus = []
    for index, samples in per_gpu.items():
        if not samples:
            missing_gpus.append(index)
            summary[str(index)] = {"samples": 0, "missing": True}
            continue
        gaps_ms = [
            (right["monotonic_ns"] - left["monotonic_ns"]) / 1_000_000
            for left, right in zip(samples, samples[1:], strict=False)
        ]
        sorted_gaps = sorted(gaps_ms)
        p95_gap = (
            sorted_gaps[max(0, math.ceil(len(sorted_gaps) * 0.95) - 1)] if sorted_gaps else 0.0
        )
        missing_intervals = [
            {
                "after_sequence": left["sequence"],
                "before_sequence": right["sequence"],
                "gap_ms": gap,
            }
            for left, right, gap in zip(samples, samples[1:], gaps_ms, strict=False)
            if gap > MISSING_SAMPLE_DETECTION_MS
        ]
        estimated_missing_samples = sum(
            max(1, round(interval["gap_ms"] / GPU_SAMPLE_INTERVAL_MS) - 1)
            for interval in missing_intervals
        )
        sample_coverage = len(samples) / (len(samples) + estimated_missing_samples)
        idle_samples = sum(
            sample["utilization_gpu_percent"] < IDLE_UTILIZATION_THRESHOLD_PERCENT for sample in samples
        )
        summary[str(index)] = {
            "samples": len(samples),
            "missing": False,
            "idle_fraction": idle_samples / len(samples),
            "peak_memory_used_mib": max(sample["memory_used_mib"] for sample in samples),
            "mean_power_watts": sum(sample["power_watts"] for sample in samples) / len(samples),
            "mean_sample_gap_ms": sum(gaps_ms) / len(gaps_ms) if gaps_ms else 0.0,
            "p95_sample_gap_ms": p95_gap,
            "max_sample_gap_ms": max(gaps_ms, default=0.0),
            "missing_interval_count": len(missing_intervals),
            "missing_intervals": missing_intervals,
            "estimated_missing_samples": estimated_missing_samples,
            "sample_coverage": sample_coverage,
        }
    expected = set(EXPECTED_GPU_INDICES)
    incomplete_sequences = {
        str(sequence): sorted(expected - indices)
        for sequence, indices in sequences.items()
        if indices != expected
    }
    return {
        "sampler_backend": "nvidia-ml-py",
        "clock": "monotonic_ns",
        "per_gpu": summary,
        "missing_gpus": missing_gpus,
        "incomplete_sequences": incomplete_sequences,
    }


def sample_gpus(samples_path: Path, ready_path: Path, heartbeat_path: Path | None = None) -> int:
    import pynvml

    stop = threading.Event()

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    heartbeat_thread = None
    if heartbeat_path is not None:
        def write_heartbeat():
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            with heartbeat_path.open("w", encoding="utf-8") as heartbeat:
                while not stop.is_set():
                    heartbeat.write(f"{time.monotonic_ns()}\n")
                    heartbeat.flush()
                    stop.wait(GPU_SAMPLE_INTERVAL_MS / 1000)

        heartbeat_thread = threading.Thread(target=write_heartbeat, name="gpu-sampler-heartbeat")
        heartbeat_thread.start()
    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in EXPECTED_GPU_INDICES]
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        def read_device(item: tuple[int, Any]) -> tuple[int, int, float, float, int]:
            index, gpu_handle = item
            utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            power_watts = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
            memory_used_mib = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 * 1024)
            return index, utilization.gpu, power_watts, memory_used_mib, time.monotonic_ns()

        with (
            samples_path.open("w", newline="", encoding="utf-8") as handle,
            concurrent.futures.ThreadPoolExecutor(
                max_workers=len(EXPECTED_GPU_INDICES), thread_name_prefix="nvml"
            ) as pool,
        ):
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "sequence",
                    "wall_time_utc",
                    "monotonic_ns",
                    "gpu_index",
                    "utilization_gpu_percent",
                    "power_watts",
                    "memory_used_mib",
                )
            )
            handle.flush()
            deadline = time.monotonic()
            sequence = 0
            while not stop.is_set():
                wall_time = datetime.now(timezone.utc).isoformat()
                device_samples = pool.map(
                    read_device, zip(EXPECTED_GPU_INDICES, handles, strict=True)
                )
                rows = [
                    (sequence, wall_time, monotonic_ns, index, utilization, power, memory)
                    for index, utilization, power, memory, monotonic_ns in device_samples
                ]
                writer.writerows(rows)
                handle.flush()
                if sequence == 0:
                    temporary_ready = ready_path.with_suffix(ready_path.suffix + ".tmp")
                    temporary_ready.write_text("ready\n", encoding="utf-8")
                    temporary_ready.replace(ready_path)
                sequence += 1
                deadline += GPU_SAMPLE_INTERVAL_MS / 1000
                stop.wait(max(0.0, deadline - time.monotonic()))
    finally:
        stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join()
        pynvml.nvmlShutdown()
    return 0


def parse_sampler_heartbeat(path: Path) -> dict[str, Any]:
    timestamps = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    gaps = [(right - left) / 1_000_000 for left, right in zip(timestamps, timestamps[1:])]
    return {"samples": len(timestamps), "max_gap_ms": max(gaps, default=0.0)}


def verify_policy_identity_log(path: Path, *, expected_rounds: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"strict policy identity artifact is missing: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or records[0].get("event") != "publish_initial":
        raise RuntimeError("strict policy identity artifact must start with publish_initial")

    required = {
        "event",
        "global_steps",
        "policy_version",
        "weight_digest",
        "sampling_config_digest",
        "runtime_identity",
    }
    missing = sorted(required - records[0].keys())
    if missing:
        raise RuntimeError(f"strict policy identity record is missing fields: {', '.join(missing)}")
    if expected_rounds <= 0:
        raise RuntimeError(f"expected training rounds must be positive, got {expected_rounds}")
    identity_fields = (
        "policy_version",
        "weight_digest",
        "sampling_config_digest",
        "runtime_identity",
    )

    def identity(record: dict[str, Any]) -> tuple[Any, ...]:
        for field in identity_fields[1:]:
            if not isinstance(record[field], str) or not record[field].strip():
                raise RuntimeError(f"strict policy identity field {field} must be a non-empty string")
        return tuple(record[field] for field in identity_fields)

    published_version = records[0]["policy_version"]
    published_identity = identity(records[0])
    if records[0]["global_steps"] != published_version:
        raise RuntimeError("publish_initial must have global_steps == policy_version")
    train_versions = []
    expected_event = "train_begin"
    for record in records[1:]:
        missing = sorted(required - record.keys())
        if missing:
            raise RuntimeError(f"strict policy identity record is missing fields: {', '.join(missing)}")
        version = record["policy_version"]
        if record["event"] != expected_event:
            raise RuntimeError(f"strict policy identity expected {expected_event}, found {record['event']}")
        if record["event"] == "train_begin":
            if record["global_steps"] != published_version + 1 or version != published_version:
                raise RuntimeError(
                    "train_begin must consume the current publication at the next global step: "
                    f"published={published_version}, step={record['global_steps']}, training={version}"
                )
            if identity(record) != published_identity:
                raise RuntimeError("train_begin identity must exactly match the current publication")
            effective_digest = record.get("effective_sampling_digest")
            if not isinstance(effective_digest, str) or not effective_digest:
                raise RuntimeError("train_begin must record a non-empty effective_sampling_digest")
            train_versions.append(version)
            expected_event = "publish"
        else:
            if record["global_steps"] != published_version + 1 or version != published_version + 1:
                raise RuntimeError(
                    "publish must advance global_steps and policy_version exactly once: "
                    f"published={published_version}, step={record['global_steps']}, next={version}"
                )
            next_identity = identity(record)
            if next_identity[1] == published_identity[1]:
                raise RuntimeError("publish must advance the weight digest after the actor update")
            published_version = version
            published_identity = next_identity
            expected_event = "train_begin"
    if expected_event != "train_begin":
        raise RuntimeError("strict policy identity artifact ended with an unpublished training round")
    if len(train_versions) != expected_rounds:
        raise RuntimeError(
            f"strict policy identity expected {expected_rounds} training rounds, found {len(train_versions)}"
        )
    return {
        "path": str(path),
        "record_count": len(records),
        "train_round_count": len(train_versions),
        "train_versions": train_versions,
        "last_published_version": published_version,
    }


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"strict training requires {name}")
    return value


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def tee_child_output(pipe, command_log, output_errors: list[OSError]) -> None:
    """Persist child output while preserving the live remote-run stream."""

    live_output_available = True
    read_chunk = getattr(pipe, "read1", pipe.read)
    while chunk := read_chunk(65536):
        try:
            command_log.write(chunk)
            command_log.flush()
        except OSError as exc:
            if not output_errors:
                output_errors.append(exc)
            continue
        if not live_output_available:
            continue
        try:
            stdout_buffer = getattr(sys.stdout, "buffer", None)
            if stdout_buffer is not None:
                stdout_buffer.write(chunk)
                stdout_buffer.flush()
            else:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        except OSError:
            live_output_available = False


def classify_failure(child_exit_code: int, contract_error: str | None, command_log: Path) -> str:
    if child_exit_code == 0 and contract_error is None:
        return "none"
    text = command_log.read_text(encoding="utf-8", errors="replace") if command_log.is_file() else ""
    lowered = text.lower()
    if "cuda out of memory" in lowered or "cudaerroroutofmemory" in lowered:
        return "cuda_oom"
    if "out of memory" in lowered or "oom-kill" in lowered or "killed process" in lowered:
        return "host_oom"
    if contract_error is not None:
        return "contract_failure"
    return "child_failure"


def extract_vllm_capacity(command_log: Path, topology_path: Path | None = None) -> dict[str, Any]:
    text = command_log.read_text(encoding="utf-8", errors="replace") if command_log.is_file() else ""
    cache_tokens = [int(value.replace(",", "")) for value in re.findall(r"GPU KV cache size: ([0-9,]+) tokens", text)]
    maximum_concurrency = [
        float(value)
        for value in re.findall(r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x", text)
    ]
    structured = []
    structured_errors = []
    if topology_path is not None and topology_path.is_file():
        topology = json.loads(topology_path.read_text(encoding="utf-8"))
        deployments = topology.get("deployments", [])
        ranks = [deployment.get("replica_rank") for deployment in deployments]
        capacities = [deployment.get("capacity") for deployment in deployments]
        if len(deployments) == 8 and len(set(ranks)) == 8 and all(isinstance(item, dict) for item in capacities):
            required = {
                "capacity_mode",
                "kv_cache_applicable",
                "max_num_seqs",
                "max_num_batched_tokens",
                "gpu_memory_utilization",
            }
            for rank, capacity in zip(ranks, capacities, strict=True):
                missing = sorted(required - capacity.keys())
                if missing:
                    structured_errors.append(f"replica {rank} capacity missing fields: {missing}")
                    continue
                structured.append({"replica_rank": int(rank), **capacity})
        elif deployments:
            structured_errors.append("rollout topology does not contain 8 unique structured capacity records")
    modes = {item["capacity_mode"] for item in structured}
    kv_applicability = {bool(item["kv_cache_applicable"]) for item in structured}
    return {
        "replica_observation_count": (
            0 if structured_errors else (len(structured) or min(len(cache_tokens), len(maximum_concurrency)))
        ),
        "capacity_mode": next(iter(modes)) if len(modes) == 1 else "kv-cache",
        "kv_cache_applicable": next(iter(kv_applicability)) if len(kv_applicability) == 1 else True,
        "per_replica": structured,
        "structured_errors": structured_errors,
        "gpu_kv_cache_tokens": cache_tokens,
        "maximum_concurrency": maximum_concurrency,
    }


def verify_rollout_capacity_observations(observations: dict[str, Any], config: dict[str, Any]) -> None:
    if observations.get("structured_errors"):
        raise RuntimeError(f"invalid structured rollout capacity: {observations['structured_errors']}")
    records = observations.get("per_replica")
    if not isinstance(records, list) or len(records) != 8:
        raise RuntimeError("formal performance run requires structured rollout capacity from 8 replicas")
    takeoff = config["takeoff"]["grpo"]
    expected = {
        "capacity_mode": "recurrent-state-no-kv-cache",
        "kv_cache_applicable": False,
        "max_num_seqs": int(takeoff["rollout_max_num_seqs"]),
        "max_num_batched_tokens": int(takeoff["rollout_max_num_batched_tokens"]),
        "gpu_memory_utilization": float(takeoff["rollout_gpu_memory_utilization"]),
    }
    ranks = set()
    for record in records:
        rank = int(record.get("replica_rank", -1))
        ranks.add(rank)
        actual = {key: record.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"rollout replica {rank} capacity does not match resolved RWKV config: {actual} != {expected}")
        if int(record["max_num_seqs"]) <= 0 or int(record["max_num_batched_tokens"]) <= 0:
            raise RuntimeError(f"rollout replica {rank} reported non-positive scheduler capacity")
        if not 0 < float(record["gpu_memory_utilization"]) < 1:
            raise RuntimeError(f"rollout replica {rank} reported invalid gpu_memory_utilization")
    if ranks != set(range(8)):
        raise RuntimeError(f"structured rollout capacity requires replica ranks 0..7, got {sorted(ranks)}")


def verify_topology_contract(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"trainer_gpus", "rollout_replicas", "rollout_tp", "rollout_pp", "rollout_internal_dp"}
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"topology contract is missing fields: {', '.join(missing)}")
    topology = {key: int(payload[key]) for key in required}
    if topology["trainer_gpus"] != 8:
        raise RuntimeError("strict topology requires trainer_gpus=8")
    if topology["rollout_replicas"] != 8 or topology["rollout_tp"] != 1:
        raise RuntimeError("strict vLLM-RWKV topology requires 8 independent TP1 rollout replicas")
    if topology["rollout_pp"] != 1 or topology["rollout_internal_dp"] != 1:
        raise RuntimeError("vLLM-RWKV strict topology requires rollout_pp=1 and rollout_internal_dp=1")
    if topology["rollout_replicas"] * topology["rollout_tp"] != 8:
        raise RuntimeError("strict topology must consume the 8-GPU global pool exactly")
    return topology


def verify_declared_contract(
    config: dict[str, Any],
    *,
    seed: str,
    batch: dict[str, Any],
    topology: dict[str, Any],
    precision: str,
    wkv_mode: str,
) -> None:
    takeoff = config["takeoff"]["grpo"]
    if int(seed) != int(takeoff["seed"]):
        raise RuntimeError(f"declared seed {seed} does not match config seed {takeoff['seed']}")
    batch_fields = (
        "train_batch_size",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_max_token_len_per_gpu",
        "rollout_n",
        "max_prompt_length",
        "max_response_length",
    )
    expected_batch = {field: int(takeoff[field]) for field in batch_fields}
    actual_batch = {field: int(batch[field]) for field in batch_fields}
    if actual_batch != expected_batch:
        raise RuntimeError(f"declared batch does not match resolved config: {actual_batch} != {expected_batch}")
    actual_training_capacity = {
        "ppo_max_token_len_per_gpu": int(takeoff["ppo_max_token_len_per_gpu"]),
        "infctx": bool(takeoff["infctx"]),
        "chunk_ctx": int(takeoff["chunk_ctx"]),
    }
    max_response_length = int(takeoff["max_response_length"])
    if (
        not actual_training_capacity["infctx"]
        or actual_training_capacity["chunk_ctx"] != 2048
        or actual_training_capacity["ppo_max_token_len_per_gpu"] < max_response_length
    ):
        raise RuntimeError(
            "strict training capacity must use infctx state passing, chunk_ctx 2048, "
            "and an actor token budget that covers the maximum response length: "
            f"{actual_training_capacity}, max_response_length={max_response_length}"
        )
    expected_topology = {
        "trainer_gpus": int(takeoff["trainer_n_gpus_per_node"]),
        "rollout_replicas": 8 // int(takeoff["rollout_tensor_parallel_size"]),
        "rollout_tp": int(takeoff["rollout_tensor_parallel_size"]),
        "rollout_pp": int(takeoff["rollout_pipeline_parallel_size"]),
        "rollout_internal_dp": int(takeoff["rollout_data_parallel_size"]),
    }
    if topology != expected_topology:
        raise RuntimeError(f"declared topology does not match resolved config: {topology} != {expected_topology}")
    config_wkv_mode = str(takeoff["wkv_mode"])
    if precision != config_wkv_mode or wkv_mode != config_wkv_mode:
        raise RuntimeError(
            "declared precision/WKV mode does not match resolved config: "
            f"precision={precision} wkv_mode={wkv_mode} config={config_wkv_mode}"
        )


def verify_observed_topology(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"observed rollout topology artifact is missing: {path}")
    observed = json.loads(path.read_text(encoding="utf-8"))
    comparisons = {
        "replicas": expected["rollout_replicas"],
        "gpus_per_replica": expected["rollout_tp"],
        "tensor_parallel_size": expected["rollout_tp"],
        "data_parallel_size": expected["rollout_internal_dp"],
        "pipeline_parallel_size": expected["rollout_pp"],
    }
    for key, value in comparisons.items():
        if observed.get(key) != value:
            raise RuntimeError(f"observed rollout topology mismatch for {key}: expected {value}, found {observed.get(key)}")
    endpoints = observed.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != expected["rollout_replicas"]:
        raise RuntimeError("observed rollout topology has an incomplete endpoint set")
    if len(endpoints) != len(set(endpoints)):
        raise RuntimeError("observed rollout topology contains duplicate endpoints")
    deployments = observed.get("deployments")
    if not isinstance(deployments, list) or len(deployments) != expected["rollout_replicas"]:
        raise RuntimeError("observed rollout topology has an incomplete deployment set")
    gpu_bindings = [
        (deployment.get("node_id"), gpu)
        for deployment in deployments
        for gpu in deployment.get("cuda_visible_devices", [])
    ]
    if len(gpu_bindings) != 8 or len(gpu_bindings) != len(set(gpu_bindings)):
        raise RuntimeError(f"observed rollout topology does not uniquely cover 8 GPUs: {gpu_bindings}")
    port_fields = ("http_port", "master_port", "dp_rpc_port", "dp_master_port")
    runtime_ports = []
    for deployment in deployments:
        if not deployment.get("actor_id"):
            raise RuntimeError("observed rollout topology is missing runtime actor identity")
        node_id = deployment.get("node_id")
        if not node_id:
            raise RuntimeError("observed rollout topology is missing node identity")
        for field in port_fields:
            port = deployment.get(field)
            if not isinstance(port, int) or not 0 < port <= 65535:
                raise RuntimeError(
                    f"observed rollout topology has invalid {field} for replica "
                    f"{deployment.get('replica_rank')}: {port!r}"
                )
            runtime_ports.append((node_id, port, field, deployment.get("replica_rank")))
    port_bindings = [(node_id, port) for node_id, port, _, _ in runtime_ports]
    if len(port_bindings) != len(set(port_bindings)):
        duplicates = sorted(
            {binding for binding in port_bindings if port_bindings.count(binding) > 1}
        )
        raise RuntimeError(
            "observed rollout topology contains duplicate runtime ports on the same node: "
            f"{duplicates}"
        )
    return observed


def verify_correctness_metrics(path: Path, *, expected_rounds: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"training metrics artifact is missing: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rounds = [record for record in records if "training/rollout_probs_diff_valid" in record.get("data", {})]
    if len(rounds) != expected_rounds:
        raise RuntimeError(f"correctness metrics expected {expected_rounds} rounds, found {len(rounds)}")
    steps = [record.get("step") for record in rounds]
    if any(not isinstance(step, int) for step in steps) or steps != sorted(set(steps)):
        raise RuntimeError(f"correctness metric steps must be unique and strictly increasing: {steps}")
    nonzero_gradient_steps = []
    cross_runtime_diagnostics = []
    for record in rounds:
        step, data = record["step"], record["data"]
        requirements = {
            "training/rollout_probs_diff_valid": 1,
            "training/off_policy/trajectory_spans/max": 1,
            "training/off_policy/trajectory_staleness/max": 0,
            "training/off_policy/trajectory_staleness_worst/max": 0,
            "training/on_policy/version_count": 1,
            "training/on_policy/weight_digest_count": 1,
            "training/on_policy/sampling_config_count": 1,
            "training/on_policy/runtime_identity_count": 1,
        }
        for key, expected in requirements.items():
            if data.get(key) != expected:
                raise RuntimeError(f"correctness metric {key} failed at step {step}: {data.get(key)} != {expected}")
        finite_metrics = {
            "training/rollout_probs_diff_max": data.get("training/rollout_probs_diff_max"),
            "training/rollout_actor_probs_pearson_corr": data.get(
                "training/rollout_actor_probs_pearson_corr"
            ),
            "rollout_corr/rollout_is_mean": data.get("rollout_corr/rollout_is_mean"),
            "rollout_corr/rollout_is_min": data.get("rollout_corr/rollout_is_min"),
            "rollout_corr/rollout_is_max": data.get("rollout_corr/rollout_is_max"),
            "rollout_corr/rollout_is_std": data.get("rollout_corr/rollout_is_std"),
            "rollout_corr/rollout_is_eff_sample_size": data.get(
                "rollout_corr/rollout_is_eff_sample_size"
            ),
            "rollout_corr/rollout_is_ratio_fraction_high": data.get(
                "rollout_corr/rollout_is_ratio_fraction_high"
            ),
            "rollout_corr/rollout_is_ratio_fraction_low": data.get(
                "rollout_corr/rollout_is_ratio_fraction_low"
            ),
            "actor/loss": data.get("actor/loss"),
            "actor/grad_norm": data.get("actor/grad_norm"),
            "actor/optimizer_steps": data.get("actor/optimizer_steps"),
        }
        missing = sorted(key for key, value in finite_metrics.items() if value is None)
        if missing:
            raise RuntimeError(
                f"correctness metrics are missing same-version correction/update evidence at step {step}: "
                f"{', '.join(missing)}"
            )
        values = {key: float(value) for key, value in finite_metrics.items()}
        nonfinite = sorted(key for key, value in values.items() if not math.isfinite(value))
        if nonfinite:
            raise RuntimeError(
                f"correctness metrics must be finite at step {step}: {', '.join(nonfinite)}"
            )
        mean_weight = values["rollout_corr/rollout_is_mean"]
        max_weight = values["rollout_corr/rollout_is_max"]
        ess = values["rollout_corr/rollout_is_eff_sample_size"]
        if mean_weight <= 0 or max_weight <= 0:
            raise RuntimeError(f"rollout correction produced no positive training weight at step {step}")
        if max_weight > CORRECTNESS_MAX_ROLLOUT_IS_WEIGHT + 1e-6:
            raise RuntimeError(f"rollout correction exceeded the configured truncation bound at step {step}")
        if ess < CORRECTNESS_MIN_ROLLOUT_IS_ESS:
            raise RuntimeError(f"rollout correction effective sample size fell below tolerance at step {step}")
        if values["actor/optimizer_steps"] != 1:
            raise RuntimeError(f"strict correctness requires exactly one optimizer step at step {step}")
        if values["actor/grad_norm"] > 0:
            nonzero_gradient_steps.append(step)
        cross_runtime_diagnostics.append(
            {
                "step": step,
                "max_probability_diff": values["training/rollout_probs_diff_max"],
                "pearson_correlation": values["training/rollout_actor_probs_pearson_corr"],
            }
        )
    if not nonzero_gradient_steps:
        raise RuntimeError("correctness run did not produce a non-zero actor gradient in any round")
    return {
        "rounds": len(rounds),
        "minimum_rollout_is_effective_sample_size": CORRECTNESS_MIN_ROLLOUT_IS_ESS,
        "maximum_rollout_is_weight": CORRECTNESS_MAX_ROLLOUT_IS_WEIGHT,
        "nonzero_gradient_steps": nonzero_gradient_steps,
        "cross_runtime_diagnostics": cross_runtime_diagnostics,
        "steps": steps,
    }


def verify_exact_response_length(
    path: Path, *, expected_rounds: int, expected_length: int
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rounds = [record for record in records if "training/global_step" in record.get("data", {})]
    if len(rounds) != expected_rounds:
        raise RuntimeError(
            f"exact response-length contract expected {expected_rounds} rounds, found {len(rounds)}"
        )
    for record in rounds:
        data = record["data"]
        step = record.get("step")
        minimum = int(data.get("response_length/min", -1))
        maximum = int(data.get("response_length/max", -1))
        samples = int(data.get("training/actual_samples", -1))
        response_tokens = int(data.get("training/actual_response_tokens", -1))
        if minimum != expected_length or maximum != expected_length:
            raise RuntimeError(
                f"step {step} did not generate exact {expected_length}-token responses: "
                f"min={minimum}, max={maximum}"
            )
        if response_tokens != samples * expected_length:
            raise RuntimeError(
                f"step {step} response-token total does not match exact-length contract: "
                f"{response_tokens} != {samples}*{expected_length}"
            )
    return {"rounds": len(rounds), "tokens_per_response": expected_length}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def verify_performance_metrics(path: Path, *, expected_rounds: int) -> dict[str, Any]:
    if expected_rounds < FORMAL_WARMUP_STEPS + FORMAL_TIMED_STEPS:
        raise RuntimeError(
            "formal performance runs require at least "
            f"{FORMAL_WARMUP_STEPS} warmup and {FORMAL_TIMED_STEPS} timed steps"
        )
    if not path.is_file():
        raise RuntimeError(f"training metrics artifact is missing: {path}")
    on_policy_correctness = verify_correctness_metrics(path, expected_rounds=expected_rounds)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rounds = [record for record in records if "training/global_step" in record.get("data", {})]
    if len(rounds) != expected_rounds:
        raise RuntimeError(f"performance metrics expected {expected_rounds} rounds, found {len(rounds)}")
    steps = [record.get("step") for record in rounds]
    if any(not isinstance(step, int) for step in steps) or steps != sorted(set(steps)):
        raise RuntimeError(f"performance metric steps must be unique and strictly increasing: {steps}")
    timed = rounds[FORMAL_WARMUP_STEPS : FORMAL_WARMUP_STEPS + FORMAL_TIMED_STEPS]
    required = {
        "training/actual_samples",
        "training/actual_prompt_tokens",
        "training/actual_response_tokens",
        "training/actual_total_tokens",
        "training/actual_policy_loss_tokens",
        "timing/rollout_seconds",
        "timing/train_seconds",
        "timing/full_step_seconds",
        "critic/rewards/mean",
        "actor/entropy",
        "actor/grad_norm",
        "actor/actual_micro_batches",
        "training/rollout_group_completion_seconds/p95",
        "training/rollout_group_completion_seconds/max",
        "training/rollout_group_tail_seconds",
        "training/rollout_preemptions",
        "training/rollout_effective_concurrency",
        "training/rollout_train_overlap_seconds",
    }
    for record in timed:
        missing = sorted(required - record["data"].keys())
        if missing:
            raise RuntimeError(
                f"performance metrics step {record['step']} is missing: {', '.join(missing)}"
            )
        data = record["data"]
        token_keys = (
            "training/actual_samples",
            "training/actual_prompt_tokens",
            "training/actual_response_tokens",
            "training/actual_total_tokens",
            "training/actual_policy_loss_tokens",
        )
        if any(not isinstance(data[key], int) or isinstance(data[key], bool) or data[key] < 0 for key in token_keys):
            raise RuntimeError(f"performance metrics step {record['step']} has invalid token/sample counts")
        if data["training/actual_total_tokens"] != (
            data["training/actual_prompt_tokens"] + data["training/actual_response_tokens"]
        ):
            raise RuntimeError(f"performance metrics step {record['step']} violates total=prompt+response")
        if data["training/actual_policy_loss_tokens"] > data["training/actual_response_tokens"]:
            raise RuntimeError(f"performance metrics step {record['step']} has policy-loss tokens above responses")
        for key in (
            "timing/rollout_seconds",
            "timing/train_seconds",
            "timing/full_step_seconds",
            "critic/rewards/mean",
            "actor/entropy",
            "actor/grad_norm",
            "actor/actual_micro_batches",
            "training/rollout_group_completion_seconds/p95",
            "training/rollout_group_completion_seconds/max",
            "training/rollout_group_tail_seconds",
            "training/rollout_preemptions",
            "training/rollout_effective_concurrency",
            "training/rollout_train_overlap_seconds",
        ):
            value = float(data[key])
            if not math.isfinite(value):
                raise RuntimeError(f"performance metric {key} must be finite at step {record['step']}")
            if key.startswith("timing/") and value <= 0:
                raise RuntimeError(f"performance metric {key} must be positive at step {record['step']}")

    totals = {
        key: sum(float(record["data"][key]) for record in timed)
        for key in (
            "training/actual_samples",
            "training/actual_prompt_tokens",
            "training/actual_response_tokens",
            "training/actual_total_tokens",
            "training/actual_policy_loss_tokens",
            "timing/rollout_seconds",
            "timing/train_seconds",
            "timing/full_step_seconds",
        )
    }
    for key in ("timing/rollout_seconds", "timing/train_seconds", "timing/full_step_seconds"):
        if totals[key] <= 0:
            raise RuntimeError(f"performance metrics require positive total {key}")
    stage_seconds = {
        stage: [float(record["data"][f"timing/{stage}_seconds"]) for record in timed]
        for stage in ("rollout", "train", "full_step")
    }
    stage_summary = {
        stage: {
            "mean_seconds": sum(values) / len(values),
            "p50_seconds": _percentile(values, 0.50),
            "p95_seconds": _percentile(values, 0.95),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
        for stage, values in stage_seconds.items()
    }
    detailed_timing_keys = sorted(
        set.intersection(
            *(
                {key for key in record["data"] if key.startswith("timing_s/")}
                for record in timed
            )
        )
    )
    detailed_stage_summary = {
        key.removeprefix("timing_s/"): {
            "mean_seconds": sum(values) / len(values),
            "p50_seconds": _percentile(values, 0.50),
            "p95_seconds": _percentile(values, 0.95),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
        for key in detailed_timing_keys
        if all(
            math.isfinite(value := float(record["data"][key])) and value >= 0
            for record in timed
        )
        for values in [[float(record["data"][key]) for record in timed]]
    }
    quality_keys = (
        "critic/rewards/mean",
        "actor/entropy",
        "actor/grad_norm",
    )
    quality = {
        key: sum(float(record["data"][key]) for record in timed) / len(timed)
        for key in quality_keys
    }
    return {
        "on_policy_correctness": on_policy_correctness,
        "warmup_steps": [record["step"] for record in rounds[:FORMAL_WARMUP_STEPS]],
        "timed_steps": [record["step"] for record in timed],
        "actual_samples": int(totals["training/actual_samples"]),
        "actual_prompt_tokens": int(totals["training/actual_prompt_tokens"]),
        "actual_response_tokens": int(totals["training/actual_response_tokens"]),
        "actual_total_tokens": int(totals["training/actual_total_tokens"]),
        "actual_policy_loss_tokens": int(totals["training/actual_policy_loss_tokens"]),
        "throughput": {
            "rollout_tokens_per_second": totals["training/actual_response_tokens"]
            / totals["timing/rollout_seconds"],
            "train_tokens_per_second": totals["training/actual_policy_loss_tokens"]
            / totals["timing/train_seconds"],
            "full_step_tokens_per_second": totals["training/actual_total_tokens"]
            / totals["timing/full_step_seconds"],
            "full_step_samples_per_second": totals["training/actual_samples"]
            / totals["timing/full_step_seconds"],
        },
        "stage_timing": stage_summary,
        "detailed_stage_timing": detailed_stage_summary,
        "quality": quality,
        "capacity_observations": {
            key: [float(record["data"][key]) for record in timed]
            for key in (
                "actor/actual_micro_batches",
                "training/rollout_group_completion_seconds/p95",
                "training/rollout_group_completion_seconds/max",
                "training/rollout_group_tail_seconds",
                "training/rollout_preemptions",
                "training/rollout_effective_concurrency",
                "training/rollout_train_overlap_seconds",
            )
        }
        | {
            "rollout_preemptions_available": all(
                float(record["data"]["training/rollout_preemptions"]) >= 0
                for record in timed
            )
        },
    }


def verify_validation_curve(path: Path, *, expected_rounds: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"training metrics artifact is missing: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cumulative_samples = 0
    curve = []
    training_trajectory = []
    for record in records:
        data = record.get("data", {})
        elapsed_seconds = record.get("elapsed_seconds")
        if elapsed_seconds is None:
            raise RuntimeError("global-batch quality records require monotonic elapsed_seconds")
        elapsed_seconds = float(elapsed_seconds)
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise RuntimeError("global-batch quality elapsed_seconds must be finite and positive")
        if "training/actual_samples" in data:
            samples = data["training/actual_samples"]
            if not isinstance(samples, int) or isinstance(samples, bool) or samples < 0:
                raise RuntimeError("validation curve encountered an invalid training sample count")
            cumulative_samples += samples
        if "training/global_step" in data:
            trajectory_values = {
                key: float(data[key])
                for key in (
                    "critic/rewards/mean",
                    "actor/entropy",
                    "actor/grad_norm",
                    "actor/optimizer_steps",
                    "training/actual_samples",
                    "training/actual_prompt_tokens",
                    "training/actual_response_tokens",
                    "training/actual_total_tokens",
                )
            }
            if any(not math.isfinite(value) for value in trajectory_values.values()):
                raise RuntimeError("global-batch quality training trajectory contains a non-finite value")
            training_trajectory.append(
                {
                    "step": record.get("step"),
                    "cumulative_samples": cumulative_samples,
                    "elapsed_seconds": elapsed_seconds,
                    **trajectory_values,
                }
            )
        validation_metrics = {
            key: float(value)
            for key, value in data.items()
            if key.startswith("val-core/")
        }
        if not validation_metrics:
            continue
        if any(not math.isfinite(value) for value in validation_metrics.values()):
            raise RuntimeError("validation curve contains a non-finite metric")
        if "timing_s/testing" not in data:
            raise RuntimeError("every validation point must record timing_s/testing")
        testing_seconds = float(data["timing_s/testing"])
        if not math.isfinite(testing_seconds) or testing_seconds <= 0:
            raise RuntimeError("every validation point must record positive finite testing time")
        curve.append(
            {
                "step": record.get("step"),
                "cumulative_samples": cumulative_samples,
                "cumulative_wall_seconds": elapsed_seconds,
                "testing_seconds": testing_seconds,
                "metrics": validation_metrics,
            }
        )
    if len(curve) < 2:
        raise RuntimeError("global-batch quality runs require initial and post-training validation points")
    if curve[0]["cumulative_samples"] != 0:
        raise RuntimeError("the initial validation point must precede all training samples")
    expected_samples = sum(
        int(record.get("data", {}).get("training/actual_samples", 0))
        for record in records
    )
    if curve[-1]["cumulative_samples"] != expected_samples:
        raise RuntimeError("the final validation point must follow every training sample")
    training_steps = [
        record.get("step")
        for record in records
        if "training/global_step" in record.get("data", {})
    ]
    if len(training_steps) != expected_rounds:
        raise RuntimeError(
            f"validation curve expected {expected_rounds} training rounds, found {len(training_steps)}"
        )
    if len(training_trajectory) != expected_rounds:
        raise RuntimeError("global-batch quality training trajectory is incomplete")
    if any(
        later["elapsed_seconds"] <= earlier["elapsed_seconds"]
        for earlier, later in zip(training_trajectory, training_trajectory[1:])
    ):
        raise RuntimeError("global-batch quality elapsed_seconds must increase across training rounds")
    common_metrics = sorted(set.intersection(*(set(point["metrics"]) for point in curve)))
    accuracy_metrics = [key for key in common_metrics if key.endswith("/acc/mean@1")]
    if not accuracy_metrics:
        raise RuntimeError("validation curve is missing a common val-core/*/acc/mean@1 metric")
    return {
        "points": curve,
        "common_metrics": common_metrics,
        "accuracy_metrics": accuracy_metrics,
        "training_trajectory": training_trajectory,
        "final_cumulative_samples": expected_samples,
        "final_cumulative_wall_seconds": curve[-1]["cumulative_wall_seconds"],
    }


def verify_global_batch_quality_schedule(
    config: dict[str, Any], validation: dict[str, Any], *, expected_rounds: int
) -> None:
    takeoff = config["takeoff"]["grpo"]
    batch_size = int(takeoff["train_batch_size"])
    expected = {
        56: {"rounds": 14, "test_freq": 2},
        112: {"rounds": 7, "test_freq": 1},
    }
    if batch_size not in expected:
        raise RuntimeError("global-batch quality runs require train_batch_size 56 or 112")
    contract = expected[batch_size]
    if int(takeoff["ppo_mini_batch_size"]) != batch_size:
        raise RuntimeError("global-batch quality requires ppo_mini_batch_size == train_batch_size")
    if int(takeoff["rollout_n"]) != 8:
        raise RuntimeError("global-batch quality requires rollout_n=8")
    if expected_rounds != contract["rounds"]:
        raise RuntimeError(
            f"global-batch quality batch {batch_size} requires {contract['rounds']} rounds"
        )
    if int(takeoff["test_freq"]) != contract["test_freq"]:
        raise RuntimeError(
            f"global-batch quality batch {batch_size} requires test_freq={contract['test_freq']}"
        )
    expected_sample_axis = list(range(0, 6272 + 1, 896))
    observed_sample_axis = [point["cumulative_samples"] for point in validation["points"]]
    if observed_sample_axis != expected_sample_axis:
        raise RuntimeError(
            "global-batch quality validation must run every 896 response samples: "
            f"expected {expected_sample_axis}, found {observed_sample_axis}"
        )


def verify_gpu_telemetry(summary: dict[str, Any]) -> None:
    if summary["missing_gpus"]:
        raise RuntimeError(f"GPU telemetry is missing devices: {summary['missing_gpus']}")
    if summary["incomplete_sequences"]:
        raise RuntimeError(
            f"GPU telemetry has incomplete 8-GPU sequences: {summary['incomplete_sequences']}"
        )
    insufficient_coverage = {
        gpu: values["sample_coverage"]
        for gpu, values in summary["per_gpu"].items()
        if values.get("sample_coverage", 0) < MIN_GPU_SAMPLE_COVERAGE
    }
    if insufficient_coverage:
        raise RuntimeError(
            f"GPU telemetry coverage fell below {MIN_GPU_SAMPLE_COVERAGE:.1%}: {insufficient_coverage}"
        )


def snapshot_nsys_reports(root: Path = Path("/tmp/ray")) -> dict[str, tuple[int, int]]:
    return {
        str(path.resolve()): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.glob("session_*/logs/nsight/*.nsys-rep")
        if path.is_file()
    }


def collect_nsys_reports(
    run_dir: Path,
    before: dict[str, tuple[int, int]],
    *,
    profiled_steps: list[int],
    source_root: Path = Path("/tmp/ray"),
) -> None:
    destination = run_dir / "nsys"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    marker_text = ""
    seen_sources: set[str] = set()
    for path in source_root.glob("session_*/logs/nsight/*.nsys-rep"):
        resolved = str(path.resolve())
        if resolved in seen_sources:
            continue
        seen_sources.add(resolved)
        state = (path.stat().st_mtime_ns, path.stat().st_size)
        if before.get(resolved) == state or state[1] <= 0:
            continue
        target = destination / f"{path.parents[2].name}-{path.name}"
        shutil.copy2(path, target)
        copied.append(target)
        marker_text += command_output(
            "nsys",
            "stats",
            "--report",
            "nvtx_sum",
            "--format",
            "csv",
            str(target),
        )
    if not copied:
        return
    markers = sorted(marker for marker in NSYS_REQUIRED_MARKERS if marker in marker_text)
    manifest = {
        "profiled_steps": profiled_steps,
        "nvtx_stage_markers": markers,
        "reports": [str(path) for path in copied],
    }
    (run_dir / "nsys_trace_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_nsys_trace(run_dir: Path) -> dict[str, Any]:
    reports = [path for path in run_dir.rglob("*.nsys-rep") if path.is_file() and path.stat().st_size > 0]
    if not reports:
        raise RuntimeError("nsys phase requires at least one non-empty .nsys-rep")
    manifest_path = run_dir / "nsys_trace_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("nsys phase requires nsys_trace_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    steps = manifest.get("profiled_steps")
    markers = manifest.get("nvtx_stage_markers")
    if not isinstance(steps, list) or not steps or any(not isinstance(step, int) or step < 1 for step in steps):
        raise RuntimeError("nsys trace manifest requires positive profiled_steps")
    if not isinstance(markers, list) or not NSYS_REQUIRED_MARKERS.issubset(markers):
        raise RuntimeError(f"nsys trace manifest is missing stage markers: {sorted(NSYS_REQUIRED_MARKERS)}")
    return {
        "report_paths": [str(path) for path in reports],
        "report_size_bytes": sum(path.stat().st_size for path in reports),
        "manifest_path": str(manifest_path),
        "profiled_steps": steps,
        "nvtx_stage_markers": markers,
        "formal_performance": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-sampler", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--heartbeat-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gpu_sampler is not None:
        if args.ready_file is None:
            raise RuntimeError("--gpu-sampler requires --ready-file")
        return sample_gpus(args.gpu_sampler, args.ready_file, args.heartbeat_file)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise RuntimeError("strict training wrapper requires a command after --")

    root = Path.cwd().resolve()
    run_dir = resolve_run_dir(root, required_env("REMOTE_RUN_LOG_DIR"))
    run_dir.mkdir(parents=True, exist_ok=True)
    run_phase = validate_run_phase(required_env("HELICOPTER_RUN_PHASE"))
    parse_visible_devices(required_env("CUDA_VISIBLE_DEVICES"))
    checkpoint = verify_file(
        Path(required_env("HELICOPTER_CHECKPOINT_PATH")),
        required_env("HELICOPTER_CHECKPOINT_SHA256"),
        label="checkpoint",
    )
    dataset = verify_dataset_manifest(Path(required_env("HELICOPTER_DATASET_MANIFEST")))
    config_path = Path(required_env("HELICOPTER_CONFIG_PATH"))
    if not config_path.is_file():
        raise RuntimeError(f"resolved training config source is missing: {config_path}")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    expected_rounds = int(config["takeoff"]["grpo"]["total_training_steps"])
    topology = verify_topology_contract(json.loads(required_env("HELICOPTER_TOPOLOGY_JSON")))
    batch = json.loads(required_env("HELICOPTER_BATCH_JSON"))
    seed = required_env("HELICOPTER_SEED")
    precision = required_env("HELICOPTER_PRECISION")
    wkv_mode = required_env("HELICOPTER_WKV_MODE")
    verify_declared_contract(
        config,
        seed=seed,
        batch=batch,
        topology=topology,
        precision=precision,
        wkv_mode=wkv_mode,
    )
    if run_phase == "nsys":
        configured_steps = config["takeoff"]["grpo"].get("profiler_steps")
        declared_steps = json.loads(required_env("HELICOPTER_NSYS_PROFILE_STEPS_JSON"))
        if config["takeoff"]["grpo"].get("profiler_tool") != "nsys" or declared_steps != configured_steps:
            raise RuntimeError("declared Nsight profile steps/tool do not match the resolved config")

    source_revisions_path = root / ".helicopter-dev" / "source-revisions.json"
    source_revisions = (
        json.loads(source_revisions_path.read_text(encoding="utf-8")) if source_revisions_path.is_file() else None
    )
    metadata_path = run_dir / "metadata.json"
    samples_path = run_dir / "gpu_samples.csv"
    heartbeat_path = run_dir / "gpu_sampler_heartbeat.csv"
    metrics_path = run_dir / "metrics.jsonl"
    identity_path = run_dir / "policy_identity.jsonl"
    sampler_cpu, child_cpus = telemetry_cpu_affinity_plan()
    metadata = {
        "schema_version": 1,
        "change": CHANGE_ID,
        "run_id": required_env("HELICOPTER_RUN_ID"),
        "run_phase": run_phase,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "source": source_metadata(root, source_revisions),
        "source_revisions": source_revisions,
        "checkpoint": checkpoint,
        "dataset_manifest": dataset,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "seed": seed,
        "topology": {
            "trainer_nodes": 1,
            "cuda_visible_devices": list(EXPECTED_GPU_INDICES),
            **topology,
        },
        "batch": batch,
        "precision": precision,
        "wkv_mode": wkv_mode,
        "gemm_policy": required_env("HELICOPTER_GEMM_POLICY"),
        "rwkv_init_stagger_seconds": int(required_env("HELICOPTER_RWKV_INIT_STAGGER_SECONDS")),
        "rwkv_init_concurrency": int(required_env("HELICOPTER_RWKV_INIT_CONCURRENCY")),
        "rollout_capacity": {
            key: config["takeoff"]["grpo"][key]
            for key in (
                "rollout_gpu_memory_utilization",
                "rollout_max_num_batched_tokens",
                "rollout_max_num_seqs",
                "ppo_max_token_len_per_gpu",
            )
        },
        "environment": environment_metadata(),
        "measurement_contract": {
            "gpu_sample_interval_ms": GPU_SAMPLE_INTERVAL_MS,
            "missing_sample_detection_ms": MISSING_SAMPLE_DETECTION_MS,
            "maximum_sampler_heartbeat_gap_ms": MAX_GPU_SAMPLE_GAP_MS,
            "minimum_sample_coverage": MIN_GPU_SAMPLE_COVERAGE,
            "gpu_sampler_backend": "nvidia-ml-py persistent NVML process",
            "gpu_sample_clock": "monotonic_ns",
            "gpu_sampler_cpu": sampler_cpu,
            "training_child_cpus": list(child_cpus),
            "idle_utilization_threshold_percent": IDLE_UTILIZATION_THRESHOLD_PERCENT,
            "missing_samples": (
                "report every gap above the detection threshold; reject missing GPUs, incomplete 8-GPU "
                "sequences, a gap above the maximum, or per-GPU coverage below the minimum"
            ),
            "token_numerator": "actual non-padding prompt plus response tokens",
            "rollout_throughput": "response tokens divided by gen stage wall time",
            "train_throughput": "policy-loss tokens divided by actor update wall time",
            "full_step_throughput": "prompt plus response tokens divided by complete step wall time",
            "wall_clock_to_quality": (
                "FileLogger monotonic elapsed time from trainer Tracking creation through each validation log; "
                "includes training, metric/postprocessing, logging, and validation after initialization"
            ),
            "correctness_min_rollout_is_effective_sample_size": CORRECTNESS_MIN_ROLLOUT_IS_ESS,
            "correctness_max_rollout_is_weight": CORRECTNESS_MAX_ROLLOUT_IS_WEIGHT,
            "cross_runtime_probability_diff": "diagnostic only; native BF16 and vLLM FP16 are non-like-for-like",
        },
    }
    write_metadata(metadata_path, metadata)
    nsys_before = snapshot_nsys_reports() if run_phase == "nsys" else {}

    ready_path = run_dir / "gpu_sampler.ready"
    ready_path.unlink(missing_ok=True)
    sampler_error = None
    sampler = None
    try:
        sampler = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--gpu-sampler",
                str(samples_path),
                "--ready-file",
                str(ready_path),
                "--heartbeat-file",
                str(heartbeat_path),
            ],
            preexec_fn=partial(set_current_process_affinity, (sampler_cpu,)),
        )
    except Exception as exc:
        sampler_error = f"GPU telemetry sampler failed to start: {exc}"
    if sampler is not None:
        ready_deadline = time.monotonic() + 10
        while not ready_path.is_file():
            sampler_exit_code = sampler.poll()
            if sampler_exit_code is not None:
                sampler_error = f"GPU telemetry sampler exited before ready with code {sampler_exit_code}"
                break
            if time.monotonic() >= ready_deadline:
                sampler_error = "GPU telemetry sampler did not become ready within 10 seconds"
                break
            time.sleep(0.05)
    child_env = strict_child_environment(dict(os.environ), run_dir)
    child = None
    command_log = None
    output_thread = None
    output_errors: list[OSError] = []
    if sampler_error is None:
        try:
            command_log = (run_dir / "command.log").open("wb")
            child = subprocess.Popen(
                command,
                env=child_env,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=partial(set_current_process_affinity, child_cpus),
            )
            assert child.stdout is not None
            output_thread = threading.Thread(
                target=tee_child_output,
                args=(child.stdout, command_log, output_errors),
                name="strict-child-output",
            )
            output_thread.start()
        except Exception as exc:
            if command_log is not None:
                command_log.close()
            sampler_error = f"training child failed to start: {exc}"

    def terminate(_signum, _frame):
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if sampler is not None and sampler.poll() is None:
            sampler.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    child_exit_code = 1
    if child is not None:
        while child.poll() is None:
            assert sampler is not None
            if output_errors:
                sampler_error = f"command log writer failed during training: {output_errors[0]}"
                stop_process(child, process_group=True)
                break
            sampler_exit_code = sampler.poll()
            if sampler_exit_code is not None:
                sampler_error = f"GPU telemetry sampler exited during training with code {sampler_exit_code}"
                stop_process(child, process_group=True)
                break
            time.sleep(0.1)
        child_exit_code = child.returncode if child.returncode is not None else child.wait()
    if output_thread is not None:
        output_thread.join()
    if command_log is not None and not command_log.closed:
        command_log.close()
    sampler_exit_code = None
    if sampler is not None:
        stop_process(sampler)
        sampler_exit_code = sampler.returncode

    metadata["gpu_telemetry"] = parse_gpu_samples(samples_path)
    metadata["gpu_telemetry"]["sampler_heartbeat"] = parse_sampler_heartbeat(heartbeat_path)
    metadata["gpu_telemetry"]["sampler_exit_code"] = sampler_exit_code
    metadata["vllm_capacity_observations"] = extract_vllm_capacity(
        run_dir / "command.log",
        run_dir / "rollout_topology.json",
    )
    contract_error = None
    try:
        if sampler_error:
            raise RuntimeError(sampler_error)
        if sampler_exit_code != 0:
            raise RuntimeError(f"GPU telemetry sampler failed with code {sampler_exit_code}")
        if metadata["gpu_telemetry"]["sampler_heartbeat"]["max_gap_ms"] > MAX_GPU_SAMPLE_GAP_MS:
            raise RuntimeError(
                "GPU telemetry sampler heartbeat exceeded the "
                f"{MAX_GPU_SAMPLE_GAP_MS}ms maximum gap: "
                f"{metadata['gpu_telemetry']['sampler_heartbeat']['max_gap_ms']}"
            )
        verify_gpu_telemetry(metadata["gpu_telemetry"])
        if run_phase == "nsys":
            profiled_steps = json.loads(required_env("HELICOPTER_NSYS_PROFILE_STEPS_JSON"))
            if not isinstance(profiled_steps, list):
                raise RuntimeError("HELICOPTER_NSYS_PROFILE_STEPS_JSON must be a JSON list")
            collect_nsys_reports(run_dir, nsys_before, profiled_steps=profiled_steps)
        metadata["policy_identity"] = verify_policy_identity_log(identity_path, expected_rounds=expected_rounds)
        metadata["observed_rollout_topology"] = verify_observed_topology(
            run_dir / "rollout_topology.json", topology
        )
        takeoff = config["takeoff"]["grpo"]
        if bool(takeoff.get("rollout_ignore_eos", False)):
            metadata["exact_response_length"] = verify_exact_response_length(
                metrics_path,
                expected_rounds=expected_rounds,
                expected_length=int(takeoff["max_response_length"]),
            )
        if run_phase == "correctness":
            metadata["correctness"] = verify_correctness_metrics(metrics_path, expected_rounds=expected_rounds)
        elif run_phase in FORMAL_PERFORMANCE_PHASES:
            verify_rollout_capacity_observations(metadata["vllm_capacity_observations"], config)
            metadata["performance"] = verify_performance_metrics(metrics_path, expected_rounds=expected_rounds)
            if run_phase == "global-batch-quality":
                takeoff = config["takeoff"]["grpo"]
                if not takeoff.get("val_before_train") or int(takeoff.get("test_freq", -1)) <= 0:
                    raise RuntimeError(
                        "global-batch quality runs require val_before_train=true and test_freq>0"
                    )
                metadata["validation"] = verify_validation_curve(
                    metrics_path, expected_rounds=expected_rounds
                )
                verify_global_batch_quality_schedule(
                    config, metadata["validation"], expected_rounds=expected_rounds
                )
        elif run_phase == "nsys":
            metadata["correctness"] = verify_correctness_metrics(metrics_path, expected_rounds=expected_rounds)
            metadata["nsys"] = verify_nsys_trace(run_dir)
    except Exception as exc:
        contract_error = str(exc)
    wrapper_exit_code = child_exit_code if child_exit_code != 0 else (3 if contract_error else 0)
    metadata["status"] = "done" if wrapper_exit_code == 0 else "failed"
    metadata["child_exit_code"] = child_exit_code
    metadata["exit_code"] = wrapper_exit_code
    metadata["contract_error"] = contract_error
    metadata["failure_class"] = classify_failure(
        child_exit_code,
        contract_error,
        run_dir / "command.log",
    )
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["metrics_path"] = str(metrics_path)
    write_metadata(metadata_path, metadata)
    return wrapper_exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"strict training contract failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
