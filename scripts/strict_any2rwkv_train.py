#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple


REQUIRED_ENV = (
    "HELICOPTER_RUN_ID",
    "HELICOPTER_CHECKPOINT_PATH",
    "HELICOPTER_CHECKPOINT_SHA256",
    "HELICOPTER_DATASET_MANIFEST",
    "HELICOPTER_CONFIG_PATH",
    "HELICOPTER_SEED",
    "HELICOPTER_BATCH_JSON",
    "HELICOPTER_PRECISION",
    "HELICOPTER_WKV_MODE",
    "HELICOPTER_RUN_PHASE",
    "CUDA_VISIBLE_DEVICES",
)

NATIVE_TRAINING_ENV = {
    "RWKV_JIT_ON": "0",
    "RWKV_MY_TESTING": "x070",
    "RWKV_KERNEL": "",
    "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
    "RWKV_TRAIN_TYPE": "infctx",
    "RWKV_FLOAT_MODE": "bf16",
    "WKV_MODE": "fp32io16",
}

TRAINING_SOURCE_FILES = (
    "src/train/any2rwkv/any2rwkv/recipes/qwen35_to_rwkv7/layer_major_runner.py",
    "src/train/any2rwkv/any2rwkv/recipes/qwen35_to_rwkv7/recipe.py",
    "src/train/any2rwkv/any2rwkv/layer_schedule.py",
    "src/train/any2rwkv/any2rwkv/streaming_training.py",
    "src/train/any2rwkv/any2rwkv/core/experiment_tracking.py",
    "src/train/any2rwkv/any2rwkv/core/layer_input_cache.py",
    "src/train/any2rwkv/any2rwkv/distributed.py",
    "scripts/profile_any2rwkv_candidate.py",
    "scripts/run_any2rwkv_profile_candidate.py",
    "scripts/strict_any2rwkv_train.py",
    "scripts/profile_any2rwkv_layer_major.py",
)

PROFILER_CAPTURE_CONTRACT = (
    "nsys-full-process-exact-nvtx-rank-transition-and-input-bindings-v5"
)

# The user-facing requirement is "near 100%", not merely "busy most of the
# time". Keep these policy floors separate from the measured batch ranking:
# reject candidates below either 95% floor, then compare throughput only among
# the survivors. Nsight Systems SMs Active is not interchangeable with the
# coarse GPU-util percentage printed by nvidia-smi.
MINIMUM_KERNEL_COVERED_WALL_FRACTION = 0.95
MINIMUM_SM_ACTIVE_FRACTION = 0.95
MINIMUM_RESERVED_MEMORY_FRACTION = 0.85
MAXIMUM_RESERVED_MEMORY_FRACTION = 0.95
MAXIMUM_RANK_WALL_RATIO = 1.05
MAXIMUM_CHECKPOINT_WALL_FRACTION = 0.05

CANDIDATE_SELECTION_DERIVED_FIELDS = {
    "source_candidate",
    "experiment_artifacts",
    "kernel_covered_wall_fraction_by_rank",
    "sm_active_fraction_by_gpu",
    "sm_active_sample_count_by_gpu",
    "sm_active_metric_source_by_gpu",
    "gpu_metric_common_window_seconds",
    "profiler_window_seconds_by_rank",
    "nsys_process_ids_by_rank",
    "nsys_device_ids_by_rank",
    "nsys_kernel_intersection_count_by_rank",
    "cache_transition_kernel_covered_wall_fraction_by_rank",
    "cache_transition_sm_active_fraction_by_gpu",
    "cache_transition_sm_active_sample_count_by_gpu",
    "cache_transition_sm_active_metric_source_by_gpu",
    "cache_transition_gpu_metric_common_window_seconds",
    "cache_transition_profiler_window_seconds_by_rank",
    "cache_transition_nsys_process_ids_by_rank",
    "cache_transition_nsys_device_ids_by_rank",
    "cache_transition_nsys_kernel_intersection_count_by_rank",
    "epoch_eligible",
    "cache_transition_eligible",
    "eligible",
}

PROFILE_INPUT_BINDING_FIELDS = (
    "plan_sha256",
    "workload_sha256",
    "source_config_sha256",
    "source_checkpoint_sha256",
    "source_checkpoint_binding",
    "zero_step_checkpoint_sha256",
    "zero_step_checkpoint_binding",
    "dataset_content_binding",
    "overlay_files_sha256",
    "overlay_files",
    "train_cache_content_binding",
    "validation_cache_content_binding",
    "training_source_files",
    "source_revision",
)


def source_compatible_rwkv_head_size(checkpoint: Path) -> int:
    config_path = (
        checkpoint
        if checkpoint.name == "config.json"
        else checkpoint.parent / "config.json"
    )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"cannot derive RWKV head size from source config: {error}"
        ) from error
    text = payload.get("text_config", payload)
    if not isinstance(text, dict):
        raise SystemExit("source config text_config must be an object")
    layer_types = tuple(str(value) for value in text.get("layer_types", ()))
    if "linear_attention" in layer_types:
        key_geometry = (
            int(text.get("linear_num_key_heads", 0)),
            int(text.get("linear_key_head_dim", 0)),
        )
        value_geometry = (
            int(text.get("linear_num_value_heads", 0)),
            int(text.get("linear_value_head_dim", 0)),
        )
        attention_geometry = (
            int(text.get("num_attention_heads", 0)),
            int(text.get("head_dim", 0)),
        )
        if (
            min(*key_geometry, *value_geometry, *attention_geometry) <= 0
            or key_geometry[1] != value_geometry[1]
            or value_geometry[0] % key_geometry[0]
            or value_geometry[0] * value_geometry[1]
            != attention_geometry[0] * attention_geometry[1]
        ):
            raise SystemExit(
                "source GDN must preserve key/value head size, use integral key-head "
                "repeat, and match the full-attention recurrent width"
            )
        return value_geometry[1]
    attention_heads = int(text.get("num_attention_heads", 0))
    attention_head_size = int(text.get("head_dim", 0))
    if attention_heads <= 0 or attention_head_size <= 0:
        raise SystemExit("source attention head geometry must be positive")
    return attention_head_size


def source_layer_count(checkpoint: Path) -> int:
    config_path = (
        checkpoint
        if checkpoint.name == "config.json"
        else checkpoint.parent / "config.json"
    )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot derive source layer count: {error}") from error
    text = payload.get("text_config", payload)
    if not isinstance(text, dict):
        raise SystemExit("source config text_config must be an object")
    layers = int(text.get("num_hidden_layers", 0))
    if layers <= 0:
        raise SystemExit("source num_hidden_layers must be positive")
    return layers


def source_profile_cases(checkpoint: Path) -> dict[str, dict[str, Any]]:
    """Return the exact layer classes that a batch candidate must cover.

    Layer zero is intentionally a separate case because it consumes embedding
    output without recurrent shared state.  Later layers consume a converted
    RWKV7 prefix and are grouped by the source Qwen mixer kind.  The earliest
    layer in each class is the deterministic representative used by profiling.
    """
    config_path = (
        checkpoint
        if checkpoint.name == "config.json"
        else checkpoint.parent / "config.json"
    )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot derive performance profile cases: {error}") from error
    text = payload.get("text_config", payload)
    if not isinstance(text, dict):
        raise SystemExit("source config text_config must be an object")
    layer_types = text.get("layer_types")
    layer_count = int(text.get("num_hidden_layers", 0))
    if (
        not isinstance(layer_types, list)
        or layer_count <= 0
        or len(layer_types) != layer_count
        or not all(isinstance(value, str) and value for value in layer_types)
    ):
        raise SystemExit(
            "source performance profiling requires one mixer kind for every layer"
        )
    cases: dict[str, dict[str, Any]] = {}
    for layer, mixer_kind in enumerate(layer_types):
        input_boundary = "embedding-output" if layer == 0 else "recurrent-prefix"
        case_id = f"{input_boundary}:{mixer_kind}"
        case = cases.setdefault(
            case_id,
            {
                "profile_case_id": case_id,
                "source_mixer_kind": mixer_kind,
                "input_boundary": input_boundary,
                "representative_layer": layer,
                "layer_count": 0,
                "transition_count": 0,
            },
        )
        case["layer_count"] += 1
        if layer + 1 < layer_count:
            case["transition_count"] += 1
    return cases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_strict_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {label}: {error}") from error


def load_experiment_report_identity(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            first_line = stream.readline().strip()
    except OSError as error:
        raise SystemExit(
            f"cannot read experiment Markdown identity: {error}"
        ) from error
    prefix = "<!-- any2rwkv-report-identity: "
    if not first_line.startswith(prefix) or not first_line.endswith(" -->"):
        raise SystemExit("experiment Markdown lacks machine-readable run identity")
    try:
        identity = json.loads(first_line[len(prefix) : -4])
    except json.JSONDecodeError as error:
        raise SystemExit("experiment Markdown identity is malformed") from error
    if not isinstance(identity, dict):
        raise SystemExit("experiment Markdown identity must be a JSON object")
    return identity


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def training_source_files() -> dict[str, str]:
    paths = {Path(raw_path) for raw_path in TRAINING_SOURCE_FILES}
    paths.update(Path("src/train/any2rwkv/any2rwkv").rglob("*.py"))
    result: dict[str, str] = {}
    for path in sorted(paths):
        if not path.is_file():
            raise SystemExit(f"strict training source file is missing: {path}")
        result[str(path)] = sha256_file(path)
    return result


def source_revision_binding() -> dict[str, object]:
    def is_git_sha(value: object) -> bool:
        return (
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None
        )

    manifest_path = Path(".helicopter-dev/source-revisions.json")
    if manifest_path.is_file():
        manifest = load_strict_json(
            manifest_path, label="synced source revision manifest"
        )
        product_commit = (
            manifest.get("product_commit") if isinstance(manifest, dict) else None
        )
        submodules = manifest.get("submodules") if isinstance(manifest, dict) else None
        if (
            not is_git_sha(product_commit)
            or not isinstance(submodules, dict)
            or any(
                not isinstance(path, str) or not path or not is_git_sha(revision)
                for path, revision in submodules.items()
            )
        ):
            raise SystemExit("synced source revision manifest is malformed")
        return {
            "product_commit": product_commit,
            "submodules": dict(sorted(submodules.items())),
        }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        raw_submodules = subprocess.run(
            ["git", "submodule", "status", "--recursive"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"cannot bind Git/submodule revisions: {error}") from error
    if not is_git_sha(head):
        raise SystemExit("Git HEAD binding is malformed")
    submodules = {}
    for line in raw_submodules:
        fields = line.strip().split()
        if len(fields) < 2:
            raise SystemExit("Git submodule revision binding is malformed")
        revision = fields[0].lstrip("-+U")
        if not is_git_sha(revision):
            raise SystemExit("Git submodule revision binding is malformed")
        submodules[fields[1]] = revision
    return {"product_commit": head, "submodules": dict(sorted(submodules.items()))}


def checkpoint_content_binding(config_path: Path) -> dict[str, object]:
    root = config_path.parent
    index_path = root / "model.safetensors.index.json"
    index = load_strict_json(index_path, label="checkpoint shard index")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("performance checkpoint shard index is empty")
    names = {
        "config.json",
        "model.safetensors.index.json",
        *map(str, weight_map.values()),
    }
    for tokenizer_name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ):
        if (root / tokenizer_name).is_file():
            names.add(tokenizer_name)
    for optional in ("mapping.json", "mapping-coverage.json", "warm-start-plan.json"):
        if (root / optional).is_file():
            names.add(optional)
    files = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise SystemExit(f"performance checkpoint file is missing: {path}")
        files[name] = sha256_file(path)
    return {"files": files, "sha256": sha256_json(files)}


def dataset_content_binding(manifest_path: Path) -> dict[str, object]:
    """Bind the prepared manifest and every prepared file consumed by training."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_strict_json(manifest_path, label="prepared dataset manifest")
    if not isinstance(manifest, dict):
        raise SystemExit("prepared dataset manifest must be a JSON object")
    files = {"manifest": sha256_file(manifest_path)}
    splits = manifest.get("splits")
    if splits is not None:
        if not isinstance(splits, dict) or not splits:
            raise SystemExit("prepared dataset manifest splits must be nonempty")
        for split, entry in sorted(splits.items()):
            if not isinstance(split, str) or not isinstance(entry, dict):
                raise SystemExit("prepared dataset split binding is malformed")
            raw_path = entry.get("path")
            expected_sha = entry.get("sha256")
            if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
                raise SystemExit("prepared dataset split path/SHA-256 is missing")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (manifest_path.parent / path).resolve()
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise SystemExit(f"prepared dataset split changed: {split}")
            files[f"split:{split}:{path}"] = expected_sha
    deduplication = manifest.get("deduplication")
    if isinstance(deduplication, dict) and deduplication.get("report_path") is not None:
        raw_path = deduplication.get("report_path")
        expected_sha = deduplication.get("report_sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            raise SystemExit("prepared dataset deduplication binding is malformed")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise SystemExit("prepared dataset deduplication report changed")
        files[f"deduplication:{path}"] = expected_sha
    return {"files": files, "sha256": sha256_json(files)}


def cache_content_binding(manifest_path: Path) -> dict[str, object]:
    """Bind a cache manifest and every shard it declares, with full hashes."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_strict_json(manifest_path, label="layer-input cache manifest")
    shards = manifest.get("shards") if isinstance(manifest, dict) else None
    if not isinstance(shards, list) or not shards:
        raise SystemExit("layer-input cache manifest requires nonempty shards")
    files = {"manifest.json": sha256_file(manifest_path)}
    root = manifest_path.parent.resolve()
    for shard in shards:
        if not isinstance(shard, dict):
            raise SystemExit("layer-input cache shard binding is malformed")
        raw_path = shard.get("path")
        expected_sha = shard.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            raise SystemExit("layer-input cache shard path/SHA-256 is missing")
        path = (root / raw_path).resolve()
        if path != root and root not in path.parents:
            raise SystemExit("layer-input cache shard escapes its cache directory")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise SystemExit(f"layer-input cache shard changed: {path}")
        relative = str(path.relative_to(root))
        if relative in files:
            raise SystemExit("layer-input cache declares a duplicate shard")
        files[relative] = expected_sha
    return {"files": files, "sha256": sha256_json(files)}


def performance_workload_binding(plan: dict[str, Any]) -> dict[str, object]:
    normalized = {
        key: value
        for key, value in plan.items()
        if key
        not in {
            "micro_batch_size",
            "performance_evidence",
            "throughput_evidence",
            "notes",
        }
    }
    return {"plan": normalized, "sha256": sha256_json(normalized)}


def _nsys_process_key(value: object) -> int:
    raw = int(value)
    return raw >> 24 if raw >= 1 << 24 else raw


def _merged_interval_duration(intervals: list[tuple[int, int]]) -> int:
    active = 0
    current_start: int | None = None
    current_end: int | None = None
    for start, end in sorted(intervals):
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            active += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        active += current_end - current_start
    return active


def derive_nsys_profile_metrics(
    sqlite_path: Path, nvtx_range_names_by_rank: list[object]
) -> dict[str, Any]:
    """Bind kernel coverage and sampled SM activity to exact NVTX windows."""
    if (
        len(nvtx_range_names_by_rank) != 8
        or not all(
            isinstance(value, str) and value for value in nvtx_range_names_by_rank
        )
        or len(set(nvtx_range_names_by_rank)) != 8
    ):
        raise SystemExit("Nsight derivation requires eight unique NVTX range names")
    try:
        with sqlite3.connect(sqlite_path) as connection:
            tables = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
            table_columns = {
                table: {
                    str(row[1])
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                }
                for table in tables
            }
            kernel_table = None
            for table in tables:
                columns = table_columns[table]
                if {"start", "end", "deviceId", "globalPid"}.issubset(
                    columns
                ) and "KERNEL" in table:
                    kernel_table = table
                    break
            if kernel_table is None:
                raise SystemExit(
                    "Nsight SQLite kernel table must contain start/end/deviceId/globalPid"
                )
            nvtx_table = next(
                (
                    table
                    for table in tables
                    if "NVTX" in table.upper()
                    and {"start", "end", "globalTid"}.issubset(table_columns[table])
                    and {"text", "textId"}.intersection(table_columns[table])
                ),
                None,
            )
            if nvtx_table is None:
                raise SystemExit("Nsight SQLite has no usable NVTX event table")
            strings: dict[int, str] = {}
            string_table = next(
                (
                    table
                    for table in tables
                    if {"id", "value"}.issubset(table_columns[table])
                    and "STRING" in table.upper()
                ),
                None,
            )
            if string_table is not None:
                strings = {
                    int(identifier): str(value)
                    for identifier, value in connection.execute(
                        f'SELECT id, value FROM "{string_table}"'
                    )
                }
            nvtx_columns = table_columns[nvtx_table]
            selected_columns = ["start", "end", "globalTid"]
            selected_columns.extend(
                name for name in ("text", "textId") if name in nvtx_columns
            )
            range_events: dict[str, list[tuple[int, int, int]]] = {
                str(name): [] for name in nvtx_range_names_by_rank
            }
            query = "SELECT " + ", ".join(selected_columns) + f' FROM "{nvtx_table}"'
            for row in connection.execute(query):
                values = dict(zip(selected_columns, row, strict=True))
                text_value = values.get("text")
                if text_value is None and values.get("textId") is not None:
                    text_value = strings.get(int(values["textId"]))
                name = str(text_value) if text_value is not None else ""
                if name not in range_events:
                    continue
                start = int(values["start"])
                end = int(values["end"])
                if end <= start:
                    raise SystemExit("bound Nsight NVTX range has no positive duration")
                range_events[name].append(
                    (start, end, _nsys_process_key(values["globalTid"]))
                )
            if any(
                len(range_events[str(name)]) != 1 for name in nvtx_range_names_by_rank
            ):
                raise SystemExit(
                    "each rank must have exactly one bound NVTX range in Nsight SQLite"
                )
            windows = [range_events[str(name)][0] for name in nvtx_range_names_by_rank]
            if len({process for _, _, process in windows}) != 8:
                raise SystemExit(
                    "bound NVTX ranges do not identify eight rank processes"
                )
            kernels: dict[int, list[tuple[int, int, int]]] = {}
            kernel_query = (
                f'SELECT globalPid, deviceId, start, end FROM "{kernel_table}" '
                "ORDER BY globalPid, start"
            )
            for global_pid, device, start, end in connection.execute(kernel_query):
                start = int(start)
                end = int(end)
                if end > start:
                    kernels.setdefault(_nsys_process_key(global_pid), []).append(
                        (int(device), start, end)
                    )
            gpu_metrics_columns = table_columns.get("GPU_METRICS", set())
            gpu_metric_info_columns = table_columns.get(
                "TARGET_INFO_GPU_METRICS", set()
            )
            if not {"timestamp", "typeId", "metricId", "value"}.issubset(
                gpu_metrics_columns
            ) or not {
                "typeId",
                "sourceId",
                "metricId",
                "typeName",
                "metricName",
            }.issubset(gpu_metric_info_columns):
                raise SystemExit(
                    "Nsight SQLite requires sampled GPU_METRICS and "
                    "TARGET_INFO_GPU_METRICS tables"
                )
            common_start = max(start for start, _, _ in windows)
            common_end = min(end for _, end, _ in windows)
            if common_end <= common_start:
                raise SystemExit("rank NVTX ranges have no common GPU-metric window")
            sm_samples: dict[tuple[int, int, str], list[float]] = {}
            sm_query = (
                "SELECT samples.timestamp, samples.value, info.sourceId, "
                "info.typeId, info.typeName "
                "FROM GPU_METRICS AS samples "
                "JOIN TARGET_INFO_GPU_METRICS AS info "
                "ON samples.typeId = info.typeId "
                "AND samples.metricId = info.metricId "
                "WHERE info.metricName LIKE 'SMs Active%' "
                "AND samples.timestamp >= ? AND samples.timestamp <= ? "
                "ORDER BY info.typeId, samples.timestamp"
            )
            for _, value, source_id, type_id, type_name in connection.execute(
                sm_query, (common_start, common_end)
            ):
                numeric = float(value)
                if not math.isfinite(numeric) or not 0 <= numeric <= 100:
                    raise SystemExit("Nsight SMs Active sample is outside 0..100")
                sm_samples.setdefault(
                    (int(source_id), int(type_id), str(type_name)), []
                ).append(numeric / 100.0)
    except sqlite3.Error as error:
        raise SystemExit(f"cannot read Nsight SQLite evidence: {error}") from error
    fractions: list[object] = []
    window_seconds: list[object] = []
    process_ids: list[object] = []
    device_ids: list[object] = []
    kernel_counts: list[object] = []
    for window_start, window_end, process in windows:
        intersections: list[tuple[int, int]] = []
        devices: set[int] = set()
        for device, start, end in kernels.get(process, []):
            clipped_start = max(start, window_start)
            clipped_end = min(end, window_end)
            if clipped_end > clipped_start:
                intersections.append((clipped_start, clipped_end))
                devices.add(device)
        if not intersections or len(devices) != 1:
            raise SystemExit(
                "each bound rank window must contain kernels from exactly one device"
            )
        duration = window_end - window_start
        active = _merged_interval_duration(intersections)
        fraction = active / duration
        if not 0 <= fraction <= 1:
            raise SystemExit("Nsight busy fraction is outside the bound NVTX window")
        fractions.append(fraction)
        window_seconds.append(duration / 1e9)
        process_ids.append(process)
        device_ids.append(next(iter(devices)))
        kernel_counts.append(len(intersections))
    if len(set(device_ids)) != 8:
        raise SystemExit("bound rank windows do not cover eight distinct CUDA devices")
    sm_source_ids = {source_id for source_id, _, _ in sm_samples}
    if len(sm_samples) != 8 or len(sm_source_ids) != 8:
        raise SystemExit(
            "Nsight GPU metrics must provide one SMs Active source for each of eight GPUs"
        )
    if sm_source_ids != set(device_ids):
        raise SystemExit(
            "Nsight SMs Active source IDs do not match the eight CUDA devices"
        )
    by_source = {
        source_id: ((source_id, type_id, type_name), values)
        for (source_id, type_id, type_name), values in sm_samples.items()
    }
    ordered_sm_samples = [by_source[device] for device in device_ids]
    sample_counts = [len(values) for _, values in ordered_sm_samples]
    if min(sample_counts) < 100:
        raise SystemExit(
            "Nsight GPU metrics require at least 100 SMs Active samples per GPU"
        )
    sm_active = [sum(values) / len(values) for _, values in ordered_sm_samples]
    return {
        "kernel_covered_wall_fraction_by_rank": fractions,
        "window_seconds_by_rank": window_seconds,
        "process_ids_by_rank": process_ids,
        "device_ids_by_rank": device_ids,
        "kernel_intersection_count_by_rank": kernel_counts,
        "sm_active_fraction_by_gpu": sm_active,
        "sm_active_sample_count_by_gpu": sample_counts,
        "sm_active_metric_source_by_gpu": [
            {
                "source_id": source_id,
                "type_id": type_id,
                "type_name": type_name,
            }
            for (source_id, type_id, type_name), _ in ordered_sm_samples
        ],
        "gpu_metric_common_window_seconds": (common_end - common_start) / 1e9,
    }


def performance_candidate_passes_gates(
    *,
    kernel_coverage: list[object],
    sm_active: list[object],
    reserved: list[object],
    wall_seconds: list[object],
    checkpoint_fraction: object,
) -> bool:
    """Apply both the anti-bubble and actual-GPU-work admission gates."""
    return bool(
        min(float(value) for value in kernel_coverage)
        >= MINIMUM_KERNEL_COVERED_WALL_FRACTION
        and max(float(value) for value in kernel_coverage) <= 1.0
        and min(float(value) for value in sm_active) >= MINIMUM_SM_ACTIVE_FRACTION
        and max(float(value) for value in sm_active) <= 1.0
        and all(
            MINIMUM_RESERVED_MEMORY_FRACTION
            <= float(value)
            <= MAXIMUM_RESERVED_MEMORY_FRACTION
            for value in reserved
        )
        and min(float(value) for value in wall_seconds) > 0
        and max(float(value) for value in wall_seconds)
        / min(float(value) for value in wall_seconds)
        <= MAXIMUM_RANK_WALL_RATIO
        and 0
        <= float(checkpoint_fraction)
        <= MAXIMUM_CHECKPOINT_WALL_FRACTION
    )


def require_file(name: str, raw_path: str) -> tuple[Path, str]:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{name} must be a readable file: {path}")
    return path, sha256_file(path)


def parse_command(argv: list[str]) -> list[str]:
    if not argv or argv[0] != "--" or len(argv) == 1:
        raise SystemExit("usage: strict_any2rwkv_train.py -- COMMAND...")
    return argv[1:]


class ParsedTrainingCommand(NamedTuple):
    action: str
    source: Path
    output: Path
    dataset_manifest: Path
    training_config: Path
    recipe: str
    precision: str
    run_id: str
    parent_run: Path | None
    parent_checkpoint_sha256: str | None


def _resolved_executable(value: str) -> Path:
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            resolved = str(candidate)
    if resolved is None:
        raise SystemExit(f"strict Any2RWKV command executable is missing: {value}")
    return Path(resolved).resolve()


def _reject_duplicate_cli_options(argv: list[str]) -> None:
    seen: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        if name in seen:
            raise SystemExit(
                f"strict Any2RWKV command rejects duplicate option: {name}"
            )
        seen.add(name)


def parse_training_command(command: list[str]) -> ParsedTrainingCommand:
    """Parse the exact command that torchrun will execute, once."""
    if not command or _resolved_executable(command[0]).name != "torchrun":
        raise SystemExit("strict Any2RWKV training command must start with torchrun")
    if command.count("--no-python") != 1:
        raise SystemExit("strict Any2RWKV training requires exactly one --no-python")
    nproc_values = [
        token.split("=", 1)[1]
        for token in command
        if token.startswith("--nproc-per-node=")
    ]
    if command.count("--nproc-per-node"):
        raise SystemExit(
            "strict Any2RWKV training requires the unambiguous --nproc-per-node=8 form"
        )
    if nproc_values != ["8"]:
        raise SystemExit(
            "strict Any2RWKV training requires exactly one --nproc-per-node=8"
        )
    entry_index = command.index("--no-python") + 1
    if tuple(command[entry_index + 1 : entry_index + 3]) != (
        "-m",
        "any2rwkv.cli",
    ):
        raise SystemExit(
            "strict Any2RWKV training command must execute "
            "python -m any2rwkv.cli distill|corrective"
        )
    if entry_index >= len(command):
        raise SystemExit("strict Any2RWKV command has no Python interpreter")
    interpreter = _resolved_executable(command[entry_index])
    if interpreter != Path(sys.executable).resolve():
        raise SystemExit(
            "strict Any2RWKV command must use the wrapper's Python environment: "
            f"expected={Path(sys.executable).resolve()} actual={interpreter}"
        )
    application_argv = command[entry_index + 3 :]
    _reject_duplicate_cli_options(application_argv)
    try:
        from any2rwkv.cli import build_parser

        args = build_parser().parse_args(application_argv)
    except SystemExit as error:
        raise SystemExit(
            "strict Any2RWKV command does not match the CLI schema"
        ) from error
    if args.action not in {"distill", "corrective"}:
        raise SystemExit(
            "strict Any2RWKV training command must execute "
            "python -m any2rwkv.cli distill|corrective"
        )
    if not args.run_id:
        raise SystemExit("strict Any2RWKV command requires an explicit --run-id")
    return ParsedTrainingCommand(
        action=str(args.action),
        source=Path(args.source).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        dataset_manifest=Path(args.dataset_manifest).expanduser().resolve(),
        training_config=Path(args.training_config).expanduser().resolve(),
        recipe=str(args.recipe),
        precision=str(args.precision),
        run_id=str(args.run_id),
        parent_run=(
            Path(args.parent_run).expanduser().resolve()
            if args.action == "corrective"
            else None
        ),
        parent_checkpoint_sha256=(
            str(args.parent_checkpoint_sha256).lower()
            if args.action == "corrective"
            else None
        ),
    )


def validate_distill_command(
    parsed: ParsedTrainingCommand,
    *,
    checkpoint: Path,
    dataset_manifest: Path,
    config: Path,
    expected_run_id: str,
) -> dict[str, Any] | None:
    parent_binding: dict[str, Any] | None = None
    if parsed.run_id != expected_run_id:
        raise SystemExit("CLI --run-id differs from HELICOPTER_RUN_ID")
    if parsed.precision != "bf16":
        raise SystemExit("CLI --precision must be bf16 for strict training")
    if parsed.action == "corrective":
        assert parsed.parent_run is not None
        assert parsed.parent_checkpoint_sha256 is not None
        parent = parsed.parent_run
        parent_checkpoint = next(
            (
                candidate
                for candidate in (
                    parent / "checkpoint-global-corrective",
                    parent / "checkpoint-layerwise-local",
                )
                if (candidate / "config.json").is_file()
            ),
            None,
        )
        if parent_checkpoint is None:
            raise SystemExit("corrective --parent-run has no recurrent checkpoint")
        parent_binding = checkpoint_directory_binding(parent_checkpoint)
        expected_parent_sha = parsed.parent_checkpoint_sha256
        if parent_binding["sha256"] != expected_parent_sha:
            raise SystemExit(
                "corrective parent checkpoint SHA-256 mismatch: "
                f"expected={expected_parent_sha} actual={parent_binding['sha256']}"
            )
        try:
            parent_metadata = json.loads(
                (parent / "metadata.json").read_text(encoding="utf-8")
            )
            parent_config = json.loads(
                (parent_checkpoint / "config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"invalid corrective parent metadata/checkpoint: {error}"
            ) from error
        requested_recipe = parsed.recipe
        source_files = parent_metadata.get("source", {}).get("files", {})
        recurrence = parent_config.get("any2rwkv", {})
        if (
            parent_metadata.get("recipe", {}).get("id") != requested_recipe
            or sha256_file(checkpoint) not in source_files.values()
            or parent_metadata.get("precision") != "bf16"
            or recurrence.get("recurrence") != "native_rwkv7"
            or not (
                recurrence.get("final_recurrent") is True
                or recurrence.get("fully_recurrent_proxy") is True
            )
        ):
            raise SystemExit(
                "corrective parent source/recipe/precision/recurrence binding mismatch"
            )
    if parsed.training_config != config:
        raise SystemExit(
            "distill --training-config differs from HELICOPTER_CONFIG_PATH"
        )
    if parsed.dataset_manifest != dataset_manifest:
        raise SystemExit(
            "distill --dataset-manifest differs from HELICOPTER_DATASET_MANIFEST"
        )
    source = parsed.source
    accepted_sources = {
        checkpoint,
        checkpoint.parent if checkpoint.name == "config.json" else checkpoint,
    }
    if source not in accepted_sources:
        raise SystemExit("distill --source differs from HELICOPTER_CHECKPOINT_PATH")
    return parent_binding


def checkpoint_directory_binding(checkpoint: Path) -> dict[str, Any]:
    index_path = checkpoint / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid corrective parent shard index: {error}") from error
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("corrective parent shard index is empty")
    files = (
        "config.json",
        "model.safetensors.index.json",
        *sorted(set(weight_map.values())),
    )
    hashes = {}
    for name in files:
        path = checkpoint / str(name)
        if not path.is_file():
            raise SystemExit(f"corrective parent checkpoint file is missing: {path}")
        hashes[str(name)] = sha256_file(path)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    mixer_fingerprint = str(
        config.get("any2rwkv", {}).get("mixer_overlay_fingerprint", "")
    )
    if len(mixer_fingerprint) != 64:
        raise SystemExit("corrective parent checkpoint lacks mixer fingerprint")
    payload = {"files": hashes, "mixer_fingerprint": mixer_fingerprint}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"path": str(checkpoint), **payload, "sha256": digest}


def validate_throughput_evidence(
    plan: dict[str, Any],
    *,
    source_config: Path,
    target_config: Path,
    head_size: int,
) -> dict[str, Any] | None:
    evidence = plan.get("throughput_evidence")
    if evidence is None:
        return None
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != "measured-exploratory"
    ):
        raise SystemExit("throughput_evidence must be measured-exploratory")
    artifact_value = evidence.get("artifact")
    expected_sha = evidence.get("artifact_sha256")
    if not isinstance(artifact_value, str) or not isinstance(expected_sha, str):
        raise SystemExit("throughput_evidence requires artifact and artifact_sha256")
    artifact = Path(artifact_value).expanduser().resolve()
    if not artifact.is_file() or sha256_file(artifact) != expected_sha:
        raise SystemExit("throughput_evidence artifact SHA-256 mismatch")
    profile = load_strict_json(artifact, label="throughput evidence artifact")
    if not isinstance(profile, dict):
        raise SystemExit("throughput evidence artifact must be a JSON object")
    selected = int(plan.get("micro_batch_size", 0))
    candidates = profile.get("candidates")
    binding = profile.get("binding")
    if (
        profile.get("schema_version") != 1
        or profile.get("status") != "complete"
        or profile.get("selection_rule")
        != "highest-slowest-rank-loss-token-throughput-within-memory-limit"
        or not isinstance(binding, dict)
        or binding.get("world_size") != 8
        or binding.get("source_config_sha256") != sha256_file(source_config)
        or not target_config.is_file()
        or binding.get("zero_step_config_sha256") != sha256_file(target_config)
        or binding.get("head_size") != head_size
        or not isinstance(candidates, list)
        or len(candidates) < 3
        or profile.get("selected_per_rank_micro_batch_size") != selected
        or evidence.get("selected_per_rank_micro_batch_size") != selected
    ):
        raise SystemExit("throughput_evidence does not bind the selected 8-rank batch")
    eligible = [row for row in candidates if row.get("eligible") is True]
    if not eligible or any(
        row.get("loss_token_count") != profile.get("equal_loss_token_budget")
        for row in eligible
    ):
        raise SystemExit(
            "throughput_evidence candidates do not use an equal token budget"
        )
    winner = max(eligible, key=lambda row: float(row["loss_tokens_per_second"]))
    if winner.get("per_rank_micro_batch_size") != selected:
        raise SystemExit("throughput_evidence selection is not reproducible")
    return {
        "path": str(artifact),
        "sha256": expected_sha,
        "selected_per_rank_micro_batch_size": selected,
        "loss_tokens_per_second": winner["loss_tokens_per_second"],
    }


def validate_experiment_artifacts(
    artifacts: object,
    *,
    base_dir: Path,
    expected_run_id: str,
    expected_project: str,
    expected_entity: str | None,
    expected_group: str | None,
    expected_tags: list[str],
    expected_job_type: str,
) -> dict[str, Any]:
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise SystemExit(
            "experiment evidence requires W&B tracking and Markdown report"
        )
    paths: dict[str, Path] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise SystemExit("experiment artifact binding must be an object")
        kind = artifact.get("kind")
        raw_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if (
            kind not in {"wandb-tracking", "experiment-report"}
            or not isinstance(raw_path, str)
            or not isinstance(expected_sha, str)
        ):
            raise SystemExit("experiment artifact binding is incomplete")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise SystemExit("experiment artifact SHA-256 mismatch")
        paths[str(kind)] = path
    if set(paths) != {"wandb-tracking", "experiment-report"}:
        raise SystemExit("experiment artifact kinds are incomplete")
    tracking = load_strict_json(
        paths["wandb-tracking"], label="bound W&B tracking result"
    )
    report_identity = load_experiment_report_identity(paths["experiment-report"])
    config = tracking.get("config") if isinstance(tracking, dict) else None
    attempt_id = tracking.get("attempt_id") if isinstance(tracking, dict) else None
    config_sha256 = (
        tracking.get("config_sha256") if isinstance(tracking, dict) else None
    )
    launch_token_sha256 = (
        tracking.get("launch_token_sha256")
        if isinstance(tracking, dict)
        else None
    )
    if (
        not isinstance(tracking, dict)
        or tracking.get("schema_version") != 2
        or tracking.get("backend") != "wandb"
        or tracking.get("mode") != "online"
        or tracking.get("status") != "completed"
        or tracking.get("run_id") != expected_run_id
        or tracking.get("project") != expected_project
        or tracking.get("entity") != expected_entity
        or tracking.get("group") != expected_group
        or tracking.get("tags") != expected_tags
        or tracking.get("job_type") != expected_job_type
        or not tracking.get("url")
        or not isinstance(attempt_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None
        or not isinstance(launch_token_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", launch_token_sha256) is None
        or type(tracking.get("coordinated_world_size")) is not int
        or tracking["coordinated_world_size"] <= 0
        or not isinstance(config, dict)
        or config_sha256 != sha256_json(config)
        or report_identity.get("schema_version") != 1
        or report_identity.get("run_id") != expected_run_id
        or report_identity.get("attempt_id") != attempt_id
        or report_identity.get("tracking_status") != "completed"
        or report_identity.get("config_sha256") != config_sha256
        or not isinstance(report_identity.get("status"), str)
        or not report_identity["status"]
        or paths["experiment-report"].stat().st_size == 0
    ):
        raise SystemExit("experiment W&B identity or Markdown report is invalid")
    return tracking


def _validate_selection_candidate_source(
    row: dict[str, Any], *, base_dir: Path
) -> tuple[Path, dict[str, Any]]:
    source_binding = row.get("source_candidate")
    if not isinstance(source_binding, dict):
        raise SystemExit("selection candidate lacks its original candidate binding")
    raw_path = source_binding.get("path")
    expected_sha = source_binding.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
        raise SystemExit("selection candidate source path/SHA-256 is missing")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise SystemExit("selection candidate source SHA-256 mismatch")
    source = load_strict_json(path, label="original performance candidate")
    if not isinstance(source, dict):
        raise SystemExit("original performance candidate must be an object")
    extras = set(row) - set(source)
    if not extras.issubset(CANDIDATE_SELECTION_DERIVED_FIELDS):
        raise SystemExit("selection candidate contains unrecognized derived fields")
    mutable_derived = {
        "loss_tokens_per_second",
        "checkpoint_wall_fraction",
        "profiler_artifacts",
    }
    for key, value in source.items():
        if key in mutable_derived:
            continue
        if row.get(key) != value:
            raise SystemExit(
                f"selection candidate changed original measured field: {key}"
            )
    source_profiler = source.get("profiler_artifacts")
    row_profiler = row.get("profiler_artifacts")
    if not isinstance(source_profiler, list) or not isinstance(row_profiler, list):
        raise SystemExit("selection candidate profiler artifacts are malformed")
    projected = [
        {"kind": item.get("kind"), "path": item.get("path")}
        for item in row_profiler
        if isinstance(item, dict) and item.get("kind") in {"nsys-rep", "nsys-sqlite"}
    ]
    if projected != source_profiler:
        raise SystemExit(
            "selection candidate profiler artifacts differ from the original"
        )
    return path, source


def validate_performance_evidence(
    plan: dict[str, Any],
    *,
    action: str,
    source_config: Path,
    target_config: Path,
    head_size: int,
    dataset_manifest: Path,
) -> dict[str, Any] | None:
    """Require end-to-end, eight-rank profile evidence before real training."""
    evidence = plan.get("performance_evidence")
    if plan.get("evidence_tier") == "fixture" and evidence is None:
        return None
    if not isinstance(evidence, dict) or evidence.get("status") != "accepted":
        raise SystemExit(
            "real Any2RWKV training requires accepted performance_evidence"
        )
    artifact_value = evidence.get("artifact")
    expected_sha = evidence.get("artifact_sha256")
    if not isinstance(artifact_value, str) or not isinstance(expected_sha, str):
        raise SystemExit("performance_evidence requires artifact and artifact_sha256")
    artifact = Path(artifact_value).expanduser().resolve()
    if not artifact.is_file() or sha256_file(artifact) != expected_sha:
        raise SystemExit("performance_evidence artifact SHA-256 mismatch")
    profile = load_strict_json(artifact, label="performance evidence artifact")
    if not isinstance(profile, dict):
        raise SystemExit("performance evidence artifact must be a JSON object")
    binding = profile.get("binding")
    selected = int(plan.get("micro_batch_size", 0))
    candidates = profile.get("candidates")
    current_sources = training_source_files()
    current_revision = source_revision_binding()
    source_checkpoint_binding = checkpoint_content_binding(source_config)
    checkpoint_binding = checkpoint_content_binding(target_config)
    current_dataset_binding = dataset_content_binding(dataset_manifest)
    workload_binding = performance_workload_binding(plan)
    required_profile_cases = source_profile_cases(source_config)
    tracking = plan.get("tracking")
    tracking_job_types = (
        tracking.get("job_types") if isinstance(tracking, dict) else None
    )
    profile_run_id = profile.get("run_id")
    if (
        profile.get("schema_version") != 3
        or profile.get("status") != "accepted"
        or profile.get("selection_rule")
        != "highest-weighted-all-layer-case-end-to-end-loss-token-throughput-with-all-gates"
        or not isinstance(binding, dict)
        or binding.get("action") != action
        or binding.get("world_size") != 8
        or binding.get("source_config_sha256") != sha256_file(source_config)
        or binding.get("source_checkpoint_sha256")
        != source_checkpoint_binding["sha256"]
        or binding.get("source_checkpoint_binding") != source_checkpoint_binding
        or binding.get("target_checkpoint_sha256") != checkpoint_binding["sha256"]
        or binding.get("target_checkpoint_binding") != checkpoint_binding
        or binding.get("dataset_manifest_sha256") != sha256_file(dataset_manifest)
        or binding.get("dataset_content_binding") != current_dataset_binding
        or binding.get("training_source_files") != current_sources
        or binding.get("source_revision") != current_revision
        or binding.get("workload_sha256") != workload_binding["sha256"]
        or binding.get("head_size") != head_size
        or binding.get("burn_in_tokens") != plan.get("burn_in_tokens")
        or binding.get("supervised_tokens") != plan.get("supervised_tokens")
        or binding.get("accumulation_steps") != 1
        or binding.get("gradient_checkpointing") is not False
        or binding.get("checkpoint_interval_micro_batches")
        != plan.get("checkpoint_interval_micro_batches")
        or profile.get("selected_per_rank_micro_batch_size") != selected
        or evidence.get("selected_per_rank_micro_batch_size") != selected
        or not isinstance(candidates, list)
        or len(candidates) < 3 * len(required_profile_cases)
        or profile.get("required_profile_cases") != required_profile_cases
        or not isinstance(profile.get("batch_summaries"), list)
        or not _finite_number(profile.get("equal_loss_token_budget"))
        or float(profile["equal_loss_token_budget"]) <= 0
        or not isinstance(profile_run_id, str)
        or not profile_run_id
        or not isinstance(tracking_job_types, dict)
    ):
        raise SystemExit(
            "performance_evidence fails immutable workload or candidate-set binding"
        )
    selection_tracking = validate_experiment_artifacts(
        profile.get("experiment_artifacts"),
        base_dir=artifact.parent,
        expected_run_id=profile_run_id,
        expected_project=str(tracking.get("project")),
        expected_entity=tracking.get("entity"),
        expected_group=tracking.get("group"),
        expected_tags=list(tracking.get("tags", [])),
        expected_job_type=str(tracking_job_types.get("profile")),
    )
    candidate_artifact_bindings = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise SystemExit("performance candidate must be an object")
        source = candidate.get("source_candidate")
        raw_path = source.get("path") if isinstance(source, dict) else None
        if not isinstance(raw_path, str):
            raise SystemExit("performance candidate source path is missing")
        candidate_path = Path(raw_path).expanduser().resolve()
        files = {
            "candidate": candidate_path,
            "nsys_export_binding": candidate_path.parent
            / "profiler-export-binding.json",
            "wandb_tracking": candidate_path.parent / "experiment-tracking.json",
            "experiment_report": candidate_path.parent / "experiment-report.md",
        }
        if any(not value.is_file() for value in files.values()):
            raise SystemExit("performance candidate artifact set is incomplete")
        candidate_artifact_bindings.append(
            {
                name: {"path": str(value), "sha256": sha256_file(value)}
                for name, value in files.items()
            }
        )
    expected_selection_config = {
        "schema_version": 1,
        "action": "performance-profile-selection",
        "run_id": profile_run_id,
        "plan_sha256": binding.get("selection_plan_sha256"),
        "workload_sha256": workload_binding["sha256"],
        "source_config_sha256": sha256_file(source_config),
        "source_checkpoint_sha256": source_checkpoint_binding["sha256"],
        "source_checkpoint_binding": source_checkpoint_binding,
        "target_checkpoint_sha256": checkpoint_binding["sha256"],
        "target_checkpoint_binding": checkpoint_binding,
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "dataset_content_binding": current_dataset_binding,
        "training_source_files": current_sources,
        "source_revision": current_revision,
        "comparison": plan.get("experiment_comparison"),
        "required_profile_cases": required_profile_cases,
        "candidate_artifacts": candidate_artifact_bindings,
    }
    if (
        not isinstance(binding.get("selection_plan_sha256"), str)
        or len(binding["selection_plan_sha256"]) != 64
        or binding.get("candidate_artifacts") != candidate_artifact_bindings
        or selection_tracking.get("config") != expected_selection_config
    ):
        raise SystemExit("selection W&B config differs from performance evidence")
    candidate_artifacts_by_path = {
        str(entry["candidate"]["path"]): entry
        for entry in candidate_artifact_bindings
    }
    batches: set[int] = set()
    candidate_matrix: dict[tuple[int, str], dict[str, Any]] = {}
    cache_bindings_by_case: dict[
        str, set[tuple[object, object, object]]
    ] = {case_id: set() for case_id in required_profile_cases}
    source_signatures_by_case: dict[str, set[str]] = {
        case_id: set() for case_id in required_profile_cases
    }
    target_signatures_by_case: dict[str, set[str]] = {
        case_id: set() for case_id in required_profile_cases
    }
    overlays_by_case: dict[str, set[str]] = {
        case_id: set() for case_id in required_profile_cases
    }
    validation_budgets: set[object] = set()
    candidate_run_ids: set[str] = set()
    for row in candidates:
        if not isinstance(row, dict):
            raise SystemExit("performance candidate must be a JSON object")
        source_candidate_path, _ = _validate_selection_candidate_source(
            row, base_dir=artifact.parent
        )
        expected_candidate_artifacts = candidate_artifacts_by_path.get(
            str(source_candidate_path)
        )
        if not isinstance(expected_candidate_artifacts, dict):
            raise SystemExit("selection candidate artifact binding is missing")
        batch = row.get("per_rank_micro_batch_size")
        kernel_coverage = row.get("kernel_covered_wall_fraction_by_rank")
        sm_active = row.get("sm_active_fraction_by_gpu")
        sm_sample_counts = row.get("sm_active_sample_count_by_gpu")
        sm_metric_sources = row.get("sm_active_metric_source_by_gpu")
        reserved = row.get("peak_reserved_memory_fraction_by_rank")
        wall = row.get("epoch_wall_seconds_by_rank")
        checkpoint_fraction = row.get("checkpoint_wall_fraction")
        profiler_artifacts = row.get("profiler_artifacts")
        candidate_binding = row.get("binding")
        candidate_run_id = row.get("run_id")
        profile_nonce = row.get("profile_nonce")
        nvtx_range_names = row.get("nvtx_range_names_by_rank")
        transition_nvtx_range_names = row.get(
            "transition_nvtx_range_names_by_rank"
        )
        profile_case = (
            candidate_binding.get("profile_case")
            if isinstance(candidate_binding, dict)
            else None
        )
        case_id = (
            profile_case.get("profile_case_id")
            if isinstance(profile_case, dict)
            else None
        )
        expected_case = required_profile_cases.get(str(case_id))
        transition_required = bool(
            expected_case is not None
            and int(expected_case["transition_count"]) > 0
        )
        train_rows = (
            candidate_binding.get("measured_train_rows")
            if isinstance(candidate_binding, dict)
            else None
        )
        validation_rows = (
            candidate_binding.get("measured_validation_rows")
            if isinstance(candidate_binding, dict)
            else None
        )
        if (
            type(batch) is not int
            or batch <= 0
            or row.get("loss_token_count") != profile["equal_loss_token_budget"]
            or not _finite_number(row.get("loss_tokens_per_second"))
            or float(row["loss_tokens_per_second"]) <= 0
            or row.get("includes_validation") is not True
            or row.get("includes_checkpoint") is not True
            or type(row.get("ephemeral_optimizer_steps")) is not int
            or row["ephemeral_optimizer_steps"] <= 0
            or row.get("persistent_weight_update") is not False
            or not isinstance(kernel_coverage, list)
            or len(kernel_coverage) != 8
            or not all(_finite_number(value) for value in kernel_coverage)
            or not isinstance(sm_active, list)
            or len(sm_active) != 8
            or not all(_finite_number(value) for value in sm_active)
            or not isinstance(sm_sample_counts, list)
            or len(sm_sample_counts) != 8
            or not all(type(value) is int and value >= 100 for value in sm_sample_counts)
            or not isinstance(sm_metric_sources, list)
            or len(sm_metric_sources) != 8
            or not isinstance(reserved, list)
            or len(reserved) != 8
            or not all(_finite_number(value) for value in reserved)
            or not isinstance(wall, list)
            or len(wall) != 8
            or not all(_finite_number(value) for value in wall)
            or not _finite_number(checkpoint_fraction)
            or not isinstance(profiler_artifacts, list)
            or len(profiler_artifacts) != 3
            or row.get("profiler_capture_contract")
            != PROFILER_CAPTURE_CONTRACT
            or not isinstance(profile_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", profile_nonce) is None
            or not isinstance(nvtx_range_names, list)
            or nvtx_range_names
            != [
                f"any2rwkv-profile:{candidate_run_id}:{profile_nonce}:rank={rank}"
                for rank in range(8)
            ]
            or row.get("cache_transition_profiled") is not transition_required
            or (
                transition_required
                and transition_nvtx_range_names
                != [
                    f"any2rwkv-transition:{candidate_run_id}:{profile_nonce}:rank={rank}"
                    for rank in range(8)
                ]
            )
            or (not transition_required and transition_nvtx_range_names != [])
            or not isinstance(candidate_binding, dict)
            or candidate_binding.get("plan_sha256")
            != binding.get("selection_plan_sha256")
            or candidate_binding.get("source_config_sha256")
            != sha256_file(source_config)
            or candidate_binding.get("source_checkpoint_sha256")
            != source_checkpoint_binding["sha256"]
            or candidate_binding.get("source_checkpoint_binding")
            != source_checkpoint_binding
            or candidate_binding.get("zero_step_checkpoint_sha256")
            != checkpoint_binding["sha256"]
            or candidate_binding.get("zero_step_checkpoint_binding")
            != checkpoint_binding
            or candidate_binding.get("dataset_content_binding")
            != current_dataset_binding
            or candidate_binding.get("training_source_files") != current_sources
            or candidate_binding.get("source_revision") != current_revision
            or not isinstance(
                candidate_binding.get("train_cache_content_binding"), dict
            )
            or not isinstance(
                candidate_binding.get("validation_cache_content_binding"), dict
            )
            or candidate_binding.get("workload_sha256") != workload_binding["sha256"]
            or candidate_binding.get("optimizer") != plan.get("optimizer")
            or candidate_binding.get("learning_rate_schedule")
            != plan.get("learning_rate_schedule")
            or candidate_binding.get("gradient_clip_norm")
            != plan.get("gradient_clip_norm")
            or candidate_binding.get("accumulation_steps") != 1
            or candidate_binding.get("gradient_checkpointing") is not False
            or candidate_binding.get("burn_in_tokens") != plan.get("burn_in_tokens")
            or candidate_binding.get("supervised_tokens")
            != plan.get("supervised_tokens")
            or candidate_binding.get("head_size") != head_size
            or candidate_binding.get("world_size") != 8
            or expected_case is None
            or profile_case != expected_case
            or candidate_binding.get("layer")
            != expected_case["representative_layer"]
            or candidate_binding.get("cache_has_shared_states")
            is not (expected_case["input_boundary"] == "recurrent-prefix")
            or not isinstance(
                candidate_binding.get("source_parameter_signature_sha256"), str
            )
            or len(candidate_binding["source_parameter_signature_sha256"]) != 64
            or not isinstance(
                candidate_binding.get("target_parameter_signature_sha256"), str
            )
            or len(candidate_binding["target_parameter_signature_sha256"]) != 64
            or type(train_rows) is not int
            or train_rows <= 0
            or type(validation_rows) is not int
            or validation_rows <= 0
            or train_rows % (8 * batch)
            or row.get("ephemeral_optimizer_steps") != train_rows // (8 * batch)
            or row["ephemeral_optimizer_steps"]
            <= int(plan["optimizer"]["warmup_steps"])
            or row.get("loss_token_count")
            != train_rows * int(plan.get("supervised_tokens", 0))
            or not isinstance(candidate_run_id, str)
            or not candidate_run_id
            or candidate_run_id == profile_run_id
            or candidate_run_id in candidate_run_ids
            or not isinstance(candidate_binding.get("overlay_files"), dict)
        ):
            raise SystemExit("performance candidate has malformed measurements")
        candidate_tracking = validate_experiment_artifacts(
            row.get("experiment_artifacts"),
            base_dir=artifact.parent,
            expected_run_id=candidate_run_id,
            expected_project=str(tracking.get("project")),
            expected_entity=tracking.get("entity"),
            expected_group=tracking.get("group"),
            expected_tags=list(tracking.get("tags", [])),
            expected_job_type=str(tracking_job_types.get("profile")),
        )
        row_experiment_artifacts = {
            str(item.get("kind")): {
                "path": str(Path(str(item.get("path"))).expanduser().resolve()),
                "sha256": item.get("sha256"),
            }
            for item in row.get("experiment_artifacts", [])
            if isinstance(item, dict)
        }
        if row_experiment_artifacts != {
            "wandb-tracking": expected_candidate_artifacts["wandb_tracking"],
            "experiment-report": expected_candidate_artifacts["experiment_report"],
        }:
            raise SystemExit(
                "candidate tracking/report differs from selection artifact binding"
            )
        expected_candidate_tracking_config = {
            "schema_version": 1,
            "action": "performance-profile-candidate",
            "run_id": candidate_run_id,
            "plan_sha256": candidate_binding.get("plan_sha256"),
            "workload_sha256": workload_binding["sha256"],
            "source_config_sha256": sha256_file(source_config),
            "source_checkpoint_sha256": source_checkpoint_binding["sha256"],
            "source_checkpoint_binding": source_checkpoint_binding,
            "zero_step_checkpoint_sha256": checkpoint_binding["sha256"],
            "zero_step_checkpoint_binding": checkpoint_binding,
            "dataset_content_binding": current_dataset_binding,
            "overlay_files_sha256": candidate_binding.get("overlay_files_sha256"),
            "overlay_files": candidate_binding.get("overlay_files"),
            "train_cache_manifest_sha256": candidate_binding.get(
                "train_cache_manifest_sha256"
            ),
            "validation_cache_manifest_sha256": candidate_binding.get(
                "validation_cache_manifest_sha256"
            ),
            "train_cache_content_binding": candidate_binding.get(
                "train_cache_content_binding"
            ),
            "validation_cache_content_binding": candidate_binding.get(
                "validation_cache_content_binding"
            ),
            "training_source_files": current_sources,
            "source_revision": current_revision,
            "cache_prefix_fingerprint": candidate_binding.get(
                "cache_prefix_fingerprint"
            ),
            "optimizer": plan.get("optimizer"),
            "comparison": plan.get("experiment_comparison"),
            "micro_batch_size_per_rank": batch,
            "world_size": 8,
            "gradient_checkpointing": False,
            "measured_train_rows": train_rows,
            "measured_validation_rows": validation_rows,
            "row_permutation_sha256": candidate_binding.get(
                "row_permutation_sha256"
            ),
            "profile_case": profile_case,
            "cache_has_shared_states": candidate_binding.get(
                "cache_has_shared_states"
            ),
            "profile_nonce": profile_nonce,
            "nvtx_range_names_by_rank": nvtx_range_names,
            "transition_nvtx_range_names_by_rank": transition_nvtx_range_names,
        }
        if (
            not isinstance(candidate_binding.get("plan_sha256"), str)
            or len(candidate_binding["plan_sha256"]) != 64
            or not isinstance(
                candidate_binding.get("row_permutation_sha256"), str
            )
            or len(candidate_binding["row_permutation_sha256"]) != 64
            or not isinstance(
                candidate_binding.get("cache_prefix_fingerprint"), str
            )
            or len(candidate_binding["cache_prefix_fingerprint"]) != 64
            or candidate_tracking.get("config") != expected_candidate_tracking_config
            or candidate_tracking.get("launch_token_sha256")
            != hashlib.sha256(profile_nonce.encode("ascii")).hexdigest()
            or candidate_tracking.get("coordinated_world_size") != 8
            or candidate_tracking.get("resume_existing") is not True
        ):
            raise SystemExit(
                "candidate W&B config does not match the measured candidate"
            )
        cache_artifacts = row.get("cache_artifacts")
        expected_cache_artifacts = {
            "train-cache-manifest": (
                "train_cache_manifest_sha256",
                "train_cache_content_binding",
                train_rows,
                "distill_train",
            ),
            "validation-cache-manifest": (
                "validation_cache_manifest_sha256",
                "validation_cache_content_binding",
                validation_rows,
                "validation",
            ),
        }
        if not isinstance(cache_artifacts, list) or len(cache_artifacts) != 2:
            raise SystemExit("candidate cache manifest evidence is incomplete")
        seen_cache_kinds: set[str] = set()
        for cache_artifact in cache_artifacts:
            if not isinstance(cache_artifact, dict):
                raise SystemExit("candidate cache artifact must be an object")
            kind = str(cache_artifact.get("kind", ""))
            if kind not in expected_cache_artifacts or kind in seen_cache_kinds:
                raise SystemExit("candidate cache artifact kind is invalid")
            raw_cache_path = cache_artifact.get("path")
            if not isinstance(raw_cache_path, str):
                raise SystemExit("candidate cache artifact path is missing")
            cache_path = Path(raw_cache_path).expanduser()
            if not cache_path.is_absolute():
                cache_path = (artifact.parent / cache_path).resolve()
            (
                digest_field,
                content_field,
                expected_rows,
                expected_split,
            ) = expected_cache_artifacts[kind]
            if (
                not cache_path.is_file()
                or sha256_file(cache_path) != cache_artifact.get("sha256")
                or cache_artifact.get("sha256") != candidate_binding.get(digest_field)
                or cache_content_binding(cache_path)
                != candidate_binding.get(content_field)
            ):
                raise SystemExit("candidate cache artifact SHA-256 mismatch")
            cache_manifest = load_strict_json(cache_path, label=f"candidate {kind}")
            cache_manifest_binding = (
                cache_manifest.get("binding")
                if isinstance(cache_manifest, dict)
                else None
            )
            if (
                not isinstance(cache_manifest, dict)
                or cache_manifest.get("row_count") != expected_rows
                or cache_manifest.get("layer_index")
                != candidate_binding.get("layer")
                or cache_manifest.get("split") != expected_split
                or cache_manifest.get("has_shared_states")
                is not candidate_binding.get("cache_has_shared_states")
                or not isinstance(cache_manifest_binding, dict)
                or cache_manifest_binding.get("source_checkpoint_sha256")
                != source_checkpoint_binding["sha256"]
                or cache_manifest_binding.get("zero_step_checkpoint_sha256")
                != checkpoint_binding["sha256"]
                or cache_manifest_binding.get("training_config_sha256")
                != binding.get("selection_plan_sha256")
                or cache_manifest_binding.get("dataset_manifest_sha256")
                != current_dataset_binding["files"]["manifest"]
                or cache_manifest_binding.get("prefix_fingerprint")
                != candidate_binding.get("cache_prefix_fingerprint")
                or cache_manifest_binding.get("split") != expected_split
            ):
                raise SystemExit(
                    "candidate measured rows do not equal the complete frozen cache"
                )
            seen_cache_kinds.add(kind)
        batches.add(batch)
        candidate_run_ids.add(candidate_run_id)
        matrix_key = (batch, str(case_id))
        if matrix_key in candidate_matrix:
            raise SystemExit("duplicate batch/profile-case candidate")
        candidate_matrix[matrix_key] = row
        cache_bindings_by_case[str(case_id)].add(
            (
                candidate_binding.get("train_cache_content_binding", {}).get(
                    "sha256"
                ),
                candidate_binding.get(
                    "validation_cache_content_binding", {}
                ).get("sha256"),
                candidate_binding.get("cache_prefix_fingerprint"),
            )
        )
        overlays_by_case[str(case_id)].add(
            candidate_binding.get("overlay_files_sha256")
        )
        source_signatures_by_case[str(case_id)].add(
            candidate_binding["source_parameter_signature_sha256"]
        )
        target_signatures_by_case[str(case_id)].add(
            candidate_binding["target_parameter_signature_sha256"]
        )
        validation_budgets.add(validation_rows)
        kinds: set[str] = set()
        nsys_sqlite: Path | None = None
        nsys_report: Path | None = None
        nsys_export_binding: Path | None = None
        for profiler_artifact in profiler_artifacts:
            if not isinstance(profiler_artifact, dict):
                raise SystemExit("profiler artifact binding must be an object")
            kind = profiler_artifact.get("kind")
            raw_path = profiler_artifact.get("path")
            raw_sha = profiler_artifact.get("sha256")
            if (
                kind not in {"nsys-rep", "nsys-sqlite", "nsys-export-binding"}
                or not isinstance(raw_path, str)
                or not isinstance(raw_sha, str)
            ):
                raise SystemExit("profiler artifact binding is incomplete")
            raw_artifact = Path(raw_path).expanduser()
            if not raw_artifact.is_absolute():
                raw_artifact = (artifact.parent / raw_artifact).resolve()
            if not raw_artifact.is_file() or sha256_file(raw_artifact) != raw_sha:
                raise SystemExit("raw Nsight profiler artifact SHA-256 mismatch")
            kinds.add(str(kind))
            if kind == "nsys-sqlite":
                nsys_sqlite = raw_artifact
            elif kind == "nsys-rep":
                nsys_report = raw_artifact
            elif kind == "nsys-export-binding":
                nsys_export_binding = raw_artifact
        if kinds != {"nsys-rep", "nsys-sqlite", "nsys-export-binding"}:
            raise SystemExit(
                "each candidate requires nsys-rep, nsys-sqlite, and export binding"
            )
        assert nsys_sqlite is not None
        assert nsys_report is not None
        assert nsys_export_binding is not None
        if nsys_export_binding != Path(
            expected_candidate_artifacts["nsys_export_binding"]["path"]
        ):
            raise SystemExit(
                "candidate export binding differs from selection artifact binding"
            )
        nsys_metrics = derive_nsys_profile_metrics(nsys_sqlite, nvtx_range_names)
        transition_metrics = (
            derive_nsys_profile_metrics(nsys_sqlite, transition_nvtx_range_names)
            if transition_required
            else None
        )
        export_binding = load_strict_json(
            nsys_export_binding, label="candidate Nsight export binding"
        )
        exported_candidate = (
            export_binding.get("candidate")
            if isinstance(export_binding, dict)
            else None
        )
        exported_report = (
            export_binding.get("nsys_report")
            if isinstance(export_binding, dict)
            else None
        )
        exported_sqlite = (
            export_binding.get("nsys_sqlite")
            if isinstance(export_binding, dict)
            else None
        )
        export_command = (
            export_binding.get("export_command")
            if isinstance(export_binding, dict)
            else None
        )
        profile_command = (
            export_binding.get("profile_command")
            if isinstance(export_binding, dict)
            else None
        )
        if (
            not isinstance(export_binding, dict)
            or not isinstance(exported_candidate, dict)
            or not isinstance(exported_report, dict)
            or not isinstance(exported_sqlite, dict)
            or not isinstance(profile_command, list)
            or not profile_command
            or Path(str(profile_command[0])).name != "nsys"
            or "profile" not in profile_command
            or str(source_candidate_path) not in profile_command
            or not isinstance(export_command, list)
            or not export_command
            or Path(str(export_command[0])).name != "nsys"
            or "export" not in export_command
            or str(nsys_report) not in export_command
            or f"--output={nsys_sqlite}" not in export_command
            or not isinstance(export_binding.get("nsys_version"), str)
            or not export_binding["nsys_version"]
            or export_binding.get("schema_version") != 1
            or export_binding.get("run_id") != candidate_run_id
            or export_binding.get("profile_nonce") != profile_nonce
            or export_binding.get("launch_token_sha256")
            != hashlib.sha256(profile_nonce.encode("ascii")).hexdigest()
            or exported_candidate.get("path") != str(source_candidate_path)
            or not source_candidate_path.is_file()
            or exported_candidate.get("sha256") != sha256_file(source_candidate_path)
            or exported_report.get("sha256") != sha256_file(nsys_report)
            or exported_report.get("path") != str(nsys_report)
            or exported_sqlite.get("sha256") != sha256_file(nsys_sqlite)
            or exported_sqlite.get("path") != str(nsys_sqlite)
            or Path(str(exported_sqlite.get("export_input_file", ""))).name
            != nsys_report.name
            or export_binding.get("frozen_input_binding")
            != {
                key: candidate_binding.get(key)
                for key in PROFILE_INPUT_BINDING_FIELDS
            }
            or export_binding.get("derived_metrics")
            != {"epoch": nsys_metrics, "cache_transition": transition_metrics}
        ):
            raise SystemExit(
                "candidate Nsight report and SQLite are not bound to this run"
            )
        derived_kernel_coverage = nsys_metrics[
            "kernel_covered_wall_fraction_by_rank"
        ]
        derived_sm_active = nsys_metrics["sm_active_fraction_by_gpu"]
        derived_walls = nsys_metrics["window_seconds_by_rank"]
        if any(
            not math.isclose(float(reported), derived, rel_tol=0, abs_tol=1e-9)
            for reported, derived in zip(
                kernel_coverage, derived_kernel_coverage, strict=True
            )
        ):
            raise SystemExit(
                "reported CUDA kernel coverage differs from the bound Nsight SQLite"
            )
        if any(
            not math.isclose(float(reported), derived, rel_tol=0, abs_tol=1e-9)
            for reported, derived in zip(sm_active, derived_sm_active, strict=True)
        ):
            raise SystemExit(
                "reported SMs Active differs from the bound Nsight SQLite"
            )
        for field, derived in (
            ("profiler_window_seconds_by_rank", derived_walls),
            ("nsys_process_ids_by_rank", nsys_metrics["process_ids_by_rank"]),
            ("nsys_device_ids_by_rank", nsys_metrics["device_ids_by_rank"]),
            (
                "nsys_kernel_intersection_count_by_rank",
                nsys_metrics["kernel_intersection_count_by_rank"],
            ),
            (
                "sm_active_sample_count_by_gpu",
                nsys_metrics["sm_active_sample_count_by_gpu"],
            ),
            (
                "sm_active_metric_source_by_gpu",
                nsys_metrics["sm_active_metric_source_by_gpu"],
            ),
            (
                "gpu_metric_common_window_seconds",
                nsys_metrics["gpu_metric_common_window_seconds"],
            ),
        ):
            if row.get(field) != derived:
                raise SystemExit(
                    f"reported {field} differs from the bound Nsight SQLite"
                )
        if any(
            abs(float(measured) - float(profiled)) / float(profiled) > 0.05
            for measured, profiled in zip(wall, derived_walls, strict=True)
        ):
            raise SystemExit(
                "Python epoch wall differs by more than 5% from bound NVTX window"
            )
        expected_throughput = float(row["loss_token_count"]) / max(
            map(float, derived_walls)
        )
        if not math.isclose(
            float(row["loss_tokens_per_second"]),
            expected_throughput,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise SystemExit("candidate throughput is not based on the NVTX window")
        checkpoint_walls = row.get("checkpoint_wall_seconds_by_rank")
        if (
            not isinstance(checkpoint_walls, list)
            or len(checkpoint_walls) != 8
            or not all(_finite_number(value) for value in checkpoint_walls)
        ):
            raise SystemExit("candidate checkpoint rank walls are malformed")
        expected_checkpoint_fraction = max(map(float, checkpoint_walls)) / max(
            map(float, derived_walls)
        )
        if not math.isclose(
            float(checkpoint_fraction),
            expected_checkpoint_fraction,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise SystemExit("candidate checkpoint fraction is not reproducible")
        epoch_passes = performance_candidate_passes_gates(
            kernel_coverage=derived_kernel_coverage,
            sm_active=derived_sm_active,
            reserved=reserved,
            wall_seconds=derived_walls,
            checkpoint_fraction=checkpoint_fraction,
        )
        transition_passes = True
        if transition_metrics is not None:
            transition_field_map = {
                "cache_transition_kernel_covered_wall_fraction_by_rank": (
                    "kernel_covered_wall_fraction_by_rank"
                ),
                "cache_transition_sm_active_fraction_by_gpu": (
                    "sm_active_fraction_by_gpu"
                ),
                "cache_transition_sm_active_sample_count_by_gpu": (
                    "sm_active_sample_count_by_gpu"
                ),
                "cache_transition_sm_active_metric_source_by_gpu": (
                    "sm_active_metric_source_by_gpu"
                ),
                "cache_transition_gpu_metric_common_window_seconds": (
                    "gpu_metric_common_window_seconds"
                ),
                "cache_transition_profiler_window_seconds_by_rank": (
                    "window_seconds_by_rank"
                ),
                "cache_transition_nsys_process_ids_by_rank": (
                    "process_ids_by_rank"
                ),
                "cache_transition_nsys_device_ids_by_rank": "device_ids_by_rank",
                "cache_transition_nsys_kernel_intersection_count_by_rank": (
                    "kernel_intersection_count_by_rank"
                ),
            }
            for reported_field, derived_field in transition_field_map.items():
                if row.get(reported_field) != transition_metrics[derived_field]:
                    raise SystemExit(
                        f"reported {reported_field} differs from the transition NVTX window"
                    )
            transition_python_walls = row.get(
                "cache_transition_wall_seconds_by_rank"
            )
            transition_reserved = row.get(
                "cache_transition_peak_reserved_memory_fraction_by_rank"
            )
            transition_walls = transition_metrics["window_seconds_by_rank"]
            if not all(
                isinstance(values, list)
                and len(values) == 8
                and all(_finite_number(value) for value in values)
                for values in (transition_python_walls, transition_reserved)
            ):
                raise SystemExit("cache-transition rank measurements are malformed")
            if any(
                abs(float(measured) - float(profiled)) / float(profiled) > 0.05
                for measured, profiled in zip(
                    transition_python_walls, transition_walls, strict=True
                )
            ):
                raise SystemExit(
                    "cache-transition Python wall differs from its bound NVTX window"
                )
            transition_passes = performance_candidate_passes_gates(
                kernel_coverage=transition_metrics[
                    "kernel_covered_wall_fraction_by_rank"
                ],
                sm_active=transition_metrics["sm_active_fraction_by_gpu"],
                reserved=transition_reserved,
                wall_seconds=transition_walls,
                checkpoint_fraction=0.0,
            )
        if (
            row.get("epoch_eligible") is not epoch_passes
            or row.get("cache_transition_eligible") is not transition_passes
            or row.get("eligible") is not (epoch_passes and transition_passes)
        ):
            raise SystemExit("performance candidate eligibility is not reproducible")
    expected_matrix = {
        (batch, case_id)
        for batch in batches
        for case_id in required_profile_cases
    }
    if (
        len(batches) < 3
        or set(candidate_matrix) != expected_matrix
        or len(validation_budgets) != 1
        or any(len(values) != 1 for values in cache_bindings_by_case.values())
        or any(len(values) != 1 for values in source_signatures_by_case.values())
        or any(len(values) != 1 for values in target_signatures_by_case.values())
        or any(len(values) != 1 for values in overlays_by_case.values())
        or binding.get("overlay_files_sha256_by_case")
        != {
            case_id: next(iter(values))
            for case_id, values in overlays_by_case.items()
        }
        or any(
            not isinstance(value, str) or len(value) != 64
            for bindings in cache_bindings_by_case.values()
            for value in next(iter(bindings))
        )
    ):
        raise SystemExit(
            "performance candidates do not cover one frozen workload for every layer case"
        )
    layer_count = sum(
        int(case["layer_count"]) for case in required_profile_cases.values()
    )
    equal_loss_token_budget = float(profile["equal_loss_token_budget"])
    batch_summaries: list[dict[str, Any]] = []
    for batch in sorted(batches):
        rows = [
            candidate_matrix[(batch, case_id)]
            for case_id in required_profile_cases
        ]
        weighted_epoch_wall = sum(
            int(required_profile_cases[case_id]["layer_count"])
            * max(
                map(
                    float,
                    candidate_matrix[(batch, case_id)][
                        "profiler_window_seconds_by_rank"
                    ],
                )
            )
            for case_id in required_profile_cases
        )
        weighted_transition_wall = sum(
            int(required_profile_cases[case_id]["transition_count"])
            * (
                max(
                    map(
                        float,
                        candidate_matrix[(batch, case_id)][
                            "cache_transition_profiler_window_seconds_by_rank"
                        ],
                    )
                )
                if int(required_profile_cases[case_id]["transition_count"]) > 0
                else 0.0
            )
            for case_id in required_profile_cases
        )
        total_wall = weighted_epoch_wall + weighted_transition_wall
        batch_summaries.append(
            {
                "per_rank_micro_batch_size": batch,
                "eligible": all(bool(row["eligible"]) for row in rows),
                "weighted_epoch_wall_seconds": weighted_epoch_wall,
                "weighted_cache_transition_wall_seconds": weighted_transition_wall,
                "weighted_end_to_end_wall_seconds": total_wall,
                "weighted_end_to_end_loss_tokens_per_second": (
                    equal_loss_token_budget * layer_count / total_wall
                ),
            }
        )
    if profile.get("batch_summaries") != batch_summaries:
        raise SystemExit("weighted all-layer batch summaries are not reproducible")
    eligible_batches = [row for row in batch_summaries if row["eligible"]]
    if not eligible_batches:
        raise SystemExit("no batch passes every layer-case utilization gate")
    winner = max(
        eligible_batches,
        key=lambda row: float(row["weighted_end_to_end_loss_tokens_per_second"]),
    )
    if winner["per_rank_micro_batch_size"] != selected:
        raise SystemExit("performance evidence selected batch is not reproducible")
    winning_rows = [
        candidate_matrix[(selected, case_id)] for case_id in required_profile_cases
    ]
    return {
        "path": str(artifact),
        "sha256": expected_sha,
        "selected_per_rank_micro_batch_size": selected,
        "loss_tokens_per_second": float(
            winner["weighted_end_to_end_loss_tokens_per_second"]
        ),
        "minimum_kernel_covered_wall_fraction": min(
            float(value)
            for row in winning_rows
            for value in row["kernel_covered_wall_fraction_by_rank"]
        ),
        "minimum_sm_active_fraction": min(
            float(value)
            for row in winning_rows
            for value in row["sm_active_fraction_by_gpu"]
        ),
        "maximum_checkpoint_wall_fraction": max(
            float(row["checkpoint_wall_fraction"]) for row in winning_rows
        ),
    }


def gpu_inventory() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            inventory.append(
                {
                    "index": fields[0],
                    "name": fields[1],
                    "compute_capability": fields[2],
                    "memory_mib": fields[3],
                }
            )
    return inventory


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str]) -> int:
    from any2rwkv.preflight import require_rwkv7_runtime, runtime_binding

    runtime = runtime_binding(require_rwkv7_runtime())
    command = parse_command(argv)
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing strict training settings: {', '.join(missing)}")
    if os.environ["HELICOPTER_RUN_PHASE"] != "any2rwkv-layerwise":
        raise SystemExit(
            "strict Any2RWKV wrapper requires HELICOPTER_RUN_PHASE=any2rwkv-layerwise"
        )
    if os.environ["HELICOPTER_PRECISION"] != "bf16":
        raise SystemExit("strict Any2RWKV training requires HELICOPTER_PRECISION=bf16")
    if os.environ["HELICOPTER_WKV_MODE"] != "fp32io16":
        raise SystemExit(
            "strict Any2RWKV training requires HELICOPTER_WKV_MODE=fp32io16"
        )
    if os.environ["CUDA_VISIBLE_DEVICES"] != "0,1,2,3,4,5,6,7":
        raise SystemExit("strict Any2RWKV training requires all eight GPUs")
    checkpoint, checkpoint_sha = require_file(
        "HELICOPTER_CHECKPOINT_PATH", os.environ["HELICOPTER_CHECKPOINT_PATH"]
    )
    head_size = source_compatible_rwkv_head_size(checkpoint)
    configured_head_size = os.environ.get("RWKV_HEAD_SIZE")
    if configured_head_size not in {None, str(head_size)}:
        raise SystemExit(
            "RWKV_HEAD_SIZE conflicts with source-compatible geometry: "
            f"configured={configured_head_size!r} expected={head_size!r}"
        )
    os.environ["RWKV_HEAD_SIZE"] = str(head_size)
    native_training_environment = {
        **NATIVE_TRAINING_ENV,
        "RWKV_HEAD_SIZE": str(head_size),
    }
    mismatches = {
        name: {"expected": expected, "actual": os.environ.get(name)}
        for name, expected in native_training_environment.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}=expected:{row['expected']!r}/actual:{row['actual']!r}"
            for name, row in sorted(mismatches.items())
        )
        raise SystemExit(f"native RWKV7 training environment mismatch: {details}")
    expected_checkpoint_sha = os.environ["HELICOPTER_CHECKPOINT_SHA256"].lower()
    if checkpoint_sha != expected_checkpoint_sha:
        raise SystemExit(
            "checkpoint SHA-256 mismatch: "
            f"expected={expected_checkpoint_sha} actual={checkpoint_sha}"
        )
    dataset_manifest, dataset_sha = require_file(
        "HELICOPTER_DATASET_MANIFEST", os.environ["HELICOPTER_DATASET_MANIFEST"]
    )
    config, config_sha = require_file(
        "HELICOPTER_CONFIG_PATH", os.environ["HELICOPTER_CONFIG_PATH"]
    )
    plan = load_strict_json(config, label="strict layer-major training plan")
    if not isinstance(plan, dict):
        raise SystemExit("strict layer-major training plan must be a JSON object")
    if plan.get("schema_version") != 3:
        raise SystemExit("strict layer-major training plan must use schema_version=3")
    if plan.get("execution_mode") != "streamed_layer_store":
        raise SystemExit(
            "strict layer-major training plan requires execution_mode=streamed_layer_store"
        )
    for field in (
        "cache_shard_rows",
        "max_layer_input_cache_bytes",
        "max_cached_layer_input_bytes_per_rank",
    ):
        if not isinstance(plan.get(field), int) or plan[field] <= 0:
            raise SystemExit(
                f"strict layer-major training plan requires positive {field}"
            )
    comparison = plan.get("experiment_comparison")
    if (
        not isinstance(comparison, dict)
        or set(comparison) != {"baseline_run_id", "only_changes"}
        or not isinstance(comparison.get("baseline_run_id"), str)
        or not comparison["baseline_run_id"]
        or not isinstance(comparison.get("only_changes"), list)
        or not comparison["only_changes"]
        or any(
            not isinstance(value, str) or not value
            for value in comparison["only_changes"]
        )
    ):
        raise SystemExit("strict training requires an explicit experiment_comparison")
    optimizer_contract = plan.get("optimizer")
    optimizer_betas = (
        optimizer_contract.get("betas")
        if isinstance(optimizer_contract, dict)
        else None
    )
    if (
        not isinstance(optimizer_contract, dict)
        or set(optimizer_contract)
        != {
            "name",
            "learning_rate",
            "final_learning_rate",
            "warmup_steps",
            "betas",
            "epsilon",
            "weight_decay",
        }
        or optimizer_contract.get("name") != "adamw"
        or not _finite_number(optimizer_contract.get("learning_rate"))
        or float(optimizer_contract["learning_rate"]) <= 0
        or not _finite_number(optimizer_contract.get("final_learning_rate"))
        or float(optimizer_contract["final_learning_rate"])
        != float(optimizer_contract["learning_rate"])
        or type(optimizer_contract.get("warmup_steps")) is not int
        or optimizer_contract["warmup_steps"] < 0
        or not isinstance(optimizer_betas, list)
        or len(optimizer_betas) != 2
        or not all(
            _finite_number(value) and 0 <= float(value) < 1 for value in optimizer_betas
        )
        or not _finite_number(optimizer_contract.get("epsilon"))
        or float(optimizer_contract["epsilon"]) <= 0
        or not _finite_number(optimizer_contract.get("weight_decay"))
        or float(optimizer_contract["weight_decay"]) < 0
        or plan.get("learning_rate_schedule") != "warmup-constant"
        or not _finite_number(plan.get("learning_rate"))
        or float(plan["learning_rate"]) != float(optimizer_contract["learning_rate"])
        or type(plan.get("burn_in_tokens")) is not int
        or plan["burn_in_tokens"] < 0
        or type(plan.get("supervised_tokens")) is not int
        or plan["supervised_tokens"] < 2
        or not _finite_number(plan.get("gradient_clip_norm"))
        or float(plan["gradient_clip_norm"]) <= 0
    ):
        raise SystemExit(
            "strict layer-major training requires a complete warmup-constant AdamW contract"
        )
    if plan.get("accumulation_steps") != 1:
        raise SystemExit(
            "strict layer-major performance-selected training requires accumulation_steps=1"
        )
    if plan.get("gradient_checkpointing") is not False:
        raise SystemExit(
            "strict layer-major training requires gradient_checkpointing=false"
        )
    if plan.get("checkpoint_interval_micro_batches") != 0:
        raise SystemExit(
            "performance-gated layer-major training checkpoints only at epoch boundaries"
        )
    if plan.get("distributed_world_size") != 8:
        raise SystemExit(
            "strict layer-major training plan requires distributed_world_size=8"
        )
    if plan.get("evidence_tier") not in {"fixture", "exploratory", "p1", "scale"}:
        raise SystemExit("strict layer-major training plan requires an evidence_tier")
    tracking = plan.get("tracking")
    tracking_job_types = (
        tracking.get("job_types") if isinstance(tracking, dict) else None
    )
    if plan.get("evidence_tier") != "fixture" and (
        not isinstance(tracking, dict)
        or tracking.get("mode") != "online"
        or not tracking.get("project")
        or not isinstance(tracking_job_types, dict)
        or set(tracking_job_types) != {"profile", "distill", "corrective"}
        or any(
            not isinstance(value, str) or not value
            for value in tracking_job_types.values()
        )
        or any(
            "api" in str(key).lower() and "key" in str(key).lower() for key in tracking
        )
    ):
        raise SystemExit(
            "real Any2RWKV training requires online W&B tracking without credentials in the plan"
        )
    has_exploratory_layer_limit = "exploratory_layer_limit" in plan
    exploratory_layer_limit = plan.get("exploratory_layer_limit")
    if has_exploratory_layer_limit and "corrective" in command:
        raise SystemExit("corrective command forbids exploratory_layer_limit")
    if has_exploratory_layer_limit and (
        type(exploratory_layer_limit) is not int
        or plan.get("evidence_tier") != "exploratory"
        or not 0 < exploratory_layer_limit < source_layer_count(checkpoint)
    ):
        raise SystemExit(
            "exploratory_layer_limit must be a positive prefix shorter than the "
            "source and is only allowed for exploratory evidence"
        )
    if plan.get("evidence_tier") == "fixture":
        raise SystemExit(
            "strict eight-GPU training refuses fixture evidence; fixtures are CPU/unit only"
        )
    parsed_command = parse_training_command(command)
    try:
        seed = int(os.environ["HELICOPTER_SEED"])
        batch = json.loads(os.environ["HELICOPTER_BATCH_JSON"])
    except (ValueError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"invalid strict training seed/batch metadata: {error}"
        ) from error
    if not isinstance(batch, dict) or not batch:
        raise SystemExit("HELICOPTER_BATCH_JSON must be a non-empty JSON object")

    run_id = os.environ["HELICOPTER_RUN_ID"]
    if seed != plan.get("seed"):
        raise SystemExit("HELICOPTER_SEED differs from the distillation plan")
    for field in ("micro_batch_size", "accumulation_steps"):
        if batch.get(field) != plan.get(field):
            raise SystemExit(
                f"HELICOPTER_BATCH_JSON {field} differs from the distillation plan"
            )
    corrective_parent = validate_distill_command(
        parsed_command,
        checkpoint=checkpoint,
        dataset_manifest=dataset_manifest,
        config=config,
        expected_run_id=run_id,
    )
    target_config = (
        Path(str(corrective_parent["path"])) / "config.json"
        if corrective_parent is not None
        else parsed_command.output / "checkpoint-zero-step/config.json"
    )
    throughput_evidence = validate_throughput_evidence(
        plan,
        source_config=checkpoint,
        target_config=target_config,
        head_size=head_size,
    )
    performance_evidence = validate_performance_evidence(
        plan,
        action=parsed_command.action,
        source_config=checkpoint,
        target_config=target_config,
        head_size=head_size,
        dataset_manifest=dataset_manifest,
    )
    log_root = Path(os.environ.get("REMOTE_RUN_LOG_DIR", ".helicopter-dev/runs"))
    metadata_path = log_root / "metadata.json"
    revisions_path = Path(".helicopter-dev/source-revisions.json")
    revisions = (
        json.loads(revisions_path.read_text(encoding="utf-8"))
        if revisions_path.is_file()
        else None
    )
    inventory = gpu_inventory()
    if len(inventory) != 8 or [row["index"] for row in inventory] != [
        str(index) for index in range(8)
    ]:
        raise SystemExit(
            "strict Any2RWKV training requires the visible GPU inventory 0..7"
        )
    started = time.time()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "run_phase": os.environ["HELICOPTER_RUN_PHASE"],
        "command": command,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "dataset_manifest": {"path": str(dataset_manifest), "sha256": dataset_sha},
        "training_config": {"path": str(config), "sha256": config_sha},
        "layer_major": {
            "execution_mode": plan["execution_mode"],
            "cache_shard_rows": plan["cache_shard_rows"],
            "max_layer_input_cache_bytes": plan["max_layer_input_cache_bytes"],
            "distributed_world_size": plan["distributed_world_size"],
            "exploratory_layer_limit": exploratory_layer_limit,
        },
        "seed": seed,
        "batch": batch,
        "throughput_evidence": throughput_evidence,
        "performance_evidence": performance_evidence,
        "corrective_parent": corrective_parent,
        "precision": os.environ["HELICOPTER_PRECISION"],
        "wkv_mode": os.environ["HELICOPTER_WKV_MODE"],
        "native_training_environment": native_training_environment,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "distributed": {"backend": "nccl", "world_size": 8, "data_parallel": True},
        "gpu_inventory": inventory,
        "source_revisions": revisions,
        "rwkv7_runtime": runtime,
        "training_source_files": training_source_files(),
        "started_unix": started,
    }
    launch_token = secrets.token_hex(16)
    child_environment = os.environ.copy()
    child_environment["ANY2RWKV_ATTEMPT_TOKEN"] = launch_token
    payload["launch_token_sha256"] = hashlib.sha256(
        launch_token.encode("ascii")
    ).hexdigest()
    atomic_write_json(metadata_path, payload)
    result = subprocess.run(command, check=False, env=child_environment)
    effective_returncode = result.returncode
    report_path = parsed_command.output / "experiment-report.md"
    tracking_path = parsed_command.output / "experiment-tracking.json"
    postcondition_error: str | None = None
    tracking_result = None
    if result.returncode == 0 and tracking_path.is_file():
        try:
            tracking_result = load_strict_json(
                tracking_path, label="experiment tracking result"
            )
        except SystemExit as error:
            postcondition_error = str(error)
    report_identity = None
    if result.returncode == 0 and report_path.is_file():
        try:
            report_identity = load_experiment_report_identity(report_path)
        except SystemExit as error:
            postcondition_error = str(error)
    expected_job_type = tracking_job_types[parsed_command.action]
    tracking_config = (
        tracking_result.get("config") if isinstance(tracking_result, dict) else None
    )
    tracking_attempt = (
        tracking_result.get("attempt_id") if isinstance(tracking_result, dict) else None
    )
    tracking_config_sha = (
        tracking_result.get("config_sha256")
        if isinstance(tracking_result, dict)
        else None
    )
    optimizer_tracking_config = {
        **optimizer_contract,
        "gradient_clip_norm": plan.get("gradient_clip_norm"),
    }
    if parsed_command.action == "distill":
        expected_training_tracking_config = {
            "schema_version": 1,
            "classification": plan.get("classification"),
            "evidence_tier": plan.get("evidence_tier"),
            "seed": plan.get("seed"),
            "optimizer": optimizer_tracking_config,
            "batch": {
                "per_rank": plan.get("micro_batch_size"),
                "accumulation_steps": plan.get("accumulation_steps"),
                "world_size": plan.get("distributed_world_size"),
                "gradient_checkpointing": plan.get("gradient_checkpointing"),
            },
            "comparison": plan.get("experiment_comparison"),
            "training_config": str(config.resolve()),
            "dataset_manifest": str(dataset_manifest.resolve()),
        }
    else:
        expected_training_tracking_config = {
            "schema_version": 1,
            "action": "corrective",
            "evidence_tier": plan.get("evidence_tier"),
            "seed": plan.get("seed"),
            "optimizer": optimizer_tracking_config,
            "parent_checkpoint": str(corrective_parent["path"]),
            "gradient_checkpointing": plan.get("gradient_checkpointing"),
            "comparison": plan.get("experiment_comparison"),
        }
    if result.returncode == 0 and (
        not report_path.is_file()
        or not isinstance(tracking_result, dict)
        or tracking_result.get("schema_version") != 2
        or tracking_result.get("backend") != "wandb"
        or tracking_result.get("mode") != "online"
        or tracking_result.get("status") != "completed"
        or tracking_result.get("run_id") != run_id
        or tracking_result.get("project") != tracking.get("project")
        or tracking_result.get("entity") != tracking.get("entity")
        or tracking_result.get("group") != tracking.get("group")
        or tracking_result.get("tags") != list(tracking.get("tags", []))
        or tracking_result.get("job_type") != expected_job_type
        or not tracking_result.get("url")
        or tracking_result.get("launch_token_sha256")
        != payload.get("launch_token_sha256")
        or tracking_result.get("coordinated_world_size") != 8
        or not isinstance(tracking_attempt, str)
        or re.fullmatch(r"[0-9a-f]{32}", tracking_attempt) is None
        or not isinstance(tracking_config, dict)
        or tracking_config != expected_training_tracking_config
        or tracking_config_sha != sha256_json(tracking_config)
        or not isinstance(report_identity, dict)
        or report_identity.get("run_id") != run_id
        or report_identity.get("attempt_id") != tracking_attempt
        or report_identity.get("tracking_status") != "completed"
        or report_identity.get("config_sha256") != tracking_config_sha
    ):
        identity_error = "successful training did not publish a report and verified online W&B identity"
        postcondition_error = (
            identity_error
            if postcondition_error is None
            else f"{postcondition_error}; {identity_error}"
        )
        effective_returncode = 1
    post_run_sources = training_source_files()
    if (
        sha256_file(checkpoint) != checkpoint_sha
        or sha256_file(dataset_manifest) != dataset_sha
        or sha256_file(config) != config_sha
        or post_run_sources != payload["training_source_files"]
    ):
        input_error = (
            "training input or executable source changed while the subprocess ran"
        )
        postcondition_error = (
            input_error
            if postcondition_error is None
            else f"{postcondition_error}; {input_error}"
        )
        effective_returncode = 1
    if not report_path.is_file():
        try:
            from any2rwkv.core import write_experiment_report

            write_experiment_report(
                parsed_command.output,
                status="failed" if effective_returncode else "complete",
                reason=postcondition_error
                or (
                    f"training subprocess exited with code {result.returncode}"
                    if result.returncode
                    else None
                ),
            )
        except BaseException as error:
            fallback_error = f"cannot write fallback experiment report: {error}"
            postcondition_error = (
                fallback_error
                if postcondition_error is None
                else f"{postcondition_error}; {fallback_error}"
            )
            effective_returncode = 1
    finished = time.time()
    payload.update(
        {
            "status": "completed" if effective_returncode == 0 else "failed",
            "returncode": effective_returncode,
            "subprocess_returncode": result.returncode,
            "postcondition_error": postcondition_error,
            "finished_unix": finished,
            "duration_seconds": finished - started,
        }
    )
    atomic_write_json(metadata_path, payload)
    return effective_returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
