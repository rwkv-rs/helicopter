from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Iterator, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import write_json
from .checkpoint import CheckpointManifest, sha256_file
from .errors import ContractError
from .target import TensorSpec, canonical_text_name, is_sequence_mixer, is_vision_tensor


HF_RUNTIME_MODULES = (
    "configuration_any2rwkv.py",
    "modeling_any2rwkv.py",
    "mixer.py",
    "kernel.py",
    "errors.py",
)


def refresh_hf_runtime_files(output: Path) -> dict[str, str]:
    """Refresh checkpoint-local HF code without rewriting model weights."""
    if not (output / "config.json").is_file() or not (
        output / "model.safetensors.index.json"
    ).is_file():
        raise ContractError(f"cannot refresh incomplete HF checkpoint: {output}")
    package_root = Path(__file__).resolve().parent
    hashes: dict[str, str] = {}
    for module_name in HF_RUNTIME_MODULES:
        shutil.copy2(package_root / module_name, output / module_name)
        hashes[module_name] = sha256_file(output / module_name)
    manifest_path = output / "roundtrip-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files")
        if isinstance(files, dict):
            files.update(hashes)
            write_json(manifest_path, manifest)
    return hashes


def _dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as error:
        raise ContractError(f"unsupported target tensor dtype: {name}") from error


def _seed(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def initialize_tensor(spec: TensorSpec, *, base_seed: int = 20260714) -> torch.Tensor:
    dtype = _dtype(spec.dtype)
    if spec.initialization.startswith("zero") or spec.name.endswith("g_norm.bias"):
        return torch.zeros(spec.shape, dtype=dtype)
    if spec.name.endswith("g_norm.weight") or spec.name.endswith("k_k") or spec.name.endswith("k_a"):
        return torch.ones(spec.shape, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(_seed(spec.name, base_seed))
    scale = 0.02 if len(spec.shape) > 1 else 0.001
    return (torch.randn(spec.shape, generator=generator, dtype=torch.float32) * scale).to(dtype)


def _source_tensors(source: CheckpointManifest) -> Iterator[tuple[str, torch.Tensor]]:
    seen: set[str] = set()
    for shard in source.shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise ContractError(f"duplicate tensor across source shards: {name}")
                seen.add(name)
                if not is_sequence_mixer(name) and not is_vision_tensor(name):
                    yield canonical_text_name(name), handle.get_tensor(name)


def _teacher_text_tensors(source: CheckpointManifest) -> Iterator[tuple[str, torch.Tensor]]:
    seen: set[str] = set()
    for shard in source.shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for source_name in handle.keys():
                if is_vision_tensor(source_name):
                    continue
                name = canonical_text_name(source_name)
                if name.startswith("mtp."):
                    continue
                if name in seen:
                    raise ContractError(f"duplicate text teacher tensor: {name}")
                seen.add(name)
                yield name, handle.get_tensor(source_name)


def _flush_shard(
    output: Path,
    shard_index: int,
    tensors: dict[str, torch.Tensor],
) -> tuple[str, dict[str, str]]:
    filename = f"model-{shard_index:05d}-of-PLACEHOLDER.safetensors"
    path = output / filename
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, temporary)
    temporary.replace(path)
    return filename, {name: filename for name in tensors}


def export_hf_checkpoint(
    source: CheckpointManifest,
    output: Path,
    *,
    target_config: dict[str, object],
    target_specs: tuple[TensorSpec, ...],
    max_shard_bytes: int = 2 * 1024**3,
    seed: int = 20260714,
    target_tensors: Mapping[str, torch.Tensor] | None = None,
    target_tensor_provider: Callable[[TensorSpec], torch.Tensor] | None = None,
    resume_partial: bool = False,
    external_resume_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Stream a deterministic structural target checkpoint in standard HF layout.

    This stage copies the Qwen text shell exactly and initializes only the new
    RWKV7 mixers. Fitted/algebraic values overwrite these tensors in subsequent
    migration stages; this exporter never silently claims trained quality.
    """
    progress_path = output / ".export-progress.json"
    if output.exists() and any(output.iterdir()) and not (
        resume_partial and progress_path.is_file()
    ):
        raise ContractError(f"HF export directory must be empty or resumable: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if max_shard_bytes <= 0:
        raise ContractError("max_shard_bytes must be positive")
    if target_tensors is not None and target_tensor_provider is not None:
        raise ContractError(
            "HF export accepts either target_tensors or target_tensor_provider, not both"
        )

    binding = hashlib.sha256(
        json.dumps(
            {
                "source_files": source.file_hashes,
                "target_config": target_config,
                "target_specs": [spec.__dict__ for spec in target_specs],
                "max_shard_bytes": max_shard_bytes,
                "seed": seed,
                "external_resume_binding": external_resume_binding,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else None
    )
    if progress is not None and progress.get("binding") != binding:
        raise ContractError("HF export resume binding differs from source/config/specs")
    weight_map: dict[str, str] = (
        {} if progress is None else dict(progress.get("weight_map", {}))
    )
    shard_files: list[str] = (
        [] if progress is None else list(progress.get("shard_files", []))
    )
    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes = 0
    total_weight_bytes = int(progress.get("total_weight_bytes", 0)) if progress else 0

    def record_progress() -> None:
        write_json(
            progress_path,
            {
                "schema_version": 1,
                "binding": binding,
                "weight_map": weight_map,
                "shard_files": shard_files,
                "total_weight_bytes": total_weight_bytes,
            },
        )

    def append(name: str, tensor: torch.Tensor) -> None:
        nonlocal buffer, buffer_bytes, total_weight_bytes
        if name in weight_map or name in buffer:
            if name in weight_map:
                return
            raise ContractError(f"duplicate target tensor: {name}")
        size = tensor.numel() * tensor.element_size()
        if buffer and buffer_bytes + size > max_shard_bytes:
            filename, entries = _flush_shard(output, len(shard_files) + 1, buffer)
            shard_files.append(filename)
            weight_map.update(entries)
            record_progress()
            buffer = {}
            buffer_bytes = 0
        buffer[name] = tensor.contiguous()
        buffer_bytes += size
        total_weight_bytes += size

    for name, tensor in _source_tensors(source):
        if name in weight_map:
            continue
        append(name, tensor)
    overrides = dict(target_tensors or {})
    declared = {spec.name for spec in target_specs}
    unknown = sorted(overrides.keys() - declared)
    if unknown:
        raise ContractError(
            f"target tensor overrides are not declared by target specs: {unknown}"
        )
    for spec in target_specs:
        if spec.name in weight_map:
            continue
        tensor = (
            target_tensor_provider(spec)
            if target_tensor_provider is not None
            else overrides.get(spec.name)
        )
        if tensor is None:
            tensor = initialize_tensor(spec, base_seed=seed)
        if tuple(tensor.shape) != spec.shape or tensor.dtype != _dtype(spec.dtype):
            raise ContractError(
                f"target tensor override violates spec: {spec.name} "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )
        append(spec.name, tensor)
    if buffer:
        filename, entries = _flush_shard(output, len(shard_files) + 1, buffer)
        shard_files.append(filename)
        weight_map.update(entries)
        record_progress()

    total = len(shard_files)
    renamed: dict[str, str] = {}
    for index, old in enumerate(shard_files, start=1):
        new = f"model-{index:05d}-of-{total:05d}.safetensors"
        old_path = output / old
        new_path = output / new
        if old_path.is_file() and not new_path.exists():
            old_path.rename(new_path)
        elif not old_path.exists() and new_path.is_file():
            pass
        else:
            raise ContractError(
                f"HF export shard finalization is ambiguous: old={old_path.exists()} new={new_path.exists()}"
            )
        renamed[old] = new
    weight_map = {name: renamed[filename] for name, filename in weight_map.items()}
    serialized_size = sum((output / filename).stat().st_size for filename in renamed.values())
    write_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total_weight_bytes}, "weight_map": dict(sorted(weight_map.items()))},
    )
    write_json(output / "config.json", target_config)
    # Transformers resolves every relative import of the dynamic modeling file
    # before importing the model class. Keep this as the complete local module
    # closure: strict-loading an export must not require an installed
    # ``any2rwkv`` package.
    refresh_hf_runtime_files(output)
    for source_file in source.tokenizer_files:
        shutil.copy2(source_file, output / source_file.name)
    if not (output / "generation_config.json").is_file():
        write_json(
            output / "generation_config.json",
            {
                "_from_model_config": True,
                "do_sample": False,
                "bos_token_id": target_config.get("bos_token_id"),
                "eos_token_id": target_config.get("eos_token_id"),
            },
        )
    files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "stage": (
            "mapped-zero-step"
            if target_tensors is not None or target_tensor_provider is not None
            else "structural-zero-step"
        ),
        "quality_accepted": False,
        "tensor_count": len(weight_map),
        "shard_count": total,
        "total_weight_bytes": total_weight_bytes,
        "serialized_size": serialized_size,
        "files": {path.name: sha256_file(path) for path in files},
    }
    write_json(output / "roundtrip-manifest.json", manifest)
    progress_path.unlink(missing_ok=True)
    return manifest


def export_text_teacher_checkpoint(
    source: CheckpointManifest,
    output: Path,
    *,
    max_shard_bytes: int = 2 * 1024**3,
) -> dict[str, object]:
    """Extract the immutable Qwen3.5 text teacher without vision or MTP."""
    if output.exists() and any(output.iterdir()):
        raise ContractError(f"text teacher directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    text_config = source.config.get("text_config", source.config)
    if not isinstance(text_config, dict):
        raise ContractError("source text_config must be an object")
    config = dict(text_config)
    config["architectures"] = [
        "Qwen3_5MoeForCausalLM" if source.contract.has_moe else "Qwen3_5ForCausalLM"
    ]

    weight_map: dict[str, str] = {}
    shard_files: list[str] = []
    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes = 0
    total_weight_bytes = 0

    def flush() -> None:
        nonlocal buffer, buffer_bytes
        if not buffer:
            return
        filename, entries = _flush_shard(output, len(shard_files) + 1, buffer)
        shard_files.append(filename)
        weight_map.update(entries)
        buffer = {}
        buffer_bytes = 0

    for name, tensor in _teacher_text_tensors(source):
        size = tensor.numel() * tensor.element_size()
        if buffer and buffer_bytes + size > max_shard_bytes:
            flush()
        buffer[name] = tensor.contiguous()
        buffer_bytes += size
        total_weight_bytes += size
    flush()
    total = len(shard_files)
    renamed: dict[str, str] = {}
    for index, old in enumerate(shard_files, start=1):
        new = f"model-{index:05d}-of-{total:05d}.safetensors"
        (output / old).rename(output / new)
        renamed[old] = new
    weight_map = {name: renamed[filename] for name, filename in weight_map.items()}
    write_json(
        output / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_weight_bytes},
            "weight_map": dict(sorted(weight_map.items())),
        },
    )
    write_json(output / "config.json", config)
    for source_file in source.tokenizer_files:
        shutil.copy2(source_file, output / source_file.name)
    manifest = {
        "schema_version": 1,
        "classification": "text-only-teacher-extraction",
        "source": str(source.path),
        "tensor_count": len(weight_map),
        "shard_count": total,
        "total_weight_bytes": total_weight_bytes,
        "files": {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    write_json(output / "teacher-manifest.json", manifest)
    return manifest
