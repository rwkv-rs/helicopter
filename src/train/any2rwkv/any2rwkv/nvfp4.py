from __future__ import annotations

import copy
import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .artifacts import write_json
from .calibration import CalibrationManifest, file_sha256
from .errors import ContractError
from .roundtrip import validate_sharded_checkpoint
from .quantize import nvfp4_policy


EXCLUDED_PATTERNS = (
    "*embed_tokens*",
    "*lm_head*",
    "*mtp*",
    "*norm*",
)


def _copy_hf_assets(source: Path, output: Path) -> None:
    names = (
        "configuration_any2rwkv.py",
        "modeling_any2rwkv.py",
        "mixer.py",
        "kernel.py",
        "errors.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    )
    for name in names:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)


def build_nvfp4_quant_config(mtq) -> dict[str, object]:
    quant_config = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    patterns = quant_config.get("quant_cfg")
    if not isinstance(patterns, dict):
        raise ContractError("ModelOpt NVFP4 quant_cfg must be a pattern mapping")
    for pattern in EXCLUDED_PATTERNS:
        patterns[pattern] = {"enable": False}
    return quant_config


def _input_device(model: Any) -> torch.device:
    """Return the embedding device selected by Accelerate's sharded device map."""
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.device.type == "meta":
        raise ContractError("NVFP4 model input embedding is not materialized on a runtime device")
    return weight.device


def export_nvfp4_checkpoint(
    source: Path,
    output: Path,
    manifest: CalibrationManifest,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ContractError(f"NVFP4 output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise ContractError("NVFP4 export requires CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major < 12:
        raise ContractError("NVFP4 delivery profile requires Blackwell compute capability")

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    model = AutoModelForCausalLM.from_pretrained(
        source,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # A 397B BF16 checkpoint cannot reside on one GPU. Accelerate dispatches
        # complete modules across all visible GPUs and CPU while ModelOpt keeps
        # quantization calibration attached to the actual module devices.
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    policy = nvfp4_policy(
        (name, str(parameter.dtype).removeprefix("torch."), parameter.ndim)
        for name, parameter in model.named_parameters()
    )
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = manifest.texts()
    input_device = _input_device(model)

    def forward_loop(active_model) -> None:
        with torch.inference_mode():
            for start in range(0, len(texts), manifest.batch_size):
                encoded = tokenizer(
                    list(texts[start : start + manifest.batch_size]),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=manifest.max_length,
                )
                active_model(
                    input_ids=encoded.input_ids.to(input_device),
                    attention_mask=encoded.attention_mask.to(input_device),
                    use_cache=False,
                )

    quant_config = build_nvfp4_quant_config(mtq)
    mtq.quantize(model, quant_config, forward_loop=forward_loop)
    export_hf_checkpoint(model, export_dir=output)
    tokenizer.save_pretrained(output)
    _copy_hf_assets(source, output)

    quant_config_path = output / "hf_quant_config.json"
    if not quant_config_path.is_file():
        raise ContractError("ModelOpt export did not produce hf_quant_config.json")
    checkpoint = validate_sharded_checkpoint(output)
    exported_quant_config = json.loads(quant_config_path.read_text(encoding="utf-8"))
    files = {
        path.name: file_sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    result = {
        "schema_version": 1,
        "backend": "nvidia-modelopt",
        "backend_version": importlib.metadata.version("nvidia-modelopt"),
        "format": "NVFP4",
        "load_strategy": "accelerate-device-map-auto-low-cpu-memory",
        "input_device": str(input_device),
        "source": str(source),
        "source_config_sha256": file_sha256(source / "config.json"),
        "calibration_manifest": str(manifest.path),
        "calibration_sha256": manifest.data_sha256,
        "calibration_rows": manifest.row_count,
        "calibration_max_length": manifest.max_length,
        "excluded_patterns": list(EXCLUDED_PATTERNS),
        "policy": [decision.__dict__ for decision in policy],
        "exported_quant_config": exported_quant_config,
        "checkpoint": checkpoint,
        "files": files,
        "quality_accepted": False,
    }
    write_json(output / "nvfp4-manifest.json", result)
    return result
