from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

from ..adapters.qwen35 import Qwen35SourceAdapter
from ..artifacts import file_sha256, write_json
from ..configuration_any2rwkv import AnyToRWKVConfig
from ..errors import ContractError
from ..mapping import SourceDisposition, TargetProvenance
from ..modeling_any2rwkv import AnyToRWKVForCausalLM
from .layer_major_contract import (
    LayerMajorResumeContract,
    activate_layer_major_training,
    canonical_digest,
)
from .mapping_contract import (
    CalibrationDevelopmentFinalSplit,
    CandidateSelection,
    MaterializedTarget,
    SourceConsumption,
    StrictMappingLedger,
)

_STAGES = ("inspect", "split", "train", "ledger", "export", "verify")


def _atomic_torch(path: Path, payload: object) -> str:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def _complete_stage(root: Path, stage: str, payload: dict[str, object]) -> None:
    write_json(root / stage / "artifact.json", payload)
    write_json(
        root / stage / "stage.json",
        {
            "schema_version": 1,
            "stage": stage,
            "status": "complete",
            "artifact_sha256": file_sha256(root / stage / "artifact.json"),
        },
    )


def _read_stage(root: Path, stage: str) -> dict[str, object] | None:
    directory = root / stage
    manifest = directory / "stage.json"
    artifact = directory / "artifact.json"
    if not directory.exists():
        return None
    if not manifest.is_file() or not artifact.is_file():
        raise ContractError(f"tiny pipeline stage is partial: {stage}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("stage") != stage
        or metadata.get("status") != "complete"
        or metadata.get("artifact_sha256") != file_sha256(artifact)
    ):
        raise ContractError(f"tiny pipeline stage manifest is invalid: {stage}")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"tiny pipeline stage artifact is invalid: {stage}")
    return payload


def _audit_stages(root: Path) -> None:
    if not root.exists():
        return
    unknown = sorted(path.name for path in root.iterdir() if path.name not in _STAGES)
    if unknown:
        raise ContractError(f"tiny pipeline has unknown stage artifacts: {unknown}")
    seen_gap = False
    for stage in _STAGES:
        exists = (root / stage).exists()
        if exists and seen_gap and stage != "train":
            raise ContractError(f"tiny pipeline stage exists after a gap: {stage}")
        if not exists:
            seen_gap = True
        elif stage != "train" or (root / stage / "stage.json").is_file():
            _read_stage(root, stage)


def run_tiny_pipeline(
    source: Path,
    output: Path,
    *,
    interrupt_after_optimizer_steps: int | None = None,
) -> dict[str, object]:
    """Run a two-layer contract smoke pipeline; never represents a large-model run."""

    source, output = source.resolve(), output.resolve()
    if (
        interrupt_after_optimizer_steps is not None
        and interrupt_after_optimizer_steps <= 0
    ):
        raise ContractError("tiny pipeline interruption step must be positive")
    output.mkdir(parents=True, exist_ok=True)
    _audit_stages(output)
    inspection = _read_stage(output, "inspect")
    if inspection is None:
        observed = Qwen35SourceAdapter().inspect_checkpoint(
            source, require_final_layout=False
        )
        inspection = {
            "adapter": observed.adapter_id,
            "layers": observed.num_layers,
            "metadata": observed.metadata,
            "config_sha256": file_sha256(source / "config.json"),
        }
        _complete_stage(output, "inspect", inspection)
    elif inspection.get("config_sha256") != file_sha256(source / "config.json"):
        raise ContractError("tiny pipeline source metadata changed after inspection")

    split_payload = _read_stage(output, "split")
    split = CalibrationDevelopmentFinalSplit(("c0", "c1"), ("d0", "d1"), ("f0", "f1"))
    if split_payload is None:
        selection = CandidateSelection(split)
        selected = selection.select(
            {
                "identity-v1": {"calibration": 1.0, "development": 1.0},
                "affine-v1": {"calibration": 0.0, "development": 0.0},
            }
        )
        selection.verify_final(candidate=selected, evidence={"observable_mse": 0.0})
        split_payload = {
            "split": {
                "calibration": split.calibration,
                "development": split.development,
                "final": split.final,
            },
            "selected": selected,
            "final_verification": selection.final_verification,
        }
        _complete_stage(output, "split", split_payload)

    train_payload = (
        None
        if (output / "train").is_dir()
        and not (output / "train" / "stage.json").is_file()
        else _read_stage(output, "train")
    )
    if train_payload is None:
        train_payload = _run_training(output / "train", interrupt_after_optimizer_steps)
        if train_payload.get("status") == "interrupted":
            return {"status": "interrupted", "stage": "train"}
        _complete_stage(output, "train", train_payload)

    ledger_payload = _read_stage(output, "ledger")
    if ledger_payload is None:
        digest = str(train_payload["source_sha256"])
        ledger = StrictMappingLedger()
        for index in range(2):
            target = f"rwkv.layers.{index}.observable_adapter"
            source_name = f"qwen.layers.{index}.observable_projection"
            materialized = str(train_payload["layer_sha256"][index])
            ledger.add_target(
                MaterializedTarget(
                    target,
                    TargetProvenance.FITTED,
                    (source_name,),
                    (),
                    (2, 2),
                    "float32",
                    {source_name: digest},
                    "tiny-observable-sgd-v1",
                    materialized,
                    canonical_digest({"perturbed": materialized, "layer": index}),
                )
            )
            ledger.add_source(
                SourceConsumption(
                    source_name,
                    SourceDisposition.CONSUMED,
                    (target,),
                    (2, 2),
                    "float32",
                    digest,
                    "tiny observable-output fit",
                )
            )
        ledger_payload = ledger.validate(
            source_names=(
                "qwen.layers.0.observable_projection",
                "qwen.layers.1.observable_projection",
            ),
            target_names=(
                "rwkv.layers.0.observable_adapter",
                "rwkv.layers.1.observable_adapter",
            ),
        )
        _complete_stage(output, "ledger", ledger_payload)

    export_payload = _read_stage(output, "export")
    if export_payload is None:
        config = AnyToRWKVConfig(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            head_dim=4,
            head_size=4,
            num_heads=4,
            attention_hidden_size=16,
            mixer_types=["rwkv7", "rwkv7"],
            mtp_num_hidden_layers=0,
            torch_dtype="float32",
            any_to_rwkv={
                "artifact_contract": "any-to-rwkv-v1",
                "source_model_type": "qwen3_5_text",
                "source_architecture": "Qwen3_5ForCausalLM",
                "source_layer_types": ["linear_attention", "full_attention"],
                "source_text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "hidden_act": "silu",
                },
                "mixer_lineage": "rwkv7",
                "kernel_contract": "fla.ops.rwkv7.recurrent_rwkv7",
            },
        )
        torch.manual_seed(int(str(train_payload["state_sha256"])[:16], 16))
        model = AnyToRWKVForCausalLM(config).eval()
        artifact = output / "export" / "checkpoint"
        model.save_pretrained(
            artifact,
            safe_serialization=True,
            max_shard_size=16 * 1024,
        )
        persisted_config = json.loads(
            (artifact / "config.json").read_text(encoding="utf-8")
        )
        if (
            persisted_config.get("model_type") != "any_to_rwkv"
            or persisted_config.get("architectures") != ["AnyToRWKVForCausalLM"]
            or "auto_map" in persisted_config
        ):
            raise ContractError("tiny Any-to-RWKV export changed model identity")
        input_ids = torch.tensor([[1, 2, 3, 4]])
        expected = model(input_ids).logits.detach()
        expected_sha = _atomic_torch(
            output / "export" / "expected.pt",
            {"input_ids": input_ids, "logits": expected},
        )
        manifest = {
            "artifact_contract": "any-to-rwkv-v1",
            "model_type": "any_to_rwkv",
            "architecture": "AnyToRWKVForCausalLM",
            "files": {
                path.name: file_sha256(path)
                for path in sorted(artifact.iterdir())
                if path.is_file()
            },
        }
        export_payload = {"manifest": manifest, "expected_sha256": expected_sha}
        _complete_stage(output, "export", export_payload)
    expected_path = output / "export" / "expected.pt"
    checkpoint = output / "export" / "checkpoint"
    if export_payload.get("expected_sha256") != file_sha256(expected_path):
        raise ContractError("tiny pipeline expected observable artifact changed")
    manifest = export_payload.get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ContractError("tiny pipeline standard export manifest is incomplete")
    for name, digest in manifest["files"].items():
        path = checkpoint / str(name)
        if not path.is_file() or file_sha256(path) != digest:
            raise ContractError(f"tiny pipeline standard export file changed: {name}")

    verify_payload = _read_stage(output, "verify")
    if verify_payload is None:
        script = """
import sys, torch
import any2rwkv
import any2rwkv.modeling_any2rwkv as modeling
from any2rwkv.kernel import Rwkv7OperatorAdapter
from transformers import AutoModelForCausalLM

def test_recurrent(r, w, k, v, a, b, *, initial_state, output_final_state,
                   cu_seqlens=None, state_indices=None, mode):
    assert output_final_state and cu_seqlens is None and state_indices is None
    assert mode == 'fp32io16'
    state = initial_state
    output = []
    for token in range(r.shape[1]):
        projection = torch.einsum('bhk,bhkv->bhv', a[:, token].float(), state)
        state = (w[:, token].float().exp().unsqueeze(-1) * state
                 + b[:, token].float().unsqueeze(-1) * projection.unsqueeze(-2)
                 + k[:, token].float().unsqueeze(-1)
                 * v[:, token].float().unsqueeze(-2))
        output.append(torch.einsum('bhk,bhkv->bhv', r[:, token].float(), state))
    return torch.stack(output, dim=1).to(r.dtype), state

modeling.load_rwkv7_operator_adapter = lambda head_size: Rwkv7OperatorAdapter(
    test_recurrent, lambda: 'flash_rwkv', head_size=head_size, require_flash=True
)
artifact, expected_path = sys.argv[1:]
any2rwkv.register_any_to_rwkv_auto_classes()
expected = torch.load(expected_path, map_location='cpu', weights_only=True)
model = AutoModelForCausalLM.from_pretrained(artifact).eval()
actual = model(expected['input_ids']).logits
torch.testing.assert_close(actual, expected['logits'], rtol=0, atol=0)
assert model.__class__.__name__ == 'AnyToRWKVForCausalLM'
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(checkpoint),
                str(expected_path),
            ],
            check=True,
        )
        verify_payload = {
            "status": "complete",
            "loader": "fresh-process AutoModelForCausalLM",
            "observable": "logits",
        }
        _complete_stage(output, "verify", verify_payload)
    return {
        "status": "complete",
        "classification": "tiny-contract-pipeline",
        "output": str(output),
    }


def _run_training(root: Path, interrupt_after: int | None) -> dict[str, object]:
    torch.manual_seed(11)
    teacher = nn.Linear(2, 1, bias=False).eval().requires_grad_(False)
    layers = nn.ModuleList((nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)))
    adapter = nn.Linear(2, 1, bias=False).eval().requires_grad_(False)
    rows = torch.tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.25]])
    targets = teacher(rows).detach()
    cache, layer_index, step, total, trace = rows.clone(), 0, 0, 0, []
    resume_path, binding_path = root / "resume.pt", root / "resume.json"
    saved_optimizer = None
    if resume_path.is_file() or binding_path.is_file():
        if not resume_path.is_file() or not binding_path.is_file():
            raise ContractError("tiny pipeline training resume artifact is partial")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding.get("sha256") != file_sha256(resume_path):
            raise ContractError("tiny pipeline training resume hash differs")
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        layers.load_state_dict(saved["layers"])
        cache, layer_index, step, total, trace = (
            saved["cache"],
            saved["layer"],
            saved["step"],
            saved["total"],
            saved["trace"],
        )
        saved_optimizer = saved["optimizer"]
        LayerMajorResumeContract.from_dict(
            saved["contract"],
            expected_digests={
                "data_sha256": canonical_digest(rows.tolist()),
                "weight_sha256": canonical_digest(
                    {
                        name: value.detach().tolist()
                        for name, value in layers.state_dict().items()
                    }
                ),
                "trace_sha256": canonical_digest(trace),
                "solver_sha256": canonical_digest(saved_optimizer),
                "state_sha256": canonical_digest({"layer": layer_index, "step": step}),
                "current_cache_sha256": canonical_digest(cache.tolist()),
                "next_cache_sha256": canonical_digest(
                    layers[layer_index](cache).detach().tolist()
                ),
            },
        )
    while layer_index < 2:
        active = activate_layer_major_training(
            teacher=teacher, layers=layers, layer_index=layer_index
        )
        optimizer = torch.optim.SGD(active, lr=0.05)
        if saved_optimizer is not None:
            optimizer.load_state_dict(saved_optimizer)
            saved_optimizer = None
        while step < 3:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                adapter(layers[layer_index](cache)), targets
            )
            loss.backward()
            optimizer.step()
            step += 1
            total += 1
            trace.append(float(loss.detach()))
            contract = LayerMajorResumeContract(
                1,
                layer_index,
                step,
                step % len(rows),
                canonical_digest(rows.tolist()),
                canonical_digest(
                    {
                        name: value.detach().tolist()
                        for name, value in layers.state_dict().items()
                    }
                ),
                canonical_digest(trace),
                canonical_digest(optimizer.state_dict()),
                canonical_digest({"layer": layer_index, "step": step}),
                canonical_digest(cache.tolist()),
                canonical_digest(layers[layer_index](cache).detach().tolist()),
            )
            sha = _atomic_torch(
                resume_path,
                {
                    "layers": layers.state_dict(),
                    "cache": cache,
                    "layer": layer_index,
                    "step": step,
                    "total": total,
                    "trace": trace,
                    "optimizer": optimizer.state_dict(),
                    "contract": contract.to_dict(),
                },
            )
            write_json(binding_path, {"sha256": sha})
            if interrupt_after is not None and total == interrupt_after:
                return {"status": "interrupted"}
        cache = layers[layer_index](cache).detach()
        layer_index += 1
        step = 0
    resume_path.unlink()
    binding_path.unlink()
    return {
        "status": "complete",
        "source_sha256": canonical_digest(rows.tolist()),
        "state_sha256": canonical_digest(cache.tolist()),
        "layer_sha256": [
            canonical_digest(
                {
                    name: value.detach().tolist()
                    for name, value in layer.state_dict().items()
                }
            )
            for layer in layers
        ],
    }
