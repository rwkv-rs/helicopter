from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.configuration_any2rwkv import AnyToRWKVProxyConfig
from any2rwkv.contract import build_target_config
from any2rwkv.errors import ContractError
from any2rwkv.export import (
    CHECKPOINT_LOCAL_RUNTIME_MODULES,
    export_hf_checkpoint,
)
from any2rwkv.fixture import write_fixture
from any2rwkv.modeling_any2rwkv import AnyToRWKVProxyForCausalLM
from any2rwkv.target import build_zero_step_ledger


def _proxy_export_inputs(tmp_path: Path):
    source = read_checkpoint(
        write_fixture(tmp_path / "source", layers=2, moe=False),
        require_final_layers=False,
    )
    _, specs, _ = build_zero_step_ledger(
        tuple(source.tensor_names()),
        layer_count=2,
        hidden_size=64,
        head_dim=16,
        source_shard_hashes=tuple(
            source.file_hashes[path.name] for path in source.shards
        ),
    )
    return (
        source,
        specs,
        build_target_config(
            source.config,
            require_final_layers=False,
        ),
    )


@pytest.mark.parametrize(
    ("model_type", "architecture", "message"),
    (
        ("rwkv7", "Rwkv7ForCausalLM", "RWKV model family"),
        ("qwen3_5_text", "Qwen3_5ForCausalLM", "Qwen model family"),
    ),
)
def test_export_rejects_rwkv_and_qwen_identity_masquerades(
    tmp_path: Path,
    model_type: str,
    architecture: str,
    message: str,
) -> None:
    source, specs, config = _proxy_export_inputs(tmp_path)
    config["model_type"] = model_type
    config["architectures"] = [architecture]
    output = tmp_path / "masquerading-artifact"

    with pytest.raises(ContractError, match=message):
        export_hf_checkpoint(
            source,
            output,
            target_config=config,
            target_specs=specs,
        )

    assert not output.exists()


def test_export_rejects_auto_map_and_legacy_metadata(tmp_path: Path) -> None:
    source, specs, config = _proxy_export_inputs(tmp_path)
    config["auto_map"] = {
        "AutoModelForCausalLM": "modeling_any2rwkv.AnyToRWKVProxyForCausalLM"
    }
    with pytest.raises(ContractError, match="no auto_map"):
        export_hf_checkpoint(
            source,
            tmp_path / "auto-map-artifact",
            target_config=config,
            target_specs=specs,
        )

    legacy_config = dict(config)
    legacy_config.pop("auto_map")
    legacy_config["any2rwkv"] = legacy_config.pop("any_to_rwkv")
    with pytest.raises(ContractError, match="artifact contract"):
        export_hf_checkpoint(
            source,
            tmp_path / "legacy-artifact",
            target_config=legacy_config,
            target_specs=specs,
        )


def test_independent_export_fresh_auto_model_roundtrip(tmp_path: Path) -> None:
    source, specs, config = _proxy_export_inputs(tmp_path)
    output = tmp_path / "artifact"
    manifest = export_hf_checkpoint(
        source,
        output,
        target_config=config,
        target_specs=specs,
        max_shard_bytes=64 * 1024,
    )

    persisted_config = json.loads((output / "config.json").read_text())
    assert persisted_config["model_type"] == "any_to_rwkv_proxy"
    assert persisted_config["architectures"] == ["AnyToRWKVProxyForCausalLM"]
    assert persisted_config["any_to_rwkv"]["source_model_type"] == "qwen3_5_text"
    assert persisted_config["any_to_rwkv"]["mixer_lineage"] == "rwkv7"
    assert (
        persisted_config["any_to_rwkv"]["kernel_contract"]
        == "fla.ops.rwkv7.recurrent_rwkv7"
    )
    assert "auto_map" not in persisted_config
    assert manifest["artifact_contract"] == "any-to-rwkv-v1"
    assert manifest["trust_remote_code"] is False
    assert not any(
        (output / name).exists() for name in CHECKPOINT_LOCAL_RUNTIME_MODULES
    )

    model, loading_info = AnyToRWKVProxyForCausalLM.from_pretrained(
        output,
        output_loading_info=True,
    )
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    model = model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected_path = tmp_path / "expected.pt"
    torch.save(
        {
            "input_ids": input_ids,
            "logits": model(input_ids).logits.detach(),
        },
        expected_path,
    )

    script = """
import sys

import torch
import any2rwkv
from transformers import AutoConfig, AutoModelForCausalLM

artifact, expected_path = sys.argv[1:]
any2rwkv.register_any_to_rwkv_auto_classes()
expected = torch.load(expected_path, map_location="cpu", weights_only=True)
config = AutoConfig.from_pretrained(artifact)
assert config.__class__.__name__ == "AnyToRWKVProxyConfig"
model, loading_info = AutoModelForCausalLM.from_pretrained(
    artifact,
    output_loading_info=True,
)
assert model.__class__.__name__ == "AnyToRWKVProxyForCausalLM"
assert not loading_info["missing_keys"]
assert not loading_info["unexpected_keys"]
actual = model.eval()(expected["input_ids"]).logits
torch.testing.assert_close(actual, expected["logits"], rtol=0, atol=0)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(output), str(expected_path)],
        check=True,
    )


def test_config_rejects_checkpoint_local_loader() -> None:
    with pytest.raises(ValueError, match="auto_map is forbidden"):
        AnyToRWKVProxyConfig(
            auto_map={
                "AutoModelForCausalLM": ("modeling_any2rwkv.AnyToRWKVProxyForCausalLM")
            }
        )
