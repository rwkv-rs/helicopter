from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.contract import build_target_config
from any2rwkv.errors import ContractError
from any2rwkv.export import (
    export_hf_checkpoint,
    export_transformers_rwkv7_checkpoint,
)
from any2rwkv.fixture import write_fixture
from any2rwkv.target import build_zero_step_ledger
from any2rwkv.transformers_rwkv7 import build_transformers_rwkv7_config


def _community_classes():
    reason = "requires Transformers RWKV-7 support from d207b7a or a later release"
    configuration = pytest.importorskip(
        "transformers.models.rwkv7.configuration_rwkv7",
        reason=reason,
    )
    modeling = pytest.importorskip(
        "transformers.models.rwkv7.modeling_rwkv7",
        reason=reason,
    )
    return configuration.Rwkv7Config, modeling.Rwkv7ForCausalLM


def _tiny_config() -> dict[str, object]:
    return build_transformers_rwkv7_config(
        vocab_size=31,
        context_length=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        head_size=4,
        dtype="float32",
    )


def test_public_builder_fails_closed_without_rwkv7_interface(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.rwkv7.configuration_rwkv7",
        None,
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.rwkv7.modeling_rwkv7",
        None,
    )

    with pytest.raises(
        ContractError,
        match="public Rwkv7Config/Rwkv7ForCausalLM interface",
    ):
        _tiny_config()


def test_private_qwen_shell_export_cannot_claim_public_rwkv7(
    tmp_path: Path,
) -> None:
    source = read_checkpoint(
        write_fixture(tmp_path / "source", layers=2),
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
    private_config = build_target_config(
        source.config,
        require_final_layers=False,
    )
    private_manifest = export_hf_checkpoint(
        source,
        tmp_path / "private-artifact",
        target_config=private_config,
        target_specs=specs,
        max_shard_bytes=64 * 1024,
    )

    assert (
        private_config["any2rwkv"]["artifact_contract"]
        == "private-any2rwkv-qwen-shell-v1"
    )
    assert private_manifest["artifact_contract"] == "private-any2rwkv-qwen-shell-v1"
    assert (tmp_path / "private-artifact/modeling_any2rwkv.py").is_file()

    unmarked_config = dict(private_config)
    unmarked_metadata = dict(unmarked_config["any2rwkv"])
    del unmarked_metadata["artifact_contract"]
    unmarked_config["any2rwkv"] = unmarked_metadata
    unmarked_output = tmp_path / "unmarked-private-artifact"
    with pytest.raises(ContractError, match="artifact contract marker"):
        export_hf_checkpoint(
            source,
            unmarked_output,
            target_config=unmarked_config,
            target_specs=(),
        )
    assert not unmarked_output.exists()

    output = tmp_path / "masquerading-artifact"

    with pytest.raises(ContractError, match="private Any2RWKV.*cannot declare"):
        export_hf_checkpoint(
            source,
            output,
            target_config=_tiny_config(),
            target_specs=(),
        )

    assert not output.exists()

    config_with_private_loader = _tiny_config()
    config_with_private_loader["auto_map"] = {
        "AutoModelForCausalLM": "modeling_any2rwkv.Any2RWKV7ForCausalLM"
    }
    with pytest.raises(ContractError, match="registered community model"):
        export_transformers_rwkv7_checkpoint(
            tmp_path / "private-loader-artifact",
            config=config_with_private_loader,
            state_dict={},
        )

    assert not (tmp_path / "private-loader-artifact").exists()


def test_community_export_lists_missing_private_pipeline_weights_before_writing(
    tmp_path: Path,
) -> None:
    _community_classes()
    output = tmp_path / "incomplete-artifact"
    private_pipeline_state = {
        "model.layers.0.attn.r_k": torch.zeros(4, 4),
    }

    with pytest.raises(ContractError) as raised:
        export_transformers_rwkv7_checkpoint(
            output,
            config=_tiny_config(),
            state_dict=private_pipeline_state,
        )

    message = str(raised.value)
    assert "public Rwkv7ForCausalLM strict state_dict load failed" in message
    assert "Missing key(s) in state_dict" in message
    assert "model.embeddings.weight" in message
    assert "Unexpected key(s) in state_dict" in message
    assert "model.layers.0.attn.r_k" in message
    assert not output.exists()


def test_community_export_fresh_auto_model_strict_weight_and_logits_roundtrip(
    tmp_path: Path,
) -> None:
    Rwkv7Config, Rwkv7ForCausalLM = _community_classes()
    config_payload = _tiny_config()
    public_config = Rwkv7Config.from_dict(config_payload)
    assert config_payload == public_config.to_diff_dict()
    torch.manual_seed(20260801)
    source = Rwkv7ForCausalLM(Rwkv7Config.from_dict(config_payload)).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected_logits = source(input_ids).logits.detach()
    expected_path = tmp_path / "expected.pt"
    torch.save(
        {
            "state_dict": {
                name: tensor.detach().clone()
                for name, tensor in source.state_dict().items()
            },
            "input_ids": input_ids,
            "logits": expected_logits,
        },
        expected_path,
    )

    output = tmp_path / "artifact"
    manifest = export_transformers_rwkv7_checkpoint(
        output,
        config=config_payload,
        state_dict=source.state_dict(),
        max_shard_bytes=16 * 1024,
    )

    persisted_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert persisted_config["model_type"] == "rwkv7"
    assert persisted_config["architectures"] == ["Rwkv7ForCausalLM"]
    assert "auto_map" not in persisted_config
    assert manifest["base_model_prefix"] == "model"
    assert manifest["tensor_count"] == len(source.state_dict())
    assert not (output / "modeling_any2rwkv.py").exists()

    script = """
import sys

import torch
from transformers import AutoModelForCausalLM

artifact, expected_path = sys.argv[1:]
expected = torch.load(expected_path, map_location="cpu", weights_only=True)
model, loading_info = AutoModelForCausalLM.from_pretrained(
    artifact,
    output_loading_info=True,
)
model = model.eval()

assert model.__class__.__name__ == "Rwkv7ForCausalLM"
assert model.base_model_prefix == "model"
for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
    assert not loading_info.get(name), (name, loading_info.get(name))
actual_state = model.state_dict()
assert actual_state.keys() == expected["state_dict"].keys()
for name, tensor in expected["state_dict"].items():
    torch.testing.assert_close(actual_state[name], tensor, rtol=0, atol=0)
actual_logits = model(expected["input_ids"]).logits
torch.testing.assert_close(actual_logits, expected["logits"], rtol=0, atol=0)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(output), str(expected_path)],
        check=True,
    )
