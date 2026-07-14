from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from any2rwkv.checkpoint import read_checkpoint, sha256_file
from any2rwkv.configuration_any2rwkv import (
    Any2RWKV7Config,
    Any2RWKVHybridConfig,
    Any2RWKVProxyConfig,
)
from any2rwkv.modeling_any2rwkv import Any2RWKVProxyForCausalLM
from any2rwkv.contract import build_target_config, validate_source_config
from any2rwkv.errors import ContractError
from any2rwkv.export import (
    export_hf_checkpoint,
    export_text_teacher_checkpoint,
    refresh_hf_runtime_files,
)
from any2rwkv import export as export_module
from any2rwkv.fixture import tiny_qwen35_config, write_fixture
from any2rwkv.roundtrip import validate_sharded_checkpoint
from any2rwkv.source import fetch_source, verify_source
from any2rwkv.target import build_zero_step_ledger


class ContractTests(unittest.TestCase):
    def test_tied_source_restores_lm_head_from_preserved_embedding(self) -> None:
        source = tiny_qwen35_config(layers=4)
        source["tie_word_embeddings"] = True
        target = build_target_config(source, require_final_layers=False)
        config = Any2RWKVProxyConfig(**target)
        model = Any2RWKVProxyForCausalLM(config)

        self.assertEqual(
            model.all_tied_weights_keys,
            {"lm_head.weight": "model.embed_tokens.weight"},
        )
        self.assertEqual(
            model.lm_head.weight.data_ptr(),
            model.model.embed_tokens.weight.data_ptr(),
        )

    def test_final_target_is_text_only_and_all_60_layers_recurrent(self) -> None:
        source = tiny_qwen35_config()
        target = build_target_config(source)
        self.assertEqual(target["model_type"], "any2rwkv_qwen35_rwkv7")
        self.assertEqual(Any2RWKV7Config.model_type, target["model_type"])
        self.assertEqual(target["architectures"], ["Any2RWKV7ForCausalLM"])
        self.assertIn("AutoModelForCausalLM", target["auto_map"])
        self.assertEqual(target["layer_types"], ["rwkv7"] * 60)
        self.assertTrue(target["any2rwkv"]["final_recurrent"])
        self.assertTrue(target["any2rwkv"]["preserved"])

    def test_hybrid_cannot_masquerade_as_final_rwkv7(self) -> None:
        target = build_target_config(tiny_qwen35_config(), converted_layers=17)
        self.assertEqual(target["model_type"], "any2rwkv_hybrid")
        self.assertEqual(Any2RWKVHybridConfig.model_type, target["model_type"])
        self.assertEqual(target["architectures"], ["Any2RWKVHybridForCausalLM"])
        self.assertTrue(target["auto_map"]["AutoConfig"].endswith("Any2RWKVHybridConfig"))
        self.assertFalse(target["any2rwkv"]["final_recurrent"])
        self.assertEqual(target["layer_types"].count("rwkv7"), 17)

    def test_multimodal_unknown_and_non_60_layouts_are_rejected(self) -> None:
        source = tiny_qwen35_config()
        source["vision_config"] = {"depth": 1}
        with self.assertRaisesRegex(ContractError, "multimodal"):
            validate_source_config(source)
        unknown = tiny_qwen35_config()
        unknown["model_type"] = "mamba"
        with self.assertRaisesRegex(ContractError, "unsupported source"):
            validate_source_config(unknown)
        with self.assertRaisesRegex(ContractError, "expected 60"):
            validate_source_config(tiny_qwen35_config(layers=4))

    def test_nested_real_qwen_rope_parameters_are_preserved(self) -> None:
        text = tiny_qwen35_config()
        text["rope_parameters"] = {"rope_theta": 10_000_000, "partial_rotary_factor": 0.25}
        source = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": text,
            "vision_config": {"depth": 27},
        }
        contract = validate_source_config(source, text_backbone_only=True)
        target = build_target_config(source)
        self.assertEqual(contract.rope_theta, 10_000_000)
        self.assertEqual(contract.partial_rotary_factor, 0.25)
        self.assertEqual(target["rope_parameters"], text["rope_parameters"])

    def test_real_proxy_can_be_fully_recurrent_but_never_marked_final(self) -> None:
        target = build_target_config(tiny_qwen35_config(layers=24), require_final_layers=False)
        self.assertEqual(target["layer_types"], ["rwkv7"] * 24)
        self.assertEqual(target["model_type"], "any2rwkv_proxy")
        self.assertEqual(Any2RWKVProxyConfig.model_type, target["model_type"])
        self.assertEqual(target["architectures"], ["Any2RWKVProxyForCausalLM"])
        self.assertTrue(target["auto_map"]["AutoConfig"].endswith("Any2RWKVProxyConfig"))
        self.assertFalse(target["any2rwkv"]["final_recurrent"])
        self.assertTrue(target["any2rwkv"]["fully_recurrent_proxy"])

    def test_reader_validates_hf_safetensors_and_tokenizer_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = write_fixture(Path(temporary) / "fixture")
            manifest = read_checkpoint(root)
            self.assertEqual(manifest.contract.num_hidden_layers, 60)
            self.assertIn("model.safetensors", manifest.file_hashes)
            self.assertGreater(len(tuple(manifest.tensor_names())), 100)
            (root / "tokenizer_config.json").unlink()
            with self.assertRaisesRegex(ContractError, "tokenizer"):
                read_checkpoint(root)

    def test_structural_export_is_sharded_deterministic_and_text_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(write_fixture(root / "source"))
            source_names = tuple(source.tensor_names())
            _, specs, target_names = build_zero_step_ledger(
                source_names,
                layer_count=60,
                hidden_size=64,
                source_shard_hashes=tuple(source.file_hashes[path.name] for path in source.shards),
            )
            manifest = export_hf_checkpoint(
                source,
                root / "target",
                target_config=build_target_config(source.config),
                target_specs=specs,
                max_shard_bytes=64 * 1024,
            )
            index = json.loads((root / "target/model.safetensors.index.json").read_text())
            self.assertGreater(manifest["shard_count"], 1)
            self.assertEqual(set(index["weight_map"]), set(target_names))
            backbone_names = (name for name in index["weight_map"] if name.startswith("model.layers."))
            self.assertFalse(any("linear_attn" in name or "self_attn" in name for name in backbone_names))
            self.assertIn("mtp.layers.0.self_attn.q_proj.weight", index["weight_map"])
            self.assertTrue(all("PLACEHOLDER" not in name for name in index["weight_map"].values()))
            self.assertEqual(validate_sharded_checkpoint(root / "target")["tensor_count"], len(target_names))
            for module_name in (
                "configuration_any2rwkv.py",
                "modeling_any2rwkv.py",
                "mixer.py",
                "kernel.py",
                "errors.py",
            ):
                self.assertTrue((root / "target" / module_name).is_file(), module_name)

    def test_text_teacher_extraction_keeps_source_mixers_but_excludes_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(write_fixture(root / "source"))
            export_text_teacher_checkpoint(
                source, root / "teacher", max_shard_bytes=64 * 1024
            )
            index = json.loads(
                (root / "teacher/model.safetensors.index.json").read_text()
            )["weight_map"]
            self.assertTrue(any("linear_attn" in name for name in index))
            self.assertTrue(any("self_attn" in name for name in index))
            self.assertFalse(any(name.startswith("mtp.") for name in index))

    def test_runtime_refresh_repairs_checkpoint_code_and_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(
                write_fixture(root / "source", layers=4),
                require_final_layers=False,
            )
            _, specs, _ = build_zero_step_ledger(
                tuple(source.tensor_names()),
                layer_count=4,
                hidden_size=64,
                source_shard_hashes=tuple(
                    source.file_hashes[path.name] for path in source.shards
                ),
            )
            target = root / "target"
            export_hf_checkpoint(
                source,
                target,
                target_config=build_target_config(
                    source.config, require_final_layers=False
                ),
                target_specs=specs,
            )
            (target / "modeling_any2rwkv.py").write_text("stale\n", encoding="utf-8")

            hashes = refresh_hf_runtime_files(target)
            manifest = json.loads(
                (target / "roundtrip-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["files"]["modeling_any2rwkv.py"],
                hashes["modeling_any2rwkv.py"],
            )
            self.assertIn(
                "_tied_weights_keys",
                (target / "modeling_any2rwkv.py").read_text(encoding="utf-8"),
            )

    def test_scale_source_verification_hashes_pinned_read_only_60_layer_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "a" * 40
            source_dir = write_fixture(root / revision, layers=60)
            for path in source_dir.iterdir():
                if path.is_file():
                    path.chmod(0o444)
            manifest = root / "scale.json"
            manifest.write_text(
                json.dumps(
                    {
                        "classification": "final-scale-source-preflight-only",
                        "repository": "Qwen/Qwen3.5-397B-A17B",
                        "revision": revision,
                        "remote_read_only_path": str(source_dir),
                    }
                ),
                encoding="utf-8",
            )
            result = verify_source(manifest, source_dir)
            self.assertEqual(result["layers"], 60)
            self.assertEqual(len(result["combined_sha256"]), 64)
            self.assertTrue(result["read_only"])

    def test_scale_source_fetch_is_rejected_before_proxy_gate_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "scale.json"
            manifest.write_text(
                json.dumps(
                    {
                        "classification": "final-scale-source-preflight-only",
                        "repository": "Qwen/Qwen3.5-397B-A17B",
                        "revision": "a" * 40,
                        "remote_read_only_path": str(root / ("a" * 40)),
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("any2rwkv.source.snapshot_download") as download:
                with self.assertRaisesRegex(ContractError, "requires --scale-gate"):
                    fetch_source(manifest, root / ("a" * 40))
                download.assert_not_called()

    def test_proxy_source_verification_requires_frozen_revision_path_and_read_only_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "b" * 40
            source = write_fixture(root / revision, layers=4)
            weight = source / "model.safetensors"
            manifest = root / "proxy.json"
            manifest.write_text(
                json.dumps(
                    {
                        "classification": "real-proxy-model-not-60-layer-isomorphic",
                        "repository": "fixture/proxy",
                        "revision": revision,
                        "weight_file": weight.name,
                        "weight_sha256": sha256_file(weight),
                        "remote_read_only_path": str(source),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "writable"):
                verify_source(manifest, source)
            weight.chmod(0o444)
            self.assertTrue(verify_source(manifest, source)["read_only"])

    def test_proxy_source_fetch_can_freeze_a_manifest_bound_local_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = write_fixture(root / "seed", layers=4)
            weight = seed / "model.safetensors"
            original_mode = weight.stat().st_mode
            revision = "c" * 40
            destination = root / revision
            manifest = root / "proxy.json"
            manifest.write_text(
                json.dumps(
                    {
                        "classification": "real-proxy-model-not-60-layer-isomorphic",
                        "repository": "fixture/proxy",
                        "revision": revision,
                        "weight_file": weight.name,
                        "weight_sha256": sha256_file(weight),
                        "remote_read_only_path": str(destination),
                        "remote_seed_path": str(seed),
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("any2rwkv.source.snapshot_download") as download:
                result = fetch_source(manifest, destination)
                download.assert_not_called()
            self.assertEqual(result["acquisition"]["mode"], "verified-local-seed")
            self.assertEqual(weight.stat().st_mode, original_mode)
            self.assertFalse(
                (destination / weight.name).stat().st_mode
                & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            )
            self.assertTrue(verify_source(manifest, destination)["read_only"])

    def test_hf_export_resumes_after_a_committed_shard_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = read_checkpoint(
                write_fixture(root / "source", layers=4),
                require_final_layers=False,
            )
            ledger, specs, _ = build_zero_step_ledger(
                tuple(source.tensor_names()),
                layer_count=4,
                hidden_size=64,
                source_shard_hashes=tuple(
                    source.file_hashes[path.name] for path in source.shards
                ),
            )
            del ledger
            target_config = build_target_config(
                source.config, require_final_layers=False
            )
            interrupted = root / "interrupted"
            original_write_json = export_module.write_json
            injected = False

            def crash_after_progress(path, payload):
                nonlocal injected
                original_write_json(path, payload)
                if path.name == ".export-progress.json" and not injected:
                    injected = True
                    raise RuntimeError("injected export crash")

            with mock.patch(
                "any2rwkv.export.write_json", side_effect=crash_after_progress
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    export_hf_checkpoint(
                        source,
                        interrupted,
                        target_config=target_config,
                        target_specs=specs,
                        max_shard_bytes=64 * 1024,
                        resume_partial=True,
                        external_resume_binding={"mixer_fingerprint": "a" * 64},
                    )
            self.assertTrue((interrupted / ".export-progress.json").is_file())
            with self.assertRaisesRegex(ContractError, "resume binding differs"):
                export_hf_checkpoint(
                    source,
                    interrupted,
                    target_config=target_config,
                    target_specs=specs,
                    max_shard_bytes=64 * 1024,
                    resume_partial=True,
                    external_resume_binding={"mixer_fingerprint": "b" * 64},
                )
            resumed = export_hf_checkpoint(
                source,
                interrupted,
                target_config=target_config,
                target_specs=specs,
                max_shard_bytes=64 * 1024,
                resume_partial=True,
                external_resume_binding={"mixer_fingerprint": "a" * 64},
            )
            clean = export_hf_checkpoint(
                source,
                root / "clean",
                target_config=target_config,
                target_specs=specs,
                max_shard_bytes=64 * 1024,
                external_resume_binding={"mixer_fingerprint": "a" * 64},
            )
            self.assertEqual(resumed["files"], clean["files"])
            self.assertFalse((interrupted / ".export-progress.json").exists())


if __name__ == "__main__":
    unittest.main()
