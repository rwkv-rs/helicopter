from __future__ import annotations

import json
import math
import shutil
import gc
import subprocess
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.contract import build_target_config
from any2rwkv.export import export_hf_checkpoint
from any2rwkv.errors import ContractError
from any2rwkv.fixture import write_fixture
from any2rwkv.migration_init import (
    WarmStartTensorProvider,
    WarmStartVariant,
    apply_warm_start_plan,
    plan_warm_start,
)
from any2rwkv.mixer_store import RWKV7MixerLayerStore
from any2rwkv.mapping import is_locally_trainable
from any2rwkv.recipes.qwen35_to_rwkv7.layer_major_runner import (
    _activation_fit_full_attention,
    _activation_fit_gqa_native_zero_step_transaction,
    _commit_distributed_generation,
    _dependency_transaction_improves,
    _frozen_parameter_sha256,
    _formal_gqa_code_binding,
    _generation_mixer_state_sha256,
    _gqa_native_validation_improves,
    _module_state_hashes,
    _load_generation_state,
    _require_independent_activation_fit_caches,
    _require_frozen_parameter_sha256,
    _retain_gate_fit_candidate,
    _resolve_best_generation,
    _split_gqa_validation_protocol,
    _sha256_json,
    _write_generation_integrity,
    prepare_performance_profile_caches,
    run_gqa_zero_step_validation,
    run_suffix_free_layer_major,
)
from any2rwkv.recipes.qwen35_to_rwkv7 import (
    layer_major_runner as layer_major_runner_module,
)
from any2rwkv.artifacts import file_sha256, write_json
from any2rwkv.recipes.qwen35_to_rwkv7.global_corrective_runner import (
    _commit_distributed_layer_generation,
    _next_token_prediction_window,
    run_global_corrective,
)
from any2rwkv.distributed import DistributedContext
from any2rwkv.streaming_training import ActiveLayerOptimizerSnapshot
from any2rwkv.streaming_training import ActiveLayerOptimizer
from any2rwkv.streamed_teacher import Qwen35TeacherLayerLoader
from any2rwkv.target import build_zero_step_ledger, rwkv7_mixer_specs
from any2rwkv.distill_runner import (
    _binding_sha256,
    _checkpoint_binding,
    _materialize_corrective_base,
    _prepare_or_validate_corrective_output,
    _select_parent_recurrent_checkpoint,
    _validate_parent_run,
    _validate_initialized_run_binding,
    _zero_step_checkpoint_binding,
)
from any2rwkv.recipes import resolve_recipe


class PlannedInterruption(RuntimeError):
    pass


def test_new_synchronized_generation_uses_one_optimizer_copy_without_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, zero_step, _ = _prepare_fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = RWKV7MixerLayerStore(zero_step, run_dir / "mixer-overlays")
    mixer = store.load_mixer(0, device="cpu", dtype=torch.float32)
    optimizer = ActiveLayerOptimizer(learning_rate=1e-3)
    optimizer.activate(0, mixer)
    assert optimizer.backward(
        mixer.r_proj.weight.float().square().mean(),
        accumulation_steps=1,
    )
    cursor = {
        "active_layer": 0,
        "epoch_index": 0,
        "next_train_row": 1,
        "permutation_sha256": "a" * 64,
    }
    distributed = DistributedContext(rank=0, local_rank=0, world_size=1)

    def reject_immediate_reload(*_args, **_kwargs):
        raise AssertionError("new generation was read back immediately")

    monkeypatch.setattr(store, "load_mixer", reject_immediate_reload)
    generation = _commit_distributed_generation(
        distributed,
        run_dir,
        store,
        mixer,
        optimizer,
        cursor,
    )

    layout = json.loads(
        (generation / "training-state-layout.json").read_text(encoding="utf-8")
    )
    assert layout["layout"] == "canonical-optimizer-with-rank-rng"
    assert (generation / "training-state-canonical.pt").is_file()
    assert (generation / "rng-state-rank-000.pt").is_file()
    assert not tuple(generation.glob("training-state-rank-*.pt"))
    snapshot = _load_generation_state(
        generation,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        expected_cursor=cursor,
    )
    assert snapshot.optimizer_step == optimizer.optimizer_step
    for actual, expected in zip(
        snapshot.master_parameters,
        optimizer.snapshot().master_parameters,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mid_accumulation_generation_keeps_rank_local_full_state(
    tmp_path: Path,
) -> None:
    _, zero_step, _ = _prepare_fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = RWKV7MixerLayerStore(zero_step, run_dir / "mixer-overlays")
    mixer = store.load_mixer(0, device="cpu", dtype=torch.float32)
    optimizer = ActiveLayerOptimizer(learning_rate=1e-3)
    optimizer.activate(0, mixer)
    assert not optimizer.backward(
        mixer.r_proj.weight.float().square().mean(),
        accumulation_steps=2,
    )
    cursor = {
        "active_layer": 0,
        "epoch_index": 0,
        "next_train_row": 1,
        "permutation_sha256": "b" * 64,
    }
    generation = _commit_distributed_generation(
        DistributedContext(rank=0, local_rank=0, world_size=1),
        run_dir,
        store,
        mixer,
        optimizer,
        cursor,
    )

    layout = json.loads(
        (generation / "training-state-layout.json").read_text(encoding="utf-8")
    )
    assert layout["layout"] == "rank-local-full-state"
    assert (generation / "training-state-rank-000.pt").is_file()
    assert not (generation / "training-state-canonical.pt").exists()
    assert not tuple(generation.glob("rng-state-rank-*.pt"))


def test_gqa_full_attention_fit_does_not_fall_through_to_legacy(
    monkeypatch,
) -> None:
    selected = {
        "accepted": True,
        "selected_module_state_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        layer_major_runner_module,
        "_activation_fit_gqa_native_zero_step_transaction",
        lambda **_kwargs: selected,
    )

    def reject_legacy(**_kwargs):
        raise AssertionError("legacy attention fit must not overwrite GQA")

    monkeypatch.setattr(
        layer_major_runner_module,
        "_activation_fit_attention_dependency_transaction",
        reject_legacy,
    )
    monkeypatch.setattr(
        layer_major_runner_module,
        "_activation_fit_attention_time_mix_transaction",
        reject_legacy,
    )

    result = _activation_fit_full_attention(
        gqa_native_geometry={"query_heads": 4},
        gqa_installation_reader=object(),
        executor=object(),
        train_reader=object(),
        validation_reader=object(),
        mixer=object(),
        loaded_layer=object(),
        layer_index=3,
        burn_in_tokens=0,
        fit_rows=8,
        ridge=1e-3,
        functional_steps=2,
        functional_learning_rate=1e-3,
        micro_batch_size=1,
        loss_weights=object(),
        max_trace_bytes_per_rank=1024,
        run_time_mix_ablation=True,
        run_dir=Path("/unused"),
        distributed=object(),
    )

    assert result is selected


def test_immutable_generation_digest_matches_selected_module_state(
    tmp_path: Path,
) -> None:
    _, zero_step, _ = _prepare_fixture(tmp_path)
    store = RWKV7MixerLayerStore(zero_step, tmp_path / "overlay")
    mixer = store.load_mixer(0, device="cpu", dtype=torch.float32)
    generation = tmp_path / "generation"
    store.save_generation(
        generation,
        0,
        mixer,
        cursor={"phase": "pre-epoch-baseline"},
    )

    assert _generation_mixer_state_sha256(
        generation / "layer-000.safetensors",
        layer_index=0,
    ) == _sha256_json(_module_state_hashes(mixer))


def test_global_window_scores_each_label_from_its_preceding_position() -> None:
    logits = torch.arange(1 * 4 * 3, dtype=torch.float32).view(1, 4, 3)
    input_ids = torch.tensor([[10, 11, 12, 13]])

    warmed_logits, warmed_labels = _next_token_prediction_window(logits, input_ids, 2)
    cold_logits, cold_labels = _next_token_prediction_window(logits, input_ids, 0)

    assert torch.equal(warmed_logits, logits[:, 1:3])
    assert torch.equal(warmed_labels, input_ids[:, 2:])
    assert torch.equal(cold_logits, logits[:, :3])
    assert torch.equal(cold_labels, input_ids[:, 1:])


def _empty_optimizer_snapshot(layer_index: int) -> ActiveLayerOptimizerSnapshot:
    return ActiveLayerOptimizerSnapshot(
        layer_index=layer_index,
        optimizer={},
        scheduler={},
        gradients=(),
        micro_step=0,
        optimizer_step=0,
        accumulation_step=0,
        parameter_signature=(),
    )


def test_mutated_best_alias_recovers_unique_immutable_generation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    immutable = run_dir / "layer-generations" / "committed-best"
    immutable.mkdir(parents=True)
    (immutable / "payload.bin").write_bytes(b"committed")
    _write_generation_integrity(immutable)
    expected = file_sha256(immutable / "integrity.json")

    mutable = run_dir / "layer-best" / "layer-000"
    mutable.mkdir(parents=True)
    (mutable / "payload.bin").write_bytes(b"uncommitted-new-best")
    _write_generation_integrity(mutable)
    progress = {
        "best_generation": "layer-best/layer-000",
        "best_generation_manifest_sha256": expected,
        "convergence": {"best_epoch": 3},
    }

    recovered = _resolve_best_generation(run_dir, progress)

    assert recovered == immutable.resolve()
    assert progress["best_generation"] == "layer-generations/committed-best"


def _prepare_fixture(
    root: Path,
    *,
    layers: int = 1,
    config_overrides: dict[str, object] | None = None,
):
    source = read_checkpoint(
        write_fixture(
            root / "source", layers=layers, config_overrides=config_overrides
        ),
        require_final_layers=False,
    )
    target_config = build_target_config(source.config, require_final_layers=False)
    specs = tuple(
        spec
        for layer_index in range(layers)
        for spec in rwkv7_mixer_specs(
            layer_index, hidden_size=64, head_dim=int(target_config["head_dim"])
        )
    )
    warm_start = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
    zero_step = root / "zero-step"
    export_hf_checkpoint(
        source,
        zero_step,
        target_config=target_config,
        target_specs=specs,
        target_tensor_provider=WarmStartTensorProvider(source, specs, warm_start),
    )
    source_names = tuple(source.tensor_names())
    shard_hashes = tuple(source.file_hashes[path.name] for path in source.shards)
    ledger, _, target_names = build_zero_step_ledger(
        source_names,
        layer_count=source.contract.num_hidden_layers,
        hidden_size=source.contract.hidden_size,
        head_dim=int(target_config["head_dim"]),
        source_shard_hashes=shard_hashes,
    )
    apply_warm_start_plan(ledger, warm_start)
    ledger.write(zero_step / "mapping.json")
    write_json(
        zero_step / "mapping-coverage.json",
        ledger.validate(source_names, target_names),
    )
    write_json(zero_step / "warm-start-plan.json", warm_start.to_dict())
    trainable = [set() for _ in range(layers)]
    for entry in warm_start.to_dict()["entries"]:
        if not is_locally_trainable(entry):
            continue
        target = str(entry["target"])
        prefix, separator, local_name = target.partition(".attn.")
        assert separator
        parts = prefix.split(".")
        layer_index = int(parts[parts.index("layers") + 1])
        trainable[layer_index].add(local_name)
    assert all(trainable)
    return source, zero_step, trainable


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        cache_teacher_layers=False,
        corrective_resident_model_max_bytes=1_000_000_000,
        seed=20260714,
        cache_shard_rows=2,
        max_layer_input_cache_bytes=100_000_000,
        learning_rate=1e-4,
        burn_in_tokens=0,
        supervised_tokens=4,
        accumulation_steps=2,
        micro_batch_size=1,
        checkpoint_interval_micro_batches=1,
        activation_fit_functional_steps=2,
        activation_fit_functional_learning_rate=0.001,
        layer_min_epochs=2,
        layer_max_epochs=8,
        # Fixture-only plateau control. Production values remain plan-bound and
        # require independent training-control calibration.
        layer_min_delta=1.2,
        layer_patience=1,
        corrective_min_sweeps=1,
        corrective_max_sweeps=1,
        corrective_min_delta=0.0,
        local_loss_weights=SimpleNamespace(
            mixer_mse=1.0,
            block_mse=1.0,
            cosine=0.1,
        ),
        global_loss_weights=SimpleNamespace(token_kl=1.0, shifted_ce=0.25),
    )


def test_profile_cache_preparation_closes_pre_evidence_cycle(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path, layers=4)
    training_config = tmp_path / "profile-plan.json"
    dataset_manifest = tmp_path / "data-splits.json"
    training_config.write_text('{"schema_version": 3}\n', encoding="utf-8")
    dataset_manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")
    plan = SimpleNamespace(
        distributed_world_size=1,
        cache_shard_rows=2,
        max_layer_input_cache_bytes=100_000_000,
        max_cached_layer_input_bytes_per_rank=10_000_000,
    )
    rows = (
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (3, 4, 5, 6),
        (4, 5, 6, 7),
    )

    result = prepare_performance_profile_caches(
        source_manifest=source,
        run_dir=zero_step,
        zero_step_dir=zero_step,
        token_rows=rows,
        validation_rows=rows[:2],
        plan=plan,
        initial_trainable=trainable,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result["status"] == "prepared"
    assert [
        case["representative_layer"] for case in result["cases"]
    ] == [0, 1, 3]
    cache_root = zero_step / "performance-profile-cache"
    for layer_index in range(4):
        train_manifest = json.loads(
            (
                cache_root
                / f"layer-{layer_index:03d}"
                / "distill_train"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        validation_manifest = json.loads(
            (
                cache_root
                / f"layer-{layer_index:03d}"
                / "validation"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert train_manifest["has_shared_states"] is (layer_index > 0)
        assert validation_manifest["has_shared_states"] is (layer_index > 0)
        assert (
            train_manifest["binding"]["training_config_sha256"]
            == file_sha256(training_config)
        )


def test_gqa_native_zero_step_runs_in_formal_layer_transaction(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(
        tmp_path,
        layers=4,
        config_overrides={
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 8,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )
    training_config = tmp_path / "gqa-fit-plan.json"
    dataset_manifest = tmp_path / "gqa-data-splits.json"
    training_config.write_text('{"schema_version": 3}\n', encoding="utf-8")
    dataset_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {
                    "distill_train": {
                        "source_sample_ids_sha256": "a" * 64,
                    },
                    "validation": {
                        "source_sample_ids_sha256": "b" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    rows = tuple(
        tuple((row + token) % 61 + 3 for token in range(4))
        for row in range(9)
    )
    validation_rows = tuple(
        tuple((token + 17) % 64 for token in row)
        for row in rows[:4]
    )
    prepare_performance_profile_caches(
        source_manifest=source,
        run_dir=zero_step,
        zero_step_dir=zero_step,
        token_rows=rows,
        validation_rows=validation_rows,
        plan=SimpleNamespace(
            distributed_world_size=1,
            cache_shard_rows=2,
            max_layer_input_cache_bytes=100_000_000,
            max_cached_layer_input_bytes_per_rank=10_000_000,
        ),
        initial_trainable=trainable,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    run_dir = tmp_path / "gqa-fit-run"
    outcome = run_gqa_zero_step_validation(
        source_manifest=source,
        run_dir=zero_step,
        evidence_dir=run_dir,
        zero_step_dir=zero_step,
        plan=SimpleNamespace(
            evidence_tier="fixture",
            distributed_world_size=1,
            burn_in_tokens=0,
            activation_fit_rows=9,
            micro_batch_size=1,
            local_loss_weights=SimpleNamespace(
                mixer_mse=1.0,
                block_mse=1.0,
                cosine=0.1,
            ),
            max_cached_layer_input_bytes_per_rank=10_000_000,
        ),
        initial_trainable=trainable,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        layer_index=3,
        train_row_source_sample_ids=tuple(
            (f"train-{row}",) for row in range(len(rows))
        ),
        validation_row_source_sample_ids=tuple(
            (f"validation-{row}",)
            for row in range(len(validation_rows))
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    report = json.loads(
        (
            run_dir
            / "activation-fit"
            / "gqa-native-zero-step-layer-003.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] in {"accepted", "rejected"}
    assert report["source_geometry"] == {
        "query_heads": 4,
        "key_value_heads": 2,
        "head_dim": 16,
    }
    assert report["target_geometry"] == {
        "native_heads": 8,
        "head_dim": 8,
    }
    assert report["requested_context_lengths"] == [1, 2, 4]
    assert report["eligible_context_lengths"] == [2, 4]
    assert report["skipped_context_lengths"] == [1]
    assert {
        tuple(candidate["row_indices"])
        for candidate in report["candidate_summaries"]
    } == {tuple(range(9))}
    assert set(report["fit_report"]) >= {
        "exact_prefix_hazard_oracle",
        "bounded_hazard_surrogate",
        "observable_state_compression",
        "native_transition",
        "native_parameter_projection",
        "solver_config_sha256",
        "trace_aggregate_sha256",
        "source_weight_aggregate_sha256",
    }
    for stage in (
        "exact_prefix_hazard_oracle",
        "bounded_hazard_surrogate",
        "observable_state_compression",
        "native_transition",
    ):
        assert len(
            report["fit_report"][stage]["per_query_head"]
        ) == 4
        assert len(
            report["fit_report"][stage]["per_key_value_group"]
        ) == 2
    assert set(report["mapped_component_restoration_ablations"]) == {
        "r_proj",
        "k_proj",
        "v_proj",
        "w_a",
        "gate",
        "o_proj",
    }
    assert len(report["materialization"]["aggregate_sha256"]) == 64
    assert (
        report["train_cache_binding"]["split"] == "distill_train"
        and report["validation_cache_binding"]["split"] == "validation"
    )
    assert (
        report["validation_cache_binding"]["row_subset_role"]
        == "gqa-native-zero-step-installation"
    )
    assert (
        json.loads(
            (run_dir / "execution.json").read_text(encoding="utf-8")
        )["split_protocol"]["epoch_selection"]["row_subset_role"]
        == "layerwise-epoch-selection"
    )
    execution = json.loads(
        (run_dir / "execution.json").read_text(encoding="utf-8")
    )
    epoch_binding = execution["split_protocol"]["epoch_selection"]
    assert (
        report["validation_cache_binding"]["parent_row_indices_sha256"]
        != epoch_binding["parent_row_indices_sha256"]
    )
    assert set(
        report["validation_cache_binding"]["parent_row_indices"]
    ).isdisjoint(
        epoch_binding["parent_row_indices"]
    )
    assert (
        report["train_cache_binding"]["source_sample_ids_sha256"]
        != report["validation_cache_binding"]["source_sample_ids_sha256"]
    )
    assert (
        report["train_cache_binding"]["row_subset_role"]
        == "gqa-native-zero-step-fit"
    )
    fit_source_ids = set(
        report["train_cache_binding"]["row_subset_source_sample_ids"]
    )
    installation_source_ids = set(
        report["validation_cache_binding"][
            "row_subset_source_sample_ids"
        ]
    )
    epoch_source_ids = set(
        epoch_binding["row_subset_source_sample_ids"]
    )
    assert fit_source_ids.isdisjoint(installation_source_ids)
    assert fit_source_ids.isdisjoint(epoch_source_ids)
    assert installation_source_ids.isdisjoint(epoch_source_ids)
    assert (
        report["validation_cache_binding"]["parent_row_identity_sha256"]
        != epoch_binding["parent_row_identity_sha256"]
    )
    assert outcome["execution_report_sha256"] == file_sha256(
        run_dir / "execution.json"
    )
    assert outcome["status"] == report["status"]
    assert (
        execution["immutable_generation"][
            "selected_module_state_sha256"
        ]
        == execution["immutable_generation"][
            "recomputed_module_state_sha256"
        ]
        == outcome["selected_module_state_sha256"]
    )
    assert (
        execution["runtime"]["ranks"][0]["device"] == "cpu-fixture"
        and execution["module_dtype"] == "float32"
    )
    assert (
        execution["runtime"]["maximum_action_wall_seconds"]
        >= execution["runtime"]["maximum_transaction_wall_seconds"]
        > 0
    )
    assert (
        execution["runtime"]["measurement_scope"]["cuda_peak_kind"]
        == "PyTorch caching-allocator allocated/reserved bytes"
    )
    assert len(execution["source_sample_protocol_sha256"]) == 64
    assert (
        execution["code_commit"]
        == execution["code_binding"]["commit"]
    )
    assert len(execution["code_binding"]["code_tree_sha256"]) == 64


def test_formal_gqa_code_binding_rejects_dirty_code_scope(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "checkout"
    package = repo / "src/train/any2rwkv/any2rwkv"
    package.mkdir(parents=True)
    module = package / "solver.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (package / ".gitignore").write_text(
        "*.generated.py\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "config",
            "user.email",
            "test@example.com",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", package.relative_to(repo)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture"],
        check=True,
    )

    clean = _formal_gqa_code_binding(repo, require_clean=True)
    assert clean["clean"] is True
    assert clean["runtime_file_count"] == 2
    module.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(
        ContractError,
        match="requires a clean any2rwkv code scope",
    ):
        _formal_gqa_code_binding(repo, require_clean=True)
    dirty = _formal_gqa_code_binding(repo, require_clean=False)
    assert dirty["clean"] is False
    assert dirty["code_tree_sha256"] != clean["code_tree_sha256"]

    subprocess.run(
        ["git", "-C", str(repo), "restore", module.relative_to(repo)],
        check=True,
    )
    ignored = package / "ignored.generated.py"
    ignored.write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(
        ContractError,
        match="requires a clean any2rwkv code scope",
    ):
        _formal_gqa_code_binding(repo, require_clean=True)
    ignored.unlink()

    linked = package / "linked.py"
    linked.symlink_to(module.name)
    with pytest.raises(
        ContractError,
        match="contains symlinks",
    ):
        _formal_gqa_code_binding(repo, require_clean=True)
    linked.unlink()

    directory = package / "directory"
    directory.mkdir()
    (directory / "__init__.py").write_text("", encoding="utf-8")
    linked_directory = package / "linked_directory"
    linked_directory.symlink_to(directory.name)
    with pytest.raises(
        ContractError,
        match="contains symlinks",
    ):
        _formal_gqa_code_binding(repo, require_clean=True)
    linked_directory.unlink()

    broken = package / "broken.py"
    broken.symlink_to("missing.py")
    with pytest.raises(
        ContractError,
        match="contains symlinks",
    ):
        _formal_gqa_code_binding(repo, require_clean=True)


def test_formal_gqa_code_binding_uses_managed_sync_revision(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "deployed"
    package = repo / "src/train/any2rwkv/any2rwkv"
    package.mkdir(parents=True)
    (package / "solver.py").write_text("VALUE = 2\n", encoding="utf-8")
    revision_dir = repo / ".helicopter-dev"
    revision_dir.mkdir()
    revision = "a" * 40
    scope = "src/train/any2rwkv/any2rwkv"
    expected_tree_sha256 = _sha256_json(
        [
            {
                "path": f"{scope}/solver.py",
                "sha256": file_sha256(package / "solver.py"),
            }
        ]
    )
    write_json(
        revision_dir / "source-revisions.json",
        {
            "product_commit": revision,
            "submodules": {},
            "scopes": {
                scope: {
                    "clean": True,
                    "runtime_file_count": 1,
                    "code_tree_sha256": expected_tree_sha256,
                }
            },
        },
    )

    binding = _formal_gqa_code_binding(repo, require_clean=True)

    assert binding["commit"] == revision
    assert binding["revision_source"] == "helicopter-dev-managed-sync"
    assert binding["clean"] is None
    assert len(binding["managed_revision_manifest_sha256"]) == 64
    assert binding["runtime_file_count"] == 1

    (package / "solver.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(
        ContractError,
        match="differs from the synchronized commit-clean tree",
    ):
        _formal_gqa_code_binding(repo, require_clean=True)


def _provenance_cache_reader(
    tmp_path: Path,
    *,
    split: str,
    row_count: int,
):
    return SimpleNamespace(
        row_count=row_count,
        cache_dir=tmp_path / split,
        manifest={
            "binding": {
                "split": split,
                "dataset_manifest_sha256": "d" * 64,
                "source_sample_ids_sha256": (
                    "a" * 64 if split == "distill_train" else "b" * 64
                ),
            }
        },
    )


def test_gqa_validation_protocol_drops_rows_that_bridge_its_cut(
    tmp_path: Path,
) -> None:
    train_reader = _provenance_cache_reader(
        tmp_path,
        split="distill_train",
        row_count=2,
    )
    validation_reader = _provenance_cache_reader(
        tmp_path,
        split="validation",
        row_count=4,
    )
    fit, installation, epoch = _split_gqa_validation_protocol(
        train_reader,
        validation_reader,
        world_size=1,
        train_row_source_sample_ids=(("fit-a",), ("fit-b",)),
        validation_row_source_sample_ids=(
            ("left",),
            ("left", "bridge"),
            ("bridge", "right"),
            ("right",),
        ),
    )

    fit_ids = set(
        fit.manifest["binding"]["row_subset_source_sample_ids"]
    )
    installation_ids = set(
        installation.manifest["binding"][
            "row_subset_source_sample_ids"
        ]
    )
    epoch_ids = set(
        epoch.manifest["binding"]["row_subset_source_sample_ids"]
    )
    assert fit_ids == {"fit-a", "fit-b"}
    assert installation_ids.isdisjoint(epoch_ids)
    assert fit_ids.isdisjoint(installation_ids)
    assert fit_ids.isdisjoint(epoch_ids)
    assert installation_ids.isdisjoint(epoch_ids)
    assert set(
        installation.manifest["binding"]["excluded_parent_rows"]
    ) == {1, 2}


def test_gqa_validation_protocol_rejects_fit_validation_sample_overlap(
    tmp_path: Path,
) -> None:
    train_reader = _provenance_cache_reader(
        tmp_path,
        split="distill_train",
        row_count=1,
    )
    validation_reader = _provenance_cache_reader(
        tmp_path,
        split="validation",
        row_count=2,
    )
    with pytest.raises(
        ContractError,
        match="fit and validation source samples overlap",
    ):
        _split_gqa_validation_protocol(
            train_reader,
            validation_reader,
            world_size=1,
            train_row_source_sample_ids=(("shared",),),
            validation_row_source_sample_ids=(("shared",), ("held-out",)),
        )


def test_gqa_validation_protocol_fails_closed_without_row_provenance(
    tmp_path: Path,
) -> None:
    train_reader = _provenance_cache_reader(
        tmp_path,
        split="distill_train",
        row_count=1,
    )
    validation_reader = _provenance_cache_reader(
        tmp_path,
        split="validation",
        row_count=2,
    )
    with pytest.raises(
        ContractError,
        match="per-row source provenance for distill_train",
    ):
        _split_gqa_validation_protocol(
            train_reader,
            validation_reader,
            world_size=1,
            validation_row_source_sample_ids=(("a",), ("b",)),
        )


def test_local_stage_rejects_any_frozen_parameter_drift(tmp_path: Path) -> None:
    _source, zero_step, trainable = _prepare_fixture(tmp_path)
    store = RWKV7MixerLayerStore(zero_step, tmp_path / "frozen-probe")
    mixer = store.load_base_mixer(0, device="cpu", dtype=torch.float32)
    constrained_trainable = set(trainable[0])
    constrained_trainable.remove("k_k")
    expected = _frozen_parameter_sha256(mixer, constrained_trainable)
    assert "k_k" in expected
    with torch.no_grad():
        mixer.k_k.add_(1)
    with pytest.raises(ContractError, match="locally frozen mixer parameters changed"):
        _require_frozen_parameter_sha256(
            mixer,
            expected,
            boundary="negative-test",
        )


def test_local_stage_allows_every_current_mixer_parameter_to_train(
    tmp_path: Path,
) -> None:
    _source, zero_step, trainable = _prepare_fixture(tmp_path)
    store = RWKV7MixerLayerStore(zero_step, tmp_path / "all-trainable-probe")
    mixer = store.load_base_mixer(0, device="cpu", dtype=torch.float32)

    assert _frozen_parameter_sha256(mixer, trainable[0]) == {}


def test_activation_fit_requires_mutually_exclusive_cache_bindings() -> None:
    train = SimpleNamespace(
        manifest={"binding": {"split": "distill_train", "digest": "train"}}
    )
    validation = SimpleNamespace(
        manifest={"binding": {"split": "validation", "digest": "validation"}}
    )
    _require_independent_activation_fit_caches(train, validation)

    same = SimpleNamespace(manifest={"binding": dict(train.manifest["binding"])})
    with pytest.raises(ContractError, match="mutually exclusive"):
        _require_independent_activation_fit_caches(train, same)

    wrong_split = SimpleNamespace(
        manifest={"binding": {"split": "distill_train", "digest": "other"}}
    )
    with pytest.raises(ContractError, match="mutually exclusive"):
        _require_independent_activation_fit_caches(train, wrong_split)


def _run(
    *,
    source,
    zero_step: Path,
    trainable,
    run_dir: Path,
    training_config: Path,
    dataset_manifest: Path,
    resume: Path | None = None,
    callback=None,
    plan=None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    warm_start_plan = run_dir / "warm-start-plan.json"
    if not warm_start_plan.exists():
        shutil.copy2(zero_step / "warm-start-plan.json", warm_start_plan)
    return run_suffix_free_layer_major(
        source_manifest=source,
        run_dir=run_dir,
        zero_step_dir=zero_step,
        token_rows=((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)),
        validation_rows=((13, 14, 15, 16), (17, 18, 19, 20)),
        plan=_plan() if plan is None else plan,
        initial_trainable=trainable,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        resume=resume,
        device=torch.device("cpu"),
        dtype=torch.float32,
        progress_callback=callback,
    )


def test_exhausted_layer_persists_final_validation_before_failing(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    plan = _plan()
    plan.layer_min_epochs = 1
    plan.layer_max_epochs = 2
    plan.layer_min_delta = 0.0
    plan.local_learning_rate_by_mixer_kind = (("linear_attention", 2e-5),)
    plan.learning_rate_schedule = "warmup-cosine"
    plan.learning_rate_warmup_ratio = 0.0
    plan.min_learning_rate_ratio = 0.1
    plan.layer_learning_rate_schedule_epochs = 1
    run_dir = tmp_path / "exhausted"

    with pytest.raises(ContractError, match="layer-convergence-failed"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            plan=plan,
        )

    progress = json.loads((run_dir / "layer-major-progress.json").read_text())
    assert progress["phase"] == "epoch-complete"
    assert progress["epoch_index"] == 2
    assert progress["next_train_row"] == 0
    assert len(progress["history"]) == 2
    assert progress["history"][-1]["exhausted"] is True
    telemetry_files = sorted((run_dir / "training-telemetry").glob("*.json"))
    assert len(telemetry_files) == 2
    telemetry = json.loads(telemetry_files[-1].read_text())
    assert telemetry["world_size"] == 1
    assert telemetry["global_micro_batch_size"] == 1
    assert telemetry["epoch_row_count"] == 3
    assert telemetry["context_token_count"] == 12
    assert telemetry["loss_token_count"] == 12
    assert telemetry["peak_cuda_reserved_bytes"] == 0
    assert telemetry["segment_context_tokens_per_second"] > 0
    assert telemetry["source_mixer_kind"] == "linear_attention"
    assert telemetry["configured_learning_rate"] == pytest.approx(2e-5)
    assert telemetry["learning_rate_schedule_epochs"] == 1
    assert telemetry["learning_rate_schedule_optimizer_steps"] == 2
    assert telemetry["optimizer"]["last_learning_rate"] == pytest.approx(2e-6)
    assert progress["generation_cursor"]["next_train_row"] == 3


def test_unknown_mixer_learning_rate_profile_fails_closed(tmp_path: Path) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    plan = _plan()
    plan.local_learning_rate_by_mixer_kind = (("misspelled_mixer", 2e-5),)

    with pytest.raises(ContractError, match="absent from source layer types"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=tmp_path / "unknown-profile",
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            plan=plan,
        )


def test_local_epochs_never_silently_expand_the_trainable_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    selected = {next(iter(sorted(trainable[0])))}
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    observed: list[set[str] | None] = []
    original_activate = ActiveLayerOptimizer.activate

    def recording_activate(
        self, layer_index, module, *, snapshot=None, trainable_names=None
    ):
        observed.append(None if trainable_names is None else set(trainable_names))
        return original_activate(
            self,
            layer_index,
            module,
            snapshot=snapshot,
            trainable_names=trainable_names,
        )

    monkeypatch.setattr(ActiveLayerOptimizer, "activate", recording_activate)
    plan = _plan()
    plan.layer_min_epochs = 2
    plan.layer_max_epochs = 2
    plan.layer_min_delta = 0.0
    with pytest.raises(
        ContractError,
        match="layer-(?:convergence-failed|training-no-improvement)",
    ):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=[selected],
            run_dir=tmp_path / "frozen-local-stage",
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            plan=plan,
        )

    # Activation fit and the optimizer each bind the declared set once. The
    # optimizer remains active across same-layer epochs so it does not rebuild
    # FP32 master parameters and Adam moments every epoch.
    assert len(observed) >= 2
    assert all(names == selected for names in observed)


def test_full_attention_time_mix_is_a_dependent_atomic_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, zero_step, trainable = _prepare_fixture(
        tmp_path,
        config_overrides={"layer_types": ["full_attention"]},
    )
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    plan = _plan()
    plan.activation_fit_rows = 3
    plan.activation_fit_ridge = 0.001
    plan.activation_fit_attention_time_mix_ablation = True
    run_dir = tmp_path / "attention-time-mix-transaction"

    # This integration case verifies transaction ordering and persistence.  Its
    # tiny random fixture is not a quality benchmark, so force both complete
    # generations through the acceptance branch and inspect their contracts.
    monkeypatch.setattr(
        layer_major_runner_module,
        "_dependency_transaction_improves",
        lambda _baseline, _candidate: True,
    )
    with pytest.raises(
        ContractError,
        match="layer-(?:convergence-failed|training-no-improvement)",
    ):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            plan=plan,
        )

    fit_dir = run_dir / "activation-fit"
    projection = json.loads(
        (fit_dir / "attention-dependency-transaction-layer-000.json").read_text()
    )
    transaction = json.loads(
        (
            fit_dir / "attention-time-mix-dependency-transaction-layer-000.json"
        ).read_text()
    )
    time_mix = json.loads(
        (fit_dir / "time-mix-attention-time-mix-transaction-layer-000.json").read_text()
    )
    baseline = json.loads(
        (run_dir / "training-baselines" / "layer-000.json").read_text()
    )

    assert projection["status"] == "accepted"
    assert transaction["status"] == "accepted"
    assert transaction["boundary"] == (
        "attention-qkv-time-mix-dependent-activation-fit-generation-v1"
    )
    assert transaction["gradient_scope"] == ["x_r", "x_k", "x_v"]
    assert transaction["excluded_native_controls"] == [
        "decay",
        "erase",
        "r_k_bonus",
        "x_w",
        "x_a",
        "x_g",
    ]
    assert transaction["selected_generation"] in {"time-mix-only", "fully-refit"}
    assert (
        transaction["selected_parameter_sha256"]
        == transaction["candidate_parameter_sha256"][transaction["selected_generation"]]
    )
    assert (
        transaction["validation_candidate"]
        == transaction["validation_candidates"][transaction["selected_generation"]]
    )
    assert time_mix["status"] == "deferred"
    assert time_mix["gradient_scope"] == ["x_r", "x_k", "x_v"]
    assert (
        time_mix["selected_parameter_sha256"] == time_mix["proposed_parameter_sha256"]
    )
    assert baseline["validation"] == transaction["validation_candidate"]


def test_subthreshold_sgd_best_is_preserved_without_advancing_layer(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    plan = _plan()
    plan.activation_fit_rows = 3
    plan.activation_fit_ridge = 0.001
    run_dir = tmp_path / "activation-fit"

    with pytest.raises(ContractError, match="layer-training-no-improvement"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            plan=plan,
        )

    report = json.loads((run_dir / "activation-fit" / "layer-000.json").read_text())
    decay_report = json.loads(
        (run_dir / "activation-fit" / "decay-layer-000.json").read_text()
    )
    qkv_report = json.loads(
        (run_dir / "activation-fit" / "gdn-qkv-layer-000.json").read_text()
    )
    qkv_dependency_report = json.loads(
        (
            run_dir / "activation-fit" / "gdn-qkv-dependency-transaction-layer-000.json"
        ).read_text()
    )
    beta_report = json.loads(
        (run_dir / "activation-fit" / "gdn-a-layer-000.json").read_text()
    )
    gate_report = json.loads(
        (run_dir / "activation-fit" / "gdn-gate-layer-000.json").read_text()
    )
    gate_dependency_report = json.loads(
        (
            run_dir
            / "activation-fit"
            / "gdn-gate-dependency-transaction-layer-000.json"
        ).read_text()
    )
    time_mix_report = json.loads(
        (
            run_dir
            / "activation-fit"
            / "time-mix-dependency-transaction-layer-000.json"
        ).read_text()
    )
    norm_report = json.loads(
        (run_dir / "activation-fit" / "norm-affine-layer-000.json").read_text()
    )
    dependency_report = json.loads(
        (
            run_dir / "activation-fit" / "dependency-transaction-layer-000.json"
        ).read_text()
    )
    assert not (run_dir / "activation-fit" / "time-mix-layer-000.json").exists()
    assert not (
        run_dir / "activation-fit" / "gdn-qkv-post-time-mix-layer-000.json"
    ).exists()
    assert not (
        run_dir / "activation-fit" / "gdn-gate-post-time-mix-layer-000.json"
    ).exists()
    progress = json.loads((run_dir / "layer-major-progress.json").read_text())
    baseline = json.loads(
        (run_dir / "training-baselines" / "layer-000.json").read_text()
    )
    assert progress["active_layer"] == 0
    assert progress["phase"] == "epoch-complete"
    assert progress["convergence"]["best_epoch"] == 0
    assert progress["convergence"]["best_metric"] < baseline["validation"]["loss"]
    assert progress["best_generation"] != baseline["generation"]
    assert not (run_dir / "layer-input-cache" / "layer-001").exists()
    assert report["status"] in {"accepted", "rejected"}
    assert report["boundary"] == "source-mixer-output-to-native-o-proj-v1"
    assert report["fit_rows"] == 3
    assert sum(row["rows"] for row in report["rank_contributions"]) == 3
    assert len(report["selected_parameter_hashes"]) == 1
    expected_output_weight = (
        report["proposed_o_proj_weight_sha256"]
        if report["status"] == "accepted"
        else report["original_o_proj_weight_sha256"]
    )
    assert report["selected_o_proj_weight_sha256"] == expected_output_weight
    if report["status"] == "accepted":
        assert (
            report["validation_after"]["mixer_normalized_mse"]
            < report["validation_before"]["mixer_normalized_mse"]
        )
    else:
        assert (
            report["validation_before"]["mixer_normalized_mse"]
            < report["chain_acceptance_baseline"]["mixer_normalized_mse"]
        )
    for expected_stage, expected_statuses, qkv_stage_report in (
        ("initial", {"accepted", "rejected"}, qkv_report),
        ("dependency-transaction", {"deferred"}, qkv_dependency_report),
    ):
        assert (
            qkv_stage_report["boundary"]
            == "gdn-headnorm-equivalent-signals-to-native-projections-v4"
        )
        assert qkv_stage_report["fit_stage"] == expected_stage
        assert qkv_stage_report["target_signals"] == {
            "r": "recurrent_r*sqrt(head_dim)",
            "k": "write_key",
            "v": "write_value",
        }
        assert qkv_stage_report["status"] in expected_statuses
        assert qkv_stage_report["train_cache_binding"]["split"] == "distill_train"
        assert qkv_stage_report["validation_cache_binding"]["split"] == "validation"
        assert all(
            sum(row["rows"] for row in qkv_stage_report["rank_contributions"][role])
            == 3
            for role in ("r", "k", "v")
        )
        assert len(qkv_stage_report["selected_parameter_hashes"]) == 1
        expected_qkv = (
            qkv_stage_report["proposed_weight_sha256"]
            if qkv_stage_report["status"] in {"accepted", "deferred"}
            else qkv_stage_report["original_weight_sha256"]
        )
        assert qkv_stage_report["selected_weight_sha256"] == expected_qkv
        assert (
            qkv_stage_report["selected_parameter_hashes"][0]["weights"] == expected_qkv
        )
        if qkv_stage_report["status"] == "accepted":
            assert sum(
                qkv_stage_report["validation_after"][role]["normalized_mse"]
                for role in ("r", "k", "v")
            ) < sum(
                qkv_stage_report["validation_before"][role]["normalized_mse"]
                for role in ("r", "k", "v")
            )
    assert beta_report["boundary"] == "gdn-beta-times-decay-to-native-erase-v3"
    assert beta_report["solver"] == "source-beta-decay-basis-ridge-v1"
    assert beta_report["basis_width"] == 2 * beta_report["source_heads"]
    assert beta_report["status"] in {"accepted", "rejected"}
    assert beta_report["train_cache_binding"]["split"] == "distill_train"
    assert beta_report["validation_cache_binding"]["split"] == "validation"
    assert sum(row["rows"] for row in beta_report["rank_contributions"]) == 3
    assert len(beta_report["selected_parameter_hashes"]) == 1
    assert (
        len(beta_report["validation_after"]["per_head_normalized_mse"])
        == (beta_report["target_heads"])
    )
    expected_erase = (
        (
            beta_report["proposed_down_weight_sha256"],
            beta_report["proposed_up_weight_sha256"],
            beta_report["proposed_up_bias_sha256"],
        )
        if beta_report["status"] == "accepted"
        else (
            beta_report["original_down_weight_sha256"],
            beta_report["original_up_weight_sha256"],
            beta_report["original_up_bias_sha256"],
        )
    )
    assert (
        beta_report["selected_down_weight_sha256"],
        beta_report["selected_up_weight_sha256"],
        beta_report["selected_up_bias_sha256"],
    ) == expected_erase
    for expected_stage, expected_statuses, gate_stage_report in (
        ("initial", {"accepted", "rejected"}, gate_report),
        ("dependency-transaction", {"deferred"}, gate_dependency_report),
    ):
        assert gate_stage_report["boundary"] == "gdn-silu-z-to-native-gate-up-v1"
        assert gate_stage_report["fit_stage"] == expected_stage
        assert gate_stage_report["status"] in expected_statuses
        assert len(gate_stage_report["selected_parameter_hashes"]) == 1
        assert gate_stage_report["functional_fit"]["status"] in {
            "accepted",
            "rejected",
            "not-run",
        }
        assert gate_stage_report["train_cache_binding"]["split"] == "distill_train"
        assert gate_stage_report["validation_cache_binding"]["split"] == "validation"
        if gate_stage_report["status"] == "accepted":
            assert (
                gate_stage_report["validation_after"]["normalized_mse"]
                < gate_stage_report["validation_before"]["normalized_mse"]
            )
            assert (
                gate_stage_report["selected_weight_sha256"]
                == gate_stage_report["proposed_weight_sha256"]
            )
            assert (
                gate_stage_report["selected_down_weight_sha256"]
                == gate_stage_report["proposed_down_weight_sha256"]
            )
        elif gate_stage_report["status"] == "deferred":
            assert (
                gate_stage_report["selected_weight_sha256"]
                == gate_stage_report["proposed_weight_sha256"]
            )
            assert (
                gate_stage_report["selected_down_weight_sha256"]
                == gate_stage_report["proposed_down_weight_sha256"]
            )
        else:
            assert (
                gate_stage_report["selected_weight_sha256"]
                == gate_stage_report["original_weight_sha256"]
            )
            assert (
                gate_stage_report["selected_down_weight_sha256"]
                == gate_stage_report["original_down_weight_sha256"]
            )
        assert gate_stage_report["selected_parameter_hashes"][0] == {
            "rank": 0,
            "down_weight_sha256": gate_stage_report["selected_down_weight_sha256"],
            "up_weight_sha256": gate_stage_report["selected_weight_sha256"],
        }
        if gate_stage_report["functional_fit"]["status"] == "accepted":
            functional = gate_stage_report["functional_fit"]
            assert (
                functional["validation_after"]["normalized_mse"]
                < functional["validation_before"]["normalized_mse"]
            )
            assert (
                functional["layer_validation_after"]["mixer_normalized_mse"]
                <= functional["layer_validation_before"]["mixer_normalized_mse"]
            )
    assert norm_report["boundary"] == "source-pre-output-to-native-groupnorm-affine-v1"
    assert norm_report["status"] in {"accepted", "rejected"}
    assert norm_report["train_cache_binding"]["split"] == "distill_train"
    assert norm_report["validation_cache_binding"]["split"] == "validation"
    assert sum(row["rows"] for row in norm_report["rank_contributions"]) == 3
    assert len(norm_report["selected_parameter_hashes"]) == 1
    if norm_report["status"] == "accepted":
        assert (
            norm_report["selected_weight_sha256"]
            == norm_report["proposed_weight_sha256"]
        )
        assert (
            norm_report["selected_bias_sha256"] == norm_report["proposed_bias_sha256"]
        )
    else:
        assert (
            norm_report["selected_weight_sha256"]
            == norm_report["original_weight_sha256"]
        )
        assert (
            norm_report["selected_bias_sha256"] == norm_report["original_bias_sha256"]
        )
    assert time_mix_report["boundary"] == "source-mixer-output-to-native-time-mix-v1"
    assert time_mix_report["fit_stage"] == "dependency-transaction"
    assert time_mix_report["status"] == "deferred"
    assert time_mix_report["gradient_scope"] == ["x_r", "x_k", "x_v"]
    assert time_mix_report["train_cache_binding"]["split"] == "distill_train"
    assert time_mix_report["validation_cache_binding"]["split"] == "validation"
    expected_time_mix_hashes = (
        time_mix_report["proposed_parameter_sha256"]
        if time_mix_report["status"] in {"accepted", "deferred"}
        else time_mix_report["original_parameter_sha256"]
    )
    assert time_mix_report["selected_parameter_sha256"] == expected_time_mix_hashes
    assert decay_report["boundary"] == "gdn-decay-logit-to-native-w-up-v1"
    assert decay_report["status"] in {"accepted", "rejected"}
    assert decay_report["train_cache_binding"]["split"] == "distill_train"
    assert decay_report["validation_cache_binding"]["split"] == "validation"
    assert sum(row["rows"] for row in decay_report["rank_contributions"]) == 3
    assert len(decay_report["selected_parameter_hashes"]) == 1
    assert len(decay_report["validation_after"]["per_head_normalized_mse"]) > 0
    expected_decay = (
        (
            decay_report["proposed_weight_sha256"],
            decay_report["proposed_bias_sha256"],
        )
        if decay_report["status"] == "accepted"
        else (
            decay_report["original_weight_sha256"],
            decay_report["original_bias_sha256"],
        )
    )
    assert (
        decay_report["selected_weight_sha256"],
        decay_report["selected_bias_sha256"],
    ) == expected_decay
    if decay_report["status"] == "accepted":
        assert (
            decay_report["validation_after"]["decay_normalized_mse"]
            < (decay_report["validation_before"]["decay_normalized_mse"])
        )
        assert all(
            after <= before
            for before, after in zip(
                decay_report["validation_before"]["per_head_normalized_mse"],
                decay_report["validation_after"]["per_head_normalized_mse"],
                strict=True,
            )
        )
        assert (
            decay_report["layer_validation_after"]["mixer_normalized_mse"]
            <= (decay_report["layer_validation_before"]["mixer_normalized_mse"])
        )
    assert dependency_report["boundary"] == (
        "gdn-time-mix-dependent-activation-fit-generation-v2"
    )
    assert dependency_report["status"] in {"accepted", "rejected"}
    assert len(dependency_report["component_report_sha256"]) == 5
    expected_dependency_state = (
        dependency_report["proposed_parameter_sha256"]
        if dependency_report["status"] == "accepted"
        else dependency_report["baseline_parameter_sha256"]
    )
    assert dependency_report["selected_parameter_sha256"] == (expected_dependency_state)
    assert _dependency_transaction_improves(
        dependency_report["validation_baseline"],
        dependency_report["validation_candidate"],
    ) is (dependency_report["status"] == "accepted")
    for component_name, expected_sha256 in dependency_report[
        "component_report_sha256"
    ].items():
        component_path = run_dir / "activation-fit" / component_name
        assert file_sha256(component_path) == expected_sha256


@pytest.mark.parametrize(
    ("candidate", "expected"),
    (
        (
            {"loss": 0.9, "normalized_mse": 0.9, "block_normalized_mse": 1.0},
            True,
        ),
        (
            {"loss": 0.9, "normalized_mse": 0.9, "block_normalized_mse": 1.1},
            False,
        ),
        (
            {"loss": 1.0, "normalized_mse": 0.9, "block_normalized_mse": 0.9},
            False,
        ),
        (
            {"loss": math.nan, "normalized_mse": 0.9, "block_normalized_mse": 0.9},
            False,
        ),
    ),
)
def test_dependency_transaction_requires_joint_held_out_improvement(
    candidate: dict[str, float], expected: bool
) -> None:
    baseline = {
        "loss": 1.0,
        "normalized_mse": 1.0,
        "block_normalized_mse": 1.0,
    }

    assert _dependency_transaction_improves(baseline, candidate) is expected


@pytest.mark.parametrize(
    ("candidate", "expected"),
    (
        (
            {
                "loss": 0.9,
                "mixer_normalized_mse": 0.9,
                "block_normalized_mse": 1.0,
            },
            True,
        ),
        (
            {
                "loss": 0.9,
                "mixer_normalized_mse": 0.9,
                "block_normalized_mse": 1.01,
            },
            False,
        ),
        (
            {
                "loss": 0.9,
                "mixer_normalized_mse": 1.0,
                "block_normalized_mse": 0.9,
            },
            False,
        ),
        (
            {
                "loss": math.nan,
                "mixer_normalized_mse": 0.9,
                "block_normalized_mse": 0.9,
            },
            False,
        ),
    ),
)
def test_gqa_native_installation_requires_block_non_regression(
    candidate: dict[str, float],
    expected: bool,
) -> None:
    baseline = {
        "loss": 1.0,
        "mixer_normalized_mse": 1.0,
        "block_normalized_mse": 1.0,
    }

    assert _gqa_native_validation_improves(baseline, candidate) is expected


@pytest.mark.parametrize(
    ("signal_after", "layer_after", "expected"),
    ((0.5, 1.0, True), (0.5, 1.01, False), (1.0, 0.9, False)),
)
def test_gate_fit_candidate_is_installed_or_rolled_back_transactionally(
    signal_after: float,
    layer_after: float,
    expected: bool,
) -> None:
    original = torch.tensor([[1.0, 2.0]])
    candidate = torch.tensor([[3.0, 4.0]])
    weight = torch.nn.Parameter(candidate.clone())

    accepted = _retain_gate_fit_candidate(
        weight=weight,
        original_weight=original,
        validation_before={"normalized_mse": 1.0},
        validation_after={"normalized_mse": signal_after},
        layer_validation_before={"mixer_normalized_mse": 1.0},
        layer_validation_after={"mixer_normalized_mse": layer_after},
    )

    assert accepted is expected
    assert torch.equal(weight.detach(), candidate if expected else original)


def test_activation_fit_rejects_lossy_gdn_width_compression(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="source GDN width"):
        _prepare_fixture(
            tmp_path,
            config_overrides={
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
            },
        )


def test_fixed_epoch_budget_continues_after_convergence_for_equal_tokens(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    plan = _plan()
    plan.layer_fixed_epochs = 3
    plan.layer_max_epochs = 3
    plan.layer_min_delta = 3.0
    run_dir = tmp_path / "fixed-budget"

    result = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=run_dir,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        plan=plan,
    )

    assert result["status"] == "layerwise-local-complete"
    assert len(result["history"]) == 3
    assert result["history"][1]["converged"] is True
    assert result["history"][1]["fixed_budget_complete"] is False
    assert result["history"][2]["convergence_observed"] is True
    assert result["history"][2]["fixed_budget_complete"] is True
    assert (
        sum(row["training_budget"]["loss_token_count"] for row in result["history"])
        == 3 * 3 * plan.supervised_tokens
    )


def test_exploratory_layer_limit_stops_without_full_checkpoint_or_next_cache(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path, layers=2)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    plan = _plan()
    plan.exploratory_layer_limit = 1
    plan.layer_fixed_epochs = 3
    plan.layer_max_epochs = 3
    plan.layer_min_delta = 3.0
    run_dir = tmp_path / "layer-calibration"

    def interrupt_after_durable_completion(phase: str, _path: Path) -> None:
        if phase == "calibration-complete":
            assert (run_dir / "layer-convergence.json").is_file()
            raise RuntimeError("interrupt-after-calibration-complete")

    with pytest.raises(RuntimeError, match="interrupt-after-calibration-complete"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            plan=plan,
            callback=interrupt_after_durable_completion,
        )

    progress = json.loads((run_dir / "layer-major-progress.json").read_text())
    assert progress["status"] == "exploratory-layer-calibration-complete"
    assert progress["phase"] == "calibration-complete"
    assert progress["active_layer"] == 1
    assert not (run_dir / "checkpoint-layerwise-local").exists()
    assert not (run_dir / "checkpoint-global-corrective").exists()
    assert not (run_dir / "layer-input-cache" / "layer-001").exists()

    resumed = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=run_dir,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        resume=run_dir / "layer-major-progress.json",
        plan=plan,
    )
    assert resumed["status"] == "exploratory-layer-calibration-complete"
    assert resumed["layers_completed"] == 1
    assert resumed["total_layers"] == 2
    assert resumed["next_stage"] == "compare-training-controls"
    assert "checkpoint" not in resumed


def test_explicit_missing_resume_is_rejected(tmp_path: Path) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")

    with pytest.raises(ContractError, match="explicit resume progress is missing"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=tmp_path / "missing-resume",
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            resume=tmp_path / "does-not-exist.json",
        )


def test_initialized_run_binds_source_and_complete_zero_step(tmp_path: Path) -> None:
    source, zero_step, _ = _prepare_fixture(tmp_path)
    run_dir = tmp_path / "initialized-run"
    run_dir.mkdir()
    zero_step_binding = _zero_step_checkpoint_binding(zero_step)
    metadata = {
        "source": {"files": source.file_hashes},
        "zero_step": {
            "binding": zero_step_binding,
            "sha256": _binding_sha256(zero_step_binding),
        },
        "warm_start_plan": {
            "path": "warm-start-plan.json",
            "sha256": file_sha256(zero_step / "warm-start-plan.json"),
        },
    }
    shutil.copy2(zero_step / "warm-start-plan.json", run_dir / "warm-start-plan.json")
    write_json(run_dir / "metadata.json", metadata)

    _validate_initialized_run_binding(
        run_dir=run_dir, source_manifest=source, zero_step=zero_step
    )

    metadata["source"]["files"] = {"config.json": "0" * 64}
    write_json(run_dir / "metadata.json", metadata)
    with pytest.raises(ContractError, match="source files differ"):
        _validate_initialized_run_binding(
            run_dir=run_dir, source_manifest=source, zero_step=zero_step
        )

    metadata["source"]["files"] = source.file_hashes
    write_json(run_dir / "metadata.json", metadata)
    mapping = zero_step / "mapping.json"
    mapping.write_text(mapping.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="zero-step checkpoint differs"):
        _validate_initialized_run_binding(
            run_dir=run_dir, source_manifest=source, zero_step=zero_step
        )


def test_initialized_run_rejects_changed_warm_start_plan(tmp_path: Path) -> None:
    source, zero_step, _ = _prepare_fixture(tmp_path)
    run_dir = tmp_path / "initialized-run"
    run_dir.mkdir()
    shutil.copy2(zero_step / "warm-start-plan.json", run_dir / "warm-start-plan.json")
    zero_step_binding = _zero_step_checkpoint_binding(zero_step)
    metadata = {
        "source": {"files": source.file_hashes},
        "zero_step": {
            "binding": zero_step_binding,
            "sha256": _binding_sha256(zero_step_binding),
        },
        "warm_start_plan": {
            "path": "warm-start-plan.json",
            "sha256": file_sha256(run_dir / "warm-start-plan.json"),
        },
    }
    write_json(run_dir / "metadata.json", metadata)
    (run_dir / "warm-start-plan.json").write_text(
        (run_dir / "warm-start-plan.json").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="warm-start plan differs"):
        _validate_initialized_run_binding(
            run_dir=run_dir, source_manifest=source, zero_step=zero_step
        )


def test_recipe_corrective_rejects_exploratory_layer_limit() -> None:
    recipe = resolve_recipe("qwen35_to_rwkv7").recipe
    request = SimpleNamespace(plan=SimpleNamespace(exploratory_layer_limit=1))
    with pytest.raises(ContractError, match="forbids exploratory_layer_limit"):
        recipe.run_corrective_distillation(request)


def test_resume_rejects_changed_zero_step_checkpoint(tmp_path: Path) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    run_dir = tmp_path / "changed-zero-step"

    def interrupt(phase: str, _path: Path) -> None:
        if phase == "epoch-complete":
            raise RuntimeError("interrupt-after-epoch")

    with pytest.raises(RuntimeError, match="interrupt-after-epoch"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            callback=interrupt,
        )

    mapping = zero_step / "mapping.json"
    mapping.write_text(mapping.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="resume progress binding differs"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            resume=run_dir / "layer-major-progress.json",
        )


def test_resume_rejects_changed_warm_start_plan(tmp_path: Path) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    run_dir = tmp_path / "changed-warm-start"

    def interrupt(phase: str, _path: Path) -> None:
        if phase == "epoch-complete":
            raise RuntimeError("interrupt-after-epoch")

    with pytest.raises(RuntimeError, match="interrupt-after-epoch"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            callback=interrupt,
        )

    plan_path = run_dir / "warm-start-plan.json"
    plan_path.write_text(plan_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="resume progress binding differs"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            resume=run_dir / "layer-major-progress.json",
        )


def test_resume_rejects_changed_local_trainable_set(tmp_path: Path) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    run_dir = tmp_path / "changed-local-trainable"

    def interrupt(phase: str, _path: Path) -> None:
        if phase == "epoch-complete":
            raise RuntimeError("interrupt-after-epoch")

    with pytest.raises(RuntimeError, match="interrupt-after-epoch"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            callback=interrupt,
        )

    changed_trainable = [set(names) for names in trainable]
    assert "r_k" in changed_trainable[0]
    changed_trainable[0].remove("r_k")
    with pytest.raises(ContractError, match="resume progress binding differs"):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=changed_trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            resume=run_dir / "layer-major-progress.json",
        )


def test_mid_epoch_resume_matches_uninterrupted_digest(tmp_path: Path) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")

    torch.manual_seed(123)
    uninterrupted = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=tmp_path / "uninterrupted",
        training_config=training_config,
        dataset_manifest=dataset_manifest,
    )
    assert len(tuple((tmp_path / "uninterrupted" / "layer-generations").iterdir())) <= 2

    interrupted_once = False

    def interrupt(phase: str, _path: Path) -> None:
        nonlocal interrupted_once
        if phase == "train" and not interrupted_once:
            interrupted_once = True
            raise PlannedInterruption("durable mid-epoch interruption")

    torch.manual_seed(123)
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(PlannedInterruption):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=interrupted_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            callback=interrupt,
        )
    resumed = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=interrupted_dir,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        resume=interrupted_dir / "layer-major-progress.json",
    )
    assert len(tuple((interrupted_dir / "layer-generations").iterdir())) <= 2

    reference_store = RWKV7MixerLayerStore(
        zero_step, tmp_path / "uninterrupted" / "mixer-overlays"
    )
    resumed_store = RWKV7MixerLayerStore(zero_step, interrupted_dir / "mixer-overlays")
    assert resumed_store.fingerprint() == reference_store.fingerprint()
    assert resumed["optimizer_steps"] == uninterrupted["optimizer_steps"]
    assert resumed["history"] == uninterrupted["history"]


def test_layer_transition_resume_matches_uninterrupted_and_cleans_old_cache(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path, layers=2)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")

    torch.manual_seed(456)
    uninterrupted = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=tmp_path / "uninterrupted",
        training_config=training_config,
        dataset_manifest=dataset_manifest,
    )

    interrupted_once = False

    def interrupt(phase: str, _path: Path) -> None:
        nonlocal interrupted_once
        if phase == "epoch-complete" and not interrupted_once:
            assert not (interrupted_dir / "layer-input-cache/layer-001").exists()
        if phase == "layer-ready" and not interrupted_once:
            interrupted_once = True
            raise PlannedInterruption("durable layer transition interruption")

    torch.manual_seed(456)
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(PlannedInterruption):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=interrupted_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            callback=interrupt,
        )
    assert (interrupted_dir / "layer-input-cache/layer-000").is_dir()
    resumed = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=interrupted_dir,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        resume=interrupted_dir / "layer-major-progress.json",
    )
    assert not (interrupted_dir / "layer-input-cache/layer-000").exists()
    reference_store = RWKV7MixerLayerStore(
        zero_step, tmp_path / "uninterrupted" / "mixer-overlays"
    )
    resumed_store = RWKV7MixerLayerStore(zero_step, interrupted_dir / "mixer-overlays")
    assert resumed_store.fingerprint() == reference_store.fingerprint()
    assert resumed["optimizer_steps"] == uninterrupted["optimizer_steps"]
    assert resumed["history"] == uninterrupted["history"]


def test_layer_transition_releases_old_source_and_target_before_next_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path, layers=2)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    plan = _plan()
    plan.layer_fixed_epochs = 3
    plan.layer_max_epochs = 3
    plan.layer_min_delta = 3.0

    source_refs: list[weakref.ReferenceType] = []
    target_refs: list[weakref.ReferenceType] = []
    original_source_load = Qwen35TeacherLayerLoader.load_layer
    original_base_load = RWKV7MixerLayerStore.load_base_mixer
    original_target_load = RWKV7MixerLayerStore.load_mixer

    def require_released(refs: list[weakref.ReferenceType], label: str) -> None:
        gc.collect()
        assert not [reference for reference in refs if reference() is not None], label
        refs.clear()

    def tracked_source_load(self, *args, **kwargs):
        require_released(source_refs, "previous source layer remained resident")
        value = original_source_load(self, *args, **kwargs)
        source_refs.append(weakref.ref(value))
        return value

    def tracked_base_load(self, *args, **kwargs):
        require_released(target_refs, "previous target mixer remained resident")
        value = original_base_load(self, *args, **kwargs)
        target_refs.append(weakref.ref(value))
        return value

    def tracked_target_load(self, *args, **kwargs):
        require_released(target_refs, "previous target mixer remained resident")
        value = original_target_load(self, *args, **kwargs)
        target_refs.append(weakref.ref(value))
        return value

    monkeypatch.setattr(Qwen35TeacherLayerLoader, "load_layer", tracked_source_load)
    monkeypatch.setattr(RWKV7MixerLayerStore, "load_base_mixer", tracked_base_load)
    monkeypatch.setattr(RWKV7MixerLayerStore, "load_mixer", tracked_target_load)

    result = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=tmp_path / "resident-boundary",
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        plan=plan,
    )

    assert result["status"] == "layerwise-local-complete"
    require_released(source_refs, "final source layer remained resident")
    require_released(target_refs, "final target mixer remained resident")


def test_epoch_validation_commit_resume_matches_uninterrupted(tmp_path: Path) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")

    torch.manual_seed(789)
    uninterrupted = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=tmp_path / "uninterrupted",
        training_config=training_config,
        dataset_manifest=dataset_manifest,
    )

    interrupted_once = False

    def interrupt(phase: str, _path: Path) -> None:
        nonlocal interrupted_once
        if phase == "epoch-complete" and not interrupted_once:
            interrupted_once = True
            raise PlannedInterruption("durable epoch validation interruption")

    torch.manual_seed(789)
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(PlannedInterruption):
        _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=interrupted_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            callback=interrupt,
        )
    resumed = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=interrupted_dir,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        resume=interrupted_dir / "layer-major-progress.json",
    )
    reference_store = RWKV7MixerLayerStore(
        zero_step, tmp_path / "uninterrupted" / "mixer-overlays"
    )
    resumed_store = RWKV7MixerLayerStore(zero_step, interrupted_dir / "mixer-overlays")
    assert resumed_store.fingerprint() == reference_store.fingerprint()
    assert resumed["optimizer_steps"] == uninterrupted["optimizer_steps"]
    assert resumed["history"] == uninterrupted["history"]


def test_fully_recurrent_global_corrective_runs_reverse_sweep_and_exports(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path, layers=2)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    run_dir = tmp_path / "run"
    torch.manual_seed(321)
    local = _run(
        source=source,
        zero_step=zero_step,
        trainable=trainable,
        run_dir=run_dir,
        training_config=training_config,
        dataset_manifest=dataset_manifest,
    )
    assert local["status"] == "layerwise-local-complete"
    local_fingerprint = RWKV7MixerLayerStore(
        zero_step, run_dir / "mixer-overlays"
    ).fingerprint()
    result = run_global_corrective(
        source_manifest=source,
        run_dir=run_dir,
        zero_step_dir=zero_step,
        token_rows=((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)),
        validation_rows=((13, 14, 15, 16), (17, 18, 19, 20)),
        plan=_plan(),
        training_config=training_config,
        dataset_manifest=dataset_manifest,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert result["status"] == "fully-recurrent-global-corrective-complete"
    assert result["history"][0]["order"] == "1..0"
    assert result["history"][0]["token_budget"] == 24
    validation = result["history"][0]["validation"]
    assert validation["kl_token_count"] == 6
    assert validation["nll_token_count"] == 6
    assert validation["token_kl"] == pytest.approx(
        validation["token_kl_sum"] / validation["kl_token_count"]
    )
    assert validation["ppl_ratio"] == pytest.approx(
        math.exp(validation["nll_delta_sum"] / validation["nll_token_count"])
    )
    config = json.loads(
        (run_dir / "checkpoint-global-corrective/config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["any2rwkv"]["training_stage"] == "fully-recurrent-global-corrective"
    assert (
        config["any2rwkv"]["mixer_overlay_fingerprint"]
        == result["selected_fingerprint"]
    )
    if result["selected_checkpoint"] == "global-sweeps/pre-sweep":
        assert result["selected_fingerprint"] == local_fingerprint
    residency = json.loads((run_dir / "corrective-residency.json").read_text())
    assert residency["mode"] == "resident"
    assert residency["world_size"] == 1
    assert residency["per_rank"]["teacher_layer_loads"] == [2.0]
    assert residency["per_rank"]["teacher_cache_hits"][0] > 0
    assert residency["per_rank"]["student_mixer_loads"][0] <= 2
    assert residency["per_rank"]["student_mixer_cache_hits"][0] > 0
    assert len(tuple((run_dir / "global-generations").iterdir())) <= 2
    assert len(tuple((run_dir / "global-snapshots").iterdir())) <= 3
    assert (
        json.loads((run_dir / "global-generation-retention.json").read_text())["policy"]
        == "latest-generation-per-layer"
    )
    assert (
        json.loads((run_dir / "global-snapshot-retention.json").read_text())["policy"]
        == "pre-sweep-current-start-best"
    )
    parent = _select_parent_recurrent_checkpoint(run_dir)
    assert parent == (run_dir / "checkpoint-global-corrective").resolve()
    binding = _checkpoint_binding(parent)
    assert binding["mixer_fingerprint"] == result["selected_fingerprint"]
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source": {
                    "path": str(source.path),
                    "files": source.file_hashes,
                    "layers": source.contract.num_hidden_layers,
                    "classification": "fixture-extra-metadata",
                },
                "recipe": {
                    "id": "qwen35_to_rwkv7",
                    "source_adapter": "qwen35",
                    "target_adapter": "rwkv7",
                },
                "precision": "bf16",
            }
        ),
        encoding="utf-8",
    )
    _validate_parent_run(
        parent_run=run_dir,
        parent_checkpoint=parent,
        source_manifest=source,
        recipe=resolve_recipe("qwen35_to_rwkv7"),
        precision="bf16",
    )
    continuation = tmp_path / "corrective-continuation"
    continuation.mkdir()
    _materialize_corrective_base(
        parent_checkpoint=parent,
        output=continuation,
        layer_count=2,
    )
    continuation_store = RWKV7MixerLayerStore(parent, continuation / "mixer-overlays")
    assert continuation_store.fingerprint() == result["selected_fingerprint"]
    atomic_continuation = tmp_path / "atomic-corrective-continuation"
    resolved = resolve_recipe("qwen35_to_rwkv7")
    _prepare_or_validate_corrective_output(
        output=atomic_continuation,
        parent_run=run_dir,
        parent_checkpoint=parent,
        parent_binding=binding,
        source_manifest=source,
        recipe=resolved,
        plan_path=training_config,
        dataset_manifest=dataset_manifest,
        precision="bf16",
        run_id="atomic-corrective-continuation",
        rwkv_hf_sha="1" * 40,
        rwkv_lm_sha="2" * 40,
    )
    # Identical bindings resume the atomically published base without rewriting it.
    _prepare_or_validate_corrective_output(
        output=atomic_continuation,
        parent_run=run_dir,
        parent_checkpoint=parent,
        parent_binding=binding,
        source_manifest=source,
        recipe=resolved,
        plan_path=training_config,
        dataset_manifest=dataset_manifest,
        precision="bf16",
        run_id="atomic-corrective-continuation",
        rwkv_hf_sha="1" * 40,
        rwkv_lm_sha="2" * 40,
    )
    assert not (
        tmp_path / ".atomic-corrective-continuation.corrective-base.tmp"
    ).exists()
    assert (
        RWKV7MixerLayerStore(
            parent, atomic_continuation / "mixer-overlays"
        ).fingerprint()
        == result["selected_fingerprint"]
    )


def test_global_generation_replay_reuses_published_transaction(tmp_path: Path) -> None:
    _, zero_step, _ = _prepare_fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = RWKV7MixerLayerStore(zero_step, run_dir / "overlays")
    mixer = store.load_mixer(0, device="cpu", dtype=torch.float32)
    cursor = {
        "schedule": "fully-recurrent-global-corrective-v1",
        "sweep_index": 0,
        "visit_index": 0,
        "layer_index": 0,
        "permutation_sha256": "0" * 64,
        "row_count": 3,
    }
    distributed = DistributedContext(rank=0, local_rank=0, world_size=1)

    first = _commit_distributed_layer_generation(
        distributed=distributed,
        run_dir=run_dir,
        store=store,
        mixer=mixer,
        optimizer_snapshot=_empty_optimizer_snapshot(0),
        cursor=cursor,
    )
    first_integrity = file_sha256(first / "integrity.json")
    second = _commit_distributed_layer_generation(
        distributed=distributed,
        run_dir=run_dir,
        store=store,
        mixer=mixer,
        optimizer_snapshot=_empty_optimizer_snapshot(0),
        cursor=cursor,
    )

    assert second == first
    assert file_sha256(second / "integrity.json") == first_integrity
    assert not second.with_name(second.name + ".tmp").exists()


def test_global_generation_rank_write_failure_is_reported_without_publish(
    tmp_path: Path, monkeypatch
) -> None:
    _, zero_step, _ = _prepare_fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = RWKV7MixerLayerStore(zero_step, run_dir / "overlays")
    mixer = store.load_mixer(0, device="cpu", dtype=torch.float32)
    cursor = {
        "schedule": "fully-recurrent-global-corrective-v1",
        "sweep_index": 0,
        "visit_index": 0,
        "layer_index": 0,
        "permutation_sha256": "0" * 64,
        "row_count": 3,
    }
    distributed = DistributedContext(rank=0, local_rank=0, world_size=1)
    monkeypatch.setattr(
        "any2rwkv.recipes.qwen35_to_rwkv7.global_corrective_runner._write_rank_training_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(ContractError, match="rank-state write failed.*disk full"):
        _commit_distributed_layer_generation(
            distributed=distributed,
            run_dir=run_dir,
            store=store,
            mixer=mixer,
            optimizer_snapshot=_empty_optimizer_snapshot(0),
            cursor=cursor,
        )

    destination = run_dir / "global-generations" / "s000-v000-l000"
    assert not destination.exists()
    assert destination.with_name(destination.name + ".tmp").is_dir()


def test_global_corrective_resume_after_layer_commit_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    source, zero_step, trainable = _prepare_fixture(tmp_path, layers=2)
    training_config = tmp_path / "training.json"
    dataset_manifest = tmp_path / "dataset.json"
    training_config.write_text(json.dumps({"fixture": "training"}), encoding="utf-8")
    dataset_manifest.write_text(json.dumps({"fixture": "dataset"}), encoding="utf-8")
    rows = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12))
    validation = ((13, 14, 15, 16), (17, 18, 19, 20))

    def prepare(run_dir: Path) -> None:
        torch.manual_seed(321)
        local = _run(
            source=source,
            zero_step=zero_step,
            trainable=trainable,
            run_dir=run_dir,
            training_config=training_config,
            dataset_manifest=dataset_manifest,
        )
        assert local["status"] == "layerwise-local-complete"

    def global_run(run_dir: Path, callback=None):
        return run_global_corrective(
            source_manifest=source,
            run_dir=run_dir,
            zero_step_dir=zero_step,
            token_rows=rows,
            validation_rows=validation,
            plan=_plan(),
            training_config=training_config,
            dataset_manifest=dataset_manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
            progress_callback=callback,
        )

    uninterrupted_dir = tmp_path / "uninterrupted-global"
    prepare(uninterrupted_dir)
    uninterrupted = global_run(uninterrupted_dir)

    interrupted_dir = tmp_path / "interrupted-global"
    prepare(interrupted_dir)
    interrupted_once = False

    def interrupt(phase: str, progress_path: Path) -> None:
        nonlocal interrupted_once
        if phase == "global-layer-committed" and not interrupted_once:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            assert progress["next_visit"] == 1
            interrupted_once = True
            raise PlannedInterruption("durable global layer interruption")

    with pytest.raises(PlannedInterruption):
        global_run(interrupted_dir, interrupt)
    resumed = global_run(interrupted_dir)
    assert resumed["selected_fingerprint"] == uninterrupted["selected_fingerprint"]
    assert resumed["history"] == uninterrupted["history"]
