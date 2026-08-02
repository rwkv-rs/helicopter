#!/usr/bin/env python3
from __future__ import annotations

# The RWKV/CUDA environment must be fixed before importing any2rwkv modules.
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

for name, value in {
    "RWKV_JIT_ON": "0",
    "RWKV_MY_TESTING": "x070",
    "RWKV_KERNEL": "",
    "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
    "RWKV_TRAIN_TYPE": "infctx",
    "RWKV_FLOAT_MODE": "bf16",
    "WKV_MODE": "fp32io16",
}.items():
    os.environ.setdefault(name, value)

from any2rwkv.checkpoint import read_checkpoint
from any2rwkv.artifacts import file_sha256, write_json
from any2rwkv.contract import build_target_config
from any2rwkv.distributed import DistributedContext
from any2rwkv.export import export_hf_checkpoint
from any2rwkv.fixture import write_fixture
from any2rwkv.migration_init import (
    WarmStartTensorProvider,
    WarmStartVariant,
    apply_warm_start_plan,
    plan_warm_start,
)
from any2rwkv.mixer_store import RWKV7MixerLayerStore
from any2rwkv.recipes.qwen35_to_rwkv7.layer_major_runner import (
    run_suffix_free_layer_major,
)
from any2rwkv.recipes.qwen35_to_rwkv7.recipe import Qwen35ToRWKV7Recipe
from any2rwkv.recipes.qwen35_to_rwkv7.global_corrective_runner import (
    run_global_corrective,
)
from any2rwkv.target import build_zero_step_ledger, rwkv7_mixer_specs


def plan() -> SimpleNamespace:
    return SimpleNamespace(
        cache_teacher_layers=False,
        corrective_resident_model_max_bytes=1_000_000_000,
        distributed_world_size=8,
        seed=20260714,
        cache_shard_rows=8,
        max_layer_input_cache_bytes=100_000_000,
        learning_rate=1e-4,
        burn_in_tokens=0,
        supervised_tokens=4,
        accumulation_steps=2,
        micro_batch_size=1,
        checkpoint_interval_micro_batches=1,
        activation_fit_rows=16,
        activation_fit_ridge=0.001,
        activation_fit_time_mix_steps=4,
        activation_fit_time_mix_learning_rate=0.001,
        gradient_clip_norm=None,
        max_parameter_update_relative_l2=None,
        layer_min_epochs=1,
        layer_max_epochs=2,
        layer_min_delta=1e9,
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


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    if contiguous.dtype == torch.bfloat16:
        contiguous = contiguous.view(torch.uint16)
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_rank_contributions(
    rows: object, *, fit_rows: int, evidence: Path
) -> list[dict[str, object]]:
    if not isinstance(rows, list) or len(rows) != 8:
        raise SystemExit(f"rank contribution count is invalid: {evidence}")
    ordered = sorted(rows, key=lambda row: row.get("rank", -1))
    if [row.get("rank") for row in ordered] != list(range(8)):
        raise SystemExit(f"rank contribution identities are invalid: {evidence}")
    for rank, row in enumerate(ordered):
        expected_indices = list(range(fit_rows))[rank::8]
        if (
            row.get("rows") != len(expected_indices)
            or row.get("tokens", 0) <= 0
            or row.get("row_indices_sha256") != json_sha256(expected_indices)
            or not row.get("local_gram_sha256")
            or not row.get("local_rhs_sha256")
            or not row.get("local_target_squared_sum_sha256")
        ):
            raise SystemExit(f"rank contribution payload is invalid: {evidence}")
    if sum(int(row["rows"]) for row in ordered) != fit_rows:
        raise SystemExit(f"rank row coverage is invalid: {evidence}")
    return ordered


def frozen_parameter_sha256(
    mixer: torch.nn.Module, trainable_names: set[str]
) -> dict[str, str]:
    parameters = dict(mixer.named_parameters())
    unknown = sorted(trainable_names - parameters.keys())
    if unknown:
        raise SystemExit(f"production trainable set has unknown parameters: {unknown}")
    return {
        name: tensor_sha256(parameter)
        for name, parameter in parameters.items()
        if name not in trainable_names
    }


def validate_frozen_parameters(
    checkpoint: Path,
    *,
    trainable: list[set[str]],
    expected: list[dict[str, str]],
    probe_dir: Path,
) -> None:
    store = RWKV7MixerLayerStore(checkpoint, probe_dir)
    for layer_index, expected_hashes in enumerate(expected):
        mixer = store.load_base_mixer(
            layer_index, device="cpu", dtype=torch.bfloat16
        )
        actual = frozen_parameter_sha256(mixer, trainable[layer_index])
        if actual != expected_hashes:
            changed = sorted(
                name
                for name in set(actual) | set(expected_hashes)
                if actual.get(name) != expected_hashes.get(name)
            )
            raise SystemExit(
                f"locally frozen parameters changed in layer {layer_index}: {changed}"
            )


def validate_fitted_checkpoint(path: Path, *, expected_layers: int) -> None:
    mapping = json.loads((path / "mapping.json").read_text())
    coverage = json.loads((path / "mapping-coverage.json").read_text())
    provenance = json.loads((path / "activation-fit-provenance.json").read_text())
    materialization = json.loads((path / "materialization.json").read_text())
    fitted = [
        row for row in mapping.get("targets", []) if row.get("provenance") == "fitted"
    ]
    if (
        not fitted
        or any(
            "activation_fit_manifest_sha256=" not in str(row.get("evidence", ""))
            for row in fitted
        )
        or coverage.get("provenance", {}).get("fitted") != len(fitted)
        or coverage.get("activation_fit_manifest_sha256")
        != provenance.get("manifest_sha256")
        or coverage.get("mixer_overlay_fingerprint")
        != provenance.get("mixer_overlay_fingerprint")
        or len(provenance.get("activation_fit_reports", [])) < expected_layers * 5
    ):
        raise SystemExit(f"trained mapping provenance is incomplete: {path}")
    files = materialization.get("files", {})
    for name in (
        "mapping.json",
        "mapping-coverage.json",
        "activation-fit-provenance.json",
    ):
        if files.get(name) != file_sha256(path / name):
            raise SystemExit(f"materialization did not bind final {name}: {path}")


def prepare(root: Path, *, layers: int, gdn_source_head_dim: int | None) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    overrides = None
    if gdn_source_head_dim is not None:
        overrides = {
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": gdn_source_head_dim,
            "linear_value_head_dim": gdn_source_head_dim,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": gdn_source_head_dim,
        }
    source = read_checkpoint(
        write_fixture(root / "source", layers=layers, config_overrides=overrides),
        require_final_layers=False,
    )
    target_config = build_target_config(source.config, require_final_layers=False)
    specs = tuple(
        spec
        for layer_index in range(layers)
        for spec in rwkv7_mixer_specs(
            layer_index,
            hidden_size=64,
            attention_hidden_size=int(target_config["attention_hidden_size"]),
            head_dim=int(target_config["head_dim"]),
        )
    )
    warm_start = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
    export_hf_checkpoint(
        source,
        root / "zero-step",
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
        attention_hidden_size=int(target_config["attention_hidden_size"]),
        source_shard_hashes=shard_hashes,
    )
    apply_warm_start_plan(ledger, warm_start)
    ledger.write(root / "zero-step" / "mapping.json")
    write_json(
        root / "zero-step" / "mapping-coverage.json",
        ledger.validate(source_names, target_names),
    )
    write_json(root / "zero-step" / "warm-start-plan.json", warm_start.to_dict())
    (root / "run").mkdir()
    for name in ("mapping.json", "mapping-coverage.json", "warm-start-plan.json"):
        shutil.copy2(root / "zero-step" / name, root / "run" / name)
    (root / "training.json").write_text('{"distributed_world_size":8}\n', encoding="utf-8")
    (root / "dataset.json").write_text('{"fixture":"distributed"}\n', encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--gdn-source-head-dim", type=int)
    args = parser.parse_args()
    if args.layers <= 0:
        raise SystemExit("--layers must be positive")
    if args.gdn_source_head_dim is not None and args.gdn_source_head_dim < 64:
        raise SystemExit("--gdn-source-head-dim must be at least the native 64")
    context = DistributedContext.initialize()
    root = args.output.resolve()
    if context.is_primary:
        prepare(
            root,
            layers=args.layers,
            gdn_source_head_dim=args.gdn_source_head_dim,
        )
    context.barrier()

    source = read_checkpoint(root / "source", require_final_layers=False)
    target_config = json.loads((root / "zero-step/config.json").read_text())
    expected_head_size = int(target_config.get("head_size", 0))
    if expected_head_size <= 0 or os.environ.get("RWKV_HEAD_SIZE") != str(
        expected_head_size
    ):
        raise SystemExit(
            "RWKV_HEAD_SIZE must match the generated zero-step checkpoint head_size"
        )
    trainable = Qwen35ToRWKV7Recipe._initial_trainable_names(
        root / "zero-step", args.layers
    )
    store = RWKV7MixerLayerStore(
        root / "zero-step", root / f"probe-rank-{context.rank}"
    )
    frozen_reference = []
    for layer_index in range(args.layers):
        mixer = store.load_base_mixer(
            layer_index, device="cpu", dtype=torch.bfloat16
        )
        frozen_reference.append(
            frozen_parameter_sha256(mixer, trainable[layer_index])
        )
    token_rows = tuple(
        tuple((index + offset) % 31 for offset in range(4))
        for index in range(1, 93, 4)
    )
    validation_rows = tuple(
        tuple((index + offset) % 31 for offset in range(4))
        for index in range(101, 161, 4)
    )
    mid_progress = root / "mid-accumulation-progress.json"
    mid_generation = root / "mid-accumulation-generation"

    def capture_mid_accumulation(phase: str, progress_path: Path) -> None:
        if phase == "train" and not mid_progress.exists():
            shutil.copy2(progress_path, mid_progress)
            progress = json.loads(progress_path.read_text())
            shutil.copytree(
                root / "run" / progress["generation"],
                mid_generation,
            )

    result = run_suffix_free_layer_major(
        source_manifest=source,
        run_dir=root / "run",
        zero_step_dir=root / "zero-step",
        token_rows=token_rows,
        validation_rows=validation_rows,
        plan=plan(),
        initial_trainable=trainable,
        training_config=root / "training.json",
        dataset_manifest=root / "dataset.json",
        resume=None,
        dtype=torch.bfloat16,
        progress_callback=capture_mid_accumulation,
    )
    context.barrier()
    if context.is_primary:
        validate_frozen_parameters(
            root / "run" / "checkpoint-layerwise-local",
            trainable=trainable,
            expected=frozen_reference,
            probe_dir=root / "final-frozen-probe",
        )
    resume_parity = False
    if args.layers == 1:
        resume_run = root / "resume-run"
        if context.is_primary:
            resume_run.mkdir()
            for name in (
                "mapping.json",
                "mapping-coverage.json",
                "warm-start-plan.json",
            ):
                shutil.copy2(root / "run" / name, resume_run / name)
            shutil.copytree(root / "run" / "layer-input-cache", resume_run / "layer-input-cache")
            resume_generations = resume_run / "layer-generations"
            resume_generations.mkdir()
            mid = json.loads(mid_progress.read_text())
            shutil.copytree(
                mid_generation,
                resume_run / mid["generation"],
            )
            shutil.copy2(mid_progress, resume_run / "layer-major-progress.json")
        context.barrier()
        resumed = run_suffix_free_layer_major(
            source_manifest=source,
            run_dir=resume_run,
            zero_step_dir=root / "zero-step",
            token_rows=token_rows,
            validation_rows=validation_rows,
            plan=plan(),
            initial_trainable=trainable,
            training_config=root / "training.json",
            dataset_manifest=root / "dataset.json",
            resume=resume_run / "layer-major-progress.json",
            dtype=torch.bfloat16,
        )
        context.barrier()
        if context.is_primary:
            uninterrupted_config = json.loads(
                (root / "run" / "checkpoint-layerwise-local" / "config.json").read_text()
            )
            resumed_config = json.loads(
                (resume_run / "checkpoint-layerwise-local" / "config.json").read_text()
            )
            uninterrupted_fingerprint = uninterrupted_config["any2rwkv"]["mixer_overlay_fingerprint"]
            resumed_fingerprint = resumed_config["any2rwkv"]["mixer_overlay_fingerprint"]
            if resumed["status"] != "layerwise-local-complete":
                raise SystemExit(f"distributed resume did not complete: {resumed}")
            if resumed_fingerprint != uninterrupted_fingerprint:
                raise SystemExit("mid-accumulation distributed resume changed the final mixer digest")
            validate_frozen_parameters(
                resume_run / "checkpoint-layerwise-local",
                trainable=trainable,
                expected=frozen_reference,
                probe_dir=root / "resume-frozen-probe",
            )
            mid = json.loads(mid_progress.read_text())
            generation = mid_generation
            rank_states = sorted(generation.glob("training-state-rank-*.pt"))
            rank_zero_state = torch.load(rank_states[0], map_location="cpu", weights_only=False)
            if len(rank_states) != 8 or rank_zero_state["optimizer"].accumulation_step != 1:
                raise SystemExit("mid-accumulation generation lacks eight rank-local states")
            resume_parity = True
        context.barrier()
    corrective = run_global_corrective(
        source_manifest=source,
        run_dir=root / "run",
        zero_step_dir=root / "zero-step",
        token_rows=token_rows,
        validation_rows=validation_rows,
        plan=plan(),
        training_config=root / "training.json",
        dataset_manifest=root / "dataset.json",
        dtype=torch.bfloat16,
    )
    context.barrier()
    if context.is_primary:
        progress = json.loads((root / "run" / "layer-major-progress.json").read_text())
        if result["status"] != "layerwise-local-complete":
            raise SystemExit(f"distributed layer-major run did not complete: {result}")
        if progress["phase"] != "local-complete" or len(progress["history"]) != 2 * args.layers:
            raise SystemExit(f"distributed layer-major progress is incomplete: {progress}")
        if any(row["train_row_count"] != len(token_rows) for row in progress["history"]):
            raise SystemExit("distributed epoch did not cover every train row exactly once")
        if any(
            row["training_budget"]["world_size"] != 8
            or row["training_budget"]["global_micro_batch_size"] != 8
            or row["training_budget"]["loss_token_count"]
            != len(token_rows) * plan().supervised_tokens
            for row in progress["history"]
        ):
            raise SystemExit("distributed epoch training budget is incomplete")
        telemetry_files = sorted((root / "run" / "training-telemetry").glob("*.json"))
        if len(telemetry_files) != len(progress["history"]):
            raise SystemExit("distributed epoch telemetry count is incomplete")
        telemetry = json.loads(telemetry_files[-1].read_text())
        if (
            telemetry["world_size"] != 8
            or len(telemetry["train_wall_seconds_by_rank"]) != 8
            or len(telemetry["checkpoint_wall_seconds_by_rank"]) != 8
            or len(telemetry["peak_cuda_reserved_bytes_by_rank"]) != 8
            or telemetry["segment_loss_tokens_per_second"] <= 0
            or telemetry["max_checkpoint_wall_seconds"] < 0
            or not 0 <= telemetry["checkpoint_fraction_of_train_wall"] <= 1
        ):
            raise SystemExit("distributed epoch telemetry is invalid")
        activation_fit_files = sorted(
            (root / "run" / "activation-fit").glob("layer-*.json")
        )
        if len(activation_fit_files) != args.layers:
            raise SystemExit("distributed activation-fit report count is incomplete")
        for path in activation_fit_files:
            report = json.loads(path.read_text())
            train_binding = report.get("train_cache_binding", {})
            validation_binding = report.get("validation_cache_binding", {})
            validate_rank_contributions(
                report.get("rank_contributions"),
                fit_rows=int(report.get("fit_rows", 0)),
                evidence=path,
            )
            output_parameter_hashes = report.get(
                "selected_parameter_hashes", []
            )
            output_accepted = report.get("status") == "accepted"
            output_rejected_with_improved_chain = (
                report.get("status") == "rejected"
                and isinstance(report.get("chain_acceptance_baseline"), dict)
                and report["validation_before"]["mixer_normalized_mse"]
                < report["chain_acceptance_baseline"]["mixer_normalized_mse"]
            )
            expected_output_hash = (
                report.get("proposed_o_proj_weight_sha256")
                if output_accepted
                else report.get("original_o_proj_weight_sha256")
            )
            if (
                not (output_accepted or output_rejected_with_improved_chain)
                or report.get("world_size") != 8
                or report.get("boundary")
                != "source-mixer-output-to-native-o-proj-v1"
                or report.get("fit_rows") != plan().activation_fit_rows
                or (
                    output_accepted
                    and report["validation_after"]["mixer_normalized_mse"]
                    >= report["validation_before"]["mixer_normalized_mse"]
                )
                or train_binding.get("split") != "distill_train"
                or validation_binding.get("split") != "validation"
                or train_binding == validation_binding
                or not is_sha256(report.get("original_o_proj_weight_sha256"))
                or not is_sha256(report.get("proposed_o_proj_weight_sha256"))
                or not is_sha256(report.get("selected_o_proj_weight_sha256"))
                or report.get("selected_o_proj_weight_sha256")
                != expected_output_hash
                or len(output_parameter_hashes) != 8
                or sorted(
                    row.get("rank") for row in output_parameter_hashes
                )
                != list(range(8))
                or len(
                    {
                        row.get("weight_sha256")
                        for row in output_parameter_hashes
                    }
                )
                != 1
                or output_parameter_hashes[0].get("weight_sha256")
                != report.get("selected_o_proj_weight_sha256")
                or any(
                    not is_sha256(row.get("weight_sha256"))
                    for row in output_parameter_hashes
                )
            ):
                raise SystemExit(
                    f"distributed activation-fit report is invalid: {path}"
                )
        linear_layer_count = sum(
            kind == "linear_attention"
            for kind in source.config.get("text_config", source.config)["layer_types"]
        )
        decay_fit_files = sorted(
            (root / "run" / "activation-fit").glob("decay-layer-*.json")
        )
        if len(decay_fit_files) != linear_layer_count:
            raise SystemExit("distributed decay-fit report count is incomplete")
        for path in decay_fit_files:
            report = json.loads(path.read_text())
            train_binding = report.get("train_cache_binding")
            validation_binding = report.get("validation_cache_binding")
            validate_rank_contributions(
                report.get("rank_contributions"),
                fit_rows=int(report.get("fit_rows", 0)),
                evidence=path,
            )
            before_heads = report["validation_before"].get(
                "per_head_normalized_mse", []
            )
            after_heads = report["validation_after"].get(
                "per_head_normalized_mse", []
            )
            every_head_non_regressed = (
                len(before_heads) == len(after_heads) > 0
                and all(
                    after <= before
                    for before, after in zip(
                        before_heads, after_heads, strict=True
                    )
                )
            )
            improved = (
                report["validation_after"]["decay_normalized_mse"]
                < report["validation_before"]["decay_normalized_mse"]
                and every_head_non_regressed
                and report["layer_validation_after"]["mixer_normalized_mse"]
                <= report["layer_validation_before"]["mixer_normalized_mse"]
            )
            parameter_hashes = report.get("selected_parameter_hashes", [])
            selected = (
                report.get("selected_weight_sha256"),
                report.get("selected_bias_sha256"),
            )
            expected = (
                (
                    report.get("proposed_weight_sha256"),
                    report.get("proposed_bias_sha256"),
                )
                if report.get("status") == "accepted"
                else (
                    report.get("original_weight_sha256"),
                    report.get("original_bias_sha256"),
                )
            )
            if (
                report.get("world_size") != 8
                or report.get("boundary")
                != "gdn-decay-logit-to-native-w-up-v1"
                or report.get("solver")
                != "augmented-bias-normal-equations-ridge-v1"
                or report.get("status") not in {"accepted", "rejected"}
                or not isinstance(train_binding, dict)
                or not isinstance(validation_binding, dict)
                or train_binding.get("split") != "distill_train"
                or validation_binding.get("split") != "validation"
                or train_binding == validation_binding
                or (report["status"] == "accepted") != improved
                or report.get("every_head_non_regressed")
                != every_head_non_regressed
                or selected != expected
                or len(parameter_hashes) != 8
                or sorted(row.get("rank") for row in parameter_hashes)
                != list(range(8))
                or len({row.get("weight_sha256") for row in parameter_hashes})
                != 1
                or len({row.get("bias_sha256") for row in parameter_hashes})
                != 1
                or parameter_hashes[0].get("weight_sha256") != selected[0]
                or parameter_hashes[0].get("bias_sha256") != selected[1]
                or not all(
                    is_sha256(report.get(name))
                    for name in (
                        "original_weight_sha256",
                        "original_bias_sha256",
                        "proposed_weight_sha256",
                        "proposed_bias_sha256",
                        "selected_weight_sha256",
                        "selected_bias_sha256",
                    )
                )
                or any(
                    not is_sha256(row.get("weight_sha256"))
                    or not is_sha256(row.get("bias_sha256"))
                    for row in parameter_hashes
                )
            ):
                raise SystemExit(f"distributed decay-fit report is invalid: {path}")
        qkv_fit_files_by_stage = {
            "initial": (
                sorted(
                    (root / "run" / "activation-fit").glob(
                        "gdn-qkv-layer-*.json"
                    )
                ),
                {"accepted", "rejected"},
            ),
            "dependency-transaction": (
                sorted(
                    (root / "run" / "activation-fit").glob(
                        "gdn-qkv-dependency-transaction-layer-*.json"
                    )
                ),
                {"deferred"},
            ),
        }
        if any(
            len(files) != linear_layer_count
            for files, _ in qkv_fit_files_by_stage.values()
        ):
            raise SystemExit(
                "distributed GDN QKV-fit report count is incomplete by stage"
            )
        qkv_fit_files = [
            (stage, statuses, path)
            for stage, (files, statuses) in qkv_fit_files_by_stage.items()
            for path in files
        ]
        for expected_stage, expected_statuses, path in qkv_fit_files:
            report = json.loads(path.read_text())
            if (
                report.get("world_size") != 8
                or report.get("boundary")
                != "gdn-headnorm-equivalent-signals-to-native-projections-v4"
                or report.get("status") not in expected_statuses
                or report.get("fit_stage") != expected_stage
            ):
                raise SystemExit(f"distributed GDN QKV-fit report is invalid: {path}")
            train_binding = report.get("train_cache_binding")
            validation_binding = report.get("validation_cache_binding")
            if (
                not isinstance(train_binding, dict)
                or not isinstance(validation_binding, dict)
                or train_binding.get("split") != "distill_train"
                or validation_binding.get("split") != "validation"
                or train_binding == validation_binding
            ):
                raise SystemExit(
                    f"distributed GDN QKV held-out binding is invalid: {path}"
                )
            if report["status"] in {"accepted", "rejected"}:
                for role in ("r", "k", "v"):
                    validate_rank_contributions(
                        report.get("rank_contributions", {}).get(role),
                        fit_rows=int(report.get("fit_rows", 0)),
                        evidence=path,
                    )
                if report.get("target_signals") != {
                    "r": "recurrent_r*sqrt(head_dim)",
                    "k": "write_key",
                    "v": "write_value",
                }:
                    raise SystemExit(
                        f"distributed GDN recurrence targets are invalid: {path}"
                    )
                before = sum(
                    report["validation_before"][role]["normalized_mse"]
                    for role in ("r", "k", "v")
                )
                after = sum(
                    report["validation_after"][role]["normalized_mse"]
                    for role in ("r", "k", "v")
                )
                layer_non_regression = (
                    report["layer_validation_after"]["mixer_normalized_mse"]
                    <= report["layer_validation_before"]["mixer_normalized_mse"]
                )
                every_head_non_regressed = all(
                    after_head <= before_head
                    for role in ("r", "k", "v")
                    for before_head, after_head in zip(
                        report["validation_before"][role][
                            "per_target_head_normalized_mse"
                        ],
                        report["validation_after"][role][
                            "per_target_head_normalized_mse"
                        ],
                        strict=True,
                    )
                )
                if (report["status"] == "accepted") != (
                    after < before
                    and every_head_non_regressed
                    and layer_non_regression
                ):
                    raise SystemExit(
                        f"distributed GDN QKV-fit acceptance is inconsistent: {path}"
                    )
                if report.get("head_compression") is not None:
                    raise SystemExit(
                        f"distributed GDN fit must preserve source head geometry: {path}"
                    )
                if report.get("every_head_non_regressed") != (
                    every_head_non_regressed
                ):
                    raise SystemExit(
                        f"distributed GDN QKV per-head gate is inconsistent: {path}"
                    )
                qkv_parameter_hashes = report.get(
                    "selected_parameter_hashes", []
                )
                selected_weights = report.get("selected_weight_sha256", {})
                expected_weights = (
                    report.get("proposed_weight_sha256", {})
                    if report["status"] == "accepted"
                    else report.get("original_weight_sha256", {})
                )
                if (
                    set(report.get("original_weight_sha256", {}))
                    != {"r", "k", "v"}
                    or set(report.get("proposed_weight_sha256", {}))
                    != {"r", "k", "v"}
                    or selected_weights != expected_weights
                    or len(qkv_parameter_hashes) != 8
                    or sorted(
                        row.get("rank") for row in qkv_parameter_hashes
                    )
                    != list(range(8))
                    or any(
                        row.get("weights") != selected_weights
                        for row in qkv_parameter_hashes
                    )
                    or any(
                        not is_sha256(value)
                        for value in (
                            *report.get("original_weight_sha256", {}).values(),
                            *report.get("proposed_weight_sha256", {}).values(),
                            *selected_weights.values(),
                        )
                    )
                ):
                    raise SystemExit(
                        f"distributed GDN QKV parameter evidence is invalid: {path}"
                    )
        erase_fit_files = sorted(
            (root / "run" / "activation-fit").glob("gdn-a-layer-*.json")
        )
        if len(erase_fit_files) != linear_layer_count:
            raise SystemExit("distributed GDN erase-fit report count is incomplete")
        for path in erase_fit_files:
            report = json.loads(path.read_text())
            if (
                report.get("world_size") != 8
                or report.get("boundary")
                != "gdn-beta-times-decay-to-native-erase-v3"
                or report.get("solver") != "source-beta-decay-basis-ridge-v1"
                or report.get("basis_width") != 2 * report.get("source_heads", 0)
                or report.get("status") not in {"accepted", "rejected"}
                or report.get("train_cache_binding", {}).get("split")
                != "distill_train"
                or report.get("validation_cache_binding", {}).get("split")
                != "validation"
                or report.get("train_cache_binding")
                == report.get("validation_cache_binding")
            ):
                raise SystemExit(f"distributed GDN erase-fit report is invalid: {path}")
            every_head_non_regressed = all(
                after <= before
                for before, after in zip(
                    report["validation_before"]["per_head_normalized_mse"],
                    report["validation_after"]["per_head_normalized_mse"],
                    strict=True,
                )
            )
            improved = (
                report["validation_after"]["normalized_mse"]
                < report["validation_before"]["normalized_mse"]
                and every_head_non_regressed
                and report["layer_validation_after"]["mixer_normalized_mse"]
                <= report["layer_validation_before"]["mixer_normalized_mse"]
            )
            contributions = report.get("rank_contributions", [])
            validate_rank_contributions(
                contributions,
                fit_rows=int(report.get("fit_rows", 0)),
                evidence=path,
            )
            parameter_hashes = report.get("selected_parameter_hashes", [])
            if (
                len(contributions) != 8
                or sorted(row.get("rank") for row in contributions) != list(range(8))
                or any(row.get("rows", 0) <= 0 for row in contributions)
                or any(row.get("tokens", 0) <= 0 for row in contributions)
                or sum(row["rows"] for row in contributions) != report.get("fit_rows")
                or not all(
                    row.get("local_gram_sha256")
                    and row.get("local_rhs_sha256")
                    and row.get("local_target_squared_sum_sha256")
                    for row in contributions
                )
                or len(parameter_hashes) != 8
                or len({row.get("down_weight_sha256") for row in parameter_hashes}) != 1
                or len({row.get("up_weight_sha256") for row in parameter_hashes}) != 1
                or len({row.get("up_bias_sha256") for row in parameter_hashes}) != 1
                or report.get("every_head_non_regressed")
                != every_head_non_regressed
                or any(
                    not is_sha256(row.get("down_weight_sha256"))
                    or not is_sha256(row.get("up_weight_sha256"))
                    or not is_sha256(row.get("up_bias_sha256"))
                    for row in parameter_hashes
                )
            ):
                raise SystemExit(f"distributed direct erase evidence is invalid: {path}")
            selected = (
                report.get("selected_down_weight_sha256"),
                report.get("selected_up_weight_sha256"),
                report.get("selected_up_bias_sha256"),
            )
            expected = (
                (
                    report.get("proposed_down_weight_sha256"),
                    report.get("proposed_up_weight_sha256"),
                    report.get("proposed_up_bias_sha256"),
                )
                if report["status"] == "accepted"
                else (
                    report.get("original_down_weight_sha256"),
                    report.get("original_up_weight_sha256"),
                    report.get("original_up_bias_sha256"),
                )
            )
            if (
                selected != expected
                or not all(
                    is_sha256(report.get(name))
                    for name in (
                        "original_down_weight_sha256",
                        "original_up_weight_sha256",
                        "original_up_bias_sha256",
                        "proposed_down_weight_sha256",
                        "proposed_up_weight_sha256",
                        "proposed_up_bias_sha256",
                        "selected_down_weight_sha256",
                        "selected_up_weight_sha256",
                        "selected_up_bias_sha256",
                    )
                )
                or any(
                    (
                        row.get("down_weight_sha256"),
                        row.get("up_weight_sha256"),
                        row.get("up_bias_sha256"),
                    )
                    != selected
                    for row in parameter_hashes
                )
            ):
                raise SystemExit(
                    f"distributed direct erase install/rollback failed: {path}"
                )
            per_head = report["validation_after"].get("per_head_normalized_mse", ())
            if len(per_head) != int(report.get("target_heads", len(per_head))):
                raise SystemExit(
                    f"distributed direct erase per-head evidence is invalid: {path}"
                )
            if (report["status"] == "accepted") != improved:
                raise SystemExit(
                    f"distributed GDN erase-fit acceptance is inconsistent: {path}"
                )
        attention_layer_count = args.layers - linear_layer_count
        attention_qkv_files_by_stage = {
            "initial": sorted(
                (root / "run" / "activation-fit").glob(
                    "attention-qkv-layer-*.json"
                )
            ),
            "post-time-mix": sorted(
                (root / "run" / "activation-fit").glob(
                    "attention-qkv-post-time-mix-layer-*.json"
                )
            ),
        }
        if any(
            len(files) != attention_layer_count
            for files in attention_qkv_files_by_stage.values()
        ):
            raise SystemExit(
                "distributed attention QKV-fit report count is incomplete by stage"
            )
        attention_qkv_files = [
            (stage, path)
            for stage, files in attention_qkv_files_by_stage.items()
            for path in files
        ]
        for expected_stage, path in attention_qkv_files:
            report = json.loads(path.read_text())
            train_binding = report.get("train_cache_binding")
            validation_binding = report.get("validation_cache_binding")
            for role in ("r", "k", "v"):
                validate_rank_contributions(
                    report.get("rank_contributions", {}).get(role),
                    fit_rows=int(report.get("fit_rows", 0)),
                    evidence=path,
                )
            if (
                report.get("world_size") != 8
                or report.get("boundary")
                != "attention-norm-qkv-to-native-projections-v1"
                or report.get("status") not in {"accepted", "rejected"}
                or report.get("fit_stage") != expected_stage
                or not isinstance(train_binding, dict)
                or not isinstance(validation_binding, dict)
                or train_binding.get("split") != "distill_train"
                or validation_binding.get("split") != "validation"
                or train_binding == validation_binding
            ):
                raise SystemExit(
                    f"distributed attention QKV-fit report is invalid: {path}"
                )
            before = sum(
                report["validation_before"][role]["normalized_mse"]
                for role in ("r", "k", "v")
            )
            after = sum(
                report["validation_after"][role]["normalized_mse"]
                for role in ("r", "k", "v")
            )
            layer_non_regression = (
                report["layer_validation_after"]["mixer_normalized_mse"]
                <= report["layer_validation_before"]["mixer_normalized_mse"]
            )
            every_head_non_regressed = all(
                after_head <= before_head
                for role in ("r", "k", "v")
                for before_head, after_head in zip(
                    report["validation_before"][role][
                        "per_target_head_normalized_mse"
                    ],
                    report["validation_after"][role][
                        "per_target_head_normalized_mse"
                    ],
                    strict=True,
                )
            )
            if (report["status"] == "accepted") != (
                after < before
                and every_head_non_regressed
                and layer_non_regression
            ):
                raise SystemExit(
                    f"distributed attention QKV-fit acceptance is inconsistent: {path}"
                )
            if report.get("every_head_non_regressed") != every_head_non_regressed:
                raise SystemExit(
                    f"distributed attention QKV per-head gate is inconsistent: {path}"
                )
            selected_weights = report.get("selected_weight_sha256", {})
            expected_weights = (
                report.get("proposed_weight_sha256", {})
                if report["status"] == "accepted"
                else report.get("original_weight_sha256", {})
            )
            parameter_hashes = report.get("selected_parameter_hashes", [])
            if (
                set(report.get("original_weight_sha256", {}))
                != {"r", "k", "v"}
                or set(report.get("proposed_weight_sha256", {}))
                != {"r", "k", "v"}
                or set(selected_weights) != {"r", "k", "v"}
                or selected_weights != expected_weights
                or len(parameter_hashes) != 8
                or sorted(row.get("rank") for row in parameter_hashes)
                != list(range(8))
                or any(
                    row.get("weights") != selected_weights
                    for row in parameter_hashes
                )
                or any(
                    not is_sha256(value)
                    for value in (
                        *report.get("original_weight_sha256", {}).values(),
                        *report.get("proposed_weight_sha256", {}).values(),
                        *selected_weights.values(),
                    )
                )
            ):
                raise SystemExit(
                    f"distributed attention QKV parameter evidence is invalid: {path}"
                )
            query_heads = report["source_geometry"]["query_heads"]
            kv_heads = report["source_geometry"]["key_value_heads"]
            if (
                len(set(report.get("fit_context_lengths", ()))) < 3
                or len(set(report.get("validation_context_lengths", ()))) < 3
            ):
                raise SystemExit(
                    f"distributed attention multi-context evidence is incomplete: {path}"
                )
            if any(
                len(report["validation_after"][role].get(
                    "per_source_query_head_normalized_mse", ()
                ))
                != query_heads
                or len(report["validation_after"][role].get(
                    "per_source_kv_group_normalized_mse", ()
                ))
                != kv_heads
                for role in ("r", "k", "v")
            ):
                raise SystemExit(
                    f"distributed attention head/group errors are incomplete: {path}"
                )
        gate_fit_files_by_stage = {
            "initial": (
                sorted(
                    (root / "run" / "activation-fit").glob(
                        "gdn-gate-layer-*.json"
                    )
                ),
                {"accepted", "rejected"},
            ),
            "dependency-transaction": (
                sorted(
                    (root / "run" / "activation-fit").glob(
                        "gdn-gate-dependency-transaction-layer-*.json"
                    )
                ),
                {"deferred"},
            ),
        }
        if any(
            len(files) != linear_layer_count
            for files, _ in gate_fit_files_by_stage.values()
        ):
            raise SystemExit(
                "distributed GDN gate-fit report count is incomplete by stage"
            )
        gate_fit_files = [
            (stage, statuses, path)
            for stage, (files, statuses) in gate_fit_files_by_stage.items()
            for path in files
        ]
        for expected_stage, expected_statuses, path in gate_fit_files:
            report = json.loads(path.read_text())
            functional = (
                report.get("boundary")
                == "source-gated-pre-output-to-native-gate-network-v1"
            )
            if (
                report.get("world_size") != 8
                or report.get("boundary") not in {
                    "gdn-silu-z-to-native-gate-up-v1",
                    "source-gated-pre-output-to-native-gate-network-v1",
                }
                or report.get("status") not in expected_statuses
                or report.get("fit_stage") != expected_stage
            ):
                raise SystemExit(f"distributed GDN gate-fit report is invalid: {path}")
            if functional:
                improved = (
                    report["validation_after"]["mixer_normalized_mse"]
                    < report["validation_before"]["mixer_normalized_mse"]
                    and report["validation_after"]["normalized_mse"]
                    <= report["validation_before"]["normalized_mse"]
                )
                contributions = report.get("rank_contributions", [])
                rank_hashes = report.get("rank_parameter_hashes", [])
                if (
                    report.get("fit_mode")
                    != "noncommuting-functional-width-compression"
                    or report.get("source_width") == report.get("target_width")
                    or len(contributions) != 8
                    or sum(row.get("rows_per_step", 0) for row in contributions)
                    != report.get("fit_rows")
                    or any(row.get("tokens_per_step", 0) <= 0 for row in contributions)
                    or len(rank_hashes) != 8
                    or any(
                        row.get("parameters") != rank_hashes[0].get("parameters")
                        for row in rank_hashes[1:]
                    )
                    or report.get("train_cache_binding", {}).get("split")
                    != "distill_train"
                    or report.get("validation_cache_binding", {}).get("split")
                    != "validation"
                ):
                    raise SystemExit(
                        f"distributed functional gate evidence is invalid: {path}"
                    )
                expected = (
                    report["proposed_parameter_sha256"]
                    if report["status"] == "accepted"
                    else report["original_parameter_sha256"]
                )
                if report["selected_parameter_sha256"] != expected:
                    raise SystemExit(
                        f"distributed functional gate install/rollback failed: {path}"
                    )
            else:
                improved = (
                    report["validation_after"]["normalized_mse"]
                    < report["validation_before"]["normalized_mse"]
                    and report["layer_validation_after"]["mixer_normalized_mse"]
                    <= report["layer_validation_before"]["mixer_normalized_mse"]
                )
                if (report["status"] == "accepted") != improved:
                    raise SystemExit(
                        f"distributed GDN gate-fit acceptance is inconsistent: {path}"
                    )
                selected = report.get("selected_weight_sha256")
                proposed = report.get("proposed_weight_sha256")
                original = report.get("original_weight_sha256")
                selected_down = report.get("selected_down_weight_sha256")
                proposed_down = report.get("proposed_down_weight_sha256")
                original_down = report.get("original_down_weight_sha256")
                functional_fit = report.get("functional_fit", {})
                gate_parameter_hashes = report.get(
                    "selected_parameter_hashes", []
                )
                if report["status"] == "accepted" and selected != proposed:
                    raise SystemExit(
                        f"accepted GDN gate-fit weight was not installed: {path}"
                    )
                if report["status"] == "rejected" and selected != original:
                    raise SystemExit(
                        f"rejected GDN gate-fit weight was not rolled back: {path}"
                    )
                expected_down = (
                    proposed_down
                    if report["status"] == "accepted"
                    else original_down
                )
                if (
                    not all(
                        is_sha256(value)
                        for value in (
                            selected_down,
                            proposed_down,
                            original_down,
                        )
                    )
                    or selected_down != expected_down
                    or len(gate_parameter_hashes) != 8
                    or sorted(
                        row.get("rank") for row in gate_parameter_hashes
                    )
                    != list(range(8))
                    or any(
                        row.get("down_weight_sha256") != selected_down
                        or row.get("up_weight_sha256") != selected
                        for row in gate_parameter_hashes
                    )
                    or functional_fit.get("status")
                    not in {"accepted", "rejected", "not-run"}
                    or (
                        functional_fit.get("status") == "accepted"
                        and (
                            functional_fit["validation_after"]["normalized_mse"]
                            >= functional_fit["validation_before"]["normalized_mse"]
                            or functional_fit["layer_validation_after"][
                                "mixer_normalized_mse"
                            ]
                            > functional_fit["layer_validation_before"][
                                "mixer_normalized_mse"
                            ]
                        )
                    )
                ):
                    raise SystemExit(
                        f"distributed GDN functional gate evidence is invalid: {path}"
                    )
            if (report["status"] == "accepted") != improved:
                raise SystemExit(
                    f"distributed GDN gate-fit acceptance is inconsistent: {path}"
                )
        attention_gate_files_by_stage = {
            "initial": sorted(
                (root / "run" / "activation-fit").glob(
                    "attention-gate-layer-*.json"
                )
            ),
            "post-time-mix": sorted(
                (root / "run" / "activation-fit").glob(
                    "attention-gate-post-time-mix-layer-*.json"
                )
            ),
        }
        if any(
            len(files) != attention_layer_count
            for files in attention_gate_files_by_stage.values()
        ):
            raise SystemExit(
                "distributed attention gate-fit report count is incomplete by stage"
            )
        attention_gate_files = [
            (stage, path)
            for stage, files in attention_gate_files_by_stage.items()
            for path in files
        ]
        for expected_stage, path in attention_gate_files:
            report = json.loads(path.read_text())
            if (
                report.get("world_size") != 8
                or report.get("boundary")
                != "attention-sigmoid-q-gate-to-native-gate-up-v1"
                or report.get("status") not in {"accepted", "rejected"}
                or report.get("fit_stage") != expected_stage
                or len(set(report.get("fit_context_lengths", ()))) < 3
                or len(set(report.get("validation_context_lengths", ()))) < 3
            ):
                raise SystemExit(
                    f"distributed attention gate-fit report is invalid: {path}"
                )
            improved = (
                report["validation_after"]["normalized_mse"]
                < report["validation_before"]["normalized_mse"]
                and report["layer_validation_after"]["mixer_normalized_mse"]
                <= report["layer_validation_before"]["mixer_normalized_mse"]
            )
            selected = report.get("selected_weight_sha256")
            proposed = report.get("proposed_weight_sha256")
            original = report.get("original_weight_sha256")
            selected_down = report.get("selected_down_weight_sha256")
            proposed_down = report.get("proposed_down_weight_sha256")
            original_down = report.get("original_down_weight_sha256")
            functional_fit = report.get("functional_fit", {})
            gate_parameter_hashes = report.get(
                "selected_parameter_hashes", []
            )
            if (report["status"] == "accepted") != improved:
                raise SystemExit(
                    f"distributed attention gate-fit acceptance is inconsistent: {path}"
                )
            if report["status"] == "accepted" and selected != proposed:
                raise SystemExit(
                    f"accepted attention gate-fit weight was not installed: {path}"
                )
            if report["status"] == "rejected" and selected != original:
                raise SystemExit(
                    f"rejected attention gate-fit weight was not rolled back: {path}"
                )
            expected_down = (
                proposed_down
                if report["status"] == "accepted"
                else original_down
            )
            if (
                not all(
                    is_sha256(value)
                    for value in (selected_down, proposed_down, original_down)
                )
                or selected_down != expected_down
                or len(gate_parameter_hashes) != 8
                or sorted(
                    row.get("rank") for row in gate_parameter_hashes
                )
                != list(range(8))
                or any(
                    row.get("down_weight_sha256") != selected_down
                    or row.get("up_weight_sha256") != selected
                    for row in gate_parameter_hashes
                )
                or functional_fit.get("status")
                not in {"accepted", "rejected", "not-run"}
                or (
                    functional_fit.get("status") == "accepted"
                    and (
                        functional_fit["validation_after"]["normalized_mse"]
                        >= functional_fit["validation_before"]["normalized_mse"]
                        or functional_fit["layer_validation_after"][
                            "mixer_normalized_mse"
                        ]
                        > functional_fit["layer_validation_before"][
                            "mixer_normalized_mse"
                        ]
                    )
                )
            ):
                raise SystemExit(
                    f"distributed attention functional gate evidence is invalid: {path}"
                )
        norm_fit_files = sorted(
            (root / "run" / "activation-fit").glob("norm-affine-layer-*.json")
        )
        standalone_time_mix_files = sorted(
            (root / "run" / "activation-fit").glob("time-mix-layer-*.json")
        )
        dependency_time_mix_files = sorted(
            (root / "run" / "activation-fit").glob(
                "time-mix-dependency-transaction-layer-*.json"
            )
        )
        time_mix_files = [
            *((path, {"accepted", "rejected"}) for path in standalone_time_mix_files),
            *((path, {"deferred"}) for path in dependency_time_mix_files),
        ]
        if len(time_mix_files) != args.layers:
            raise SystemExit("distributed time-mix report count is incomplete")
        for path, expected_statuses in time_mix_files:
            report = json.loads(path.read_text())
            expected_gradient_scope = (
                ["x_r", "x_k", "x_v"]
                if report.get("source_layer_type") == "linear_attention"
                else ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]
            )
            improved = (
                report["validation_after"]["mixer_normalized_mse"]
                < report["validation_before"]["mixer_normalized_mse"]
                and report["validation_after"]["normalized_mse"]
                <= report["validation_before"]["normalized_mse"]
            )
            contributions = report.get("rank_contributions", [])
            rank_hashes = report.get("rank_parameter_hashes", [])
            if (
                report.get("world_size") != 8
                or report.get("boundary")
                != "source-mixer-output-to-native-time-mix-v1"
                or report.get("status") not in expected_statuses
                or report.get("gradient_scope") != expected_gradient_scope
                or len(contributions) != 8
                or sum(row.get("rows_per_step", 0) for row in contributions)
                != report.get("fit_rows")
                or any(row.get("tokens_per_step", 0) <= 0 for row in contributions)
                or len(rank_hashes) != 8
                or any(
                    row.get("parameters") != rank_hashes[0].get("parameters")
                    for row in rank_hashes[1:]
                )
                or report.get("train_cache_binding", {}).get("split") != "distill_train"
                or report.get("validation_cache_binding", {}).get("split") != "validation"
            ):
                raise SystemExit(f"distributed time-mix report is invalid: {path}")
            if (report["status"] == "accepted") != improved:
                raise SystemExit(f"distributed time-mix acceptance is inconsistent: {path}")
            selected = report["selected_parameter_sha256"]
            expected = (
                report["proposed_parameter_sha256"]
                if report["status"] == "accepted"
                else report["original_parameter_sha256"]
            )
            if selected != expected:
                raise SystemExit(f"distributed time-mix install/rollback failed: {path}")
        if len(norm_fit_files) != args.layers:
            raise SystemExit("distributed norm-affine report count is incomplete")
        for path in norm_fit_files:
            report = json.loads(path.read_text())
            if (
                report.get("world_size") != 8
                or report.get("boundary")
                != "source-pre-output-to-native-groupnorm-affine-v1"
                or report.get("status")
                not in {"accepted", "rejected", "unsupported-geometry"}
            ):
                raise SystemExit(f"distributed norm-affine report is invalid: {path}")
            if report["status"] == "unsupported-geometry":
                continue
            functional = (
                report.get("fit_mode")
                == "noncommuting-functional-norm-width-compression"
            )
            if functional:
                rank_contributions = report.get("rank_contributions", [])
                rank_hashes = report.get("rank_parameter_hashes", [])
                improved = (
                    report["validation_after"]["mixer_normalized_mse"]
                    < report["validation_before"]["mixer_normalized_mse"]
                    and report["validation_after"]["normalized_mse"]
                    <= report["validation_before"]["normalized_mse"]
                )
                if (
                    report.get("source_width") == report.get("target_width")
                    or len(rank_contributions) != 8
                    or sum(
                        row.get("rows_per_step", 0) for row in rank_contributions
                    )
                    != report.get("fit_rows")
                    or any(
                        row.get("tokens_per_step", 0) <= 0
                        for row in rank_contributions
                    )
                    or len(rank_hashes) != 8
                    or any(
                        row.get("parameters") != rank_hashes[0].get("parameters")
                        for row in rank_hashes[1:]
                    )
                    or report.get("train_cache_binding", {}).get("split")
                    != "distill_train"
                    or report.get("validation_cache_binding", {}).get("split")
                    != "validation"
                ):
                    raise SystemExit(
                        f"distributed functional norm evidence is invalid: {path}"
                    )
                expected = (
                    report["proposed_parameter_sha256"]
                    if report["status"] == "accepted"
                    else report["original_parameter_sha256"]
                )
                if report["selected_parameter_sha256"] != expected:
                    raise SystemExit(
                        f"distributed functional norm install/rollback failed: {path}"
                    )
                if (report["status"] == "accepted") != improved:
                    raise SystemExit(
                        f"distributed functional norm acceptance is inconsistent: {path}"
                    )
                continue
            rank_contributions = report.get("rank_contributions", [])
            if (
                len(rank_contributions) != 8
                or sorted(row.get("rank") for row in rank_contributions) != list(range(8))
                or any(row.get("rows", 0) <= 0 for row in rank_contributions)
                or any(row.get("tokens", 0) <= 0 for row in rank_contributions)
                or sum(row["rows"] for row in rank_contributions) != report["fit_rows"]
                or not all(row.get("local_statistics_sha256") for row in rank_contributions)
            ):
                raise SystemExit(f"distributed norm-affine rank evidence is invalid: {path}")
            parameter_hashes = report.get("selected_parameter_hashes", [])
            if (
                len(parameter_hashes) != 8
                or len({row.get("weight_sha256") for row in parameter_hashes}) != 1
                or len({row.get("bias_sha256") for row in parameter_hashes}) != 1
            ):
                raise SystemExit(f"distributed norm-affine parameters diverged: {path}")
            if (
                report.get("train_cache_binding", {}).get("split") != "distill_train"
                or report.get("validation_cache_binding", {}).get("split") != "validation"
                or report["train_cache_binding"] == report["validation_cache_binding"]
            ):
                raise SystemExit(f"distributed norm-affine cache binding is invalid: {path}")
            improved = (
                report["validation_after"]["normalized_mse"]
                < report["validation_before"]["normalized_mse"]
                and report["layer_validation_after"]["mixer_normalized_mse"]
                <= report["layer_validation_before"]["mixer_normalized_mse"]
            )
            selected = (
                report.get("selected_weight_sha256"),
                report.get("selected_bias_sha256"),
            )
            proposed = (
                report.get("proposed_weight_sha256"),
                report.get("proposed_bias_sha256"),
            )
            original = (
                report.get("original_weight_sha256"),
                report.get("original_bias_sha256"),
            )
            if (report["status"] == "accepted") != improved:
                raise SystemExit(
                    f"distributed norm-affine acceptance is inconsistent: {path}"
                )
            if report["status"] == "accepted" and selected != proposed:
                raise SystemExit(f"accepted norm-affine values were not installed: {path}")
            if report["status"] == "rejected" and selected != original:
                raise SystemExit(f"rejected norm-affine values were not rolled back: {path}")
        cache_telemetry_files = sorted(
            (root / "run" / "cache-transition-telemetry").glob("*.json")
        )
        if len(cache_telemetry_files) != args.layers or any(
            json.loads(path.read_text()).get("writer")
            != "distributed-row-sharded"
            for path in cache_telemetry_files
        ):
            raise SystemExit("distributed cache-transition telemetry is incomplete")
        retained_generations = tuple(
            path
            for path in (root / "run" / "layer-generations").iterdir()
            if path.is_dir()
        )
        if len(retained_generations) > 2:
            raise SystemExit("layer generation retention exceeded two references")
        corrective_progress = json.loads(
            (root / "run" / "global-corrective-progress.json").read_text()
        )
        if corrective["status"] != "fully-recurrent-global-corrective-complete":
            raise SystemExit(f"distributed corrective run did not complete: {corrective}")
        if corrective_progress["world_size"] != 8:
            raise SystemExit("distributed corrective progress did not bind world_size=8")
        if len(corrective_progress["layer_generations"]) != args.layers or any(
            entry["cursor"]["row_count"] != len(token_rows)
            for entry in corrective_progress["layer_generations"].values()
        ):
            raise SystemExit("distributed corrective sweep did not cover every train row")
        residency = json.loads(
            (root / "run" / "corrective-residency.json").read_text()
        )
        if (
            residency["mode"] != "resident"
            or residency["world_size"] != 8
            or len(residency["per_rank"]["teacher_layer_loads"]) != 8
            or any(
                loads != args.layers
                for loads in residency["per_rank"]["teacher_layer_loads"]
            )
            or not all(
                hits > 0
                for hits in residency["per_rank"]["teacher_cache_hits"]
            )
        ):
            raise SystemExit("distributed corrective residency telemetry is invalid")
        validate_fitted_checkpoint(
            root / "run" / "checkpoint-layerwise-local",
            expected_layers=args.layers,
        )
        validate_fitted_checkpoint(
            root / "run" / "checkpoint-global-corrective",
            expected_layers=args.layers,
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "world_size": 8,
                    "epochs": len(progress["history"]),
                    "train_rows_per_epoch": len(token_rows),
                    "best_generation": progress["best_generation"],
                    "corrective_status": corrective["status"],
                    "layers": args.layers,
                    "mid_accumulation_resume_parity": resume_parity,
                },
                sort_keys=True,
            )
        )
    context.barrier()
    context.close()


if __name__ == "__main__":
    main()
