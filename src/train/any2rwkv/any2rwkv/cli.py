from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch.distributed as dist

from .artifacts import file_sha256, initialize_run, verify_run_bundle, write_json
from .distill_runner import (
    _binding_sha256,
    _checkpoint_binding,
    _zero_step_checkpoint_binding,
    run_corrective_continuation,
    run_distillation,
)
from .distributed import DistributedContext
from .errors import ContractError
from .evaluator_runner import (
    evaluate_hf_checkpoints,
    evaluate_hf_migration_stage,
)
from .export import export_hf_checkpoint
from .fixture import write_fixture
from .migration_init import (
    WarmStartVariant,
    WarmStartTensorProvider,
    apply_warm_start_plan,
    plan_warm_start,
)
from .oracle import run_gdn_oracle
from .p0_runner import P0ValidationInputs, run_p0_validation
from .preflight import collect_preflight
from .recipes import resolve_recipe
from .source import fetch_source, verify_source
from .target import build_zero_step_ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="any2rwkv")
    subparsers = parser.add_subparsers(dest="action", required=True)
    commands: dict[str, argparse.ArgumentParser] = {}
    for action in (
        "preflight",
        "convert",
        "distill",
        "corrective",
        "validate-p0",
        "evaluate",
    ):
        command = subparsers.add_parser(action)
        commands[action] = command
        command.add_argument("--source", required=True)
        command.add_argument("--recipe", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--precision", required=True, choices=("bf16", "fp32io16"))
        command.add_argument("--rwkv-hf-sha", required=True)
        command.add_argument("--rwkv-lm-sha", required=True)
        command.add_argument("--contract")
        command.add_argument("--run-id")
        command.add_argument("--allow-proxy-layers", action="store_true")
    commands["distill"].add_argument("--dataset-manifest", required=True)
    commands["distill"].add_argument("--training-config", required=True)
    commands["distill"].add_argument("--resume")
    commands["corrective"].add_argument("--parent-run", required=True)
    commands["corrective"].add_argument(
        "--parent-checkpoint-sha256", required=True
    )
    commands["corrective"].add_argument("--dataset-manifest", required=True)
    commands["corrective"].add_argument("--training-config", required=True)
    commands["validate-p0"].add_argument("--kernel-oracle", required=True)
    commands["evaluate"].add_argument("--teacher", required=True)
    commands["evaluate"].add_argument("--evaluation-manifest", required=True)
    commands["evaluate"].add_argument("--p0-evidence", required=True)
    commands["evaluate"].add_argument("--migration-baselines", required=True)
    commands["evaluate"].add_argument("--quality-threshold-profile", required=True)
    commands["evaluate"].add_argument("--ruler-scores")
    commands["evaluate"].add_argument("--downstream-scores")
    fixture = subparsers.add_parser("fixture", help="write deterministic 60-layer Qwen3.5-like test input")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--layers", type=int, default=60)
    binding = subparsers.add_parser(
        "checkpoint-binding", help="print the immutable recurrent checkpoint binding"
    )
    binding.add_argument("--checkpoint", required=True)
    oracle = subparsers.add_parser("oracle", help="run the frozen 32-case FP64 GDN/RWKV7 oracle")
    oracle.add_argument("--output", required=True)
    oracle.add_argument("--seed", type=int, default=20260714)
    baseline = subparsers.add_parser(
        "evaluate-baseline",
        help="evaluate one raw migration-baseline stage without a circular quality dependency",
    )
    baseline.add_argument("--teacher", required=True)
    baseline.add_argument("--candidate", required=True)
    baseline.add_argument("--evaluation-manifest", required=True)
    baseline.add_argument("--stage", required=True)
    baseline.add_argument("--evidence")
    baseline.add_argument("--precision", required=True)
    baseline.add_argument("--output", required=True)
    for action in ("fetch-source", "verify-source"):
        source_command = subparsers.add_parser(action)
        source_command.add_argument("--manifest", required=True)
        source_command.add_argument("--destination", required=True)
        source_command.add_argument("--scale-gate")
    return parser


def _product_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _quality_gate_passed(path: Path, level: str) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gates", {}).get(level)
    return isinstance(gate, dict) and gate.get("passed") is True


def prepare_conversion(args: argparse.Namespace) -> int:
    resolved = resolve_recipe(args.recipe)
    source = resolved.source.load_checkpoint(
        Path(args.source), require_final_layout=not args.allow_proxy_layers
    )
    inspection = resolved.source.inspect_checkpoint(
        Path(args.source), require_final_layout=not args.allow_proxy_layers
    )
    resolved.recipe.validate_source(inspection)
    output = Path(args.output).resolve()
    metadata = initialize_run(
        output,
        run_id=args.run_id or output.name,
        source={
            "path": str(source.path),
            "files": source.file_hashes,
            "classification": (
                "real-60-layer-source" if source.contract.num_hidden_layers == 60 else "real-non-isomorphic-proxy"
            ),
            "layers": source.contract.num_hidden_layers,
            "extracted_text_backbone": source.contract.extracted_text_backbone,
        },
        precision=args.precision,
        command=sys.argv,
        product_root=_product_root(),
        rwkv_hf_sha=args.rwkv_hf_sha,
        rwkv_lm_sha=args.rwkv_lm_sha,
    )
    metadata["recipe"] = {
        "id": resolved.recipe.recipe_id,
        "source_adapter": resolved.source.adapter_id,
        "target_adapter": resolved.target.adapter_id,
    }
    target = resolved.target.build_target_config(
        source, require_final_layout=not args.allow_proxy_layers
    )
    write_json(output / "target-config.json", target)
    source_names = tuple(source.tensor_names())
    shard_hashes = tuple(source.file_hashes[path.name] for path in source.shards)
    ledger, specs, target_names = build_zero_step_ledger(
        source_names,
        layer_count=source.contract.num_hidden_layers,
        hidden_size=source.contract.hidden_size,
        head_dim=int(target["head_dim"]),
        attention_hidden_size=int(target["attention_hidden_size"]),
        source_shard_hashes=shard_hashes,
    )
    warm_start = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
    apply_warm_start_plan(ledger, warm_start)
    coverage = ledger.validate(source_names, target_names)
    ledger.write(output / "mapping.json")
    write_json(output / "mapping-coverage.json", coverage)
    write_json(
        output / "target-tensor-specs.json",
        {
            "stage": "mapped-zero-step-initialization",
            "tensors": [spec.__dict__ for spec in specs],
        },
    )
    write_json(output / "warm-start-plan.json", warm_start.to_dict())
    for variant in WarmStartVariant:
        plan = plan_warm_start(source, specs, variant=variant)
        write_json(output / "warm-start-plans" / f"{variant.value}.json", plan.to_dict())
    write_json(
        output / "source-manifest.json",
        {
            "classification": "real-source-checkpoint",
            "path": str(source.path),
            "contract": source.contract.__dict__,
            "shards": [path.name for path in source.shards],
            "tokenizer_files": [path.name for path in source.tokenizer_files],
            "file_hashes": source.file_hashes,
        },
    )
    roundtrip = export_hf_checkpoint(
        source,
        output / "checkpoint-zero-step",
        target_config=target,
        target_specs=specs,
        target_tensor_provider=WarmStartTensorProvider(source, specs, warm_start),
    )
    ledger.write(output / "checkpoint-zero-step" / "mapping.json")
    write_json(
        output / "checkpoint-zero-step" / "mapping-coverage.json", coverage
    )
    write_json(output / "roundtrip-manifest.json", roundtrip)
    zero_step_binding = _zero_step_checkpoint_binding(
        output / "checkpoint-zero-step"
    )
    metadata["zero_step"] = {
        "binding": zero_step_binding,
        "sha256": _binding_sha256(zero_step_binding),
    }
    metadata["warm_start_plan"] = {
        "path": "warm-start-plan.json",
        "sha256": file_sha256(output / "warm-start-plan.json"),
    }
    metadata["status"] = "structural-zero-step"
    metadata["next_stage"] = "zero-step-baselines"
    write_json(output / "metadata.json", metadata)
    print(json.dumps({"status": metadata["status"], "output": str(output)}, sort_keys=True))
    return 0


def run_preflight(args: argparse.Namespace) -> int:
    resolved = resolve_recipe(args.recipe)
    source = resolved.source.load_checkpoint(
        Path(args.source), require_final_layout=not args.allow_proxy_layers
    )
    inspection = resolved.source.inspect_checkpoint(
        Path(args.source), require_final_layout=not args.allow_proxy_layers
    )
    resolved.recipe.validate_source(inspection)
    output = Path(args.output).resolve()
    metadata = initialize_run(
        output,
        run_id=args.run_id or output.name,
        source={
            "path": str(source.path),
            "files": source.file_hashes,
            "layers": source.contract.num_hidden_layers,
        },
        precision=args.precision,
        command=sys.argv,
        product_root=_product_root(),
        rwkv_hf_sha=args.rwkv_hf_sha,
        rwkv_lm_sha=args.rwkv_lm_sha,
    )
    result = collect_preflight(
        _product_root(),
        expected_rwkv_hf_sha=args.rwkv_hf_sha,
        expected_rwkv_lm_sha=args.rwkv_lm_sha,
    )
    result["recipe"] = {
        "id": resolved.recipe.recipe_id,
        "source_adapter": resolved.source.adapter_id,
        "target_adapter": resolved.target.adapter_id,
    }
    write_json(output / "preflight.json", result)
    metadata["status"] = "preflight-passed" if result["passed"] else "preflight-failed"
    metadata["preflight"] = "preflight.json"
    write_json(output / "metadata.json", metadata)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


def run_existing_stage(args: argparse.Namespace) -> int:
    resolved = resolve_recipe(args.recipe)
    output = Path(args.output).resolve()
    if not (output / "metadata.json").is_file():
        raise ContractError(f"run metadata not found: {output / 'metadata.json'}; run convert first")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("submodules", {}).get("rwkv-hf") != args.rwkv_hf_sha:
        raise ContractError("rwkv-hf SHA differs from initialized run metadata")
    if metadata.get("submodules", {}).get("rwkv-lm") != args.rwkv_lm_sha:
        raise ContractError("rwkv-lm SHA differs from initialized run metadata")
    expected_recipe = metadata.get("recipe", {}).get("id")
    if expected_recipe != resolved.recipe.recipe_id:
        raise ContractError(
            f"run recipe differs from initialized metadata: "
            f"expected={expected_recipe!r} requested={resolved.recipe.recipe_id!r}"
        )
    if args.action == "distill":
        distributed: DistributedContext | None = None
        try:
            result = run_distillation(
                source=Path(args.source),
                run_dir=output,
                dataset_manifest=Path(args.dataset_manifest),
                training_config=Path(args.training_config),
                recipe_id=resolved.recipe.recipe_id,
                allow_proxy_layers=args.allow_proxy_layers,
                resume=Path(args.resume) if args.resume else None,
            )
            distributed = DistributedContext.initialize()
            if distributed.is_primary:
                metadata["distillation"] = result
                metadata["status"] = result["status"]
                write_json(output / "metadata.json", metadata)
                print(json.dumps(result, sort_keys=True))
            distributed.barrier()
            return 0
        finally:
            if distributed is not None:
                distributed.close()
            elif dist.is_initialized():
                DistributedContext.initialize().close()
    if args.action == "validate-p0":
        result = run_p0_validation(
            P0ValidationInputs(
                run_dir=output,
                student=Path(args.source).resolve(),
                kernel_oracle=Path(args.kernel_oracle).resolve(),
                package_root=Path(__file__).resolve().parents[1],
            )
        )
        metadata["p0_evidence"] = "p0-evidence.json"
        metadata["status"] = "p0-passed"
        write_json(output / "metadata.json", metadata)
        print(json.dumps({"status": metadata["status"], "evidence": result}, sort_keys=True))
        return 0
    if args.action == "evaluate":
        distributed = DistributedContext.initialize()
        try:
            quality = evaluate_hf_checkpoints(
                teacher_path=Path(args.teacher),
                student_path=Path(args.source),
                manifest_path=Path(args.evaluation_manifest),
                p0_evidence_path=Path(args.p0_evidence),
                migration_baselines_path=Path(args.migration_baselines),
                quality_threshold_profile_path=Path(args.quality_threshold_profile),
                output_path=output / "quality.json",
                ruler_scores_path=Path(args.ruler_scores) if args.ruler_scores else None,
                downstream_scores_path=(
                    Path(args.downstream_scores) if args.downstream_scores else None
                ),
                distributed=distributed,
            )
            publish_status = None
            if distributed.is_primary:
                try:
                    write_json(output / "quality.json", quality)
                    metadata["evaluation"] = {
                        "quality_path": str(output / "quality.json"),
                        "gates": quality["gates"],
                    }
                    metadata["status"] = (
                        "quality-p2"
                        if quality["gates"]["P2"]["passed"]
                        else "evaluated"
                    )
                    write_json(output / "metadata.json", metadata)
                    if quality["gates"]["P0"]["passed"]:
                        verify_run_bundle(output)
                    publish_status = {"status": "ok"}
                except BaseException as error:
                    publish_status = {"status": "error", "error": repr(error)}
            publish_status = distributed.broadcast_object(publish_status)
            if publish_status["status"] != "ok":
                raise ContractError(
                    "distributed evaluation publish failed: "
                    + publish_status["error"]
                )
            if distributed.is_primary:
                print(
                    json.dumps(
                        {
                            "status": "validated",
                            "action": args.action,
                            "output": str(output),
                        },
                        sort_keys=True,
                    )
                )
            distributed.barrier()
            return 0
        finally:
            distributed.close()
    else:
        raise ContractError(f"unsupported existing stage: {args.action}")
    print(json.dumps({"status": "validated", "action": args.action, "output": str(output)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "fixture":
            write_fixture(Path(args.output), layers=args.layers)
            return 0
        if args.action == "checkpoint-binding":
            binding = _checkpoint_binding(Path(args.checkpoint).resolve())
            print(
                json.dumps(
                    {"binding": binding, "sha256": _binding_sha256(binding)},
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "oracle":
            result = run_gdn_oracle(seed=args.seed)
            write_json(Path(args.output), result)
            print(json.dumps({"status": "passed" if result["passed"] else "failed", "fixture_count": result["fixture_count"]}, sort_keys=True))
            return 0 if result["passed"] else 1
        if args.action == "evaluate-baseline":
            distributed = DistributedContext.initialize()
            try:
                result = evaluate_hf_migration_stage(
                    teacher_path=Path(args.teacher).resolve(),
                    candidate_path=Path(args.candidate).resolve(),
                    manifest_path=Path(args.evaluation_manifest).resolve(),
                    stage=args.stage,
                    output_path=Path(args.output).resolve(),
                    evidence_path=(
                        Path(args.evidence).resolve() if args.evidence else None
                    ),
                    precision=args.precision,
                    distributed=distributed,
                )
                if distributed.is_primary:
                    print(json.dumps(result, sort_keys=True))
                return 0
            finally:
                distributed.close()
        if args.action in {"fetch-source", "verify-source"}:
            function = fetch_source if args.action == "fetch-source" else verify_source
            kwargs = (
                {"scale_gate": Path(args.scale_gate) if args.scale_gate else None}
                if args.action == "fetch-source"
                else {}
            )
            result = function(Path(args.manifest), Path(args.destination), **kwargs)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.action == "preflight":
            return run_preflight(args)
        if args.action == "convert":
            return prepare_conversion(args)
        if args.action == "corrective":
            return run_corrective_continuation(args)
        return run_existing_stage(args)
    except (ContractError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
