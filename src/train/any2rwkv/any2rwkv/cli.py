from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import initialize_run, verify_run_bundle, write_json
from .checkpoint import read_checkpoint
from .calibration import read_calibration_manifest
from .contract import build_target_config
from .distill_runner import run_distillation
from .errors import ContractError
from .evaluator_runner import evaluate_hf_checkpoints
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
from .nvfp4 import export_nvfp4_checkpoint
from .source import fetch_source, verify_source
from .target import build_zero_step_ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="any2rwkv")
    subparsers = parser.add_subparsers(dest="action", required=True)
    commands: dict[str, argparse.ArgumentParser] = {}
    for action in ("preflight", "convert", "distill", "validate-p0", "quantize", "evaluate"):
        command = subparsers.add_parser(action)
        commands[action] = command
        command.add_argument("--source", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--precision", required=True, choices=("bf16", "fp16", "fp32io16", "nvfp4"))
        command.add_argument("--rwkv-hf-sha", required=True)
        command.add_argument("--rwkv-lm-sha", required=True)
        command.add_argument("--contract")
        command.add_argument("--calibration-manifest")
        command.add_argument("--run-id")
        command.add_argument("--allow-proxy-layers", action="store_true")
    commands["distill"].add_argument("--dataset-manifest", required=True)
    commands["distill"].add_argument("--training-config", required=True)
    commands["distill"].add_argument("--resume")
    commands["validate-p0"].add_argument("--kernel-oracle", required=True)
    commands["evaluate"].add_argument("--teacher", required=True)
    commands["evaluate"].add_argument("--evaluation-manifest", required=True)
    commands["evaluate"].add_argument("--p0-evidence", required=True)
    commands["evaluate"].add_argument("--migration-baselines", required=True)
    commands["evaluate"].add_argument("--ruler-scores")
    commands["evaluate"].add_argument("--downstream-scores")
    fixture = subparsers.add_parser("fixture", help="write deterministic 60-layer Qwen3.5-like test input")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--layers", type=int, default=60)
    oracle = subparsers.add_parser("oracle", help="run the frozen 32-case FP64 GDN/RWKV7 oracle")
    oracle.add_argument("--output", required=True)
    oracle.add_argument("--seed", type=int, default=20260714)
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
    source = read_checkpoint(Path(args.source), require_final_layers=not args.allow_proxy_layers)
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
    target = build_target_config(source.config, require_final_layers=not args.allow_proxy_layers)
    write_json(output / "target-config.json", target)
    source_names = tuple(source.tensor_names())
    shard_hashes = tuple(source.file_hashes[path.name] for path in source.shards)
    ledger, specs, target_names = build_zero_step_ledger(
        source_names,
        layer_count=source.contract.num_hidden_layers,
        hidden_size=source.contract.hidden_size,
        source_shard_hashes=shard_hashes,
    )
    warm_start = plan_warm_start(source, specs, variant=WarmStartVariant.MAPPED)
    apply_warm_start_plan(ledger, warm_start)
    ledger.write(output / "mapping.json")
    write_json(output / "mapping-coverage.json", ledger.validate(source_names, target_names))
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
    write_json(output / "roundtrip-manifest.json", roundtrip)
    metadata["status"] = "structural-zero-step"
    metadata["next_stage"] = "zero-step-baselines"
    write_json(output / "metadata.json", metadata)
    print(json.dumps({"status": metadata["status"], "output": str(output)}, sort_keys=True))
    return 0


def run_preflight(args: argparse.Namespace) -> int:
    source = read_checkpoint(
        Path(args.source), require_final_layers=not args.allow_proxy_layers
    )
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
    result = collect_preflight(_product_root())
    write_json(output / "preflight.json", result)
    metadata["status"] = "preflight-passed" if result["passed"] else "preflight-failed"
    metadata["preflight"] = "preflight.json"
    write_json(output / "metadata.json", metadata)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


def run_existing_stage(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if not (output / "metadata.json").is_file():
        raise ContractError(f"run metadata not found: {output / 'metadata.json'}; run convert first")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("submodules", {}).get("rwkv-hf") != args.rwkv_hf_sha:
        raise ContractError("rwkv-hf SHA differs from initialized run metadata")
    if args.action == "distill":
        result = run_distillation(
            source=Path(args.source),
            run_dir=output,
            dataset_manifest=Path(args.dataset_manifest),
            training_config=Path(args.training_config),
            allow_proxy_layers=args.allow_proxy_layers,
            resume=Path(args.resume) if args.resume else None,
        )
        metadata["distillation"] = result
        metadata["status"] = result["status"]
        write_json(output / "metadata.json", metadata)
        print(json.dumps(result, sort_keys=True))
        return 0
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
    if args.action == "quantize":
        quality_path = output / "quality.json"
        if not quality_path.is_file() or not _quality_gate_passed(quality_path, "P1"):
            raise ContractError("experimental NVFP4 export requires an accepted BF16 P1 checkpoint")
        if not args.calibration_manifest:
            raise ContractError("NVFP4 export requires a disjoint calibration manifest")
        source = Path(args.source).resolve()
        try:
            source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(f"cannot read accepted BF16 source config: {error}") from error
        if (
            source_config.get("model_type") != "any2rwkv_qwen35_rwkv7"
            or source_config.get("layer_types") != ["rwkv7"] * 60
            or source_config.get("any2rwkv", {}).get("final_recurrent") is not True
        ):
            raise ContractError(
                "NVFP4 source must be an accepted final 60-layer Any2RWKV BF16 checkpoint"
            )
        calibration = read_calibration_manifest(Path(args.calibration_manifest))
        result = export_nvfp4_checkpoint(
            source,
            output / "checkpoint-nvfp4",
            calibration,
        )
        metadata["quantization"] = result
        metadata["status"] = "experimental-nvfp4"
        write_json(output / "metadata.json", metadata)
        print(
            json.dumps(
                {
                    "status": metadata["status"],
                    "action": args.action,
                    "output": str(output / "checkpoint-nvfp4"),
                },
                sort_keys=True,
            )
        )
        return 0
    elif args.action == "evaluate":
        quality = evaluate_hf_checkpoints(
            teacher_path=Path(args.teacher),
            student_path=Path(args.source),
            manifest_path=Path(args.evaluation_manifest),
            p0_evidence_path=Path(args.p0_evidence),
            migration_baselines_path=Path(args.migration_baselines),
            output_path=output / "quality.json",
            ruler_scores_path=Path(args.ruler_scores) if args.ruler_scores else None,
            downstream_scores_path=(
                Path(args.downstream_scores) if args.downstream_scores else None
            ),
        )
        metadata["evaluation"] = {
            "quality_path": str(output / "quality.json"),
            "gates": quality["gates"],
        }
        metadata["status"] = (
            "quality-p2" if quality["gates"]["P2"]["passed"] else "evaluated"
        )
        write_json(output / "metadata.json", metadata)
        if quality["gates"]["P0"]["passed"]:
            verify_run_bundle(output)
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
        if args.action == "oracle":
            result = run_gdn_oracle(seed=args.seed)
            write_json(Path(args.output), result)
            print(json.dumps({"status": "passed" if result["passed"] else "failed", "fixture_count": result["fixture_count"]}, sort_keys=True))
            return 0 if result["passed"] else 1
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
        return run_existing_stage(args)
    except (ContractError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
