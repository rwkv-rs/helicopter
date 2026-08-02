#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"read_error": str(error)}


def summarize_json(path: Path):
    payload = read_json(path)
    if payload is None or not isinstance(payload, dict):
        return payload
    return {
        "keys": sorted(payload),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "target_count": len(payload.get("targets", ())),
        "source_count": len(payload.get("sources", ())),
        "read_error": payload.get("read_error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    activation_fit = run / "activation-fit"
    telemetry = run / "training-telemetry"
    generations = run / "layer-generations"
    convergence = read_json(run / "layer-convergence.json")
    progress = read_json(run / "layer-major-progress.json")
    if isinstance(progress, dict):
        progress = {key: value for key, value in progress.items() if key != "history"}
    epochs = convergence.get("epochs", []) if isinstance(convergence, dict) else []
    latest_by_layer = {}
    for row in epochs:
        if isinstance(row, dict) and isinstance(row.get("layer"), int):
            latest_by_layer[row["layer"]] = row
    checkpoint_details = {}
    if run.is_dir():
        for checkpoint in sorted(
            path for path in run.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ):
            checkpoint_details[checkpoint.name] = {
                "files": sorted(path.name for path in checkpoint.iterdir()),
                "config": summarize_json(checkpoint / "config.json"),
                "mapping": summarize_json(checkpoint / "mapping.json"),
                "mapping_coverage": summarize_json(checkpoint / "mapping-coverage.json"),
                "activation_fit_provenance": summarize_json(
                    checkpoint / "activation-fit-provenance.json"
                ),
                "materialization": summarize_json(
                    checkpoint / "materialization.json"
                ),
            }
    metadata = read_json(run / "metadata.json")
    if isinstance(metadata, dict) and isinstance(metadata.get("distillation"), dict):
        metadata = dict(metadata)
        metadata["distillation"] = {
            key: value
            for key, value in metadata["distillation"].items()
            if key != "history"
        }
    trainable_by_layer: dict[int, list[str]] = defaultdict(list)
    provenance_by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    warm_start = read_json(run / "warm-start-plan.json")
    if isinstance(warm_start, dict):
        for entry in warm_start.get("entries", []):
            if not isinstance(entry, dict):
                continue
            target = str(entry.get("target", ""))
            prefix, separator, local_name = target.partition(".attn.")
            if not separator:
                continue
            parts = prefix.split(".")
            try:
                layer = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            provenance = str(entry.get("provenance", ""))
            provenance_by_layer[layer][provenance] += 1
            explicit_trainable = entry.get("local_trainable")
            if explicit_trainable is not None and not isinstance(
                explicit_trainable, bool
            ):
                raise ValueError(
                    "warm-start local_trainable must be boolean or null"
                )
            locally_trainable = (
                explicit_trainable
                if explicit_trainable is not None
                else provenance in {"fitted", "initialized"}
                or entry.get("is_semantically_lossless") is False
            )
            if locally_trainable:
                trainable_by_layer[layer].append(local_name)
    baseline_dir = run / "training-baselines"
    activation_fit_payloads = {}
    if activation_fit.is_dir():
        for path in sorted(activation_fit.glob("*.json")):
            if args.summary_only:
                continue
            if args.layer is not None and f"layer-{args.layer:03d}" not in path.name:
                continue
            activation_fit_payloads[path.name] = read_json(path)
    payload = {
        "run": str(run),
        "exists": run.is_dir(),
        "metadata": metadata,
        "progress": progress,
        "warm_start": {
            str(layer): {
                "provenance_counts": dict(sorted(provenance_by_layer[layer].items())),
                "trainable_count": len(names),
                "trainable_names": sorted(names),
            }
            for layer, names in sorted(trainable_by_layer.items())
            if args.layer is None or layer == args.layer
        },
        "training_baselines": (
            {
                path.name: read_json(path)
                for path in sorted(baseline_dir.glob("*.json"))
                if args.layer is None or path.name == f"layer-{args.layer:03d}.json"
            }
            if baseline_dir.is_dir()
            else {}
        ),
        "layer_convergence": {
            "world_size": convergence.get("world_size")
            if isinstance(convergence, dict) else None,
            "epoch_count": len(epochs),
            "layers": {
                str(layer): {
                    key: row.get(key)
                    for key in (
                        "epoch", "best_epoch", "best_metric", "converged",
                        "convergence_reason", "exhausted",
                    )
                }
                for layer, row in sorted(latest_by_layer.items())
            },
        },
        "global_corrective_progress": read_json(
            run / "global-corrective-progress.json"
        ),
        "activation_fit_reports": (
            [path.name for path in sorted(activation_fit.glob("*.json"))]
            if activation_fit.is_dir()
            else []
        ),
        "activation_fit_payloads": activation_fit_payloads,
        "training_telemetry": (
            [
                {"name": path.name, "payload": read_json(path)}
                for path in sorted(telemetry.glob("*.json"))[-3:]
            ]
            if telemetry.is_dir()
            else []
        ),
        "optimizer_curve": (
            [
                {
                    "layer": payload.get("layer"),
                    "epoch": payload.get("epoch"),
                    "optimizer": payload.get("optimizer"),
                }
                for path in sorted(telemetry.glob("*.json"))
                if isinstance((payload := read_json(path)), dict)
                and (args.layer is None or payload.get("layer") == args.layer)
            ]
            if telemetry.is_dir()
            else []
        ),
        "layer_generations": (
            sorted(path.name for path in generations.iterdir() if path.is_dir())
            if generations.is_dir()
            else []
        ),
        "checkpoints": checkpoint_details,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
