#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from any2rwkv.distributed import DistributedContext
from any2rwkv.distill_runner import prepare_performance_profile_caches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the frozen embedding/recurrent-prefix cache cases required "
            "by the 8-rank Any2RWKV performance matrix."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--allow-proxy-layers", action="store_true")
    parser.add_argument("--train-row-limit", type=int)
    parser.add_argument("--validation-row-limit", type=int)
    args = parser.parse_args()

    try:
        result = prepare_performance_profile_caches(
            source=args.source.resolve(),
            run_dir=args.run_dir.resolve(),
            dataset_manifest=args.dataset_manifest.resolve(),
            training_config=args.training_config.resolve(),
            recipe_id=args.recipe,
            allow_proxy_layers=args.allow_proxy_layers,
            train_row_limit=args.train_row_limit,
            validation_row_limit=args.validation_row_limit,
        )
    finally:
        if torch.distributed.is_initialized():
            DistributedContext.initialize().close()
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
