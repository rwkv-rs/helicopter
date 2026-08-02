#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from any2rwkv.core.quality_calibration import build_quality_threshold_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a hash-bound quality threshold profile from eligible seed runs."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed-result", type=Path, action="append", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    build_quality_threshold_profile(
        protocol_path=args.protocol,
        seed_result_paths=args.seed_result,
        artifact_path=args.artifact,
        profile_path=args.profile,
    )


if __name__ == "__main__":
    main()
