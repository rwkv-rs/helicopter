#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from any2rwkv.core.training_control_calibration import (
    build_training_control_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select hash-bound training controls from multi-seed ablations."
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--result", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_training_control_calibration(
        protocol_path=args.protocol,
        result_paths=args.result,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
