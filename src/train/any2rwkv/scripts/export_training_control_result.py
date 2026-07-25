#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from any2rwkv.core.training_control_export import export_training_control_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export one completed layer-major pilot as calibration evidence."
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_training_control_result(
        protocol_path=args.protocol,
        candidate_id=args.candidate_id,
        run_dir=args.run_dir,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
