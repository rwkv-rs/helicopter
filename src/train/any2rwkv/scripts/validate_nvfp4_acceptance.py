#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from any2rwkv.nvfp4_acceptance import validate_nvfp4_acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-checkpoint", required=True, type=Path)
    parser.add_argument("--nvfp4-checkpoint", required=True, type=Path)
    parser.add_argument("--bf16-quality", required=True, type=Path)
    parser.add_argument("--nvfp4-quality", required=True, type=Path)
    parser.add_argument("--p0-evidence", required=True, type=Path)
    parser.add_argument("--service-evidence", required=True, type=Path)
    parser.add_argument("--roundtrip", required=True, type=Path)
    parser.add_argument("--performance", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate_nvfp4_acceptance(
        bf16_checkpoint=args.bf16_checkpoint,
        nvfp4_checkpoint=args.nvfp4_checkpoint,
        bf16_quality_path=args.bf16_quality,
        nvfp4_quality_path=args.nvfp4_quality,
        p0_evidence_path=args.p0_evidence,
        service_evidence_path=args.service_evidence,
        roundtrip_path=args.roundtrip,
        performance_path=args.performance,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    if not result["quality_compatible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
