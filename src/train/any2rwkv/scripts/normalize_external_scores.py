#!/usr/bin/env python3
"""Normalize official RULERv2 or lm-eval raw sample outputs for paired scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from any2rwkv.external_scores import normalize_lm_eval, normalize_ruler2, write_score_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=("ruler", "downstream"))
    parser.add_argument("--quality-suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--results-json", action="append", type=Path)
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument(
        "--ruler-input",
        action="append",
        default=[],
        metavar="LENGTH:TASK:PATH",
        help="repeat once for every frozen RULERv2 context/task bucket",
    )
    parser.add_argument("--scale", action="store_true", help="require scale context lengths")
    args = parser.parse_args()
    suite = json.loads(args.quality_suite.read_text(encoding="utf-8"))
    if args.suite == "downstream":
        if not args.results_json or args.samples_dir is None or args.ruler_input:
            parser.error("downstream requires one or more --results-json and --samples-dir only")
        task_metrics = [(row["name"], row["metric"]) for row in suite["downstream"]["tasks"]]
        rows = normalize_lm_eval(
            results_json=args.results_json,
            samples_dir=args.samples_dir,
            task_metrics=task_metrics,
        )
    else:
        if args.results_json or args.samples_dir is not None:
            parser.error("ruler accepts only repeated --ruler-input values")
        inputs = []
        for value in args.ruler_input:
            try:
                length_text, task, path = value.split(":", 2)
                inputs.append((int(length_text), task, Path(path)))
            except ValueError as error:
                parser.error(f"invalid --ruler-input {value!r}: {error}")
        ruler = suite["ruler"]
        lengths = ruler["scale_required_lengths" if args.scale else "proxy_required_lengths"]
        rows = normalize_ruler2(
            inputs=inputs,
            expected_tasks=ruler["tasks"],
            expected_lengths=lengths,
            samples_per_bucket=int(ruler["samples_per_task_length"]),
        )
    write_score_rows(args.output, rows)


if __name__ == "__main__":
    main()
