#!/usr/bin/env python3
"""Materialize exact pinned RULERv2 and lm-eval commands for one served checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from any2rwkv.calibration import file_sha256


def checkout_sha(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot inspect pinned evaluator checkout {path}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-suite", required=True, type=Path)
    parser.add_argument("--ruler-checkout", required=True, type=Path)
    parser.add_argument("--nemo-skills-checkout", required=True, type=Path)
    parser.add_argument("--lm-eval-checkout", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--role", required=True, choices=("teacher", "student", "bf16", "fp16", "nvfp4"))
    parser.add_argument("--target", required=True, choices=("proxy", "scale"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    suite = json.loads(args.quality_suite.read_text(encoding="utf-8"))
    pinned = {
        "ruler": (args.ruler_checkout, suite["ruler"]["revision"]),
        "nemo_skills": (args.nemo_skills_checkout, suite["ruler"]["implementation_revision"]),
        "lm_eval": (args.lm_eval_checkout, suite["downstream"]["revision"]),
    }
    revisions = {}
    for name, (path, expected) in pinned.items():
        actual = checkout_sha(path.resolve())
        if actual != expected:
            raise SystemExit(f"{name} checkout revision mismatch: expected {expected}, got {actual}")
        revisions[name] = actual
    if not args.tokenizer.is_dir():
        raise SystemExit(f"tokenizer directory not found: {args.tokenizer}")
    output = args.output.resolve()
    ruler = suite["ruler"]
    lengths = ruler[f"{args.target}_required_lengths"]
    commands: list[dict[str, object]] = []
    model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    server_gpus = 8 if args.target == "scale" else 1
    for length in lengths:
        setup = f"{model_name}-{length}"
        data_dir = output / "ruler-data" / str(length)
        result_dir = output / "ruler-results" / args.role / str(length)
        commands.extend(
            (
                {
                    "suite": "ruler",
                    "stage": "prepare_data",
                    "length": length,
                    "argv": [
                        "ns", "prepare_data", "ruler2", f"--cluster={args.cluster}",
                        f"--expname=ruler2-data-{setup}", f"--data_dir={data_dir}",
                        f"--setup={setup}", f"--tokenizer_path={args.tokenizer}",
                        f"--max_seq_length={length}",
                    ],
                },
                {
                    "suite": "ruler",
                    "stage": "evaluate",
                    "length": length,
                    "argv": [
                        "ns", "eval", f"--cluster={args.cluster}",
                        f"--expname=ruler2-{args.role}-{setup}", f"--data_dir={data_dir}",
                        f"--output_dir={result_dir}", f"--benchmarks=ruler2.{setup}",
                        f"--model={args.model}", "--server_nodes=1", f"--server_gpus={server_gpus}",
                        "--server_type=vllm",
                        f"--server_args=--tensor-parallel-size {server_gpus} --max-model-len {length} --trust-remote-code",
                        "++inference.tokens_to_generate=16384", "++inference.top_p=1.0",
                        "++inference.temperature=0.0", "++skip_filled=True",
                    ],
                },
                {
                    "suite": "ruler",
                    "stage": "summarize",
                    "length": length,
                    "argv": ["ns", "summarize_results", f"--cluster={args.cluster}", str(result_dir)],
                },
            )
        )
    completions_url = args.base_url.rstrip("/") + "/v1/completions"
    for task in suite["downstream"]["tasks"]:
        task_output = output / "lm-eval" / args.role / task["name"]
        commands.append(
            {
                "suite": "downstream",
                "stage": "evaluate",
                "task": task["name"],
                "metric": task["metric"],
                "argv": [
                    "lm-eval", "run", "--model", "local-completions", "--model_args",
                    f"model={args.model}", f"base_url={completions_url}",
                    f"tokenizer={args.tokenizer}", "tokenizer_backend=huggingface",
                    "tokenized_requests=False", "num_concurrent=16", "max_retries=3",
                    "--tasks", task["name"], "--num_fewshot", str(task["num_fewshot"]),
                    "--batch_size", "16", "--seed", str(suite["generation"]["seed"]),
                    "--apply_chat_template", "--log_samples", "--output_path", str(task_output),
                ],
            }
        )
    payload = {
        "schema_version": 1,
        "quality_suite": str(args.quality_suite.resolve()),
        "quality_suite_sha256": file_sha256(args.quality_suite),
        "revisions": revisions,
        "model": args.model,
        "tokenizer": str(args.tokenizer.resolve()),
        "base_url": args.base_url,
        "role": args.role,
        "target": args.target,
        "commands": commands,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "quality-command-plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
