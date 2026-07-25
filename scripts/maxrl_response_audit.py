#!/usr/bin/env python3
"""Emit an auditable prompt, rollout, extraction, and reward JSON record."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torchdata.stateful_dataloader.sampler import RandomSampler

from math_verify.grader import verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
from vllm import LLM, SamplingParams
from vllm.sampling_params import RepetitionDetectionParams

from verl.utils.ngram_repetition import (
    NGramRepetitionDetector,
    vllm_repetition_detection_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--responses", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=10240)
    parser.add_argument("--vllm-source-revision", required=True)
    parser.add_argument("--historical-run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_revision(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "path": str(path),
            "metadata_available": True,
            "revision": run("rev-parse", "HEAD"),
            "describe": run("describe", "--tags", "--long", "--always"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except subprocess.CalledProcessError as error:
        return {
            "path": str(path),
            "metadata_available": False,
            "error": str(error),
        }


def json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "as_py"):
        return json_value(value.as_py())
    return str(value)


def extraction_record(value: Any, gold: list[Any]) -> dict[str, Any]:
    return {
        "type": type(value).__name__,
        "text": str(value),
        "repr": repr(value),
        "matches_ground_truth": any(
            verify(target, value, strict=False) for target in gold
        ),
    }


def score_response(
    response_text: str,
    ground_truth: str,
    *,
    repetition_truncated: bool,
) -> dict[str, Any]:
    gold = parse(f"\\boxed{{{ground_truth}}}", (LatexExtractionConfig(),))
    predictions = parse(
        response_text,
        (ExprExtractionConfig(), LatexExtractionConfig()),
    )
    extraction_records = [extraction_record(value, gold) for value in predictions]
    raw_parser_score = float(
        any(record["matches_ground_truth"] for record in extraction_records)
    )
    training_parser_invoked = not repetition_truncated
    training_score = 0.0 if repetition_truncated else raw_parser_score
    return {
        "ground_truth_extractions": [
            {"type": type(value).__name__, "text": str(value), "repr": repr(value)}
            for value in gold
        ],
        "diagnostic_full_text_extractions": extraction_records,
        "diagnostic_full_text_score": raw_parser_score,
        "training_parser_invoked": training_parser_invoked,
        "training_extractions": extraction_records if training_parser_invoked else [],
        "training_final_score": training_score,
        "binary_success": training_score > 0,
        "termination_override": (
            "repetition_truncated_before_math_parser" if repetition_truncated else None
        ),
    }


def load_historical_run(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"available": False, "reason": "historical run directory not provided"}
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.jsonl"
    policy_path = run_dir / "policy_identity.jsonl"
    metadata = json.loads(metadata_path.read_text())
    metrics_rows = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    policy_rows = [
        json.loads(line)
        for line in policy_path.read_text().splitlines()
        if line.strip()
    ]
    first_step = metrics_rows[0]["data"]
    selected_metrics = {
        key: value
        for key, value in first_step.items()
        if key.startswith("training/effective_sampling/")
        or key.startswith("training/on_policy/")
        or key.startswith("training/actual_")
        or key.startswith("prompt_length/")
        or key.startswith("response_length/")
        or key in {"actor/loss", "actor/grad_norm", "actor/optimizer_steps"}
    }
    return {
        "available": True,
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "source_revisions": metadata["source_revisions"],
        "checkpoint": metadata["checkpoint"],
        "dataset_manifest": metadata["dataset_manifest"],
        "seed": metadata["seed"],
        "batch": metadata["batch"],
        "precision": metadata["precision"],
        "rollout_capacity": metadata["rollout_capacity"],
        "first_step_metrics": selected_metrics,
        "policy_identity": policy_rows,
        "sample_level_prompt_response_available": False,
        "sample_level_absence_reason": (
            "The historical run disabled rollout dumps and log_val_generations, "
            "and its artifact retained aggregate metrics but no prompt UID list or "
            "per-response text/token records."
        ),
    }


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        run_log_dir = os.environ.get("REMOTE_RUN_LOG_DIR")
        if not run_log_dir:
            raise RuntimeError("--output or REMOTE_RUN_LOG_DIR is required")
        output_path = Path(run_log_dir) / "maxrl-response-audit.json"
    table = pq.read_table(
        args.dataset,
        columns=["prompt", "source_prompt", "reward_model", "extra_info"],
    )
    generator = torch.Generator().manual_seed(args.dataset_seed)
    sampler = RandomSampler(range(len(table)), generator=generator)
    dataset_position = next(
        position for rank, position in enumerate(sampler) if rank == args.candidate_rank
    )
    row = table.take(pa.array([dataset_position])).to_pylist()[0]

    llm = LLM(
        model=str(args.model),
        tokenizer_mode="rwkv",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.context_length,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        distributed_executor_backend="uni",
        seed=args.dataset_seed,
    )
    tokenizer = llm.get_tokenizer()
    messages = row["source_prompt"]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        rwkv_generation_prompt="open_think",
    )
    prompt_token_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            rwkv_generation_prompt="open_think",
        )
    )
    max_tokens = args.context_length - len(prompt_token_ids)
    if max_tokens < 1:
        raise RuntimeError(
            f"prompt leaves no generation capacity: {len(prompt_token_ids)} >= "
            f"{args.context_length}"
        )

    repetition_config = vllm_repetition_detection_config()
    sampling_params = SamplingParams(
        n=args.responses,
        temperature=1.0,
        top_k=-1,
        top_p=0.95,
        max_tokens=max_tokens,
        ignore_eos=False,
        repetition_detection=RepetitionDetectionParams(**repetition_config),
    )
    request_output = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        sampling_params,
    )[0]

    responses = []
    for sample_index, sample in enumerate(request_output.outputs):
        detector_truncation_length = NGramRepetitionDetector().observe(sample.token_ids)
        engine_repetition_truncated = (
            sample.finish_reason == "repetition"
            or sample.stop_reason == "repetition_detected"
        )
        repetition_truncated = (
            engine_repetition_truncated or detector_truncation_length is not None
        )
        score = score_response(
            sample.text,
            row["reward_model"]["ground_truth"],
            repetition_truncated=repetition_truncated,
        )
        responses.append(
            {
                "sample_index": sample_index,
                "output_text": sample.text,
                "output_token_ids": list(sample.token_ids),
                "output_token_count": len(sample.token_ids),
                "finish_reason": sample.finish_reason,
                "stop_reason": sample.stop_reason,
                "contains_eos_token": tokenizer.eos_token_id in sample.token_ids,
                "last_token_id": sample.token_ids[-1] if sample.token_ids else None,
                "engine_repetition_truncated": engine_repetition_truncated,
                "detector_repetition_truncation_length": detector_truncation_length,
                "scoring": score,
            }
        )

    vllm_source = Path(__file__).resolve().parents[1] / "src/infer/vllm-rwkv"
    scorer_source = (
        Path(__file__).resolve().parents[1]
        / "src/train/verl-rwkv/verl/utils/reward_score/math_verify.py"
    )
    payload = {
        "schema_version": 1,
        "purpose": "same-input MaxRL prompt, output, extraction, and reward audit",
        "limitations": {
            "stochastic_replay": (
                "The historical server used global seed 42 without per-request "
                "seeds. Concurrent scheduling advances RNG state, so this probe "
                "reconstructs the same first candidate prompt and sampling contract "
                "but cannot reproduce the historical token stream."
            ),
            "historical_samples": (
                "Historical response text was not retained; only aggregate metrics "
                "can be compared."
            ),
        },
        "current_runtime": {
            "vllm_package_version": version("vllm"),
            "vllm_source_revision_declared_by_control": args.vllm_source_revision,
            "vllm_source": git_revision(vllm_source),
            "model": str(args.model),
            "context_length": args.context_length,
            "tokenizer_eos_token": tokenizer.eos_token,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "reward_scorer_source": str(scorer_source),
            "reward_scorer_source_sha256": subprocess.run(
                ["sha256sum", str(scorer_source)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()[0],
        },
        "historical_run": load_historical_run(args.historical_run_dir),
        "candidate_reconstruction": {
            "method": "torchdata.stateful_dataloader.sampler.RandomSampler",
            "dataset_seed": args.dataset_seed,
            "candidate_rank_zero_based": args.candidate_rank,
            "dataset_position_zero_based": dataset_position,
            "dataset_rows": len(table),
            "extra_info_index": row["extra_info"]["index"],
        },
        "input": {
            "dataset_path": str(args.dataset),
            "problem": row["prompt"],
            "source_prompt_messages": json_value(messages),
            "ground_truth": row["reward_model"]["ground_truth"],
            "chat_template_options": {
                "add_generation_prompt": True,
                "rwkv_generation_prompt": "open_think",
            },
            "rendered_prompt": rendered_prompt,
            "decoded_prompt_with_special_tokens": tokenizer.decode(
                prompt_token_ids,
                skip_special_tokens=False,
            ),
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_count": len(prompt_token_ids),
            "request_max_tokens": max_tokens,
        },
        "sampling": {
            "n": args.responses,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 0.95,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "penalty_decay": 1.0,
            "ignore_eos": False,
            "global_engine_seed": args.dataset_seed,
            "per_request_seed": None,
            "max_tokens": max_tokens,
            "repetition_detection": asdict(
                RepetitionDetectionParams(**repetition_config)
            ),
            "sampling_params_repr": repr(sampling_params),
        },
        "responses": responses,
        "summary": {
            "responses": len(responses),
            "binary_successes": sum(
                response["scoring"]["binary_success"] for response in responses
            ),
            "diagnostic_full_text_matches": sum(
                response["scoring"]["diagnostic_full_text_score"] > 0
                for response in responses
            ),
            "repetition_truncated": sum(
                response["engine_repetition_truncated"] for response in responses
            ),
            "responses_containing_eos": sum(
                response["contains_eos_token"] for response in responses
            ),
            "output_token_counts": [
                response["output_token_count"] for response in responses
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "candidate": payload["candidate_reconstruction"],
                "summary": payload["summary"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
