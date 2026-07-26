#!/usr/bin/env python3
"""Merge native-template and exact-token MaxRL response audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-prompt-body", type=Path, required=True)
    parser.add_argument("--historical-prompt-body", type=Path, required=True)
    parser.add_argument("--current-full-prompt", type=Path, required=True)
    parser.add_argument("--historical-full-prompt-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simple-output", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sampling_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in audit["sampling"].items()
        if key != "sampling_params_repr"
    }


def pair_contract(
    current: dict[str, Any],
    historical: dict[str, Any],
) -> dict[str, Any]:
    return {
        "same_candidate": (
            current["candidate_reconstruction"]
            == historical["candidate_reconstruction"]
        ),
        "same_source_messages": (
            current["input"]["source_prompt_messages"]
            == historical["input"]["source_prompt_messages"]
        ),
        "same_problem": current["input"]["problem"] == historical["input"]["problem"],
        "same_ground_truth": (
            current["input"]["ground_truth"] == historical["input"]["ground_truth"]
        ),
        "same_rendered_prompt": (
            current["input"]["rendered_prompt"]
            == historical["input"]["rendered_prompt"]
        ),
        "same_prompt_token_ids": (
            current["input"]["prompt_token_ids"]
            == historical["input"]["prompt_token_ids"]
        ),
        "same_prompt_token_count": (
            current["input"]["prompt_token_count"]
            == historical["input"]["prompt_token_count"]
        ),
        "same_sampling_contract": (
            sampling_contract(current) == sampling_contract(historical)
        ),
        "same_reward_scorer_sha256": (
            current["current_runtime"]["reward_scorer_source_sha256"]
            == historical["current_runtime"]["reward_scorer_source_sha256"]
        ),
        "same_model_path": (
            current["current_runtime"]["model"]
            == historical["current_runtime"]["model"]
        ),
    }


def pair_summary(
    current: dict[str, Any],
    historical: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract": pair_contract(current, historical),
        "current_revision": current["current_runtime"][
            "vllm_source_revision_declared_by_control"
        ],
        "historical_revision": historical["current_runtime"][
            "vllm_source_revision_declared_by_control"
        ],
        "current_summary": current["summary"],
        "historical_summary": historical["summary"],
    }


def main() -> None:
    args = parse_args()
    current_body = load(args.current_prompt_body)
    historical_body = load(args.historical_prompt_body)
    current_full = load(args.current_full_prompt)
    historical_full = load(args.historical_full_prompt_replay)

    payload = {
        "schema_version": 1,
        "purpose": (
            "Full MaxRL prompt, response, extraction, and reward evidence across "
            "the historical and current vLLM-RWKV revisions."
        ),
        "question": current_body["input"]["problem"],
        "ground_truth": current_body["input"]["ground_truth"],
        "historical_training_run": current_full["historical_run"],
        "limitations": {
            "historical_training_response_text_unavailable": (
                "The original 159-candidate training run retained aggregate metrics "
                "but no sample-level prompt or response dump. Its first candidate is "
                "reconstructed deterministically from sampler seed 42 and replayed "
                "on the exact historical source revisions."
            ),
            "stochastic_outputs": (
                "Both revisions use global engine seed 42 without per-request seeds. "
                "The comparison controls candidate, prompt token IDs, batch width, "
                "checkpoint path, and sampling parameters, but token streams need not "
                "match sample-by-sample across different engine implementations."
            ),
        },
        "comparisons": {
            "native_prompt_body_after_requested_fix": pair_summary(
                current_body,
                historical_body,
            ),
            "exact_current_full_prompt_token_replay": pair_summary(
                current_full,
                historical_full,
            ),
        },
        "runs": {
            "current_prompt_body": current_body,
            "historical_prompt_body": historical_body,
            "current_full_prompt_before_requested_fix": current_full,
            "historical_exact_full_prompt_token_replay": historical_full,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.simple_output is not None:
        simple_payload = {
            "prompt": current_body["input"]["rendered_prompt"],
            "previous_version_outputs": [
                response["output_text"]
                for response in historical_body["responses"]
            ],
            "current_version_outputs": [
                response["output_text"] for response in current_body["responses"]
            ],
        }
        args.simple_output.parent.mkdir(parents=True, exist_ok=True)
        args.simple_output.write_text(
            json.dumps(simple_payload, ensure_ascii=False, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "comparisons": payload["comparisons"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
