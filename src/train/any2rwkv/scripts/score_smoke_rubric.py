#!/usr/bin/env python3
"""Score the 32 deterministic generations with the frozen minimum-usability judge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from any2rwkv.artifacts import file_sha256
from any2rwkv.source import verify_source


def judge_json(model, tokenizer, *, system_prompt: str, prompt: str) -> dict:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=32,
            use_cache=True,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"judge did not return JSON: {text!r}")
    result = json.loads(text[start : end + 1])
    if set(result) != {"coherent", "relevant"} or any(
        type(result[name]) is not bool for name in result
    ):
        raise ValueError(f"judge did not return the frozen boolean schema: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", required=True, type=Path)
    parser.add_argument("--quality-suite", required=True, type=Path)
    parser.add_argument("--judge-source-manifest", required=True, type=Path)
    parser.add_argument("--judge-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    suite = json.loads(args.quality_suite.read_text(encoding="utf-8"))
    rubric = suite["smoke"]
    judge_manifest = json.loads(args.judge_source_manifest.read_text(encoding="utf-8"))
    if (
        judge_manifest.get("repository") != rubric["judge_repository"]
        or judge_manifest.get("revision") != rubric["judge_revision"]
    ):
        raise SystemExit("judge source manifest differs from the frozen smoke rubric")
    judge_verification = verify_source(args.judge_source_manifest, args.judge_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    judge_tokenizer = AutoTokenizer.from_pretrained(
        args.judge_checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_checkpoint,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    quality = json.loads(args.quality.read_text(encoding="utf-8"))
    rows = quality.get("smoke", {}).get("raw_outputs")
    if not isinstance(rows, list) or len(rows) != int(rubric["prompt_count"]):
        raise SystemExit("quality.json does not contain the frozen 32 smoke outputs")
    results = []
    for row in rows:
        if row.get("status") != "passed" or len(row.get("generated_token_ids", [])) != int(
            rubric["new_tokens_per_prompt"]
        ):
            judgment = {"coherent": False, "relevant": False}
        else:
            judgment = judge_json(
                judge_model,
                judge_tokenizer,
                system_prompt=rubric["judge_prompt"],
                prompt=(
                    f"PROMPT:\n{row['prompt']}\n\n"
                    f"RESPONSE:\n{row['generated_text']}"
                ),
            )
        results.append(
            {
                "prompt_index": int(row["prompt_index"]),
                "prompt_sha256": hashlib.sha256(str(row.get("prompt", "")).encode()).hexdigest(),
                "response_sha256": hashlib.sha256(
                    str(row.get("generated_text", "")).encode()
                ).hexdigest(),
                **judgment,
                "passed": judgment["coherent"] and judgment["relevant"],
            }
        )
    passed = sum(row["passed"] for row in results)
    payload = {
        "schema_version": 1,
        "rubric_id": rubric["rubric_id"],
        "judge_repository": rubric["judge_repository"],
        "judge_revision": rubric["judge_revision"],
        "judge_backend": "transformers-direct",
        "judge_source_manifest_sha256": file_sha256(args.judge_source_manifest),
        "judge_checkpoint_verification": judge_verification,
        "temperature": rubric["judge_temperature"],
        "quality_sha256": file_sha256(args.quality),
        "student_sha256": quality.get("binding", {}).get("student_sha256"),
        "pass_count": passed,
        "pass_rate": passed / len(results),
        "status": "scored",
        "gate_note": "pass/fail is decided only by the hash-bound quality threshold profile",
        "rows": results,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
