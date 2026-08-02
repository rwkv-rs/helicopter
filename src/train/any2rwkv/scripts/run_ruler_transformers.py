#!/usr/bin/env python3
"""Generate RULER responses with Transformers while preserving task-native rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from any2rwkv import register_any_to_rwkv_auto_classes


def _prompt(row: dict, tokenizer) -> str:
    if isinstance(row.get("input"), str):
        return row["input"]
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    messages = row.get("messages")
    if isinstance(messages, list):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    raise ValueError("RULER row has no supported input/prompt/messages field")


def _generate(model, tokenizer, prompt: str, *, max_context: int, max_new: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if inputs["input_ids"].shape[1] > max_context:
        raise ValueError(
            f"RULER prompt length {inputs['input_ids'].shape[1]} exceeds {max_context}"
        )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new,
            use_cache=True,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-context-length", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    args = parser.parse_args()
    if args.max_context_length <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("context and generation lengths must be positive")
    files = sorted(args.data_dir.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"RULER data directory has no JSONL files: {args.data_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    register_any_to_rwkv_auto_classes()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    for source in files:
        destination = args.output_dir / source.relative_to(args.data_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with source.open(encoding="utf-8") as reader, temporary.open(
            "w", encoding="utf-8"
        ) as writer:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                response = _generate(
                    model,
                    tokenizer,
                    _prompt(row, tokenizer),
                    max_context=args.max_context_length,
                    max_new=args.max_new_tokens,
                )
                if "generation" in row or "predicted_answer" in row:
                    raise ValueError("RULER source row already contains generated output")
                row["generation"] = response
                row["predicted_answer"] = response
                writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)


if __name__ == "__main__":
    main()
