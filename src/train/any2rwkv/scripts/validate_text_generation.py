#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from any2rwkv import register_any_to_rwkv_auto_classes
from any2rwkv.artifacts import write_json
from any2rwkv.distributed import DistributedContext

DEFAULT_PROMPTS = (
    "请用两句话解释为什么天空看起来是蓝色的。",
    "What is 1 + 1? Answer in one short sentence.",
    "写一段简短、连贯的话，说明每天阅读的一个好处。",
    "Name the capital of France and explain where it is located.",
)


def token_statistics(token_ids: list[int]) -> dict[str, float | int]:
    if not token_ids:
        return {"token_count": 0, "unique_token_ratio": 0.0, "max_token_run": 0}
    max_run = 1
    current_run = 1
    for previous, current in zip(token_ids, token_ids[1:], strict=False):
        if current == previous:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return {
        "token_count": len(token_ids),
        "unique_token_ratio": len(Counter(token_ids)) / len(token_ids),
        "max_token_run": max_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt-mode", choices=("chat", "raw", "both"), default="chat"
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    distributed = DistributedContext.initialize()
    if torch.cuda.is_available() and distributed.world_size != 8:
        distributed.close()
        raise SystemExit("real CUDA generation validation requires 8 ranks")
    checkpoint = args.checkpoint.resolve()
    register_any_to_rwkv_auto_classes()
    device = distributed.device if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    model, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        output_loading_info=True,
    )
    model = model.to(device).eval()
    normalized_loading = {
        key: list(loading.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    rows = []
    modes = ("chat", "raw") if args.prompt_mode == "both" else (args.prompt_mode,)
    jobs = tuple(
        (prompt_index, prompt, prompt_mode)
        for prompt_index, prompt in enumerate(DEFAULT_PROMPTS)
        for prompt_mode in modes
    )
    if distributed.world_size > 1 and len(jobs) < distributed.world_size:
        distributed.close()
        raise SystemExit("generation validation requires at least one job per rank")
    for prompt_index, prompt, prompt_mode in jobs[
        distributed.rank :: distributed.world_size
    ]:
        rendered = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if prompt_mode == "chat"
                else prompt
            )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
        generated = output[0, inputs["input_ids"].shape[1] :].tolist()
        rows.append(
            {
                    "prompt_index": prompt_index,
                    "prompt_mode": prompt_mode,
                    "prompt": prompt,
                    "rendered_prompt": rendered,
                    "generated_token_ids": generated,
                    "generated_text": tokenizer.decode(
                        generated, skip_special_tokens=True
                    ).strip(),
                    **token_statistics(generated),
            }
        )
    if distributed.world_size > 1:
        rows = [
            row
            for shard in distributed.all_gather_objects(rows)
            for row in shard
        ]
    rows.sort(key=lambda row: (row["prompt_index"], row["prompt_mode"]))
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "backend": "transformers-direct",
        "dtype": str(dtype),
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        },
        "loading_info": normalized_loading,
        "rows": rows,
    }
    if distributed.is_primary:
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    technical_failure = any(normalized_loading.values()) or any(
        not row["generated_token_ids"] for row in rows
    )
    distributed.barrier()
    distributed.close()
    return 1 if technical_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
