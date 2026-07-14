#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from any2rwkv.artifacts import write_json
from any2rwkv.roundtrip import validate_sharded_checkpoint


def digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--new-tokens", type=int, default=128)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    shard_report = validate_sharded_checkpoint(checkpoint)
    model, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        output_loading_info=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    generator = torch.Generator().manual_seed(20260714)
    prompts = torch.randint(
        0,
        model.config.vocab_size,
        (args.prompt_count, args.prompt_length),
        generator=generator,
    )
    generations = []
    logits_rows = []
    ce_rows = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = prompt.view(1, -1).to(device)
            output = model(input_ids=input_ids, use_cache=False).logits
            logits_rows.append(output[:, -1].cpu())
            ce_rows.append(
                torch.nn.functional.cross_entropy(
                    output[:, :-1].float().reshape(-1, output.shape[-1]), input_ids[:, 1:].reshape(-1)
                ).cpu()
            )
            generated_tokens = []
            current = input_ids
            past = None
            attention_mask = torch.ones_like(input_ids)
            for _ in range(args.new_tokens):
                step = model(
                    input_ids=current,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                next_token = step.logits[:, -1].argmax(dim=-1, keepdim=True)
                generated_tokens.append(next_token)
                past = step.past_key_values
                current = next_token
                attention_mask = torch.ones(
                    1,
                    input_ids.shape[1] + len(generated_tokens),
                    dtype=input_ids.dtype,
                    device=device,
                )
            generated = torch.cat((input_ids, *generated_tokens), dim=1)
            generations.append(generated.cpu())
    normalized_loading = {
        key: list(loading.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "shards": shard_report,
        "loading_info": normalized_loading,
        "prompt_count": args.prompt_count,
        "new_tokens": args.new_tokens,
        "greedy_digest": digest(torch.cat(generations)),
        "logits_digest": digest(torch.cat(logits_rows)),
        "ppl": math.exp(float(torch.stack(ce_rows).mean())),
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if not any(normalized_loading.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
