#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from any2rwkv import register_any_to_rwkv_auto_classes
from any2rwkv.artifacts import checkpoint_sha256, write_json
from any2rwkv.roundtrip import validate_sharded_checkpoint


def digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _logits(output) -> torch.Tensor:
    value = getattr(output, "logits", None)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("Transformers model output has no logits tensor")
    return value


def _past(output):
    value = getattr(output, "past_key_values", None)
    if value is None:
        raise RuntimeError("Transformers model did not return a recurrent cache")
    return value


def _full_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    return _logits(model(input_ids=input_ids, use_cache=False))


def _chunked_logits(model, input_ids: torch.Tensor, *, split: int) -> torch.Tensor:
    if not 0 < split < input_ids.shape[1]:
        raise ValueError("chunk split must be inside the prompt")
    prefix = model(input_ids=input_ids[:, :split], use_cache=True)
    suffix = model(
        input_ids=input_ids[:, split:],
        attention_mask=torch.ones_like(input_ids),
        past_key_values=_past(prefix),
        use_cache=True,
    )
    return torch.cat((_logits(prefix), _logits(suffix)), dim=1)


def _cached_greedy(model, input_ids: torch.Tensor, *, new_tokens: int) -> torch.Tensor:
    generated = [input_ids]
    current = input_ids
    past = None
    attention_mask = torch.ones_like(input_ids)
    for _ in range(new_tokens):
        step = model(
            input_ids=current,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        next_token = _logits(step)[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        past = _past(step)
        current = next_token
        attention_mask = torch.ones(
            input_ids.shape[0],
            input_ids.shape[1] + len(generated) - 1,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
    return torch.cat(generated, dim=1)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise RuntimeError(
            f"inference contract shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    return float((left.float() - right.float()).abs().max().cpu())


def main() -> int:
    from any2rwkv.preflight import require_rwkv7_runtime, runtime_binding

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--cache-atol", type=float, default=0.03125)
    parser.add_argument("--batch-atol", type=float, default=0.03125)
    args = parser.parse_args()
    if (
        args.prompt_count != 32
        or args.prompt_length < 2
        or args.new_tokens != 128
        or args.cache_atol < 0
        or args.batch_atol < 0
    ):
        raise ValueError(
            "strict HF roundtrip requires 32 prompts, at least two prompt tokens, "
            "128 new tokens and non-negative tolerances"
        )
    checkpoint = Path(args.checkpoint).resolve()
    register_any_to_rwkv_auto_classes()
    runtime = runtime_binding(require_rwkv7_runtime())
    shard_report = validate_sharded_checkpoint(checkpoint)
    model, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=False,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
    logits_rows = []
    ce_rows = []
    with torch.no_grad():
        prompts_device = prompts.to(device)
        full_batch_logits = _full_logits(model, prompts_device)
        single_full_logits = []
        for prompt in prompts:
            input_ids = prompt.view(1, -1).to(device)
            output = _full_logits(model, input_ids)
            single_full_logits.append(output)
            logits_rows.append(output[:, -1].cpu())
            ce_rows.append(
                torch.nn.functional.cross_entropy(
                    output[:, :-1].float().reshape(-1, output.shape[-1]), input_ids[:, 1:].reshape(-1)
                ).cpu()
            )
        single_full = torch.cat(single_full_logits, dim=0)
        batch_max_abs = _max_abs(single_full, full_batch_logits)
        split = args.prompt_length // 2
        chunked_logits = _chunked_logits(model, prompts_device, split=split)
        cache_max_abs = _max_abs(full_batch_logits, chunked_logits)
        reset_tokens = min(8, args.new_tokens)
        reset_reference = _cached_greedy(
            model,
            prompts_device[:1],
            new_tokens=reset_tokens,
        )
        _cached_greedy(
            model,
            prompts_device[1:2],
            new_tokens=reset_tokens,
        )
        reset_repeated = _cached_greedy(
            model,
            prompts_device[:1],
            new_tokens=reset_tokens,
        )
        reset_equal = torch.equal(reset_reference, reset_repeated)
        single_generations = torch.cat(
            [
                _cached_greedy(
                    model,
                    prompt.view(1, -1).to(device),
                    new_tokens=args.new_tokens,
                )
                for prompt in prompts
            ],
            dim=0,
        )
        batch_generations = _cached_greedy(
            model,
            prompts_device,
            new_tokens=args.new_tokens,
        )
        single_batch_greedy_equal = torch.equal(
            single_generations, batch_generations
        )
    contract_passed = (
        batch_max_abs <= args.batch_atol
        and cache_max_abs <= args.cache_atol
        and reset_equal
        and single_batch_greedy_equal
    )
    normalized_loading = {
        key: list(loading.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "model_sha256": checkpoint_sha256(checkpoint),
        "backend": "transformers",
        "transformers_version": transformers.__version__,
        "runtime": runtime,
        "strict_reload": not any(normalized_loading.values()),
        "shards": shard_report,
        "loading_info": normalized_loading,
        "prompt_count": args.prompt_count,
        "new_tokens": args.new_tokens,
        "greedy_digest": digest(batch_generations),
        "single_greedy_digest": digest(single_generations),
        "batch_greedy_digest": digest(batch_generations),
        "logits_digest": digest(torch.cat(logits_rows)),
        "ppl": math.exp(float(torch.stack(ce_rows).mean())),
        "single_batch_greedy_equal": single_batch_greedy_equal,
        "batch_isolation": {
            "passed": batch_max_abs <= args.batch_atol,
            "max_abs": batch_max_abs,
            "atol": args.batch_atol,
        },
        "full_chunked_cache": {
            "passed": cache_max_abs <= args.cache_atol,
            "split": split,
            "max_abs": cache_max_abs,
            "atol": args.cache_atol,
        },
        "state_reset": {
            "passed": reset_equal,
            "generated_tokens": reset_tokens,
            "reference_digest": digest(reset_reference),
            "repeated_digest": digest(reset_repeated),
        },
        "passed": contract_passed and not any(normalized_loading.values()),
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
