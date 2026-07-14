#!/usr/bin/env python3
"""Produce direct-vLLM evidence and validate the OpenAI-compatible service."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

from any2rwkv.artifacts import checkpoint_sha256


def _json_request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def _health_request(url: str) -> None:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        if not 200 <= int(response.status) < 300:
            raise RuntimeError(f"vLLM health endpoint returned HTTP {response.status}")


def _memory_mib() -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(float(line) for line in result.stdout.splitlines() if line.strip())


def _stream_text(url: str, payload: dict) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps({**payload, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    pieces = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            pieces.append(json.loads(line[6:])["choices"][0]["text"])
    return "".join(pieces)


def write_direct(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))["prompts"]
    engine = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, logprobs=1)
    outputs = engine.generate(prompts, params)
    rows = []
    for prompt, output in zip(prompts, outputs, strict=True):
        completion = output.outputs[0]
        first = completion.logprobs[0][completion.token_ids[0]].logprob
        rows.append(
            {
                "prompt": prompt,
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "first_token_logprob": float(first),
            }
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": args.model,
                "model_sha256": checkpoint_sha256(Path(args.model)),
                "tensor_parallel_size": args.tensor_parallel_size,
                "temperature": 0,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_service(args: argparse.Namespace) -> None:
    direct = json.loads(Path(args.direct).read_text(encoding="utf-8"))
    if direct.get("schema_version") != 1 or not direct.get("rows"):
        raise RuntimeError("direct vLLM evidence must use schema_version=1 and contain rows")
    checkpoint_digest = checkpoint_sha256(Path(args.checkpoint))
    if checkpoint_digest != direct.get("model_sha256"):
        raise RuntimeError("service checkpoint differs from direct-loader evidence")
    rows = direct["rows"]
    base = args.base_url.rstrip("/")
    _health_request(base.removesuffix("/v1") + "/health")

    def invoke(row: dict) -> dict:
        result = _json_request(
            base + "/completions",
            {
                "model": args.served_model,
                "prompt": row["prompt"],
                "max_tokens": len(row["token_ids"]),
                "temperature": 0,
                "logprobs": 1,
            },
        )["choices"][0]
        token_logprobs = result["logprobs"]["token_logprobs"]
        if not token_logprobs or not math.isfinite(float(token_logprobs[0])):
            raise RuntimeError("service returned no finite first-token logprob")
        return {"text": result["text"], "first_token_logprob": float(token_logprobs[0])}

    for index in range(20):
        invoke(rows[index % len(rows)])
    comparisons = []
    for row in rows:
        result = invoke(row)
        comparisons.append(
            {
                "prompt": row["prompt"],
                "text_equal": result["text"] == row["text"],
                "first_token_logprob_abs": abs(
                    result["first_token_logprob"] - row["first_token_logprob"]
                ),
            }
        )
    batch = _json_request(
        base + "/completions",
        {
            "model": args.served_model,
            "prompt": [row["prompt"] for row in rows[:4]],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
    )
    if len(batch.get("choices", [])) != 4:
        raise RuntimeError("service batch request did not return four choices")
    reset_left = invoke(rows[0])
    invoke(rows[1])
    reset_right = invoke(rows[0])
    if reset_left != reset_right:
        raise RuntimeError("interleaved request changed a repeated prompt result")
    streaming_text = _stream_text(
        base + "/completions",
        {
            "model": args.served_model,
            "prompt": rows[0]["prompt"],
            "max_tokens": len(rows[0]["token_ids"]),
            "temperature": 0,
        },
    )
    memories = []
    outcomes = []
    isolation_passed = True
    for start in range(0, 100, args.concurrency):
        selected = [rows[index % len(rows)] for index in range(start, min(start + args.concurrency, 100))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(invoke, row): row for row in selected}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                result = future.result()
                outcomes.append(result)
                isolation_passed &= (
                    result["text"] == row["text"]
                    or abs(result["first_token_logprob"] - row["first_token_logprob"])
                    <= 1e-3
                )
                memories.append(_memory_mib())
    threshold = statistics.median(memories[:20]) + max(
        statistics.median(memories[:20]) * 0.02, 256.0
    )
    passed = (
        all(row["text_equal"] or row["first_token_logprob_abs"] <= 1e-3 for row in comparisons)
        and max(memories[-20:]) <= threshold
        and len(outcomes) == 100
        and isolation_passed
        and streaming_text == rows[0]["text"]
    )
    payload = {
        "schema_version": 1,
        "passed": passed,
        "model_sha256": checkpoint_digest,
        "served_model": args.served_model,
        "direct_manifest_sha256": hashlib.sha256(
            Path(args.direct).read_bytes()
        ).hexdigest(),
        "health": True,
        "warmups": 20,
        "requests": 100,
        "concurrency": args.concurrency,
        "single_comparisons": comparisons,
        "batch_choices": 4,
        "state_reset_equal": reset_left == reset_right,
        "streaming_equal": streaming_text == rows[0]["text"],
        "concurrent_isolation_passed": isolation_passed,
        "memory_mib": memories,
        "memory_drift_threshold_mib": threshold,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("vLLM service acceptance failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    direct = sub.add_parser("direct")
    direct.add_argument("--model", required=True)
    direct.add_argument("--prompts", required=True)
    direct.add_argument("--output", required=True)
    direct.add_argument("--tensor-parallel-size", type=int, default=1)
    direct.add_argument("--max-tokens", type=int, default=32)
    service = sub.add_parser("service")
    service.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    service.add_argument("--served-model", required=True)
    service.add_argument("--checkpoint", required=True)
    service.add_argument("--direct", required=True)
    service.add_argument("--output", required=True)
    service.add_argument("--max-tokens", type=int, default=32)
    service.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    (write_direct if args.action == "direct" else validate_service)(args)


if __name__ == "__main__":
    main()
