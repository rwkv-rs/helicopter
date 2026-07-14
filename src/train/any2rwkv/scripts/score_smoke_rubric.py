#!/usr/bin/env python3
"""Score the 32 deterministic generations with the frozen minimum-usability judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from any2rwkv.calibration import file_sha256
from any2rwkv.source import verify_source


def request_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        value = json.loads(response.read())
    result = json.loads(value["choices"][0]["message"]["content"])
    if set(result) != {"coherent", "relevant"} or any(
        type(result[name]) is not bool for name in result
    ):
        raise ValueError(f"judge did not return the frozen boolean schema: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", required=True, type=Path)
    parser.add_argument("--quality-suite", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model", required=True)
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
            judgment = request_json(
                args.base_url.rstrip("/") + "/v1/chat/completions",
                {
                    "model": args.served_model,
                    "temperature": float(rubric["judge_temperature"]),
                    "max_tokens": 32,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": rubric["judge_prompt"]},
                        {
                            "role": "user",
                            "content": f"PROMPT:\n{row['prompt']}\n\nRESPONSE:\n{row['generated_text']}",
                        },
                    ],
                },
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
        "judge_served_model": args.served_model,
        "judge_source_manifest_sha256": file_sha256(args.judge_source_manifest),
        "judge_checkpoint_verification": judge_verification,
        "temperature": rubric["judge_temperature"],
        "quality_sha256": file_sha256(args.quality),
        "student_sha256": quality.get("binding", {}).get("student_sha256"),
        "pass_count": passed,
        "pass_rate": passed / len(results),
        "passed": passed / len(results) >= float(rubric["minimum_pass_rate"]),
        "rows": results,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
