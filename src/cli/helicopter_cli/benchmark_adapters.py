from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


SWE_BENCH_DATASETS: dict[str, dict[str, str]] = {
    "swe_bench": {
        "hf_dataset": "princeton-nlp/SWE-bench",
        "harness_dataset": "princeton-nlp/SWE-bench",
    },
    "swe_bench_lite": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
    "swe_bench_verified": {
        "hf_dataset": "princeton-nlp/SWE-bench_Verified",
        "harness_dataset": "princeton-nlp/SWE-bench_Verified",
    },
    "swe_bench_lite_oracle": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite_oracle",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
    "swe_bench_lite_bm25_13k": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite_bm25_13K",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
}

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCED_BLOCK_RE = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class SweBenchRecord:
    task_id: str
    instance_id: str
    prompt: str
    repo: str
    base_commit: str
    hints_text: str
    retrieved_context: str
    source_dataset: str
    harness_dataset_name: str
    patch: str
    metadata: dict[str, Any]


def load_suite(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def suite_adapter_entries(suite: Mapping[str, Any], benchmark_names: Sequence[str]) -> list[tuple[str, dict[str, Any]]]:
    raw_benchmarks = suite.get("benchmarks", {})
    if not isinstance(raw_benchmarks, Mapping):
        raise SystemExit("suite file must contain a [benchmarks] table")
    wanted = {name for name in benchmark_names if name}
    selected: list[tuple[str, dict[str, Any]]] = []
    for name, raw_entry in raw_benchmarks.items():
        if wanted and name not in wanted:
            continue
        entry = dict(raw_entry) if isinstance(raw_entry, Mapping) else {}
        if str(entry.get("status") or "") == "adapter" or entry.get("adapter"):
            selected.append((str(name), entry))
    missing = sorted(wanted - {name for name, _entry in selected})
    if missing:
        raise SystemExit(f"selected benchmarks are not configured as adapters: {', '.join(missing)}")
    if not selected:
        raise SystemExit("suite adapter selection is empty")
    return selected


def _read_json_or_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    payload = json.loads(text)
                    if isinstance(payload, Mapping):
                        rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping) and isinstance(payload.get("instances"), list):
        return [item for item in payload["instances"] if isinstance(item, Mapping)]
    raise ValueError(f"unsupported JSON source for SWE-Bench rows: {path}")


def _load_local_rows(path: Path) -> list[Mapping[str, Any]]:
    target = path.expanduser().resolve()
    if target.is_dir():
        candidates = sorted(target.glob("*.jsonl")) + sorted(target.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"no JSON/JSONL SWE-Bench source files under {target}")
        target = candidates[0]
    return _read_json_or_jsonl(target)


def load_swebench_source_rows(dataset_name: str, split: str) -> list[Mapping[str, Any]]:
    source_override = os.environ.get("HELICOPTER_SWEBENCH_SOURCE") or os.environ.get("RWKV_SKILLS_SWEBENCH_SOURCE")
    if source_override:
        return _load_local_rows(Path(source_override))
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install datasets to load SWE-Bench sources") from exc
    dataset = load_dataset(dataset_name, split=split)
    return sorted(dataset, key=lambda item: str(item.get("instance_id", "")))


def _extract_context(row: Mapping[str, Any]) -> str:
    for key in (
        "retrieved_context",
        "context",
        "text",
        "file_context",
        "oracle_context",
        "bm25_context",
        "repo_context",
    ):
        value = row.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)
    return ""


def normalize_swebench_row(
    row: Mapping[str, Any],
    *,
    source_dataset: str,
    harness_dataset: str,
) -> SweBenchRecord:
    instance_id = str(row.get("instance_id") or row.get("task_id") or row.get("id") or "").strip()
    if not instance_id:
        raise ValueError(f"SWE-Bench row missing instance_id: {row}")
    problem_statement = str(row.get("problem_statement") or row.get("prompt") or row.get("issue") or "").strip()
    metadata = {
        "instance_id": instance_id,
        "repo": str(row.get("repo") or "").strip(),
        "base_commit": str(row.get("base_commit") or "").strip(),
        "source_dataset": source_dataset,
        "harness_dataset_name": harness_dataset,
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": row.get("PASS_TO_PASS"),
        "environment_setup_commit": row.get("environment_setup_commit"),
        "difficulty": row.get("difficulty"),
    }
    return SweBenchRecord(
        task_id=instance_id,
        instance_id=instance_id,
        prompt=problem_statement,
        repo=str(metadata["repo"]),
        base_commit=str(metadata["base_commit"]),
        hints_text=str(row.get("hints_text") or "").strip(),
        retrieved_context=_extract_context(row),
        source_dataset=source_dataset,
        harness_dataset_name=harness_dataset,
        patch=str(row.get("patch") or ""),
        metadata=metadata,
    )


def load_swebench_records(benchmark_name: str, entry: Mapping[str, Any], *, split: str) -> list[SweBenchRecord]:
    spec = SWE_BENCH_DATASETS.get(benchmark_name, {})
    source_dataset = str(entry.get("hf_dataset") or spec.get("hf_dataset") or "")
    harness_dataset = str(entry.get("harness_dataset") or spec.get("harness_dataset") or "")
    if not source_dataset or not harness_dataset:
        raise ValueError(f"{benchmark_name} is missing hf_dataset/harness_dataset adapter metadata")
    rows = load_swebench_source_rows(source_dataset, split)
    return [
        normalize_swebench_row(row, source_dataset=source_dataset, harness_dataset=harness_dataset)
        for row in rows
    ]


def select_records(records: Sequence[SweBenchRecord], *, max_samples: int | None, sample_seed: int | None) -> list[tuple[int, SweBenchRecord]]:
    indexed = list(enumerate(records))
    if max_samples is None or max_samples <= 0 or max_samples >= len(indexed):
        return indexed
    if sample_seed is None:
        return indexed[:max_samples]
    rng = random.Random(sample_seed)
    return sorted(rng.sample(indexed, max_samples), key=lambda item: item[0])


def build_swebench_prompt(record: SweBenchRecord, *, max_context_chars: int | None = None, prompt_profile: str = "normal") -> str:
    if str(prompt_profile or "normal").strip().lower() == "naive":
        return f"User: {record.prompt.strip()}\n\nAssistant: <think>\n</think>\n```diff\n"
    retrieved_context = record.retrieved_context.strip()
    if max_context_chars is not None and max_context_chars > 0 and len(retrieved_context) > max_context_chars:
        retrieved_context = retrieved_context[:max_context_chars].rstrip()
    lines = [
        "User: You are resolving a real GitHub issue from SWE-bench.",
        "Return only a unified git diff patch. Do not include prose, commands, or markdown outside the patch.",
        "The patch must be applicable with git apply from the repository root.",
        "",
        f"Instance: {record.instance_id}",
    ]
    if record.repo:
        lines.append(f"Repository: {record.repo}")
    if record.base_commit:
        lines.append(f"Base commit: {record.base_commit}")
    lines.extend(["", "Issue:", record.prompt.strip()])
    if record.hints_text:
        lines.extend(["", "Hints:", record.hints_text])
    if retrieved_context:
        lines.extend(["", "Retrieved repository context:", retrieved_context])
    lines.extend(["", "Assistant: <think>\n</think>\n```diff\n"])
    return "\n".join(lines)


def extract_swebench_patch(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    matches = list(_FENCED_BLOCK_RE.finditer(cleaned))
    if matches:
        cleaned = matches[-1].group(1).strip()
    for marker in ("diff --git ", "--- a/"):
        index = cleaned.find(marker)
        if index >= 0:
            return cleaned[index:].strip()
    return ""


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[str, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1, int(max_tokens)),
        "temperature": float(temperature),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "authorization": f"Bearer {api_key or 'EMPTY'}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible endpoint HTTP {exc.code}: {detail}") from exc
    choices = data.get("choices") if isinstance(data, Mapping) else None
    if not isinstance(choices, list) or not choices:
        return "", ""
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    content = str(message.get("content") or choice.get("text") or "")
    finish_reason = str(choice.get("finish_reason") or "")
    return content, finish_reason


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            file.write("\n")


def run_swebench_harness(
    *,
    predictions_path: Path,
    dataset_name: str,
    split: str,
    run_id: str,
    max_workers: int,
    timeout_s: float | None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max(1, int(max_workers))),
        "--run_id",
        run_id,
    ]
    subprocess.run(cmd, check=True, timeout=timeout_s)


def run_swebench_adapter(name: str, entry: Mapping[str, Any], args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    split = str(getattr(args, "split", None) or entry.get("split") or "test")
    records = load_swebench_records(name, entry, split=split)
    selected = select_records(records, max_samples=args.max_samples, sample_seed=args.sample_seed)
    benchmark_dir = run_root / name
    completions: list[dict[str, Any]] = []
    predictions: list[dict[str, str]] = []
    for ordinal, (dataset_index, record) in enumerate(selected):
        prompt = build_swebench_prompt(
            record,
            max_context_chars=args.swebench_max_context_chars,
            prompt_profile=args.swebench_prompt_profile,
        )
        started = time.monotonic()
        completion, finish_reason = call_chat_completion(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model_name,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        latency_s = time.monotonic() - started
        patch = extract_swebench_patch(completion)
        completions.append(
            {
                "benchmark_name": name,
                "dataset_split": split,
                "sample_index": ordinal,
                "dataset_index": dataset_index,
                "instance_id": record.instance_id,
                "prompt": prompt,
                "completion0": completion,
                "finish_reason": finish_reason,
                "latency_s": latency_s,
                "model_name": args.model_name,
                "source_dataset": record.source_dataset,
                "harness_dataset_name": record.harness_dataset_name,
                "patch_chars": len(patch),
                "metadata": record.metadata,
            }
        )
        predictions.append(
            {
                "instance_id": record.instance_id,
                "model_name_or_path": args.model_name,
                "model_patch": patch,
            }
        )
        print(f"{name}: generated {ordinal + 1}/{len(selected)} instance={record.instance_id} patch_chars={len(patch)}")

    completions_path = benchmark_dir / "completions.jsonl"
    predictions_path = benchmark_dir / "predictions.jsonl"
    write_jsonl(completions_path, completions)
    write_jsonl(predictions_path, predictions)
    nonempty = sum(1 for item in predictions if item["model_patch"].strip())
    metrics: dict[str, Any] = {
        "adapter": "swebench",
        "benchmark": name,
        "samples": len(selected),
        "source_rows": len(records),
        "predictions": len(predictions),
        "nonempty_patches": nonempty,
        "nonempty_patch_rate": (nonempty / len(predictions)) if predictions else 0.0,
        "swebench_harness_ran": 0.0,
        "source_dataset": records[0].source_dataset if records else entry.get("hf_dataset"),
        "harness_dataset_name": records[0].harness_dataset_name if records else entry.get("harness_dataset"),
        "split": split,
        "completions_path": str(completions_path),
        "predictions_path": str(predictions_path),
    }
    if args.swebench_run_harness:
        run_swebench_harness(
            predictions_path=predictions_path,
            dataset_name=str(metrics["harness_dataset_name"]),
            split=split,
            run_id=args.run_id,
            max_workers=args.swebench_harness_workers,
            timeout_s=args.swebench_harness_timeout_s,
        )
        metrics["swebench_harness_ran"] = 1.0
    metrics_path = benchmark_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite_path)
    entries = suite_adapter_entries(suite, args.benchmark or [])
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.run_id = run_id
    run_root = Path(args.output_dir).expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    for name, entry in entries:
        adapter = str(entry.get("adapter") or "")
        if adapter != "swebench":
            raise SystemExit(f"unsupported adapter for {name}: {adapter}")
        all_metrics.append(run_swebench_adapter(name, entry, args, run_root))
    summary = {
        "run_id": run_id,
        "model": args.model_name,
        "base_url": args.base_url,
        "benchmarks": [item["benchmark"] for item in all_metrics],
        "metrics": all_metrics,
        "run_root": str(run_root),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m helicopter_cli.benchmark_adapters")
    parser.add_argument("--suite-path", required=True)
    parser.add_argument("--benchmark", action="append", help="suite benchmark name; repeatable")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--swebench-max-context-chars", type=int)
    parser.add_argument("--swebench-prompt-profile", choices=("normal", "naive"), default="normal")
    parser.add_argument("--swebench-run-harness", action="store_true")
    parser.add_argument("--swebench-harness-workers", type=int, default=1)
    parser.add_argument("--swebench-harness-timeout-s", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
