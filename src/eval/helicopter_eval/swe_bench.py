from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from .openai_client import chat_completion
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


SWE_BENCH_DATASETS = {
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


@dataclass(frozen=True, slots=True)
class SweBenchSample:
    sample_index: int
    task_id: str
    instance_id: str
    prompt: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    retrieved_context: str
    harness_dataset_name: str
    reference_patch: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SweBenchResult:
    sample_index: int
    task_id: str
    instance_id: str
    prompt: str
    completion: str
    patch: str
    reference_patch: str
    is_passed: bool
    fail_reason: str
    harness_result: dict[str, Any]

    def to_scoreboard(self) -> ScoreboardEvalResult:
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=self.patch,
            reference_answer=self.reference_patch,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class SweBenchHarnessResult:
    metrics: dict[str, float]
    instance_results: dict[str, dict[str, Any]]
    results_path: Path | None = None
    instance_results_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SweBenchRunConfig:
    base_url: str
    model: str
    benchmark: str
    dataset_name: str
    limit: int | None = None
    split: str = "test"
    source_path: str | None = None
    source_root: str | None = None
    run_harness: bool = False
    predictions_dir: str | None = None
    harness_run_id: str | None = None
    harness_max_workers: int = 1
    harness_cache_level: str | None = None
    harness_clean: bool = False
    harness_timeout_s: float | None = None
    max_context_chars: int = 24000
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout_s: float = 600.0
    scoreboard_dataset: str | None = None
    job_name: str = "code_swe_bench"
    job_id: str | None = None
    runner: str = "helicopter_eval.swe_bench"
    cot_mode: str = "CoT"


def load_samples(config: SweBenchRunConfig) -> list[SweBenchSample]:
    if config.split != "test":
        raise ValueError("SWE-Bench only provides test split in the rwkv-skills catalog")
    if config.dataset_name not in SWE_BENCH_DATASETS:
        raise ValueError(f"unknown SWE-Bench dataset: {config.dataset_name}")
    if config.limit is not None and int(config.limit) < 0:
        raise ValueError("limit must be non-negative")
    rows = _load_rows(config)
    samples = [
        _sample_from_row(index, row, dataset_name=config.dataset_name)
        for index, row in enumerate(rows)
    ]
    if config.limit is not None:
        samples = samples[: int(config.limit)]
    if not samples:
        raise ValueError("SWE-Bench run selected zero samples")
    return samples


def build_prompt(sample: SweBenchSample, *, max_context_chars: int) -> str:
    retrieved_context = sample.retrieved_context.strip()
    if max_context_chars > 0 and len(retrieved_context) > max_context_chars:
        retrieved_context = retrieved_context[:max_context_chars].rstrip()
    lines = [
        "You are resolving a real GitHub issue from SWE-Bench.",
        "Return only a unified git diff patch. Do not include prose, commands, or markdown outside the patch.",
        "The patch must be applicable with git apply from the repository root.",
        "",
        f"Instance: {sample.instance_id}",
    ]
    if sample.repo:
        lines.append(f"Repository: {sample.repo}")
    if sample.base_commit:
        lines.append(f"Base commit: {sample.base_commit}")
    lines.extend(["", "Issue:", sample.problem_statement.strip()])
    if sample.hints_text:
        lines.extend(["", "Hints:", sample.hints_text])
    if retrieved_context:
        lines.extend(["", "Retrieved repository context:", retrieved_context])
    lines.extend(["", "Patch:"])
    return "\n".join(lines)


def extract_swebench_patch(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", str(text or "")).strip()
    fenced = _extract_last_fenced_block(cleaned)
    if fenced is not None:
        cleaned = fenced.strip()
    for marker in ("diff --git ", "--- a/"):
        index = cleaned.find(marker)
        if index >= 0:
            return cleaned[index:].strip()
    return ""


def generate_sample(sample: SweBenchSample, config: SweBenchRunConfig) -> SweBenchResult:
    prompt = build_prompt(sample, max_context_chars=config.max_context_chars)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    patch = extract_swebench_patch(completion)
    return SweBenchResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        instance_id=sample.instance_id,
        prompt=prompt,
        completion=completion,
        patch=patch,
        reference_patch=sample.reference_patch,
        is_passed=False,
        fail_reason="" if patch else "empty_patch",
        harness_result={},
    )


def evaluate_samples(samples: Sequence[SweBenchSample], config: SweBenchRunConfig) -> list[SweBenchResult]:
    return [generate_sample(sample, config) for sample in samples]


def write_predictions(
    results: Sequence[SweBenchResult],
    *,
    config: SweBenchRunConfig,
) -> Path:
    root = Path(config.predictions_dir or os.getenv("HELICOPTER_SWEBENCH_PREDICTIONS_DIR") or "out/swebench_predictions")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{job_id(config)}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "instance_id": result.instance_id,
                        "model_name_or_path": config.model,
                        "model_patch": result.patch,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
    return path


def run_swebench_harness(
    *,
    predictions_path: str | Path,
    dataset_name: str,
    split: str = "test",
    run_id: str | None = None,
    max_workers: int = 1,
    cache_level: str | None = None,
    clean: bool = False,
    timeout_s: float | None = None,
) -> SweBenchHarnessResult:
    if importlib.util.find_spec("swebench") is None:
        raise ModuleNotFoundError("Install the official SWE-Bench harness first: pip install swebench")
    resolved_run_id = run_id or f"helicopter-{Path(predictions_path).stem}"
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
        resolved_run_id,
    ]
    if cache_level:
        cmd.extend(["--cache_level", cache_level])
    if clean:
        cmd.extend(["--clean", "True"])
    subprocess.run(cmd, check=True, timeout=timeout_s)
    return load_swebench_harness_result(run_id=resolved_run_id)


def load_swebench_harness_result(*, run_id: str, root: str | Path = "evaluation_results") -> SweBenchHarnessResult:
    root_path = Path(root)
    result_path = _latest_matching_path(root_path, run_id, "results.json")
    instance_path = _latest_matching_path(root_path, run_id, "instance_results.jsonl")
    result_payload: Mapping[str, Any] = {}
    if result_path is not None:
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    instance_results = _load_instance_results(instance_path)
    return SweBenchHarnessResult(
        metrics=_normalize_harness_metrics(result_payload, instance_results),
        instance_results=instance_results,
        results_path=result_path,
        instance_results_path=instance_path,
    )


def apply_harness_results(
    results: Sequence[SweBenchResult],
    harness: SweBenchHarnessResult,
) -> list[SweBenchResult]:
    updated: list[SweBenchResult] = []
    for result in results:
        instance_result = harness.instance_results.get(result.instance_id, {})
        resolved = _is_instance_resolved(instance_result)
        fail_reason = "" if resolved else _instance_fail_reason(instance_result)
        updated.append(
            SweBenchResult(
                sample_index=result.sample_index,
                task_id=result.task_id,
                instance_id=result.instance_id,
                prompt=result.prompt,
                completion=result.completion,
                patch=result.patch,
                reference_patch=result.reference_patch,
                is_passed=resolved,
                fail_reason=fail_reason,
                harness_result=dict(instance_result),
            )
        )
    return updated


def scoreboard_dataset_name(config: SweBenchRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    return dataset


def job_id(config: SweBenchRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: SweBenchRunConfig) -> dict[str, Any]:
    return {
        "patch": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
            "max_context_chars": config.max_context_chars,
        }
    }


def task_sampling_config(config: SweBenchRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_swebench_patch",
        "harness_dataset": SWE_BENCH_DATASETS[config.dataset_name]["harness_dataset"],
        "run_harness": bool(config.run_harness),
        "sampling_config": completion_sampling_config(config),
    }


def write_results(
    results: Sequence[SweBenchResult],
    *,
    config: SweBenchRunConfig,
    repo_root: Path,
    harness_metrics: Mapping[str, float],
) -> int:
    task_id = asyncio.run(
        write_scoreboard_results(
            [result.to_scoreboard() for result in results],
            config=ScoreboardWriteConfig(
                dataset=scoreboard_dataset_name(config),
                model=config.model,
                job_name=config.job_name,
                job_id=job_id(config),
                benchmark=config.benchmark,
                runner=config.runner,
                cot_mode=config.cot_mode,
                sampling_config=task_sampling_config(config),
                completion_sampling_config=completion_sampling_config(config),
                extra_metrics=dict(harness_metrics),
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_swe_bench(config: SweBenchRunConfig, *, repo_root: Path) -> dict[str, Any]:
    if not config.run_harness:
        raise ValueError("SWE-Bench formal scoring requires --swebench-run-harness or HELICOPTER_SWEBENCH_RUN_HARNESS=1")
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    predictions_path = write_predictions(results, config=config)
    harness = run_swebench_harness(
        predictions_path=predictions_path,
        dataset_name=str(SWE_BENCH_DATASETS[config.dataset_name]["harness_dataset"]),
        split=config.split,
        run_id=config.harness_run_id or job_id(config),
        max_workers=config.harness_max_workers,
        cache_level=config.harness_cache_level,
        clean=config.harness_clean,
        timeout_s=config.harness_timeout_s,
    )
    scored_results = apply_harness_results(results, harness)
    task_id = write_results(scored_results, config=config, repo_root=repo_root, harness_metrics=harness.metrics)
    passed = sum(1 for result in scored_results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "source_dataset": config.dataset_name,
        "predictions_path": str(predictions_path),
        "harness_metrics": dict(harness.metrics),
        "total": len(scored_results),
        "passed": passed,
        "accuracy": passed / len(scored_results) if scored_results else 0.0,
    }


def dry_run_summary(config: SweBenchRunConfig) -> dict[str, Any]:
    samples = load_samples(config)
    source = _resolve_rows_path(config)
    return {
        "benchmark": config.benchmark,
        "source": str(source) if source is not None else str(SWE_BENCH_DATASETS[config.dataset_name]["hf_dataset"]),
        "split": config.split,
        "source_dataset": config.dataset_name,
        "hf_dataset": SWE_BENCH_DATASETS[config.dataset_name]["hf_dataset"],
        "harness_dataset": SWE_BENCH_DATASETS[config.dataset_name]["harness_dataset"],
        "limit": config.limit,
        "available_samples": len(samples),
        "base_url": config.base_url,
        "model": config.model,
        "run_harness": bool(config.run_harness),
        "harness_installed": importlib.util.find_spec("swebench") is not None,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


def _load_rows(config: SweBenchRunConfig) -> list[Mapping[str, Any]]:
    source_path = _resolve_rows_path(config)
    if source_path is not None:
        return _load_local_rows(source_path)
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install `datasets` or provide prepared SWE-Bench JSONL manifests") from exc
    dataset = load_dataset(str(SWE_BENCH_DATASETS[config.dataset_name]["hf_dataset"]), split=config.split)
    return sorted(dataset, key=lambda item: str(item.get("instance_id", "")))


def _load_local_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"no JSON/JSONL SWE-Bench source files under {path}")
        path = candidates[0]
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if raw:
                    rows.append(json.loads(raw))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("instances"), list):
        return payload["instances"]
    raise ValueError(f"unsupported SWE-Bench local source format: {path}")


def _sample_from_row(index: int, row: Mapping[str, Any], *, dataset_name: str) -> SweBenchSample:
    instance_id = str(row.get("instance_id") or row.get("task_id") or row.get("id") or "").strip()
    if not instance_id:
        raise ValueError(f"SWE-Bench row missing instance_id: {row}")
    dataset_info = SWE_BENCH_DATASETS[dataset_name]
    return SweBenchSample(
        sample_index=index,
        task_id=instance_id,
        instance_id=instance_id,
        prompt=str(row.get("prompt") or row.get("problem_statement") or ""),
        repo=str(row.get("repo") or ""),
        base_commit=str(row.get("base_commit") or ""),
        problem_statement=str(row.get("problem_statement") or row.get("prompt") or row.get("issue") or "").strip(),
        hints_text=str(row.get("hints_text") or "").strip(),
        retrieved_context=_extract_context(row),
        harness_dataset_name=str(row.get("harness_dataset_name") or dataset_info["harness_dataset"]),
        reference_patch=str(row.get("patch") or ""),
        metadata=dict(row),
    )


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


def _resolve_rows_path(config: SweBenchRunConfig) -> Path | None:
    if config.source_path:
        candidate = Path(config.source_path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"SWE-Bench source_path does not exist: {candidate}")
    dataset_env = config.dataset_name.upper()
    for key in (
        f"HELICOPTER_{dataset_env}_MANIFEST",
        f"RWKV_{dataset_env}_MANIFEST",
        f"{dataset_env}_MANIFEST",
        "HELICOPTER_SWEBENCH_MANIFEST",
        "RWKV_SKILLS_SWEBENCH_SOURCE",
    ):
        raw = os.getenv(key)
        if raw:
            candidate = Path(raw).expanduser()
            if candidate.exists():
                return candidate.resolve()
    for raw in (
        config.source_root,
        os.getenv("HELICOPTER_SWEBENCH_DATA_ROOT"),
        os.getenv("RWKV_SWEBENCH_DATA_ROOT"),
        os.getenv("SWEBENCH_DATA_ROOT"),
    ):
        if raw:
            candidate = Path(raw).expanduser() / config.dataset_name / "test.jsonl"
            if candidate.is_file():
                return candidate.resolve()
    for candidate in (
        _repo_root() / "data" / config.dataset_name / "test.jsonl",
        Path("/home/chase/GitHub/rwkv-skills/data") / config.dataset_name / "test.jsonl",
        Path("/home/chase/rwkv-skills/data") / config.dataset_name / "test.jsonl",
        Path("/tmp/rwkv-skills/data") / config.dataset_name / "test.jsonl",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _extract_last_fenced_block(text: str) -> str | None:
    matches = list(_FENCED_BLOCK_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1)


def _latest_matching_path(root: Path, run_id: str, name: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.rglob(name) if run_id in str(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_instance_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            instance_id = str(row.get("instance_id") or row.get("id") or "")
            if instance_id:
                results[instance_id] = row
    return results


def _normalize_harness_metrics(
    result_payload: Mapping[str, Any],
    instance_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    submitted = _numeric_metric(result_payload, ("submitted", "instances_submitted", "total_instances"))
    completed = _numeric_metric(result_payload, ("completed", "instances_completed"))
    resolved = _numeric_metric(result_payload, ("resolved", "instances_resolved"))
    if instance_results:
        submitted = submitted or float(len(instance_results))
        resolved = resolved or float(sum(1 for row in instance_results.values() if _is_instance_resolved(row)))
        completed = completed or float(sum(1 for row in instance_results.values() if not _instance_failed_to_run(row)))
    rate = _numeric_metric(result_payload, ("resolved_rate", "resolution_rate"))
    if rate == 0.0 and submitted:
        rate = resolved / submitted
    return {
        "swebench_instances_submitted": float(submitted),
        "swebench_instances_completed": float(completed),
        "swebench_instances_resolved": float(resolved),
        "swebench_resolution_rate": float(rate),
    }


def _numeric_metric(payload: Mapping[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return float(len(value))
    return 0.0


def _is_instance_resolved(row: Mapping[str, Any]) -> bool:
    for key in ("resolved", "is_resolved", "passed", "is_passed"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "resolved"}:
            return True
    return False


def _instance_failed_to_run(row: Mapping[str, Any]) -> bool:
    for key in ("completed", "ran", "success"):
        value = row.get(key)
        if isinstance(value, bool):
            return not value
    return False


def _instance_fail_reason(row: Mapping[str, Any]) -> str:
    for key in ("fail_reason", "error", "status", "result"):
        value = row.get(key)
        if value:
            return str(value)
    return "unresolved"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = [
    "SWE_BENCH_DATASETS",
    "SweBenchRunConfig",
    "build_prompt",
    "dry_run_summary",
    "evaluate_samples",
    "extract_swebench_patch",
    "load_samples",
    "run_swe_bench",
    "run_swebench_harness",
]
