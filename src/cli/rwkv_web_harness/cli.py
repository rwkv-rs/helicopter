from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .batch import TaskSpec, load_task_specs, summarize_cases, validate_case
from .models import RWKVLocalBackend
from .runner import AgentConfig, AgentRunner
from .tools import WebToolkit
from .trace import TraceWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rwkv-web-harness",
        description="Run a local RWKV model through a read-only live web research loop.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one research task")
    run.add_argument("--task", help="research question")
    run.add_argument("--tasks-file", type=Path, help="JSONL file containing task_id/question records")
    run.add_argument("--task-id", help="task id when --tasks-file is used")
    run.add_argument("--model-url", default=os.environ.get("RWKV_MODEL_URL", "http://127.0.0.1:8000/v1"))
    run.add_argument("--model", default=os.environ.get("RWKV_MODEL_NAME", "RWKV"))
    run.add_argument("--api-key", default=os.environ.get("RWKV_MODEL_API_KEY"))
    run.add_argument(
        "--interface",
        choices=("chat", "completion", "rwkv-json", "g1h"),
        default=os.environ.get("RWKV_MODEL_INTERFACE", "chat"),
        help="chat uses native tools; g1h uses User✿/Bot✿ JSON calls; rwkv-json uses generic JSON; completion uses legacy tags",
    )
    run.add_argument(
        "--endpoint",
        help="optional endpoint path override, e.g. /v1/chat/completions or /v1/completions",
    )
    run.add_argument("--search-url", default=os.environ.get("RWKV_WEB_SEARCH_URL"))
    run.add_argument(
        "--search-backend",
        choices=("html", "searxng"),
        default=os.environ.get("RWKV_WEB_SEARCH_BACKEND", "html"),
        help="html uses a normal search webpage; searxng expects a self-hosted SearXNG endpoint",
    )
    run.add_argument("--trace", type=Path, help="JSONL trace path")
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-context-chars", type=int, default=12000)
    run.add_argument("--max-new-tokens", type=int, default=768)
    run.add_argument("--max-page-chars", type=int, default=6000)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=120.0)
    run.set_defaults(handler=_run)

    batch = subparsers.add_parser("batch", help="run a JSONL research suite")
    batch.add_argument("--tasks-file", type=Path, required=True, help="JSONL task suite")
    batch.add_argument("--summary", type=Path, default=Path("results/web_harness/batch_summary.json"))
    batch.add_argument("--trace-dir", type=Path, default=Path("results/web_harness/batch"))
    batch.add_argument("--max-tasks", type=int, help="only run the first N tasks")
    batch.add_argument("--retries", type=int, default=1, help="retries after a failed validation")
    batch.add_argument("--resume", action="store_true", help="reuse passed cases from an existing summary")
    batch.add_argument(
        "--resume-from",
        type=Path,
        help="read passed cases from this summary while writing a new --summary",
    )
    _add_backend_arguments(batch)
    batch.set_defaults(handler=_batch)

    preflight = subparsers.add_parser("preflight", help="check the local model and web search endpoints")
    preflight.add_argument("--model-url", default=os.environ.get("RWKV_MODEL_URL", "http://127.0.0.1:8000/v1"))
    preflight.add_argument("--search-url", default=os.environ.get("RWKV_WEB_SEARCH_URL"))
    preflight.add_argument(
        "--search-backend",
        choices=("html", "searxng"),
        default=os.environ.get("RWKV_WEB_SEARCH_BACKEND", "html"),
    )
    preflight.add_argument("--timeout", type=float, default=10.0)
    preflight.set_defaults(handler=_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        print(f"rwkv-web-harness: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    task_id, question = _load_task(args)
    trace_path = args.trace or Path("results/web_harness") / f"{task_id}.jsonl"
    result = _run_task(args, task_id=task_id, question=question, trace_path=trace_path)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-url", default=os.environ.get("RWKV_MODEL_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("RWKV_MODEL_NAME", "RWKV"))
    parser.add_argument("--api-key", default=os.environ.get("RWKV_MODEL_API_KEY"))
    parser.add_argument(
        "--interface",
        choices=("chat", "completion", "rwkv-json", "g1h"),
        default=os.environ.get("RWKV_MODEL_INTERFACE", "chat"),
        help="chat uses native tools; g1h uses User✿/Bot✿ JSON calls; rwkv-json uses generic JSON; completion uses legacy tags",
    )
    parser.add_argument("--endpoint", help="optional endpoint path override")
    parser.add_argument("--search-url", default=os.environ.get("RWKV_WEB_SEARCH_URL"))
    parser.add_argument(
        "--search-backend",
        choices=("html", "searxng"),
        default=os.environ.get("RWKV_WEB_SEARCH_BACKEND", "html"),
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-page-chars", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)


def _run_task(
    args: argparse.Namespace,
    *,
    task_id: str,
    question: str,
    trace_path: Path,
    tool_sequence: tuple[str, ...] | None = None,
):
    search_url = args.search_url
    if search_url is None:
        search_url = (
            "https://lite.duckduckgo.com/lite/"
            if args.search_backend == "html"
            else "http://127.0.0.1:8080/search"
        )
    backend = RWKVLocalBackend(
        base_url=args.model_url,
        model=args.model,
        timeout=args.timeout,
        api_key=args.api_key,
        interface=args.interface,
        endpoint=args.endpoint,
    )
    toolkit = WebToolkit(
        search_url=search_url,
        search_backend=args.search_backend,
        timeout=min(args.timeout, 30.0),
        max_page_chars=args.max_page_chars,
    )
    config = AgentConfig(
        max_steps=args.max_steps,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tool_sequence=tool_sequence,
    )
    with TraceWriter(trace_path) as trace:
        return AgentRunner(backend=backend, toolkit=toolkit, config=config, trace=trace).run(
            task_id=task_id,
            question=question,
        )


def _batch(args: argparse.Namespace) -> int:
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("--max-tasks must be positive")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    specs = load_task_specs(args.tasks_file)
    if args.max_tasks is not None:
        specs = specs[: args.max_tasks]

    existing: dict[str, dict[str, Any]] = {}
    resume_path = args.resume_from or args.summary
    if (args.resume or args.resume_from) and resume_path.is_file():
        previous = json.loads(resume_path.read_text(encoding="utf-8"))
        for case in previous.get("cases", []):
            if isinstance(case, dict):
                task = case.get("task", {})
                task_id = task.get("task_id") if isinstance(task, dict) else None
                if isinstance(task_id, str) and case.get("validation", {}).get("passed") is True:
                    existing[task_id] = case

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        if spec.task_id in existing:
            cases.append(existing[spec.task_id])
            _write_batch_report(args.summary, cases, args.tasks_file)
            print(json.dumps({"index": index, "task_id": spec.task_id, "resumed": True, "passed": True}, ensure_ascii=False))
            continue
        attempts: list[dict[str, Any]] = []
        final_case: dict[str, Any] | None = None
        for attempt in range(1, args.retries + 2):
            trace_path = args.trace_dir / f"{_safe_name(spec.task_id)}.attempt{attempt}.jsonl"
            result = _run_task(
                args,
                task_id=spec.task_id,
                question=spec.question,
                trace_path=trace_path,
                tool_sequence=spec.expected_tools,
            )
            validation = validate_case(result, spec, trace_path)
            final_case = {
                "task": asdict(spec),
                "attempt": attempt,
                "result": result.as_dict(),
                "validation": validation,
                "trace_path": str(trace_path),
            }
            attempts.append(
                {
                    "attempt": attempt,
                    "status": result.status,
                    "error": result.error,
                    "passed": validation["passed"],
                    "validation_error": validation["error"],
                    "trace_path": str(trace_path),
                }
            )
            if validation["passed"]:
                break
        if final_case is None:
            raise RuntimeError(f"batch task did not execute: {spec.task_id}")
        final_case["attempts"] = attempts
        cases.append(final_case)
        _write_batch_report(args.summary, cases, args.tasks_file)
        print(
            json.dumps(
                {
                    "index": index,
                    "task_id": spec.task_id,
                    "attempt": final_case["attempt"],
                    "status": final_case["result"]["status"],
                    "passed": final_case["validation"]["passed"],
                    "validation_error": final_case["validation"]["error"],
                },
                ensure_ascii=False,
            )
        )

    report = _write_batch_report(args.summary, cases, args.tasks_file)
    print(
        json.dumps(
            {
                "suite": report["suite"],
                "total": report["total"],
                "passed": report["passed"],
                "failed": report["failed"],
                "network_verified": report["network_verified"],
                "summary": str(args.summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] == report["total"] else 1


def _write_batch_report(summary_path: Path, cases: list[dict[str, Any]], tasks_file: Path) -> dict[str, Any]:
    report = summarize_cases(cases, suite=str(tasks_file))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "task"


def _load_task(args: argparse.Namespace) -> tuple[str, str]:
    if args.task and args.tasks_file:
        raise ValueError("use either --task or --tasks-file, not both")
    if args.task:
        return args.task_id or "interactive", args.task
    if not args.tasks_file:
        raise ValueError("run requires --task or --tasks-file")
    selected: dict[str, Any] | None = None
    with args.tasks_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("task_id") or row.get("id") or "")
            if args.task_id is None or row_id == args.task_id:
                selected = row
                break
    if selected is None:
        raise ValueError(f"task not found: {args.task_id or '<first non-empty record>'}")
    question = selected.get("question", selected.get("instruction", selected.get("task")))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("task record must contain question, instruction, or task")
    return str(selected.get("task_id") or selected.get("id") or args.task_id or "task"), question


def _preflight(args: argparse.Namespace) -> int:
    search_url = args.search_url
    if search_url is None:
        search_url = "https://lite.duckduckgo.com/lite/" if args.search_backend == "html" else "http://127.0.0.1:8080/search"
    checks = {
        "model": _probe(f"{args.model_url.rstrip('/')}/models", args.timeout),
        "search": _probe(f"{search_url}{'&' if '?' in search_url else '?'}{urlencode({'q': 'RWKV'})}", args.timeout),
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


def _probe(url: str, timeout: float) -> bool:
    try:
        request = Request(url, headers={"User-Agent": "rwkv-web-harness/0.1"})
        with urlopen(request, timeout=timeout) as response:
            response.read(64)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
