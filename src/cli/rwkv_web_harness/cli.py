from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
    run.add_argument("--search-url", default=os.environ.get("RWKV_WEB_SEARCH_URL"))
    run.add_argument(
        "--search-backend",
        choices=("html", "searxng"),
        default=os.environ.get("RWKV_WEB_SEARCH_BACKEND", "html"),
        help="html uses a normal search webpage; searxng expects a self-hosted SearXNG endpoint",
    )
    run.add_argument("--trace", type=Path, help="JSONL trace path")
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-context-chars", type=int, default=24000)
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--max-page-chars", type=int, default=6000)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=120.0)
    run.set_defaults(handler=_run)

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
    search_url = args.search_url
    if search_url is None:
        search_url = "https://lite.duckduckgo.com/lite/" if args.search_backend == "html" else "http://127.0.0.1:8080/search"
    trace_path = args.trace or Path("results/web_harness") / f"{task_id}.jsonl"
    backend = RWKVLocalBackend(
        base_url=args.model_url,
        model=args.model,
        timeout=args.timeout,
        api_key=args.api_key,
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
    )
    with TraceWriter(trace_path) as trace:
        result = AgentRunner(backend=backend, toolkit=toolkit, config=config, trace=trace).run(
            task_id=task_id,
            question=question,
        )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


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
