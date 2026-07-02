from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from pprint import pformat


def load_registry(*, tasks: str | None = None, custom_tasks: str | None = None, load_multilingual: bool = False):
    from lighteval.tasks.registry import Registry

    return Registry(
        tasks=tasks,
        custom_tasks=custom_tasks,
        load_multilingual=load_multilingual,
    )


def inspect_tasks(args: argparse.Namespace) -> int:
    registry = load_registry(
        tasks=args.tasks,
        custom_tasks=args.custom_tasks,
        load_multilingual=args.load_multilingual,
    )
    task_dict = registry.load_tasks()
    for name, task in task_dict.items():
        print("-" * 10, name, "-" * 10)
        if args.show_config:
            print("-" * 10, "CONFIG")
            print(str(task.config), end="")
        for index, sample in enumerate(task.eval_docs()[: int(args.num_samples)]):
            if index == 0:
                print("-" * 10, "SAMPLES")
            print(f"-- sample {index} --")
            print(pformat(asdict(sample), indent=2))
    return 0


def selected_task_rows(args: argparse.Namespace) -> list[tuple[str, str]]:
    registry = load_registry(
        custom_tasks=args.custom_tasks,
        load_multilingual=args.load_multilingual,
    )
    rows = [("task", name) for name in sorted(registry._task_registry)]
    if args.include_supersets:
        rows.extend(("superset", name) for name in sorted(registry._task_superset_dict))

    patterns = [pattern.casefold() for pattern in args.contains or []]
    if patterns:
        rows = [(kind, name) for kind, name in rows if any(pattern in name.casefold() for pattern in patterns)]
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    return rows


def format_export(rows: list[tuple[str, str]], output_format: str) -> str:
    if output_format == "jsonl":
        return "".join(json.dumps({"kind": kind, "task": task}, sort_keys=True) + "\n" for kind, task in rows)
    return "".join(task + "\n" for _, task in rows)


def export_tasks(args: argparse.Namespace) -> int:
    text = format_export(selected_task_rows(args), args.format)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m helicopter_cli.lighteval_tasks")
    subparsers = parser.add_subparsers(dest="task_action", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("tasks")
    inspect_parser.add_argument("--load-multilingual", action="store_true")
    inspect_parser.add_argument("--custom-tasks")
    inspect_parser.add_argument("--num-samples", type=int, default=10)
    inspect_parser.add_argument("--show-config", action="store_true")
    inspect_parser.set_defaults(handler=inspect_tasks)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--load-multilingual", action="store_true")
    export_parser.add_argument("--custom-tasks")
    export_parser.add_argument("--output")
    export_parser.add_argument("--format", choices=("text", "jsonl"), default="text")
    export_parser.add_argument("--contains", action="append")
    export_parser.add_argument("--limit", type=int)
    export_parser.add_argument("--include-supersets", action="store_true")
    export_parser.set_defaults(handler=export_tasks)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
