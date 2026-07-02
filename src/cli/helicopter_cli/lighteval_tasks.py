from __future__ import annotations

import argparse
from dataclasses import asdict
from pprint import pformat


def inspect_tasks(args: argparse.Namespace) -> int:
    from lighteval.tasks.registry import Registry

    registry = Registry(
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
