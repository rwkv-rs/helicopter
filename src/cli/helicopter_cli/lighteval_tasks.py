from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any


@dataclass(frozen=True)
class SourceBenchmark:
    name: str
    field: str | None = None


@dataclass(frozen=True)
class CoverageRow:
    source: str
    field: str | None
    status: str
    target_kind: str | None
    targets: tuple[str, ...]
    candidates: tuple[str, ...]


OFFICIAL_LIGHTEVAL_ALIASES: dict[str, tuple[str, ...]] = {
    # The source names are rwkv-skills benchmark ids; the target names are
    # official LightEval task or superset ids verified in the registry.
    "ceval": ("ceval_zho_mcf",),
    "cmmlu": ("cmmlu_zho_mcf",),
    "ifbench": ("ifbench_test", "ifbench_multiturn"),
    "include": ("include_tgl_mcf",),
    "livecodebench": ("lcb",),
    "mmmlu": (
        "openai_mmlu_ara_mcf",
        "openai_mmlu_ben_mcf",
        "openai_mmlu_deu_mcf",
        "openai_mmlu_fra_mcf",
        "openai_mmlu_hin_mcf",
        "openai_mmlu_ind_mcf",
        "openai_mmlu_ita_mcf",
        "openai_mmlu_jpn_mcf",
        "openai_mmlu_kor_mcf",
        "openai_mmlu_por_mcf",
        "openai_mmlu_spa_mcf",
        "openai_mmlu_swa_mcf",
        "openai_mmlu_yor_mcf",
        "openai_mmlu_zho_mcf",
    ),
    "mmlu_redux": ("mmlu_redux_2",),
}

DIRECT_COVERAGE_STATUSES = {
    "exact_task",
    "exact_superset",
    "normalized_task",
    "normalized_superset",
    "compact_task",
    "compact_superset",
    "alias_task",
    "alias_superset",
    "alias_task_list",
    "alias_superset_list",
    "alias_mixed_list",
}


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


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def literal_call_arg(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and node.args:
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def rwkv_field_from_value(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        function_name = func.id
    elif isinstance(func, ast.Attribute):
        function_name = func.attr
    else:
        return None
    mapping = {
        "_knowledge": "knowledge",
        "_math": "maths",
        "_coding_human_eval": "coding",
        "_coding_mbpp": "coding",
        "_coding_livecodebench": "coding",
        "_coding_swe_bench": "coding",
        "_instruction_following": "instruction_following",
        "_function_calling": "function_calling",
    }
    return mapping.get(function_name)


def load_rwkv_skills_registry(path: Path) -> list[SourceBenchmark]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_EXPLICIT_METADATA" for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_EXPLICIT_METADATA"
        ):
            value = node.value
        if value is None:
            continue
        if not isinstance(value, ast.Dict):
            raise SystemExit(f"{path} _EXPLICIT_METADATA is not a dict literal")
        rows = []
        for key, item_value in zip(value.keys, value.values):
            if key is None:
                continue
            name = literal_call_arg(key)
            if not name:
                continue
            rows.append(SourceBenchmark(name=name, field=rwkv_field_from_value(item_value)))
        return rows
    raise SystemExit(f"could not find _EXPLICIT_METADATA in {path}")


def load_json_source(path: Path) -> list[SourceBenchmark]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("benchmarks", raw.get("tasks", []))
    if not isinstance(raw, list):
        raise SystemExit(f"{path} must contain a JSON list or an object with benchmarks/tasks")
    return [source_benchmark_from_value(item) for item in raw]


def load_jsonl_source(path: Path) -> list[SourceBenchmark]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            rows.append(source_benchmark_from_value(json.loads(text)))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def load_text_source(path: Path) -> list[SourceBenchmark]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        name, _, field = text.partition(",")
        rows.append(SourceBenchmark(name=name.strip(), field=field.strip() or None))
    return rows


def source_benchmark_from_value(value: Any) -> SourceBenchmark:
    if isinstance(value, str):
        return SourceBenchmark(name=value)
    if isinstance(value, dict):
        name = value.get("name", value.get("benchmark", value.get("task")))
        if not name:
            raise SystemExit(f"source benchmark row is missing name/benchmark/task: {value!r}")
        field = value.get("field", value.get("category"))
        return SourceBenchmark(name=str(name), field=str(field) if field else None)
    raise SystemExit(f"unsupported source benchmark row: {value!r}")


def load_source_benchmarks(path_text: str | None, source_format: str) -> list[SourceBenchmark]:
    if not path_text:
        raise SystemExit("coverage requires --source")
    path = Path(path_text)
    if not path.exists():
        raise SystemExit(f"source file not found: {path}")

    resolved_format = source_format
    if resolved_format == "auto":
        if path.name == "benchmark_registry.py":
            resolved_format = "rwkv-skills-registry"
        elif path.suffix == ".json":
            resolved_format = "json"
        elif path.suffix == ".jsonl":
            resolved_format = "jsonl"
        else:
            resolved_format = "text"

    if resolved_format == "rwkv-skills-registry":
        return load_rwkv_skills_registry(path)
    if resolved_format == "json":
        return load_json_source(path)
    if resolved_format == "jsonl":
        return load_jsonl_source(path)
    return load_text_source(path)


def unique_lookup(values: list[str], key_func) -> dict[str, str]:
    lookup: dict[str, str] = {}
    duplicate_keys: set[str] = set()
    for value in values:
        key = key_func(value)
        if key in lookup:
            duplicate_keys.add(key)
        else:
            lookup[key] = value
    for key in duplicate_keys:
        lookup.pop(key, None)
    return lookup


def candidate_tasks(source: str, names: list[str], *, limit: int) -> tuple[str, ...]:
    if limit <= 0:
        return tuple()
    source_norm = normalized_name(source)
    source_compact = compact_name(source)
    source_tokens = [token for token in source_norm.split("_") if token and token not in {"bench", "benchmark", "eval"}]
    scored = []
    for name in names:
        name_norm = normalized_name(name)
        name_compact = compact_name(name)
        name_tokens = [token for token in name_norm.split("_") if token]
        score = 0
        if source_norm and source_norm in name_norm:
            score += 100
        if source_compact and source_compact in name_compact:
            score += 80
        if len(source_tokens) >= 2 and all(
            any(
                len(source_token) >= 3
                and len(name_token) >= 3
                and (source_token.startswith(name_token) or name_token.startswith(source_token))
                for name_token in name_tokens
            )
            for source_token in source_tokens
        ):
            score += 40 + len(source_tokens) * 5
        if score:
            scored.append((-score, len(name), name))
    return tuple(item[2] for item in sorted(scored)[:limit])


def coverage_rows(args: argparse.Namespace) -> list[CoverageRow]:
    registry = load_registry(
        custom_tasks=args.custom_tasks,
        load_multilingual=args.load_multilingual,
    )
    sources = load_source_benchmarks(args.source, args.source_format)
    task_names = sorted(registry._task_registry)
    superset_names = sorted(registry._task_superset_dict)
    all_target_names = task_names + superset_names
    task_set = set(task_names)
    superset_set = set(superset_names)
    normalized_tasks = unique_lookup(task_names, normalized_name)
    normalized_supersets = unique_lookup(superset_names, normalized_name)
    compact_tasks = unique_lookup(task_names, compact_name)
    compact_supersets = unique_lookup(superset_names, compact_name)
    candidate_limit = max(0, int(args.candidate_limit))

    rows = []
    for source in sources:
        name = source.name
        target_kind: str | None = None
        targets: tuple[str, ...] = tuple()
        status = "missing"
        norm = normalized_name(name)
        compact = compact_name(name)
        if name in task_set:
            status, target_kind, targets = "exact_task", "task", (name,)
        elif name in superset_set:
            status, target_kind, targets = "exact_superset", "superset", (name,)
        elif norm in normalized_tasks:
            status, target_kind, targets = "normalized_task", "task", (normalized_tasks[norm],)
        elif norm in normalized_supersets:
            status, target_kind, targets = "normalized_superset", "superset", (normalized_supersets[norm],)
        elif compact in compact_tasks:
            status, target_kind, targets = "compact_task", "task", (compact_tasks[compact],)
        elif compact in compact_supersets:
            status, target_kind, targets = "compact_superset", "superset", (compact_supersets[compact],)
        else:
            alias_targets = OFFICIAL_LIGHTEVAL_ALIASES.get(norm)
            if alias_targets and all(target in task_set or target in superset_set for target in alias_targets):
                target_types = {"task" if target in task_set else "superset" for target in alias_targets}
                if len(alias_targets) == 1:
                    target_kind = next(iter(target_types))
                    status = f"alias_{target_kind}"
                else:
                    target_kind = "mixed" if len(target_types) > 1 else next(iter(target_types))
                    status = f"alias_{target_kind}_list"
                targets = alias_targets
        candidates = candidate_tasks(name, all_target_names, limit=candidate_limit)
        if status == "missing" and candidates:
            status = "candidate_only"
        rows.append(
            CoverageRow(
                source=name,
                field=source.field,
                status=status,
                target_kind=target_kind,
                targets=targets,
                candidates=candidates,
            )
        )
    return rows


def format_coverage(rows: list[CoverageRow], output_format: str) -> str:
    if output_format == "jsonl":
        return "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows)
    if output_format == "tasks":
        seen = set()
        task_lines = []
        for row in rows:
            if row.status not in DIRECT_COVERAGE_STATUSES:
                continue
            for target in row.targets:
                if target not in seen:
                    seen.add(target)
                    task_lines.append(target)
        return "".join(task + "\n" for task in task_lines)
    if output_format == "summary":
        status_counts = sorted(
            ((status, sum(1 for row in rows if row.status == status)) for status in {row.status for row in rows}),
            key=lambda item: (-item[1], item[0]),
        )
        direct_count = sum(1 for row in rows if row.status in DIRECT_COVERAGE_STATUSES)
        lines = [
            f"total\t{len(rows)}",
            f"direct\t{direct_count}",
            f"not_direct\t{len(rows) - direct_count}",
        ]
        lines.extend(f"status\t{status}\t{count}" for status, count in status_counts)
        return "\n".join(lines) + "\n"
    lines = ["source\tfield\tstatus\ttarget_kind\ttargets\tcandidates"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.source,
                    row.field or "",
                    row.status,
                    row.target_kind or "",
                    ",".join(row.targets),
                    ",".join(row.candidates),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def coverage_tasks(args: argparse.Namespace) -> int:
    text = format_coverage(coverage_rows(args), args.format)
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

    coverage_parser = subparsers.add_parser("coverage")
    coverage_parser.add_argument("--load-multilingual", action="store_true")
    coverage_parser.add_argument("--custom-tasks")
    coverage_parser.add_argument("--output")
    coverage_parser.add_argument("--format", choices=("text", "jsonl", "summary", "tasks"), default="text")
    coverage_parser.add_argument("--source", required=True)
    coverage_parser.add_argument(
        "--source-format",
        choices=("auto", "text", "json", "jsonl", "rwkv-skills-registry"),
        default="auto",
    )
    coverage_parser.add_argument("--candidate-limit", type=int, default=5)
    coverage_parser.set_defaults(handler=coverage_tasks)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
