from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def detail_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("details_*.parquet")))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"LightEval details path not found: {path}")
    if not files:
        raise SystemExit("no LightEval details parquet files found")
    return files


def first_text(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    if value is None:
        return None
    return str(value)


def is_correct(metric: dict[str, Any]) -> bool | None:
    numeric_values: list[float] = []
    for value in metric.values():
        if isinstance(value, bool):
            numeric_values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            numeric_values.append(float(value))
    if not numeric_values:
        return None
    return any(value > 0 for value in numeric_values)


def export_rows(details_path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas and pyarrow are required; run with `uv run --group eval`") from exc

    frame = pd.read_parquet(details_path)
    rows: list[dict[str, Any]] = []
    for sample_index, row in frame.iterrows():
        doc = jsonable(row.get("doc") or {})
        metric = jsonable(row.get("metric") or {})
        response = jsonable(row.get("model_response") or {})
        specific = doc.get("specific") if isinstance(doc.get("specific"), dict) else {}
        rows.append(
            {
                "details_path": str(details_path),
                "sample_index": int(sample_index),
                "sample_id": doc.get("id"),
                "task_name": doc.get("task_name"),
                "is_correct": is_correct(metric),
                "metric": metric,
                "query": doc.get("query"),
                "choices": doc.get("choices"),
                "gold_index": doc.get("gold_index"),
                "extracted_golds": specific.get("extracted_golds"),
                "extracted_predictions": specific.get("extracted_predictions"),
                "model_text": first_text(response.get("text")),
                "model_text_post_processed": first_text(response.get("text_post_processed")),
                "input": response.get("input"),
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], output: Path | None) -> None:
    handle = output.open("w", encoding="utf-8") if output else sys.stdout
    try:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        if output:
            handle.close()


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(rows: list[dict[str, Any]], output: Path | None) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    handle = output.open("w", encoding="utf-8", newline="") if output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})
    finally:
        if output:
            handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helicopter eval lighteval-export")
    parser.add_argument("details", nargs="+", help="LightEval details parquet file or directory")
    parser.add_argument("--output", help="output file; defaults to stdout")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for details_path in detail_files(Path(path) for path in args.details):
        rows.extend(export_rows(details_path))
    if args.format == "csv":
        write_csv(rows, output)
    else:
        write_jsonl(rows, output)
    print(f"exported {len(rows)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
