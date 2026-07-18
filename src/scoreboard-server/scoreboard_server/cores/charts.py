from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .normalize import (
    display_param,
    entry_domain,
    metric_from_context,
    numeric_value,
    parse_model_tags,
)


def serialize_charts(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "knowledge": _knowledge_chart(entries),
        "math": _math_chart(entries),
        "instruction_following": _instruction_chart(entries),
        "coding": _coding_chart(entries),
        "agent": _agent_chart(entries),
    }


def _knowledge_chart(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    subject_scores: dict[str, dict[str, float]] = {}
    for entry in entries:
        if entry_domain(entry) != "knowledge":
            continue
        details = _task_details(entry)
        acc_map = details.get("accuracy_by_subject")
        if not isinstance(acc_map, Mapping):
            continue
        model = _chart_model_label(entry)
        for raw_subject, raw_score in acc_map.items():
            score = numeric_value(raw_score)
            if score is None:
                continue
            subject = str(raw_subject).replace("_", " ").strip().lower()
            if not subject:
                continue
            bucket = subject_scores.setdefault(model, {})
            if subject not in bucket or score > bucket[subject]:
                bucket[subject] = score

    data: list[dict[str, Any]] = []
    subject_sums: dict[str, list[float]] = {}
    for model, scores in subject_scores.items():
        for subject, score in scores.items():
            label = subject.title()
            data.append({"model": model, "subject": label, "score": score})
            subject_sums.setdefault(label, []).append(score)
    if not data:
        return None
    subjects = sorted(subject_sums, key=lambda subject: sum(subject_sums[subject]) / len(subject_sums[subject]), reverse=True)
    return {"type": "knowledge_bar", "subjects": subjects, "models": _chart_models(data), "data": data}


def _math_chart(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    series: dict[str, list[dict[str, float]]] = {}
    order: dict[str, tuple[str, tuple[Any, ...]]] = {}
    ks: set[int] = set()
    for entry in entries:
        base = _dataset_base(str(entry.get("dataset") or "")).lower()
        if base not in {"aime24", "aime25"}:
            continue
        curve = _pass_curve(entry.get("metrics") or {}, _task_details(entry).get("pass_curve"))
        if not curve:
            continue
        name = f"{base.upper()} · {_chart_model_label(entry)}"
        order.setdefault(name, (base.upper(), _chart_model_sort_key(str(entry.get("model") or ""))))
        points = series.setdefault(name, [])
        for k, acc in curve.items():
            points.append({"k": int(k), "acc": float(acc)})
            ks.add(int(k))
    if not series:
        return None
    ordered_names = sorted(series, key=lambda name: order[name])
    return {
        "type": "aime_line",
        "ks": sorted(ks),
        "series": [{"name": name, "points": sorted(series[name], key=lambda item: item["k"])} for name in ordered_names],
    }


def _instruction_chart(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    data: list[dict[str, Any]] = []
    domains: set[str] = set()
    for entry in entries:
        if entry_domain(entry) != "instruction_following":
            continue
        details = _task_details(entry)
        buckets: dict[str, list[float]] = {}
        for tier_key in ("tier0_accuracy", "tier1_accuracy"):
            acc_map = details.get(tier_key)
            if not isinstance(acc_map, Mapping):
                continue
            for raw_name, raw_score in acc_map.items():
                score = numeric_value(raw_score)
                if score is None:
                    continue
                domain = str(raw_name).split(":", 1)[0].replace("_", " ")
                buckets.setdefault(domain, []).append(score)
        for domain, scores in buckets.items():
            domains.add(domain)
            data.append({"domain": domain, "model": _chart_model_label(entry), "score": sum(scores) / len(scores)})
    if not data:
        return None
    domain_order = [domain for domain in _INSTRUCTION_DOMAIN_ORDER if domain in domains]
    domain_order.extend(domain for domain in sorted(domains) if domain not in domain_order)
    return {"type": "instruction_bar", "domains": domain_order, "models": _chart_models(data), "data": data}


def _coding_chart(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _benchmark_score_chart(entries, domain="coding", chart_type="coding_bar")


def _agent_chart(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _benchmark_score_chart(entries, domain="agent", chart_type="agent_bar")


def _benchmark_score_chart(entries: list[dict[str, Any]], *, domain: str, chart_type: str) -> dict[str, Any] | None:
    data: list[dict[str, Any]] = []
    datasets: set[str] = set()
    for entry in entries:
        dataset = str(entry.get("dataset") or "")
        if entry_domain(entry) != domain:
            continue
        metric, value = metric_from_context(entry.get("metrics") or {}, entry.get("sampling_config"))
        if value is None:
            continue
        label = _dataset_base(dataset).upper()
        datasets.add(label)
        data.append(
            {
                "dataset": label,
                "model": _chart_model_label(entry),
                "score": float(value),
                "metric": metric or "score",
            }
        )
    if not data:
        return None
    return {"type": chart_type, "datasets": sorted(datasets), "models": _chart_models(data), "data": data}


def _dataset_base(dataset: str) -> str:
    for suffix in ("_test", "_eval", "_val"):
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)]
    return dataset


def _chart_model_label(entry: Mapping[str, Any]) -> str:
    model = str(entry.get("model") or "")
    arch, data, params = parse_model_tags(model)
    if arch != "unknown" and data != "unknown" and params != "unknown":
        return f"{arch.lower()}-{data.lower()}-{display_param(params).lower()}"
    parts = model.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else model


def _chart_model_sort_key(model: str) -> tuple[int, float, str, str]:
    arch, data, params = parse_model_tags(model)
    return (_param_rank(params), _arch_rank(arch), data, model)


def _chart_models(data: list[dict[str, Any]]) -> list[str]:
    labels = {str(row["model"]) for row in data}
    return sorted(labels, key=_chart_model_sort_key)


def _arch_rank(arch: str) -> float:
    if arch == "unknown":
        return float("inf")
    digits = "".join(ch for ch in arch if ch.isdigit())
    return float(digits) if digits else float("inf")


def _param_rank(params: str) -> int:
    if params == "unknown":
        return 10**9
    token = params.lower().replace("_", ".")
    try:
        return int(float(token.removesuffix("b")) * 1000)
    except ValueError:
        return 10**9


def _task_details(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    details = entry.get("task_details")
    if isinstance(details, Mapping):
        return details
    metrics = entry.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    return {}


def _pass_curve(*sources: Any) -> dict[int, float]:
    curve: dict[int, float] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            raw_key = str(key).lower().strip()
            if raw_key.startswith("pass@"):
                raw_suffix = raw_key.split("@", 1)[1]
            elif raw_key.startswith("pass"):
                raw_suffix = raw_key[4:]
                if raw_suffix.startswith("at"):
                    raw_suffix = raw_suffix[2:]
            else:
                continue
            try:
                k = int(raw_suffix)
            except ValueError:
                continue
            score = numeric_value(value)
            if score is not None:
                curve.setdefault(k, score)
    return dict(sorted(curve.items()))


_INSTRUCTION_DOMAIN_ORDER = [
    "prompt",
    "keywords",
    "language",
    "length_constraints",
    "detectable_format",
    "punctuation",
    "startend",
    "change_case",
    "combination",
]
