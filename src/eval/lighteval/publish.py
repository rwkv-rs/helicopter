from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class PublicationError(RuntimeError):
    pass


def prepare_staging(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, mode=0o700)
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
        or path.is_symlink()
    ):
        raise PublicationError(
            "evaluation staging root must be an owned 0700 regular directory"
        )
    return path.resolve()


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as error:
        raise PublicationError("evaluation result is not canonical JSON") from error


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        return None


class ScoreboardClient:
    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._opener = build_opener(_NoRedirects())

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        if payload is not None:
            body = gzip.compress(canonical_json(payload))
            headers["Content-Type"] = "application/json"
            headers["Content-Encoding"] = "gzip"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        request = Request(
            f"{self._base_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with self._opener.open(request, timeout=60) as response:
                if response.headers.get_content_type() != "application/json":
                    raise PublicationError(
                        "Scoreboard response Content-Type must be application/json"
                    )
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise PublicationError("Scoreboard response exceeds size limit")
                value = json.loads(raw)
        except HTTPError as error:
            if error.code == 409:
                raise PublicationError(
                    "Scoreboard rejected conflicting content"
                ) from error
            raise PublicationError(f"Scoreboard HTTP {error.code}") from error
        except PublicationError:
            raise
        except (OSError, TimeoutError, UnicodeDecodeError, ValueError) as error:
            name = f"{type(error).__module__}.{type(error).__qualname__}"
            raise PublicationError(f"Scoreboard request failed: {name}") from error
        if not isinstance(value, dict):
            raise PublicationError("Scoreboard response must be a JSON object")
        return value

    def preflight(
        self,
        evaluator: str = "lighteval",
        version: str = "0.13.0",
    ) -> dict[str, Any]:
        response = self._request(
            "GET",
            "/api/v1/evaluation-publication-preflight",
        )
        if evaluator == "lighteval" and (
            response.get("status") != "ready"
            or response.get("schema_version") != "lighteval-campaign-v3"
            or response.get("lighteval_version") != version
        ):
            raise PublicationError("Scoreboard publication API is incompatible")
        if evaluator != "lighteval" and (
            response.get("status") != "ready"
            or f"{evaluator}-campaign-v1"
            not in response.get("supported_campaign_schemas", [])
            or response.get("evaluator_versions", {}).get(evaluator) != version
        ):
            raise PublicationError("Scoreboard publication API is incompatible")
        return response

    def create_campaign(
        self,
        payload: dict[str, Any],
        run_key: str,
    ) -> dict[str, Any]:
        receipt = self._request(
            "POST",
            "/api/v1/evaluation-campaigns",
            payload=payload,
            idempotency_key=f"campaign:{run_key}",
        )
        if not isinstance(receipt.get("campaign_id"), str) or receipt.get(
            "expected_task_count"
        ) != len(payload["expected_tasks"]):
            raise PublicationError("Scoreboard returned an invalid campaign receipt")
        return receipt

    def publish_task(
        self,
        campaign_id: str,
        task_identity: str,
        payload: dict[str, Any],
    ) -> None:
        digest = content_digest(payload)
        receipt = self._request(
            "PUT",
            (
                f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}"
                f"/tasks/{quote(task_identity, safe='')}"
            ),
            payload=payload,
            idempotency_key=f"publish:{digest}",
        )
        if (
            receipt.get("task_identity") != task_identity
            or receipt.get("content_digest") != digest
            or receipt.get("disposition") not in {"created", "unchanged"}
        ):
            raise PublicationError("Scoreboard returned an invalid task receipt")

    def finalize(self, campaign_id: str, expected_count: int) -> None:
        receipt = self._request(
            "POST",
            f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/finalize",
            idempotency_key=f"finalize:{campaign_id}",
        )
        if (
            receipt.get("campaign_id") != campaign_id
            or receipt.get("status") != "complete"
            or receipt.get("task_count") != expected_count
        ):
            raise PublicationError("Scoreboard did not finalize the campaign")


def publish_results(
    *,
    output_dir: Path,
    campaign_id: str,
    expected_tasks: list[dict[str, object]],
    model: dict[str, object],
    sampling_config: dict[str, object],
    client: ScoreboardClient,
) -> int:
    results, rows, artifact = _read_standard_results(output_dir)
    task_results = results.get("results")
    task_configs = results.get("config_tasks")
    if not isinstance(task_results, dict) or not isinstance(task_configs, dict):
        raise PublicationError("LightEval results lack task results or configs")

    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise PublicationError("LightEval details must contain objects")
        try:
            task_name = row["doc"]["task_name"]
        except (KeyError, TypeError) as error:
            raise PublicationError("LightEval detail lacks doc.task_name") from error
        if not isinstance(task_name, str):
            raise PublicationError("LightEval detail task_name must be a string")
        rows_by_task.setdefault(task_name, []).append(row)

    expected_names = {str(task["task_name"]) for task in expected_tasks}
    if not expected_names.issubset(task_results) or not expected_names.issubset(
        task_configs
    ):
        raise PublicationError("LightEval output is missing an expected task")

    for task in expected_tasks:
        task_name = str(task["task_name"])
        task_rows = rows_by_task.get(task_name, [])
        raw_aggregates = task_results[task_name]
        task_config = task_configs[task_name]
        if (
            not task_rows
            or not isinstance(raw_aggregates, dict)
            or not isinstance(task_config, dict)
        ):
            raise PublicationError(f"LightEval output is incomplete for {task_name}")
        aggregates = {
            name: float(value)
            for name, value in raw_aggregates.items()
            if isinstance(name, str)
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        }
        primary_metric = next(
            (name for name in aggregates if not name.endswith("_stderr")),
            None,
        )
        if primary_metric is None:
            raise PublicationError(f"LightEval output has no metric for {task_name}")

        details: list[dict[str, object]] = []
        for sample_index, row in enumerate(task_rows):
            try:
                document_index = row["doc"]["specific"]["helicopter_document_index"]
                metric = row["metric"]
                model_response = row["model_response"]
            except (KeyError, TypeError) as error:
                raise PublicationError(
                    f"LightEval detail is incomplete for {task_name}"
                ) from error
            if (
                isinstance(document_index, bool)
                or not isinstance(document_index, int)
                or not isinstance(metric, dict)
                or not isinstance(model_response, dict)
            ):
                raise PublicationError(f"LightEval detail is invalid for {task_name}")
            details.append(
                {
                    "sample_index": sample_index,
                    "document_index": document_index,
                    "doc": row["doc"],
                    "metric": metric,
                    "model_response": model_response,
                }
            )

        payload = {
            "schema_version": "lighteval-task-v2",
            "campaign_id": campaign_id,
            "task": task,
            "artifact": artifact,
            "task_config": task_config,
            "model": model,
            "sampling_config": sampling_config,
            "primary_metric": primary_metric,
            "aggregates": aggregates,
            "diagnostics": _diagnostics(
                details,
                int(sampling_config["max_new_tokens"]),
                str(sampling_config["stop"][0]),
            ),
            "details": details,
        }
        client.publish_task(campaign_id, str(task["identity"]), payload)
    return len(expected_tasks)


def read_aggregate_metrics(
    *,
    output_dir: Path,
    task_names: list[str],
) -> dict[str, float]:
    results, _, _ = _read_standard_results(output_dir)
    task_results = results.get("results")
    if not isinstance(task_results, dict):
        raise PublicationError("LightEval results lack task results")

    metrics: dict[str, float] = {}
    for task_name in task_names:
        raw_aggregates = task_results.get(task_name)
        if not isinstance(raw_aggregates, dict):
            raise PublicationError(f"LightEval output is missing task {task_name}")
        selector = task_name.rsplit("|", 1)[0]
        for metric_name, value in raw_aggregates.items():
            if (
                isinstance(metric_name, str)
                and not metric_name.endswith("_stderr")
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                metrics[f"{selector}/{metric_name}"] = float(value)
    if not metrics:
        raise PublicationError("LightEval output contains no finite aggregate metrics")
    return metrics


def write_sample_audit(
    *,
    output_dir: Path,
    destination: Path,
    task_names: list[str],
    weight_sha256: str,
    wkv_mode: str,
    samples_per_task: int = 10,
) -> None:
    if samples_per_task <= 0:
        raise PublicationError("sample audit size must be positive")
    _, rows, _ = _read_standard_results(output_dir)
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise PublicationError("LightEval details must contain objects")
        try:
            task_name = row["doc"]["task_name"]
        except (KeyError, TypeError) as error:
            raise PublicationError("LightEval detail lacks doc.task_name") from error
        if not isinstance(task_name, str):
            raise PublicationError("LightEval detail task_name must be a string")
        rows_by_task.setdefault(task_name, []).append(row)

    tasks: dict[str, list[dict[str, object]]] = {}
    for task_name in task_names:
        task_rows = rows_by_task.get(task_name, [])
        try:
            task_rows.sort(
                key=lambda row: row["doc"]["specific"][
                    "helicopter_document_index"
                ]
            )
        except (KeyError, TypeError) as error:
            raise PublicationError(
                f"LightEval detail lacks a document index for {task_name}"
            ) from error
        if len(task_rows) < samples_per_task:
            raise PublicationError(
                f"LightEval output has fewer than {samples_per_task} samples "
                f"for {task_name}"
            )
        tasks[task_name] = [
            _sample_audit_row(task_name, row)
            for row in task_rows[:samples_per_task]
        ]

    payload = {
        "schema_version": 1,
        "weight_sha256": weight_sha256,
        "wkv_mode": wkv_mode,
        "samples_per_task": samples_per_task,
        "tasks": tasks,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_audit_row(
    task_name: str,
    row: dict[str, Any],
) -> dict[str, object]:
    try:
        doc = row["doc"]
        metric = row["metric"]
        model_response = row["model_response"]
        document_index = doc["specific"]["helicopter_document_index"]
        question = doc["query"]
        choices = doc["choices"]
        gold_index = doc["gold_index"]
        model_input = model_response["input"]
        model_outputs = model_response["text"]
        post_processed = model_response.get("text_post_processed")
    except (KeyError, TypeError) as error:
        raise PublicationError(
            f"LightEval detail is incomplete for {task_name}"
        ) from error
    if (
        isinstance(document_index, bool)
        or not isinstance(document_index, int)
        or not isinstance(question, str)
        or not isinstance(choices, (str, list))
        or not isinstance(metric, dict)
        or not isinstance(model_outputs, list)
        or any(not isinstance(value, str) for value in model_outputs)
    ):
        raise PublicationError(f"LightEval detail is invalid for {task_name}")

    indices = gold_index if isinstance(gold_index, list) else [gold_index]
    choice_count = 1 if isinstance(choices, str) else len(choices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= choice_count
        for index in indices
    ):
        raise PublicationError(f"LightEval gold index is invalid for {task_name}")
    if isinstance(choices, str):
        standard_answers = [choices]
        scorer_golds = [choices[index] for index in indices]
    else:
        standard_answers = [
            answer
            for index in indices
            for answer in (
                choices[index]
                if isinstance(choices[index], list)
                else [choices[index]]
            )
        ]
        scorer_golds = list(standard_answers)
    if any(not isinstance(answer, str) for answer in standard_answers):
        raise PublicationError(f"LightEval gold answer is invalid for {task_name}")

    predictions = post_processed if post_processed is not None else model_outputs
    if not isinstance(predictions, list) or any(
        not isinstance(value, str) for value in predictions
    ):
        raise PublicationError(
            f"LightEval scorer predictions are invalid for {task_name}"
        )
    return {
        "document_index": document_index,
        "question": question,
        "model_input_text": _model_input_text(task_name, model_input),
        "model_output_text": model_outputs,
        "scorer_input": {
            "golds": scorer_golds,
            "predictions": predictions,
        },
        "scorer_output": metric,
        "standard_answer": standard_answers,
    }


def _model_input_text(task_name: str, value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for message in value:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
            ):
                raise PublicationError(
                    f"LightEval model input is invalid for {task_name}"
                )
            parts.append(f"{message['role']}: {message['content']}")
        return "\n".join(parts)
    raise PublicationError(f"LightEval model input is invalid for {task_name}")


def _read_standard_results(
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, object]]:
    import pyarrow.parquet as parquet

    result_files = list(output_dir.glob("results/**/results_*.json"))
    if len(result_files) != 1:
        raise PublicationError("expected one standard LightEval results JSON")
    result_file = result_files[0]
    stamp = result_file.stem.removeprefix("results_")
    model_dir = result_file.parent.relative_to(output_dir / "results")
    detail_files = sorted(
        (output_dir / "details" / model_dir / stamp).glob(f"details_*_{stamp}.parquet")
    )
    if not detail_files:
        raise PublicationError("expected standard LightEval details parquet")
    try:
        results = json.loads(result_file.read_text(encoding="utf-8"))
        rows = [
            row
            for detail_file in detail_files
            for row in parquet.read_table(detail_file).to_pylist()
        ]
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PublicationError("cannot read standard LightEval results") from error
    if not isinstance(results, dict):
        raise PublicationError("LightEval results JSON must be an object")
    return (
        results,
        rows,
        {
            "lighteval_version": "0.13.1.dev0",
            "results_path": str(result_file.relative_to(output_dir)),
            "details_paths": [
                str(path.relative_to(output_dir)) for path in detail_files
            ],
        },
    )


def _diagnostics(
    details: list[dict[str, object]],
    output_limit: int,
    stop: str,
) -> dict[str, int | float]:
    completions = 0
    truncated = 0
    violations = 0
    for detail in details:
        response = detail["model_response"]
        if not isinstance(response, dict):
            raise PublicationError("model response must be an object")
        texts = response.get("text")
        tokens = response.get("output_tokens")
        if texts in (None, []):
            continue
        if (
            not isinstance(texts, list)
            or not isinstance(tokens, list)
            or len(texts) != len(tokens)
        ):
            raise PublicationError("completion text and tokens do not align")
        for text, token_ids in zip(texts, tokens, strict=True):
            if not isinstance(text, str) or not isinstance(token_ids, list):
                raise PublicationError("completion text or tokens are invalid")
            completions += 1
            truncated += int(len(token_ids) >= output_limit)
            violations += int(stop in text)
    return {
        "samples": len(details),
        "completions": completions,
        "truncated": truncated,
        "non_truncated": completions - truncated,
        "truncation_rate": truncated / completions if completions else 0.0,
        "turn_boundary_violations": violations,
        "turn_boundary_violation_rate": (
            violations / completions if completions else 0.0
        ),
    }
