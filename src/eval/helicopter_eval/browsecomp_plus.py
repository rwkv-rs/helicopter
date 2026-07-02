from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request

from .apibank import decode_tool_calls
from .browsecomp import BrowseCompRunConfig, BrowseCompSample, judge_answer, parse_final_answer
from .openai_client import chat_completion
from .sampling import apply_limit_or_sample, dataset_sample_suffix
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


OFFICIAL_BROWSECOMP_PLUS_SOURCE = "texttron/BrowseComp-Plus"
BROWSECOMP_PLUS_DATA_URL = (
    "https://huggingface.co/datasets/texttron/BrowseComp-Plus/resolve/main/"
    "data/browsecomp_plus_decrypted.jsonl"
)
DEFAULT_BROWSECOMP_PLUS_ROOT = Path("/tmp/rwkv-official-refs/BrowseComp-Plus")
DEFAULT_BROWSECOMP_PLUS_MAX_STEPS = 12
DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS = 1400
DEFAULT_BROWSECOMP_PLUS_CHUNK_OVERLAP = 220
DEFAULT_BROWSECOMP_PLUS_TOP_K = 5


BROWSECOMP_PLUS_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "search",
        "description": "Search the fixed BrowseComp-Plus corpus and return relevant evidence chunks.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "get_document",
        "description": "Retrieve one BrowseComp-Plus document by docid.",
        "parameters": {"type": "object", "properties": {"docid": {"type": "string"}}, "required": ["docid"]},
    },
    {
        "name": "get_document_chunks",
        "description": "Read chunked passages from one retrieved document id.",
        "parameters": {
            "type": "object",
            "properties": {"docid": {"type": "string"}, "query": {"type": "string"}},
            "required": ["docid"],
        },
    },
    {
        "name": "final_answer",
        "description": "Finish the benchmark task with the final answer.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
)


@dataclass(frozen=True, slots=True)
class BrowseCompPlusSample:
    sample_index: int
    task_id: str
    query_id: str
    question: str
    reference_answer: str
    documents: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BrowseCompPlusResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str
    judge_reason: str

    def to_scoreboard(self) -> ScoreboardEvalResult:
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=self.answer,
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class BrowseCompPlusRunConfig:
    base_url: str
    model: str
    benchmark: str = "browsecomp_plus"
    source_path: str | None = None
    source_root: str | None = None
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout_s: float = 600.0
    max_steps: int = DEFAULT_BROWSECOMP_PLUS_MAX_STEPS
    history_max_chars: int = 8192
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_timeout_s: float = 60.0
    scoreboard_dataset: str | None = None
    job_name: str = "function_browsecomp_plus"
    job_id: str | None = None
    runner: str = "helicopter_eval.browsecomp_plus"
    cot_mode: str = "CoT"


class BrowseCompPlusEnv:
    def __init__(self, sample: BrowseCompPlusSample) -> None:
        self.sample = sample
        self.retrieved_docids: set[str] = set()
        self.tool_call_counts: dict[str, int] = {}
        self.final_answer = ""

    def initial_user_message(self) -> str:
        return _normalize_text(
            "\n".join(
                [
                    "You are answering a BrowseComp-Plus deep-research question against a fixed corpus.",
                    "Use search or get_document as needed. When ready, call final_answer.",
                    "Final answer should include the exact answer and concise evidence citations using [docid] when available.",
                    f"Question: {self.sample.question}",
                ]
            )
        )

    def step(self, call: Mapping[str, Any]) -> dict[str, Any]:
        name = str(call.get("name") or "").strip()
        arguments = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        arguments = dict(arguments or {})
        self.tool_call_counts[name] = self.tool_call_counts.get(name, 0) + 1
        if name == "search":
            query = str(arguments.get("query") or self.sample.question)
            return {"observation": {"chunks": self.search(query, DEFAULT_BROWSECOMP_PLUS_TOP_K)}, "done": False}
        if name == "get_document":
            docid = str(arguments.get("docid") or "").strip()
            document = self.document_by_id(docid)
            if document is None:
                return {"observation": {"docid": docid, "error": "not_found"}, "done": False}
            return {"observation": _document_payload(document), "done": False}
        if name == "get_document_chunks":
            docid = str(arguments.get("docid") or "").strip()
            query = str(arguments.get("query") or self.sample.question)
            return {
                "observation": {"docid": docid, "chunks": self.document_chunks(docid, query=query)},
                "done": False,
            }
        if name == "final_answer":
            self.final_answer = str(arguments.get("answer") or "").strip()
            return {"observation": "Final answer recorded.", "done": True, "final_answer": self.final_answer}
        return {"observation": f"Unknown BrowseComp-Plus tool: {name}", "done": True, "error": "unknown_tool"}

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        documents = sorted(self.sample.documents, key=lambda item: _document_score(query, item), reverse=True)
        for item in documents[: max(20, k)]:
            docid = str(item.get("docid") or item.get("id") or "")
            if docid:
                self.retrieved_docids.add(docid)
        return _top_chunks(documents[: max(20, k)], query=query, limit=k)

    def document_chunks(self, docid: str, *, query: str) -> list[dict[str, Any]]:
        document = self.document_by_id(docid)
        if document is None:
            return [{"docid": docid, "error": "not_found"}]
        return _top_chunks([document], query=query, limit=DEFAULT_BROWSECOMP_PLUS_TOP_K)

    def document_by_id(self, docid: str) -> dict[str, Any] | None:
        for document in self.sample.documents:
            current = str(document.get("docid") or document.get("id") or "")
            if current == docid:
                self.retrieved_docids.add(docid)
                return dict(document)
        return None

    def details(self) -> dict[str, Any]:
        return {
            "browsecomp_plus_run": {
                "query_id": self.sample.query_id,
                "retrieved_docids": sorted(self.retrieved_docids),
                "tool_call_counts": dict(self.tool_call_counts),
                "document_count": len(self.sample.documents),
                "result": [{"type": "output_text", "output": self.final_answer}] if self.final_answer else [],
            }
        }


def load_samples(config: BrowseCompPlusRunConfig) -> list[BrowseCompPlusSample]:
    if config.split != "test":
        raise ValueError("BrowseComp-Plus only provides test split")
    path = _resolve_source_path(config)
    rows = _load_rows(path)
    official_source = _resolve_official_source(config, manifest_path=path)
    samples: list[BrowseCompPlusSample] = []
    for index, item in enumerate(rows):
        if config.limit is not None and config.sample_size is None and len(samples) >= int(config.limit):
            break
        if not isinstance(item, Mapping):
            continue
        metadata = dict(item.get("metadata") or {})
        query_id = str(metadata.get("query_id") or item.get("query_id") or item.get("id") or index)
        question = str(item.get("question") or item.get("instruction") or item.get("query") or "").strip()
        answer = str(item.get("answer") or metadata.get("answer") or "").strip()
        if not question:
            continue
        documents = _documents_for_item(item)
        item_source = _metadata_official_source(metadata)
        if not documents and item_source is not None:
            documents = _load_documents_for_query(item_source, query_id)
        if not documents and official_source is not None:
            documents = _load_documents_for_query(official_source, query_id)
        metadata.setdefault("browsecomp_plus_source_path", str(path))
        if item_source is not None:
            metadata.setdefault("browsecomp_plus_official_source_path", str(item_source))
        elif official_source is not None:
            metadata.setdefault("browsecomp_plus_official_source_path", str(official_source))
        samples.append(
            BrowseCompPlusSample(
                sample_index=len(samples),
                task_id=str(item.get("task_id") or f"browsecomp_plus__{query_id}"),
                query_id=query_id,
                question=question,
                reference_answer=answer,
                documents=tuple(documents),
                metadata=metadata,
            )
        )
    return apply_limit_or_sample(
        samples,
        limit=config.limit,
        sample_size=config.sample_size,
        sample_seed=config.sample_seed,
        sort_key=lambda sample: sample.sample_index,
    )


def build_prompt(
    sample: BrowseCompPlusSample,
    *,
    messages: Sequence[Mapping[str, str]],
    history_max_chars: int,
) -> str:
    history = _trim_history(messages, max_chars=history_max_chars)
    trajectory, current = _render_agent_state(history)
    return _normalize_text(
        "\n".join(
            [
                "You are controlling tools in a BrowseComp-Plus deep-research environment.",
                "Respond with exactly one JSON tool call and no extra text.",
                'Use this shape: {"name":"ToolName","arguments":{"arg":"value"}}',
                "Use search and get_document to gather evidence. Use final_answer only when ready to answer.",
                'For final_answer, use exactly {"name":"final_answer","arguments":{"answer":"<exact answer>"}}.',
                "Do not use reason, reasoning, explanation, output, or response keys for final_answer.",
                "Available tools:",
                json.dumps(BROWSECOMP_PLUS_TOOL_SCHEMAS, ensure_ascii=False, indent=2),
                "",
                "Trajectory:",
                trajectory,
                "",
                "Current observation:",
                current,
                "",
                "Assistant:",
            ]
        )
    )


def generate_completion(sample: BrowseCompPlusSample, config: BrowseCompPlusRunConfig) -> BrowseCompPlusResult:
    if not sample.documents:
        return BrowseCompPlusResult(
            sample_index=sample.sample_index,
            task_id=sample.task_id,
            prompt="",
            completion="",
            answer="",
            reference_answer=sample.reference_answer,
            is_passed=False,
            fail_reason="BrowseComp-Plus evidence documents are unavailable; set RWKV_BROWSECOMP_PLUS_ROOT or source_path with embedded documents",
            judge_reason="",
        )
    env = BrowseCompPlusEnv(sample)
    messages: list[dict[str, str]] = [{"role": "user", "content": env.initial_user_message()}]
    prompts: list[str] = []
    completions: list[str] = []
    trace: list[dict[str, Any]] = []
    fail_reason = ""
    final_answer = ""

    for step_index in range(1, max(1, int(config.max_steps)) + 1):
        prompt = build_prompt(sample, messages=messages, history_max_chars=config.history_max_chars)
        prompts.append(prompt)
        completion = chat_completion(
            base_url=config.base_url,
            model=config.model,
            prompt=prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
        )
        completions.append(completion)
        try:
            call = _decode_one_call(completion)
        except Exception as exc:  # noqa: BLE001
            recovered = _recover_final_answer_call(completion)
            if recovered is None:
                fail_reason = f"tool call parse failed: {exc}"
                trace.append({"step": step_index, "raw": completion, "parse_error": fail_reason})
                break
            call = recovered
        result = env.step(call)
        observation_text = json.dumps(result.get("observation"), ensure_ascii=False, separators=(",", ":"))
        messages.append({"role": "assistant", "content": json.dumps(call, ensure_ascii=False, separators=(",", ":"))})
        messages.append({"role": "user", "content": _tool_result_message(call, observation_text)})
        trace.append(
            {
                "step": step_index,
                "decoded_call": call,
                "observation": observation_text[:4000],
                "done": bool(result.get("done", False)),
                **env.details(),
            }
        )
        if result.get("error"):
            fail_reason = str(result["error"])
            break
        if result.get("done"):
            final_answer = str(result.get("final_answer") or "")
            break

    if not final_answer and not fail_reason:
        fail_reason = "browsecomp_plus produced no final answer"
    if final_answer:
        try:
            passed, judge_reason = _judge_answer(sample, final_answer, config)
            if not passed:
                fail_reason = judge_reason
        except Exception as exc:  # noqa: BLE001
            passed = False
            judge_reason = ""
            fail_reason = f"judge failed: {exc}"
    else:
        passed = False
        judge_reason = ""

    return BrowseCompPlusResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=json.dumps(prompts, ensure_ascii=False),
        completion=json.dumps({"completions": completions, "trace": trace}, ensure_ascii=False),
        answer=final_answer,
        reference_answer=sample.reference_answer,
        is_passed=passed,
        fail_reason="" if passed else fail_reason,
        judge_reason=judge_reason,
    )


def evaluate_samples(samples: Sequence[BrowseCompPlusSample], config: BrowseCompPlusRunConfig) -> list[BrowseCompPlusResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: BrowseCompPlusRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    dataset += dataset_sample_suffix(sample_size=config.sample_size, sample_seed=config.sample_seed)
    return dataset


def job_id(config: BrowseCompPlusRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: BrowseCompPlusRunConfig) -> dict[str, Any]:
    return {
        "tool": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
            "max_steps": config.max_steps,
        },
        "judge": {
            "model": config.judge_model or os.getenv("HELICOPTER_JUDGE_MODEL") or os.getenv("JUDGE_MODEL"),
            "base_url": config.judge_base_url or os.getenv("HELICOPTER_JUDGE_BASE_URL") or os.getenv("JUDGE_BASE_URL"),
        },
    }


def task_sampling_config(config: BrowseCompPlusRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_browsecomp_plus_tool_loop",
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[BrowseCompPlusResult], *, config: BrowseCompPlusRunConfig, repo_root: Path) -> int:
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
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_browsecomp_plus(config: BrowseCompPlusRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    if not samples:
        raise ValueError("BrowseComp-Plus source is empty")
    missing_document_count = sum(1 for sample in samples if not sample.documents)
    if missing_document_count:
        raise ValueError(
            f"BrowseComp-Plus evidence documents are unavailable for {missing_document_count} selected sample(s); "
            "set RWKV_BROWSECOMP_PLUS_ROOT or source_path with embedded documents"
        )
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: BrowseCompPlusRunConfig) -> dict[str, Any]:
    path = _resolve_source_path(config)
    probe_config = config if config.sample_size is not None else replace(config, limit=config.limit or 5)
    samples = load_samples(probe_config)
    document_counts = [len(sample.documents) for sample in samples]
    return {
        "benchmark": config.benchmark,
        "source": str(path),
        "official_source": OFFICIAL_BROWSECOMP_PLUS_SOURCE,
        "split": config.split,
        "limit": config.limit,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
        "sample_probe_count": len(samples),
        "document_counts": document_counts,
        "documents_available": bool(document_counts and all(count > 0 for count in document_counts)),
    }


def _judge_answer(sample: BrowseCompPlusSample, answer: str, config: BrowseCompPlusRunConfig) -> tuple[bool, str]:
    return judge_answer(
        BrowseCompSample(
            sample_index=sample.sample_index,
            task_id=sample.task_id,
            question=sample.question,
            reference_answer=sample.reference_answer,
            locale="en",
        ),
        answer,
        BrowseCompRunConfig(
            base_url=config.base_url,
            model=config.model,
            benchmark=config.benchmark,
            judge_base_url=config.judge_base_url,
            judge_model=config.judge_model,
            judge_api_key=config.judge_api_key,
            judge_timeout_s=config.judge_timeout_s,
        ),
    )


def _resolve_source_path(config: BrowseCompPlusRunConfig) -> Path:
    candidates: list[Path] = []
    if config.source_path:
        candidates.append(Path(config.source_path).expanduser())
    for raw in (
        os.getenv("RWKV_BROWSECOMP_PLUS_MANIFEST"),
        os.getenv("BROWSECOMP_PLUS_MANIFEST"),
    ):
        if raw:
            candidates.append(Path(raw).expanduser())
    for root in _source_root_candidates(config):
        candidates.extend(
            [
                root / "data" / "browsecomp_plus_decrypted.jsonl",
                root / "browsecomp_plus_decrypted.jsonl",
                root / "test.jsonl",
                root / "data" / "browsecomp_plus" / "test.jsonl",
            ]
        )
    candidates.extend(
        [
            Path("/home/chase/GitHub/helicopter/data/browsecomp_plus/test.jsonl"),
            Path("/home/chase/GitHub/rwkv-skills/data/browsecomp_plus/test.jsonl"),
        ]
    )
    for candidate in _dedupe_paths(candidates):
        if candidate.is_file():
            return candidate.resolve()
    download_target = _cached_hf_source_path()
    if download_target.is_file():
        return download_target.resolve()
    _download_hf_source(download_target, timeout_s=config.timeout_s)
    return download_target.resolve()


def _resolve_official_source(config: BrowseCompPlusRunConfig, *, manifest_path: Path) -> Path | None:
    candidates: list[Path] = []
    for root in _source_root_candidates(config):
        candidates.extend([root / "data" / "browsecomp_plus_decrypted.jsonl", root / "browsecomp_plus_decrypted.jsonl"])
    candidates.extend(
        [
            Path(str(os.getenv("RWKV_BROWSECOMP_PLUS_SOURCE") or "")).expanduser(),
            Path(str(os.getenv("BROWSECOMP_PLUS_SOURCE") or "")).expanduser(),
        ]
    )
    candidates.extend([manifest_path, _cached_hf_source_path()])
    for candidate in _dedupe_paths([path for path in candidates if str(path) not in {"", "."}]):
        if candidate.is_file() and candidate.name == "browsecomp_plus_decrypted.jsonl":
            return candidate.resolve()
    return None


def _metadata_official_source(metadata: Mapping[str, Any]) -> Path | None:
    for key in ("browsecomp_plus_source_path", "browsecomp_plus_official_source_path"):
        raw = metadata.get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_file() and path.name == "browsecomp_plus_decrypted.jsonl":
            return path.resolve()
    return None


def _source_root_candidates(config: BrowseCompPlusRunConfig) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for raw in (
        config.source_root,
        os.getenv("RWKV_BROWSECOMP_PLUS_ROOT"),
        os.getenv("BROWSECOMP_PLUS_ROOT"),
    ):
        if raw:
            candidates.append(Path(str(raw)).expanduser())
    candidates.extend([DEFAULT_BROWSECOMP_PLUS_ROOT, Path("/tmp/ref-BrowseComp-Plus")])
    return _dedupe_paths(candidates)


def _load_rows(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _documents_for_item(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    for value in (
        metadata.get("browsecomp_plus_documents") if isinstance(metadata, Mapping) else None,
        item.get("browsecomp_plus_documents"),
        _official_row_documents(item),
    ):
        docs = _list_of_dicts(value)
        if docs:
            return docs
    return []


def _official_row_documents(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("gold_docs", "evidence_docs", "negative_docs"):
        for document in _list_of_dicts(item.get(key)):
            docid = str(document.get("docid") or document.get("id") or "")
            if docid and docid in seen:
                continue
            if docid:
                seen.add(docid)
            documents.append(document)
    return documents


def _load_documents_for_query(source_path: Path, query_id: str) -> list[dict[str, Any]]:
    if not source_path.is_file():
        return []
    with source_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, Mapping) and str(item.get("query_id") or item.get("id") or "") == str(query_id):
                return _official_row_documents(item)
    return []


def _decode_one_call(completion: str) -> dict[str, Any]:
    if completion.strip() == "[]":
        raise ValueError("empty tool call")
    calls = decode_tool_calls(completion)
    if not calls:
        raise ValueError("model returned no tool call")
    return dict(calls[0])


def _recover_final_answer_call(completion: str) -> dict[str, Any] | None:
    try:
        answer = parse_final_answer(completion)
    except Exception:  # noqa: BLE001
        answer = _extract_answer_string(completion)
    if answer:
        return {"name": "final_answer", "arguments": {"answer": answer}, "id": "final_answer"}
    return None


def _extract_answer_string(response: str) -> str:
    text = _normalize_text(response)
    if "final_answer" not in text and "answer" not in text:
        return ""
    match = re.search(r'"answer"\s*:\s*"(?P<answer>(?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if match is None:
        return ""
    try:
        return _normalize_text(json.loads(f'"{match.group("answer")}"'))
    except json.JSONDecodeError:
        return _normalize_text(match.group("answer"))


def _tool_result_message(call: Mapping[str, Any], observation: str) -> str:
    tool_name = str(call.get("name") or "tool").strip() or "tool"
    return _normalize_text(
        "\n".join(
            [
                f"Tool result from {tool_name}.",
                "This is read-only evidence, not the next assistant JSON object.",
                "Next assistant message must be exactly one JSON tool call with keys name and arguments.",
                "Valid tool names are search, get_document, get_document_chunks, and final_answer.",
                "",
                observation,
            ]
        )
    )


def _render_agent_state(messages: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    if not messages:
        return "", ""
    current = str(messages[-1].get("content") or "")
    trajectory: list[str] = []
    for message in messages[:-1]:
        role = str(message.get("role") or "user").lower()
        content = str(message.get("content") or "")
        if not content:
            continue
        trajectory.append(("Assistant action: " if role == "assistant" else "Environment: ") + content)
    return "\n".join(trajectory), current


def _trim_history(messages: Sequence[Mapping[str, str]], *, max_chars: int) -> list[dict[str, str]]:
    normalized = [
        {"role": str(message.get("role") or "user"), "content": _normalize_text(str(message.get("content") or ""))}
        for message in messages
        if str(message.get("content") or "").strip()
    ]
    if max_chars <= 0:
        return normalized
    total = 0
    kept: list[dict[str, str]] = []
    for message in reversed(normalized):
        size = len(message["content"])
        if kept and total + size > max_chars:
            break
        kept.append(message)
        total += size
    return list(reversed(kept))


def _top_chunks(documents: Sequence[Mapping[str, Any]], *, query: str, limit: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        docid = str(document.get("docid") or document.get("id") or "")
        for chunk_index, text in enumerate(_chunk_text(_document_text(document))):
            chunks.append(
                {
                    "docid": docid,
                    "chunk_id": f"{docid}:{chunk_index}",
                    "score": _text_score(query, text),
                    "text": text,
                }
            )
    chunks.sort(key=lambda item: (float(item["score"]), str(item["docid"])), reverse=True)
    return chunks[: max(1, int(limit))]


def _chunk_text(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if len(normalized) <= DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS:
        return [normalized] if normalized else []
    chunks: list[str] = []
    step = max(1, DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS - DEFAULT_BROWSECOMP_PLUS_CHUNK_OVERLAP)
    for start in range(0, len(normalized), step):
        chunk = normalized[start : start + DEFAULT_BROWSECOMP_PLUS_CHUNK_CHARS].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _document_score(query: str, document: Mapping[str, Any]) -> float:
    return _text_score(query, _document_text(document))


def _document_payload(document: Mapping[str, Any]) -> dict[str, str]:
    return {
        "docid": str(document.get("docid") or document.get("id") or ""),
        "text": _document_text(document),
    }


def _document_text(document: Mapping[str, Any]) -> str:
    for key in ("text", "contents", "content", "snippet", "body"):
        value = document.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(dict(document), ensure_ascii=False, separators=(",", ":"))


def _text_score(query: str, text: str) -> float:
    tokens = _tokenize(query)
    if not tokens:
        return 0.0
    lowered = text.lower()
    return float(sum(1 for token in tokens if token in lowered))


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _download_hf_source(path: Path, *, timeout_s: float) -> None:
    token = os.getenv("HELICOPTER_HF_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    headers = {"authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(BROWSECOMP_PLUS_DATA_URL, headers=headers)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(
            "BrowseComp-Plus official HF source requires access; set HELICOPTER_HF_TOKEN/HF_TOKEN "
            "or provide RWKV_BROWSECOMP_PLUS_ROOT/source_path"
        ) from exc


def _cached_hf_source_path() -> Path:
    root = Path(os.getenv("HELICOPTER_CACHE_DIR") or "~/.cache/helicopter-eval").expanduser()
    return root / "BrowseComp-Plus" / "data" / "browsecomp_plus_decrypted.jsonl"


def _dedupe_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        key = expanded.resolve() if expanded.exists() else expanded
        if key not in deduped:
            deduped.append(key)
    return tuple(deduped)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


__all__ = [
    "BROWSECOMP_PLUS_TOOL_SCHEMAS",
    "BrowseCompPlusRunConfig",
    "BrowseCompPlusSample",
    "BrowseCompPlusEnv",
    "build_prompt",
    "dry_run_summary",
    "load_samples",
    "run_browsecomp_plus",
]
