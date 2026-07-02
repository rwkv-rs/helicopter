from __future__ import annotations

import ast
import asyncio
import base64
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from .openai_client import chat_completion
from .sampling import apply_limit_or_sample, dataset_sample_suffix
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
BROWSECOMP_ZH_XLSX_URL = (
    "https://raw.githubusercontent.com/PALIN2018/BrowseComp-ZH/main/data/browsecomp-zh-encrypted.xlsx"
)


@dataclass(frozen=True, slots=True)
class BrowseCompSample:
    sample_index: int
    task_id: str
    question: str
    reference_answer: str
    locale: str
    topic: str = ""


@dataclass(frozen=True, slots=True)
class BrowseCompResult:
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
class BrowseCompRunConfig:
    base_url: str
    model: str
    benchmark: str = "browsecomp"
    source_type: str = "browsecomp_csv"
    source_url: str | None = None
    source_path: str | None = None
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    temperature: float = 0.0
    top_p: float = 1.0
    cot_max_tokens: int = 2048
    answer_max_tokens: int = 1024
    timeout_s: float = 600.0
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_timeout_s: float = 60.0
    scoreboard_dataset: str | None = None
    job_name: str = "function_browsecomp"
    job_id: str | None = None
    runner: str = "helicopter_eval.browsecomp"
    cot_mode: str = "CoT"


def decrypt_xor_base64(ciphertext_b64: str, password: str) -> str:
    ciphertext = base64.b64decode(ciphertext_b64.strip())
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    plaintext = bytes(lhs ^ rhs for lhs, rhs in zip(ciphertext, _repeat_bytes(digest, len(ciphertext))))
    return plaintext.decode("utf-8")


def _repeat_bytes(raw: bytes, target_len: int) -> Iterable[int]:
    if not raw:
        return ()
    return (raw[index % len(raw)] for index in range(target_len))


def _read_source_bytes(config: BrowseCompRunConfig) -> bytes:
    if config.source_path:
        return Path(config.source_path).expanduser().read_bytes()
    url = config.source_url or (
        BROWSECOMP_ZH_XLSX_URL if config.source_type == "browsecomp_zh_xlsx" else BROWSECOMP_CSV_URL
    )
    with urllib.request.urlopen(url, timeout=config.timeout_s) as response:
        return response.read()


def _load_browsecomp_csv(raw: bytes) -> list[BrowseCompSample]:
    rows: list[BrowseCompSample] = []
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    for index, row in enumerate(reader):
        canary = str(row.get("canary") or "")
        rows.append(
            BrowseCompSample(
                sample_index=index,
                task_id=f"browsecomp_{index:04d}",
                question=decrypt_xor_base64(str(row.get("problem") or ""), canary),
                reference_answer=decrypt_xor_base64(str(row.get("answer") or ""), canary),
                locale="en",
                topic=str(row.get("problem_topic") or "").strip(),
            )
        )
    return rows


def _decrypt_optional_field(value: str | None, password: str) -> str:
    return decrypt_xor_base64(value, password) if value else ""


def _load_browsecomp_zh_xlsx(raw: bytes) -> list[BrowseCompSample]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        shared_strings = _load_shared_strings(zf)
        worksheet_rows = _load_sheet_rows(zf, shared_strings)

    samples: list[BrowseCompSample] = []
    for row_index, row in enumerate(worksheet_rows):
        canary = str(row.get("canary") or "")
        question = _decrypt_optional_field(row.get("Question"), canary)
        answer = _decrypt_optional_field(row.get("Answer"), canary)
        if not question.strip() or not answer.strip():
            continue
        samples.append(
            BrowseCompSample(
                sample_index=len(samples),
                task_id=f"browsecomp_zh_{row_index:04d}",
                question=question,
                reference_answer=answer,
                locale="zh",
            )
        )
    return samples


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        text = zf.read("xl/sharedStrings.xml").decode("utf-8")
    except KeyError:
        return []
    root = ET.fromstring(text)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared: list[str] = []
    for item in root.findall("x:si", ns):
        shared.append("".join(node.text or "" for node in item.findall(".//x:t", ns)))
    return shared


def _load_sheet_rows(zf: zipfile.ZipFile, shared_strings: Sequence[str]) -> list[dict[str, str]]:
    text = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
    root = ET.fromstring(text)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    raw_rows = root.findall(".//x:sheetData/x:row", ns)
    if not raw_rows:
        return []

    header_row = _parse_sheet_row(raw_rows[0], shared_strings, ns)
    ordered_headers = {column: value for column, value in header_row.items() if value}
    parsed_rows: list[dict[str, str]] = []
    for raw_row in raw_rows[1:]:
        cells = _parse_sheet_row(raw_row, shared_strings, ns)
        payload = {header: cells.get(column, "") for column, header in ordered_headers.items()}
        if any(value.strip() for value in payload.values()):
            parsed_rows.append(payload)
    return parsed_rows


def _parse_sheet_row(row: ET.Element, shared_strings: Sequence[str], ns: dict[str, str]) -> dict[str, str]:
    cells: dict[str, str] = {}
    for cell in row.findall("x:c", ns):
        ref = str(cell.attrib.get("r") or "")
        column = "".join(ch for ch in ref if ch.isalpha())
        cell_type = str(cell.attrib.get("t") or "")
        if cell_type == "s":
            raw = cell.findtext("x:v", default="", namespaces=ns)
            value = shared_strings[int(raw)] if raw.isdigit() and int(raw) < len(shared_strings) else ""
        elif cell_type == "inlineStr":
            value = "".join(node.text or "" for node in cell.findall(".//x:t", ns))
        else:
            value = cell.findtext("x:v", default="", namespaces=ns) or ""
        cells[column] = _xml_unescape(value)
    return cells


def _xml_unescape(text: str) -> str:
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


def load_samples(config: BrowseCompRunConfig) -> list[BrowseCompSample]:
    if config.split != "test":
        raise ValueError("BrowseComp only provides test split")
    raw = _read_source_bytes(config)
    if config.source_type == "browsecomp_csv":
        samples = _load_browsecomp_csv(raw)
    elif config.source_type == "browsecomp_zh_xlsx":
        samples = _load_browsecomp_zh_xlsx(raw)
    else:
        raise ValueError(f"unsupported BrowseComp source_type: {config.source_type}")
    return apply_limit_or_sample(
        samples,
        limit=config.limit,
        sample_size=config.sample_size,
        sample_seed=config.sample_seed,
        sort_key=lambda sample: sample.sample_index,
    )


def build_browsecomp_user_prompt(question: str, *, locale: str) -> str:
    if locale.strip().lower() == "zh":
        return (
            "你是一个浏览基准测试助手。请先仔细思考，再直接回答问题。\n\n"
            "请基于你自己的知识回答下面这个需要较强检索能力的问题。"
            "不要通过让用户自己去搜索网页来回避作答。即使你不完全确定，也要给出最具体的答案。\n\n"
            f"问题:\n{question}\n\n"
            "请按下面格式回复：\n解释: <简短说明>\n最终答案: <简洁最终答案>\n置信度: <0% 到 100%>"
        )
    return (
        "You are a browsing benchmark assistant. Think through the question carefully and answer directly.\n\n"
        "Answer the following browsing-intensive question using your own knowledge. "
        "Do not refuse by asking the user to search the web. "
        "If you are uncertain, provide your best concrete answer.\n\n"
        f"Question:\n{question}\n\n"
        "Return your response in this format:\n"
        "Explanation: <brief explanation>\n"
        "Exact Answer: <succinct final answer>\n"
        "Confidence: <0% to 100%>"
    )


def build_browsecomp_answer_prompt(question: str, cot: str, *, locale: str) -> str:
    if locale.strip().lower() == "zh":
        return (
            "下面是一个问题和模型的推理。请抽取最终答案。\n\n"
            f"问题:\n{question}\n\n推理:\n{cot}\n\n"
            '只返回一个 JSON 对象：{"name":"final_answer","arguments":{"answer":"<简洁最终答案>"},"id":"final_answer"}'
        )
    return (
        "Below is a question and model reasoning. Extract the final answer.\n\n"
        f"Question:\n{question}\n\nReasoning:\n{cot}\n\n"
        'Return exactly one JSON object: '
        '{"name":"final_answer","arguments":{"answer":"<succinct final answer>"},"id":"final_answer"}'
    )


def parse_final_answer(response: str) -> str:
    payload = _loads_json_or_literal(_extract_json_object(response))
    if isinstance(payload, Mapping):
        arguments = payload.get("arguments")
        if str(payload.get("name") or "") == "final_answer" and isinstance(arguments, Mapping):
            answer = _first_answer_value(arguments)
            if answer is not None:
                return answer
        answer = _first_answer_value(payload)
        if answer is not None:
            return answer
    raise ValueError("BrowseComp final answer response did not contain an answer")


def _extract_json_object(response: str) -> str:
    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _loads_json_or_literal(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return ast.literal_eval(candidate)


def _first_answer_value(payload: Mapping[str, Any]) -> str | None:
    for key in ("answer", "final_answer", "response", "prediction"):
        if key in payload:
            return _stringify_answer(payload[key])
    if len(payload) == 1:
        return _stringify_answer(next(iter(payload.values())))
    return None


def _stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _resolved_judge_base_url(config: BrowseCompRunConfig) -> str | None:
    return config.judge_base_url or os.getenv("HELICOPTER_JUDGE_BASE_URL") or os.getenv("JUDGE_BASE_URL")


def _resolved_judge_model(config: BrowseCompRunConfig) -> str | None:
    return config.judge_model or os.getenv("HELICOPTER_JUDGE_MODEL") or os.getenv("JUDGE_MODEL")


def _resolved_judge_api_key(config: BrowseCompRunConfig) -> str | None:
    return config.judge_api_key or os.getenv("HELICOPTER_JUDGE_API_KEY") or os.getenv("JUDGE_API_KEY")


def _judge_prompt(sample: BrowseCompSample, answer: str) -> str:
    if sample.locale.strip().lower() == "zh":
        return (
            "你是严格的答案判定器。\n"
            "请根据 correct_answer 判断 response 是否正确回答了 question。数值题可接受很小误差。\n"
            "只返回 JSON，字段为 is_passed(bool) 和 reason(string)。\n\n"
            f"[question]\n{sample.question}\n\n[response]\n{answer}\n\n[correct_answer]\n{sample.reference_answer}\n"
        )
    return (
        "You are a rigorous answer judge.\n"
        "Decide whether the response correctly answers the question according to the correct answer. "
        "Treat small numerical tolerance as acceptable.\n"
        "Return only JSON with fields is_passed(bool) and reason(string).\n\n"
        f"[question]\n{sample.question}\n\n[response]\n{answer}\n\n[correct_answer]\n{sample.reference_answer}\n"
    )


def judge_answer(sample: BrowseCompSample, answer: str, config: BrowseCompRunConfig) -> tuple[bool, str]:
    judge_base_url = _resolved_judge_base_url(config)
    judge_model = _resolved_judge_model(config)
    if not judge_base_url or not judge_model:
        raise ValueError(
            "BrowseComp requires HELICOPTER_JUDGE_BASE_URL/JUDGE_BASE_URL and "
            "HELICOPTER_JUDGE_MODEL/JUDGE_MODEL"
        )
    content = chat_completion(
        base_url=judge_base_url,
        model=judge_model,
        prompt=_judge_prompt(sample, answer),
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        timeout_s=config.judge_timeout_s,
        api_key=_resolved_judge_api_key(config),
        response_format={"type": "json_object"},
    )
    payload = _loads_json_or_literal(_extract_json_object(content))
    if not isinstance(payload, Mapping):
        raise ValueError(f"judge did not return an object: {content!r}")
    return bool(payload.get("is_passed", False)), str(payload.get("reason") or "").strip()


def generate_completion(sample: BrowseCompSample, config: BrowseCompRunConfig) -> BrowseCompResult:
    cot_prompt = build_browsecomp_user_prompt(sample.question, locale=sample.locale)
    cot = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=cot_prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.cot_max_tokens,
        timeout_s=config.timeout_s,
    )
    answer_prompt = build_browsecomp_answer_prompt(sample.question, cot, locale=sample.locale)
    answer_completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=answer_prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=config.answer_max_tokens,
        timeout_s=config.timeout_s,
    )
    try:
        answer = parse_final_answer(answer_completion)
    except Exception as exc:  # noqa: BLE001
        return _result(
            sample,
            config,
            cot_prompt,
            cot,
            answer_prompt,
            answer_completion,
            "",
            False,
            f"final answer parse failed: {exc}",
            "",
        )
    try:
        passed, judge_reason = judge_answer(sample, answer, config)
    except Exception as exc:  # noqa: BLE001
        return _result(
            sample,
            config,
            cot_prompt,
            cot,
            answer_prompt,
            answer_completion,
            answer,
            False,
            f"judge failed: {exc}",
            "",
        )
    return _result(
        sample,
        config,
        cot_prompt,
        cot,
        answer_prompt,
        answer_completion,
        answer,
        passed,
        "" if passed else judge_reason,
        judge_reason,
    )


def _result(
    sample: BrowseCompSample,
    config: BrowseCompRunConfig,
    cot_prompt: str,
    cot: str,
    answer_prompt: str,
    answer_completion: str,
    answer: str,
    is_passed: bool,
    fail_reason: str,
    judge_reason: str,
) -> BrowseCompResult:
    del config
    return BrowseCompResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=f"{cot_prompt}\n\n--- final-answer prompt ---\n{answer_prompt}",
        completion=f"{cot}\n\n--- final-answer completion ---\n{answer_completion}",
        answer=answer,
        reference_answer=sample.reference_answer,
        is_passed=is_passed,
        fail_reason=fail_reason,
        judge_reason=judge_reason,
    )


def evaluate_samples(samples: Sequence[BrowseCompSample], config: BrowseCompRunConfig) -> list[BrowseCompResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: BrowseCompRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    dataset += dataset_sample_suffix(sample_size=config.sample_size, sample_seed=config.sample_seed)
    return dataset


def job_id(config: BrowseCompRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: BrowseCompRunConfig) -> dict[str, Any]:
    return {
        "cot": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.cot_max_tokens,
        },
        "answer": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": config.answer_max_tokens,
        },
        "judge": {
            "model": _resolved_judge_model(config),
            "base_url": _resolved_judge_base_url(config),
        },
    }


def task_sampling_config(config: BrowseCompRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_browsecomp_two_stage",
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[BrowseCompResult], *, config: BrowseCompRunConfig, repo_root: Path) -> int:
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


def run_browsecomp(config: BrowseCompRunConfig, *, repo_root: Path) -> dict[str, Any]:
    if not _resolved_judge_base_url(config) or not _resolved_judge_model(config):
        raise ValueError("BrowseComp formal scoring requires judge base URL and judge model")
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "judge_model": _resolved_judge_model(config),
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: BrowseCompRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "source": config.source_url or config.source_path or config.source_type,
        "source_type": config.source_type,
        "split": config.split,
        "limit": config.limit,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "base_url": config.base_url,
        "model": config.model,
        "judge_model": _resolved_judge_model(config),
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }
