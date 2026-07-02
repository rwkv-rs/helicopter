from __future__ import annotations

import re
from pathlib import Path
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


LOCAL_DATA_ROOT = Path(__file__).resolve().parents[3] / "benchmarks/lighteval_data"

HF_MATH_TASKS = {
    "algebra222": {
        "repo": "sirdug/Algebra222",
        "split": "train",
    },
    "amc23": {
        "repo": "math-ai/amc23",
        "split": "test",
    },
    "beyond_aime": {
        "repo": "ByteDance-Seed/BeyondAIME",
        "split": "test",
    },
    "brumo25": {
        "repo": "MathArena/brumo_2025",
        "split": "train",
    },
    "college_math": {
        "repo": "di-zhang-fdu/College_Math_Test",
        "split": "test",
    },
    "comp_math_24_25": {
        "repo": str(LOCAL_DATA_ROOT / "comp_math_24_25"),
        "split": "test",
    },
    "gaokao2023en": {
        "repo": "test-time-compute/test_gaokao2023en",
        "split": "test",
    },
    "hmmt_feb25": {
        "repo": "MathArena/hmmt_feb_2025",
        "split": "train",
    },
    "math_odyssey": {
        "repo": "MathOdyssey/MathOdyssey",
        "split": "test",
    },
    "mawps": {
        "repo": str(LOCAL_DATA_ROOT / "mawps"),
        "split": "test",
    },
    "minerva_math": {
        "repo": "math-ai/minervamath",
        "split": "test",
    },
    "omni_math": {
        "repo": "KbsdJames/Omni-MATH",
        "split": "test",
    },
}

HF_POLYMATH_LANGUAGES = (
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "id",
    "it",
    "ja",
    "ko",
    "ms",
    "pt",
    "ru",
    "sw",
    "te",
    "th",
    "vi",
    "zh",
)

HF_MULTIPLE_CHOICE_TASKS = {
    "supergpqa": {
        "repo": "m-a-p/SuperGPQA",
        "split": "train",
    },
}

HF_MMMLU_TASKS = {
    "mmmlu_ar": "AR_XY",
    "mmmlu_bn": "BN_BD",
    "mmmlu_de": "DE_DE",
    "mmmlu_es": "ES_LA",
    "mmmlu_fr": "FR_FR",
    "mmmlu_hi": "HI_IN",
    "mmmlu_id": "ID_ID",
    "mmmlu_it": "IT_IT",
    "mmmlu_ja": "JA_JP",
    "mmmlu_ko": "KO_KR",
    "mmmlu_pt": "PT_BR",
    "mmmlu_sw": "SW_KE",
    "mmmlu_yo": "YO_NG",
    "mmmlu_zh": "ZH_CN",
}

WMT24PP_TARGET_LANGUAGES = {
    "de_DE": "German",
    "es_MX": "Spanish",
    "fr_FR": "French",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
}

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes)
where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to
the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_choice_letter(text: str, *, max_choices: int) -> str | None:
    valid = set(ascii_uppercase[:max_choices])
    normalized = text.strip().upper()
    marker = re.search(r"\b(?:ANSWER|ANS|OPTION)(?:\s+IS)?\s*[:\-\)]?\s*([A-Z])\b", normalized)
    if marker and marker.group(1) in valid:
        return marker.group(1)
    for match in re.finditer(r"(?:^|[^A-Z])([A-Z])(?:\s*(?:[.)\]:-]|$))", normalized):
        if match.group(1) in valid:
            return match.group(1)
    return None


class MultipleChoiceLetterMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        gold_indices = doc.gold_index if isinstance(doc.gold_index, list) else [doc.gold_index]
        gold_letters = {ascii_uppercase[index] for index in gold_indices}
        for prediction in model_response.final_text:
            letter = _extract_choice_letter(prediction, max_choices=len(doc.choices))
            if letter in gold_letters:
                return 1.0
        return 0.0


MULTIPLE_CHOICE_LETTER_MATCH = SampleLevelMetric(
    metric_name="mc_letter_match",
    sample_level_fn=MultipleChoiceLetterMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


def _mean_annotation_score(annotations: object) -> float | None:
    if not isinstance(annotations, list) or not annotations:
        return None
    scores: list[float] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        try:
            scores.append(float(annotation.get("score")))
        except (TypeError, ValueError):
            continue
    return _mean(scores) if scores else None


def _judgement_text(value: bool) -> str:
    return "Judgement: Yes" if value else "Judgement: No"


def _extract_judgement(text: str) -> str | None:
    normalized = text.strip().lower()
    marker = re.search(r"\bjudg(?:e)?ment\s*[:\-]\s*(yes|no)\b", normalized)
    if marker:
        return marker.group(1)
    marker = re.search(r"\b(yes|no)\b", normalized)
    return marker.group(1) if marker else None


class JudgementMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        gold = _extract_judgement(str(doc.choices[doc.gold_index]))
        if gold is None:
            return 0.0
        for prediction in model_response.final_text:
            if _extract_judgement(prediction) == gold:
                return 1.0
        return 0.0


JUDGEMENT_MATCH = SampleLevelMetric(
    metric_name="judgement_match",
    sample_level_fn=JudgementMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=_mean,
    higher_is_better=True,
)


def _question(record: dict) -> str:
    for key in ("question", "problem", "problem_statement", "prompt", "input"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _format_answer(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _answer(record: dict) -> str:
    for key in ("expected_answer", "answer", "final_answer", "target", "result"):
        value = record.get(key)
        if value is not None:
            return _format_answer(value)
    return ""


def qwen_math_prompt(line: dict, task_name: str | None = None) -> Doc:
    question = _question(line)
    answer = _answer(line)
    return Doc(
        task_name=task_name,
        query=f"Question: {question}\nAnswer:",
        choices=[answer],
        gold_index=0,
        specific={
            "id": line.get("id"),
            "source": line.get("source") or line.get("sourcename"),
        },
    )


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=_question(record),
        target=_answer(record),
        metadata={
            "id": record.get("id"),
            "source": record.get("source") or record.get("sourcename"),
        },
    )


def _choice_texts(record: dict) -> list[str]:
    if isinstance(record.get("options"), list):
        return [str(choice).strip() for choice in record["options"]]
    choices: list[str] = []
    for letter in ascii_uppercase:
        value = record.get(letter)
        if value is None:
            break
        choices.append(str(value).strip())
    return choices


def supergpqa_prompt(line: dict, task_name: str | None = None) -> Doc:
    choices = _choice_texts(line)
    answer_letter = str(line.get("answer_letter") or line.get("answer") or "").strip().upper()
    if not answer_letter and line.get("answer") in choices:
        answer_letter = ascii_uppercase[choices.index(str(line["answer"]).strip())]
    gold_index = ascii_uppercase.index(answer_letter)

    query = f"Question: {line['question']}"
    query += "".join(f"\n{letter}. {choice}" for letter, choice in zip(ascii_uppercase, choices))
    query += "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {letter}" for letter in ascii_uppercase[: len(choices)]],
        gold_index=gold_index,
        specific={
            "uuid": line.get("uuid"),
            "discipline": line.get("discipline"),
            "field": line.get("field"),
            "subfield": line.get("subfield"),
            "difficulty": line.get("difficulty"),
        },
    )


def mmmlu_prompt(line: dict, task_name: str | None = None) -> Doc:
    choices = [str(line[letter]).strip() for letter in ascii_uppercase[:4]]
    gold_index = ascii_uppercase.index(str(line["Answer"]).strip().upper())
    query = f"Question: {line['Question']}"
    query += "".join(f"\n{letter}. {choice}" for letter, choice in zip(ascii_uppercase, choices))
    query += "\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {letter}" for letter in ascii_uppercase[:4]],
        gold_index=gold_index,
        specific={"subject": line.get("Subject")},
    )


def wmt24pp_prompt(line: dict, task_name: str | None = None) -> Doc:
    target_language = str(line.get("lp") or "").split("-", 1)[-1]
    target_name = WMT24PP_TARGET_LANGUAGES.get(target_language, target_language)
    return Doc(
        task_name=task_name,
        query=f"English phrase: {str(line['source']).rstrip()}\n{target_name} phrase:",
        choices=[str(line["target"]).rstrip()],
        gold_index=0,
        instruction=f"Translate English to {target_name}, do not explain, only output the translation.",
        specific={
            "lp": line.get("lp"),
            "domain": line.get("domain"),
            "document_id": line.get("document_id"),
            "segment_id": line.get("segment_id"),
            "is_bad_source": line.get("is_bad_source"),
        },
    )


def answer_judge_prompt(line: dict, task_name: str | None = None) -> Doc | None:
    mean_score = _mean_annotation_score(line.get("annotations"))
    if mean_score is None:
        return None
    expected_judgement = _judgement_text(mean_score > 0.5)
    query = (
        "Problem:\n"
        f"{line.get('question') or ''}\n\n"
        "Expected answer:\n"
        f"{line.get('gt_answer') or ''}\n\n"
        "Predicted answer:\n"
        f"{line.get('gen_answer') or ''}\n\n"
        "Decide whether the predicted answer matches the expected answer. "
        "Return exactly `Judgement: Yes` or `Judgement: No`.\n"
        "Judgement:"
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=[expected_judgement],
        gold_index=0,
        specific={
            "item_name": line.get("item_name"),
            "dataset_name": line.get("dataset_name"),
            "mean_score": mean_score,
        },
    )


def svamp_prompt(line: dict, task_name: str | None = None) -> Doc:
    body = str(line.get("Body") or "").strip()
    question = str(line.get("Question") or "").strip()
    if body and question:
        full_question = f"{body.rstrip('.')}. {question}"
    else:
        full_question = body or question
    return Doc(
        task_name=task_name,
        query=f"Question: {full_question}\nAnswer:",
        choices=[_format_answer(line.get("Answer", ""))],
        gold_index=0,
        specific={
            "id": line.get("ID"),
            "type": line.get("Type"),
            "equation": line.get("Equation"),
        },
    )


def _hf_answer_judge_task() -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name="rwkv_skills:answer_judge",
        prompt_function=answer_judge_prompt,
        hf_repo="nvidia/judges-verdict",
        hf_subset=None,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=8,
        metrics=[JUDGEMENT_MATCH],
        stop_sequence=["\n"],
        version=0,
    )


def _hf_math_task(name: str, task: dict[str, object]) -> LightevalTaskConfig:
    split = str(task["split"])
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=qwen_math_prompt,
        sample_fields=record_to_sample,
        solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
        scorer=math_scorer(),
        hf_repo=str(task["repo"]),
        hf_subset=task.get("subset"),
        hf_avail_splits=[split],
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=512,
        metrics=[Metrics.expr_gold_metric],
        stop_sequence=["Question:"],
        version=0,
    )


def _hf_svamp_task() -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name="rwkv_skills:svamp",
        prompt_function=svamp_prompt,
        hf_repo="tongyx361/svamp",
        hf_subset=None,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=512,
        metrics=[Metrics.expr_gold_metric],
        stop_sequence=["Question:"],
        version=0,
    )


def _hf_polymath_task(language: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:polymath_{language}",
        prompt_function=qwen_math_prompt,
        hf_repo="Qwen/PolyMath",
        hf_subset=language,
        hf_avail_splits=["top", "high", "medium", "low"],
        evaluation_splits=["top", "high", "medium", "low"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=512,
        metrics=[Metrics.expr_gold_metric],
        stop_sequence=["Question:"],
        version=0,
    )


def _hf_multiple_choice_task(name: str, task: dict[str, str]) -> LightevalTaskConfig:
    split = task["split"]
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=supergpqa_prompt,
        hf_repo=task["repo"],
        hf_subset=None,
        hf_avail_splits=[split],
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=5,
        metrics=[MULTIPLE_CHOICE_LETTER_MATCH],
        stop_sequence=["\n"],
        version=0,
    )


def _hf_mmmlu_task(name: str, subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:{name}",
        prompt_function=mmmlu_prompt,
        hf_repo="openai/MMMLU",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=5,
        metrics=[MULTIPLE_CHOICE_LETTER_MATCH],
        stop_sequence=["\n"],
        version=0,
    )


def _hf_wmt24pp_task(target_language: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"rwkv_skills:wmt24pp_{target_language}",
        prompt_function=wmt24pp_prompt,
        hf_repo="google/wmt24pp",
        hf_subset=f"en-{target_language}",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=None,
        metrics=[Metrics.bleu, Metrics.chrf, Metrics.ter],
        stop_sequence=["\n"],
        version=0,
    )


TASKS_TABLE = [
    *[_hf_math_task(name, task) for name, task in HF_MATH_TASKS.items()],
    _hf_answer_judge_task(),
    _hf_svamp_task(),
    *[_hf_polymath_task(language) for language in HF_POLYMATH_LANGUAGES],
    *[_hf_multiple_choice_task(name, task) for name, task in HF_MULTIPLE_CHOICE_TASKS.items()],
    *[_hf_mmmlu_task(name, subset) for name, subset in HF_MMMLU_TASKS.items()],
    *[_hf_wmt24pp_task(target_language) for target_language in WMT24PP_TARGET_LANGUAGES],
]
