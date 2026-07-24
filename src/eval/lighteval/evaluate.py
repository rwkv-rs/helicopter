# ruff: noqa: E401, E501, E701, E702
import gzip, json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import pyarrow.parquet as parquet
from langdetect import DetectorFactory
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
DetectorFactory.seed = 0
# Edit these ordinary constants for an evaluation. Every run gets a unique directory.
MODEL_PATH = os.environ.get("LIGHTEVAL_MODEL_PATH", "/home/caizus/Weights/RWKV/rwkv7/pth/rwkv7-g1h-7.2b-20260710-ctx10240.pth")
TASKS = os.environ.get("LIGHTEVAL_TASKS", "gsm8k|0")
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def _output_dir(run_id=RUN_ID) -> Path:
    root = os.environ.get("LIGHTEVAL_OUTPUT_ROOT"); return Path(root) / run_id if root is not None else Path(os.environ.get("REMOTE_RUN_LOG_DIR", "outputs")) / "lighteval" / run_id
def _cache_dir(run_id=RUN_ID) -> Path: return Path(".tmp/lighteval-cache") / run_id
OUTPUT_DIR, CACHE_DIR = _output_dir(), _cache_dir()
MAX_SAMPLES = int(value) if (value := os.environ.get("LIGHTEVAL_MAX_SAMPLES")) else None
MAX_NEW_TOKENS = int(os.environ.get("LIGHTEVAL_MAX_NEW_TOKENS", "8192"))
MAX_MODEL_LENGTH = int(os.environ.get("LIGHTEVAL_MAX_MODEL_LENGTH", "10240"))
WKV_MODE = os.environ.get("VLLM_RWKV7_WKV_MODE", "fp16")
os.environ["VLLM_USE_RAPID_SAMPLER"] = "1"
CONCURRENCY_CANDIDATES = (40, 80, 160, 320, 640, 1280, 2560)
TARGET_CONCURRENCY = int(os.environ.get("LIGHTEVAL_TARGET_CONCURRENCY", "40"))
# Fill only after the seven-candidate Pro 6000 scans have produced evidence.
PRO6000_RECOMMENDED_CONCURRENCY = {"fp16": 2560, "fp32io16": 1280}
GENERATION_PARAMETERS = {
    "temperature": 0.96, "top_p": 0.76, "top_k": 32,
    "presence_penalty": 1.0, "frequency_penalty": 0.1, "penalty_decay": 0.988,
    "stop_tokens": ["\nUser:"], "max_new_tokens": MAX_NEW_TOKENS}
PUBLISH_SCOREBOARD, SCOREBOARD_URL, SCOREBOARD_TOKEN = False, "", ""
_MARKUP = re.compile(r"\*\*|__|`+")
_BOXED = re.compile(r"\\boxed\{\s*(?:([A-Z])|\\(?:text|mathrm)\{\s*([A-Z])\s*\})\s*\}", re.I)
_EXPLICIT = re.compile(r"^\s*(?:(?:thus|therefore|hence|so)[,:]?\s+)?(?:the\s+)?(?:(?:final|correct)\s+)?(?:answer|choice|option)\s*(?:is|:|=)\s*([A-Z])\b", re.I | re.M)
_BARE = re.compile(r"^\s*(?:([A-Z])\.?|\(([A-Z])\)|\[([A-Z])\])\s*$", re.I)
class RWKVGenerationParameters(GenerationParameters):
    penalty_decay: float = 0.988
    def to_vllm_dict(self) -> dict:
        backend = super().to_vllm_dict()
        backend.update(repetition_penalty=self.frequency_penalty,
                       frequency_penalty=0.0, penalty_decay=self.penalty_decay)
        return backend
class RWKVVLLMModelConfig(VLLMModelConfig): generation_parameters: RWKVGenerationParameters; wkv_mode: str
def _generation_parameters() -> RWKVGenerationParameters: return RWKVGenerationParameters(**GENERATION_PARAMETERS)
def _choice_answer(raw, tokens, choices) -> str:
    if not isinstance(raw, str) or not isinstance(tokens, list) or not tokens or len(tokens) >= MAX_NEW_TOKENS or raw.count("</think>") != 1: return ""
    suffix = _MARKUP.sub("", raw.split("</think>", 1)[1])
    matches = [value.upper() for match in _BOXED.finditer(suffix) for value in match.groups() if value]
    matches += [match.group(1).upper() for match in _EXPLICIT.finditer(suffix)]
    if match := _BARE.fullmatch(suffix): matches += [value.upper() for value in match.groups() if value]
    unique = set(matches)
    if len(unique) != 1: return ""
    label = unique.pop(); labels = [choice.strip() for choice in choices]
    return choices[labels.index(label)] if label in labels else ""
class RWKVPipeline(Pipeline):
    def _post_process_outputs(self, sampling_method_responses):
        super()._post_process_outputs(sampling_method_responses)
        for method, responses in sampling_method_responses.items():
            for doc, response in zip(self.sampling_docs[method], responses):
                choices = doc.choices; labels = [choice.strip() for choice in choices] if isinstance(choices, list) and all(isinstance(choice, str) for choice in choices) else []
                metrics = [metric for metric in self.tasks_dict[doc.task_name].metrics if metric.category == method]
                if len(labels) <= 1 or labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(labels)]) or not metrics or not all(type(metric.sample_level_fn) is ExactMatches and metric.sample_level_fn.normalize_pred is None and metric.sample_level_fn.type_exact_match == "full" for metric in metrics): continue
                response.text_post_processed = [_choice_answer(raw, response.output_tokens[i] if isinstance(response.output_tokens, list) and i < len(response.output_tokens) else None, choices) for i, raw in enumerate(response.text)]
def build_pipeline() -> Pipeline:
    if WKV_MODE not in PRO6000_RECOMMENDED_CONCURRENCY: raise ValueError("WKV_MODE must be fp16 or fp32io16")
    if TARGET_CONCURRENCY not in CONCURRENCY_CANDIDATES: raise ValueError("invalid concurrency candidate")
    os.environ["VLLM_RWKV7_WKV_MODE"] = WKV_MODE
    tracker = EvaluationTracker(output_dir=str(OUTPUT_DIR), save_details=True)
    parameters = PipelineParameters(launcher_type=ParallelismManager.VLLM,
        max_samples=MAX_SAMPLES, remove_reasoning_tags=False)
    model = RWKVVLLMModelConfig(
        model_name=Path(MODEL_PATH).as_uri(), cache_dir=str(CACHE_DIR), wkv_mode=WKV_MODE,
        dtype="float16", max_model_length=MAX_MODEL_LENGTH,
        max_num_seqs=TARGET_CONCURRENCY,
        max_num_batched_tokens=max(MAX_MODEL_LENGTH, TARGET_CONCURRENCY * 128),
        enable_prefix_caching=False, override_chat_template=True,
        generation_parameters=_generation_parameters())
    return RWKVPipeline(tasks=TASKS, pipeline_parameters=parameters,
                        evaluation_tracker=tracker, model_config=model)
def read_standard_artifacts(output_dir: Path) -> tuple[dict, list[dict]]:
    result_files = list(output_dir.glob("results/**/results_*.json"))
    if len(result_files) != 1: raise ValueError("expected one standard results JSON")
    result_file = result_files[0]; stamp = result_file.stem.removeprefix("results_"); model_dir = result_file.parent.relative_to(output_dir / "results")
    detail_files = list((output_dir / "details" / model_dir / stamp).glob(f"details_*_{stamp}.parquet"))
    if not detail_files: raise ValueError("expected matching standard details parquet")
    results = json.loads(result_file.read_text(encoding="utf-8")); rows = [row for path in detail_files for row in parquet.read_table(path).to_pylist()]
    tasks = results.get("config_tasks")
    if not isinstance(tasks, dict) or {row.get("doc", {}).get("task_name") for row in rows} != set(tasks): raise ValueError("details task is not associated with results config")
    return results, rows
def _positive_limit(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0: raise ValueError(f"{location} must be a positive integer")
    return value
def _completions(results: dict, rows: list[dict]):
    try:
        global_limit = results["config_general"]["model_config"]["generation_parameters"]["max_new_tokens"]; task_configs = results["config_tasks"]
    except (KeyError, TypeError) as error: raise ValueError("results are missing generation configuration") from error
    if global_limit is not None: global_limit = _positive_limit(global_limit, "global max_new_tokens")
    for row in rows:
        try:
            doc, response, metric = row["doc"], row["model_response"], row["metric"]; task_name = doc["task_name"]; texts, token_lists = response["text"], response["output_tokens"]
        except (KeyError, TypeError, AttributeError) as error: raise ValueError("details are missing doc/model_response/metric fields") from error
        if not isinstance(metric, dict) or not isinstance(texts, list) or not isinstance(token_lists, list): raise ValueError("details contain invalid metric/text/output_tokens fields")
        if not texts:
            if any(response.get(key) not in (None, []) for key in ("logprobs", "argmax_logits_eq_gold")): continue  # Log-likelihood rows have no generated completion.
            raise ValueError("empty completion lacks log-likelihood evidence")
        if len(texts) != len(token_lists): raise ValueError("completion and output-token counts differ")
        if global_limit is None:
            try: limit = _positive_limit(task_configs[task_name]["generation_size"], f"{task_name} generation_size")
            except (KeyError, TypeError) as error: raise ValueError(f"missing generation_size for {task_name}") from error
        else: limit = global_limit
        for text, tokens in zip(texts, token_lists):
            if not isinstance(text, str) or not isinstance(tokens, list) or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens): raise ValueError("completion text/tokens have invalid types")
            yield doc, metric, text, tokens, limit
def diagnose(results: dict, rows: list[dict]) -> dict[str, int | float]:
    completions = list(_completions(results, rows)); truncated = sum(len(tokens) >= limit for _, _, _, tokens, limit in completions)
    violations = sum("\nUser:" in text for _, _, text, _, _ in completions); count = len(completions)
    return {
        "samples": len(rows), "completions": count, "truncated": truncated, "non_truncated": count - truncated,
        "truncation_rate": truncated / count if count else 0.0, "turn_boundary_violations": violations,
        "turn_boundary_violation_rate": violations / count if count else 0.0}
def publication_payload(results: dict, rows: list[dict]) -> dict:
    task_results = {key: value for key, value in results["results"].items() if key != "all"}
    if len(task_results) != 1: raise ValueError("Scoreboard publication requires exactly one task")
    _, aggregates = next(iter(task_results.items()))
    metrics = {key: value for key, value in aggregates.items() if not key.endswith("_stderr") and isinstance(value, (int, float))}
    if len(metrics) != 1: raise ValueError("special or multi-metric results cannot be published losslessly")
    if len(list(_completions(results, rows))) != len(rows) or any(len(row["model_response"]["text"]) != 1 for row in rows): raise ValueError("multi-completion or non-generative rows cannot be published losslessly")
    raise ValueError("standard LightEval artifacts lack the identity, accounting, finish-reason, and checksum evidence required by the current Scoreboard API")
def publish(payload: dict, base_url: str, token: str, run_id: str):
    raw = json.dumps(payload, separators=(",", ":")).encode(); body = gzip.compress(raw)
    request = Request(f"{base_url.rstrip('/')}/api/v1/evaluation-publications/{run_id}", data=body, method="PUT", headers={
        "Authorization": f"Bearer {token}", "Content-Encoding": "gzip", "Content-Type": "application/json",
        "Idempotency-Key": f"publish:{payload['manifest']['digest']}"})
    return urlopen(request, timeout=30).read()
def main() -> dict:
    pipeline = build_pipeline()
    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()
    results, rows = read_standard_artifacts(OUTPUT_DIR)
    diagnostics = diagnose(results, rows)
    print(json.dumps(diagnostics, indent=2))
    if PUBLISH_SCOREBOARD:
        try:
            payload = publication_payload(results, rows)
            publish(payload, SCOREBOARD_URL, SCOREBOARD_TOKEN, RUN_ID)
        except Exception as error:
            print(f"Scoreboard publication failed: {error}", file=sys.stderr)
    return pipeline.get_results()
if __name__ == "__main__":
    main()
