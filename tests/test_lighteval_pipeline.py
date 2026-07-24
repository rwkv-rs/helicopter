# ruff: noqa: E401, E501, E701, E702
import gzip, importlib.metadata, importlib.util, json, tomllib
from pathlib import Path
from types import MethodType, SimpleNamespace
import pytest
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics import apply_metric
from lighteval.models.model_output import ModelResponse
from lighteval.models.vllm.vllm_model import VLLMModel
from lighteval.pipeline import ParallelismManager
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.registry import Registry
from lighteval.utils.imports import is_package_available
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.tokenizers.registry import get_tokenizer, resolve_tokenizer_args
from vllm.tokenizers.rwkv_defaults import normalize_rwkv_message_content
from vllm.transformers_utils.configs.rwkv7 import try_parse_rwkv7_pth_source
ROOT = Path(__file__).parents[1]; SPEC = importlib.util.spec_from_file_location("helicopter_evaluate", ROOT / "src/eval/lighteval/evaluate.py"); evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader; SPEC.loader.exec_module(evaluate)
def artifacts(limit=3):
    return {"config_general": {"model_config": {"generation_parameters": {"max_new_tokens": limit}}}, "config_tasks": {"gsm8k|0": {"generation_size": 2}}, "results": {"gsm8k|0": {"exact_match": 0.5}}}, [{"doc": {"id": "0", "query": "1+1?", "task_name": "gsm8k|0"}, "model_response": {"text": ["2", "bad\nUser:"], "output_tokens": [[1, 2, 3], [4, 5]]}, "metric": {"exact_match": 1.0}}]
def test_layout_registry_passthrough_and_generation_contract():
    component = ROOT / "src/eval/lighteval"; assert list(component.glob("*.py")) == [component / "evaluate.py"] and not (component / "pyproject.toml").exists()
    recommendations = tomllib.loads((ROOT / "configs/lighteval-pro6000.toml").read_text())["recommendation"]; assert {size: {mode: entry["LIGHTEVAL_TARGET_CONCURRENCY"] for mode, entry in modes.items()} for size, modes in recommendations.items()} == {"1.5B": {"fp16": 2560, "fp32io16": 2560}, "2.9B": {"fp16": 2560, "fp32io16": 2560}, "7.2B": {"fp16": 2560, "fp32io16": 1280}, "13.3B": {"fp16": 1280, "fp32io16": 640}} and all(entry["LIGHTEVAL_MAX_MODEL_LENGTH"] == 8192 and entry["VLLM_RWKV7_WKV_MODE"] == mode for modes in recommendations.values() for mode, entry in modes.items())
    assert all(text not in (component / "evaluate.py").read_text() for text in ("Question:", "Answer:", "DAPO")) and 'kwargs.setdefault("disable_log_stats", "VLLM_LOG_STATS_INTERVAL" not in os.environ)' in (ROOT / "src/infer/vllm-rwkv/vllm/entrypoints/llm.py").read_text()
    assert Registry(tasks=evaluate.TASKS).load_tasks() and evaluate.DetectorFactory.seed == 0
    with pytest.raises(ValueError): Registry(tasks="definitely_unknown_task|0").load_tasks()
    params = evaluate._generation_parameters(); backend = params.to_vllm_dict()
    keys = ("temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty", "frequency_penalty", "penalty_decay", "max_tokens")
    assert tuple(backend[key] for key in keys) == (0.96, 0.76, 32, 1.0, 0.1, 0.0, 0.988, 8192) and backend["stop"] == ["\nUser:"]
    config = evaluate.RWKVVLLMModelConfig(model_name="model", wkv_mode="fp16", generation_parameters=params)
    logical = config.model_dump()["generation_parameters"]; assert (logical["frequency_penalty"], logical["penalty_decay"]) == (0.1, 0.988)
def test_pipeline_receives_tasks_precision_candidate_and_remote_output(monkeypatch, tmp_path):
    captured = {}
    for name in ("EvaluationTracker", "PipelineParameters", "RWKVVLLMModelConfig"): monkeypatch.setattr(evaluate, name, lambda **kw: kw)
    monkeypatch.setattr(evaluate, "RWKVPipeline", lambda **kw: captured.update(kw) or kw); evaluate.build_pipeline()
    assert captured["tasks"] == evaluate.TASKS and captured["pipeline_parameters"]["launcher_type"] is ParallelismManager.VLLM
    assert captured["model_config"]["max_num_seqs"] in evaluate.CONCURRENCY_CANDIDATES and captured["model_config"]["override_chat_template"] is True
    assert captured["model_config"]["model_name"] == Path(evaluate.MODEL_PATH).as_uri() and (captured["model_config"]["cache_dir"], captured["model_config"]["wkv_mode"]) == (str(evaluate.CACHE_DIR), evaluate.WKV_MODE)
    monkeypatch.delenv("LIGHTEVAL_OUTPUT_ROOT", raising=False); monkeypatch.setenv("REMOTE_RUN_LOG_DIR", str(tmp_path / "runs"))
    assert evaluate._output_dir("id") == tmp_path / "runs/lighteval/id"; monkeypatch.setenv("LIGHTEVAL_OUTPUT_ROOT", str(tmp_path / "explicit")); assert evaluate._output_dir("id") == tmp_path / "explicit/id"
def test_strict_categorical_postprocessing_uses_only_closed_suffixes():
    task = next(iter(Registry(tasks="mmlu:abstract_algebra|0").load_tasks().values())); doc = task.formatter({"subject": "abstract_algebra", "question": "1+1?", "choices": ["1", "2", "3", "4"], "answer": "B"}, task.name); method = task.metrics[0].category
    raw = ["<think>x</think>\\boxed{A}", "<think>x</think>\\boxed{\\mathrm{B}}", "<think>x</think>Thus, the final choice is **C. detail**", "<think>x</think>D.", "<think>Answer: B</think>nothing", "<think>x", "<think>x</think>Correct option: B", "<think>x</think></think>Answer: B", "<think>x</think>Answer: B\n\\boxed{C}", "<think>x</think>choose B"]
    tokens = [[1], [2], [3], [4], [5], [6], [7] * evaluate.MAX_NEW_TOKENS, [8], [9], [10]]
    response = ModelResponse(text=raw.copy(), output_tokens=[item.copy() for item in tokens]); gsm = next(iter(Registry(tasks="gsm8k|0").load_tasks().values())); gsm_doc = gsm.formatter({"question": "1+1?", "answer": "work #### 2"}, gsm.name); untouched = ModelResponse(text=["<think>x</think>Answer: B"], output_tokens=[[1]])
    pipeline = object.__new__(evaluate.RWKVPipeline); pipeline.pipeline_parameters = SimpleNamespace(remove_reasoning_tags=False); pipeline.sampling_docs = {method: [doc, gsm_doc]}; pipeline.tasks_dict = {doc.task_name: task, gsm_doc.task_name: gsm}
    pipeline._post_process_outputs({method: [response, untouched]})
    assert response.text_post_processed == [" A", " B", " C", " D", "", "", "", "", "", ""] and len(response.text_post_processed) == len(raw)
    assert response.text == raw and response.output_tokens == tokens and untouched.text_post_processed is None and untouched.text == ["<think>x</think>Answer: B"]
    assert [apply_metric([ModelResponse(text_post_processed=[value])], [doc], task.metrics)[0]["em"] for value in (" B", "")] == [1, 0]
def test_official_vllm_init_bridge_cache_and_sampling(tmp_path, monkeypatch):
    assert is_package_available("vllm") and not getattr(VLLMModel, "is_dummy", False) and resolve_tokenizer_args(evaluate.MODEL_PATH)[0] == "rwkv" and resolve_tokenizer_args("facebook/opt-125m")[0] == "hf"
    checkpoint = tmp_path / Path(evaluate.MODEL_PATH).name; checkpoint.touch(); monkeypatch.chdir(tmp_path)
    cache_a, cache_b = evaluate._cache_dir("one"), evaluate._cache_dir("two"); captured = {}
    engine_globals = EngineArgs.__post_init__.__globals__; monkeypatch.setattr(engine_globals["huggingface_hub"].constants, "HF_HUB_OFFLINE", True); monkeypatch.setitem(engine_globals, "get_model_path", lambda *_: pytest.fail("RWKV file URI reached HF resolution"))
    monkeypatch.setattr(VLLMModel, "_create_auto_model", lambda self, config: captured.update(engine=EngineArgs(
        model=config.model_name, tokenizer_mode="auto", dtype=config.dtype, max_model_len=config.max_model_length).create_model_config()))
    config = evaluate.RWKVVLLMModelConfig(model_name=checkpoint.as_uri(), cache_dir=str(cache_a), wkv_mode="fp16", override_chat_template=True, max_model_length=evaluate.MAX_MODEL_LENGTH, generation_parameters=evaluate._generation_parameters())
    initialized = VLLMModel(config); source = try_parse_rwkv7_pth_source(captured["engine"].model)
    other_mode = config.model_copy(update={"wkv_mode": "fp32io16"}); root = (tmp_path / ".tmp/lighteval-cache").resolve()
    assert source.local_path == checkpoint and captured["engine"].model == str(checkpoint) and captured["engine"].hf_config.model_type == "rwkv7"
    assert config.model_dump()["wkv_mode"] == "fp16" and initialized._cache.get_model_hash(config) != initialized._cache.get_model_hash(other_mode)
    assert initialized._cache.cache_dir.resolve().is_relative_to(cache_a.resolve()) and cache_a != cache_b and all(path.resolve().is_relative_to(root) for path in (cache_a, cache_b))
    assert (initialized.tokenizer.bos_token, initialized.tokenizer.eos_token, initialized.tokenizer.pad_token) == ("<|endoftext|>",) * 3
    backend, captured = object.__new__(LLM), {}; backend.model_config = SimpleNamespace(runner_type="generate", tokenizer_mode="rwkv", hf_config=SimpleNamespace(model_type="rwkv7"))
    backend._run_completion = MethodType(lambda self, **kwargs: captured.update(kwargs) or [], backend)
    model = object.__new__(VLLMModel); model.config = config; model.data_parallel_size, model.model = 1, backend
    task = next(iter(Registry(tasks="aime24_gpassk|0").load_tasks().values()))
    model._generate(inputs=[[1]], max_new_tokens=13, stop_tokens=[], num_samples=max(task.num_samples)); params = captured["params"]
    assert captured["prompts"] == [{"prompt_token_ids": [1]}] and (params.stop, params.stop_token_ids, params.ignore_eos) == (["\nUser:"], [0], False)
    assert (params.n, params.max_tokens, params.repetition_penalty, params.frequency_penalty, params.penalty_decay) == (48, 13, 0.1, 0.0, 0.988)
def test_official_task_native_metrics_receive_raw_completions(monkeypatch):
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv"); cases = (("gsm8k|0", {"question": "1+1?", "answer": "work #### 2"}, "1+1?"), ("mmlu:abstract_algebra|0", {"subject": "abstract_algebra", "question": "1+1?", "choices": ["1", "2", "3", "4"], "answer": "B"}, "The following are multiple choice questions (with answers) about abstract algebra.\n1+1?\nA. 1\nB. 2\nC. 3\nD. 4"), ("math_500|0", {"problem": "Find 1+1.", "solution": "2"}, None), ("ifeval|0", {"prompt": "Use at least two words", "instruction_id_list": ["length_constraints:number_words"], "kwargs": [{"num_words": 2, "relation": "at least"}]}, None))
    for name, row, expected in cases:
        task = next(iter(Registry(tasks=name).load_tasks().values())); doc = task.formatter(row, task.name); query = doc.query; user = expected if expected is not None else normalize_rwkv_message_content(query)
        assert PromptManager(True, tokenizer).prepare_prompt(doc) == f"User: {user}\n\nAssistant: <think" and doc.query == query
        seen, response = {}, ModelResponse(text=[f"raw:{name}"]); monkeypatch.setattr(task.metrics[0], "compute_sample", lambda **kw: seen.update(kw) or {"probe": 1})
        assert apply_metric([response], [doc], task.metrics[:1]) == [{"probe": 1}] and seen["doc"] is doc and seen["model_response"] is response and response.final_text == [f"raw:{name}"]
    math = next(iter(Registry(tasks="gsm8k|0").load_tasks().values())); doc = math.formatter(cases[0][1], math.name); raw, truncated = "<think>open\nANSWER: 2", "<think>truncated"
    assert [apply_metric([ModelResponse(text=[text])], [doc], math.metrics)[0]["extractive_match"] for text in (raw, truncated)] == [1, 0]
    results, rows = artifacts(2); rows[0]["model_response"] = {"text": [truncated], "output_tokens": [[1, 2]]}; assert evaluate.diagnose(results, rows)["truncated"] == 1
    instruction = next(iter(Registry(tasks="ifeval|0").load_tasks().values())); if_doc = instruction.formatter(cases[3][1], instruction.name); assert apply_metric([ModelResponse(text=["hello world"])], [if_doc], instruction.metrics)[0] == {"prompt_level_strict_acc": 1, "inst_level_strict_acc": [True], "prompt_level_loose_acc": 1, "inst_level_loose_acc": [True]}
def test_parser_reads_artifacts_written_by_pinned_evaluation_tracker(tmp_path):
    assert importlib.metadata.version("lighteval") == "0.13.0"
    tasks = Registry(tasks="gsm8k|0").load_tasks(); name, task = next(iter(tasks.items())); task.config.generation_size = 2
    tracker = EvaluationTracker(output_dir=str(tmp_path), save_details=True)
    tracker.task_config_logger.log(tasks); tracker.general_config_logger.log_args_info(1, None, 0)
    tracker.general_config_logger.log_model_info(evaluate.RWKVVLLMModelConfig(
        model_name="fixture", wkv_mode="fp16", generation_parameters=evaluate.RWKVGenerationParameters(max_new_tokens=2)))
    doc = task.formatter({"question": "1+1?", "answer": "work #### 2"}, task.name); doc.id, doc.task_name = "0", name
    response = ModelResponse(input=doc.query, text=["ANSWER: 2\nUser:", "short"], output_tokens=[[1, 2], [3]])
    tracker.details_logger.log(name, doc, response, {"extractive_match": 1.0})
    tracker.metrics_logger.metric_aggregated[name] = {"extractive_match": 1.0}; tracker.save()
    decoy = tmp_path / "details/fixture/other/details_decoy_other.parquet"; decoy.parent.mkdir(parents=True); decoy.write_bytes(b"invalid")
    results, rows = evaluate.read_standard_artifacts(tmp_path); assert evaluate.diagnose(results, rows) == {"samples": 1, "completions": 2, "truncated": 1, "non_truncated": 1, "truncation_rate": 0.5, "turn_boundary_violations": 1, "turn_boundary_violation_rate": 0.5}
    config = results["config_general"]["model_config"]["generation_parameters"]; config["max_new_tokens"] = None
    assert evaluate.diagnose(results, rows)["truncated"] == 1; tokens = rows[0]["model_response"].pop("output_tokens")
    def invalid():
        with pytest.raises(ValueError): evaluate.diagnose(results, rows)
    invalid(); rows[0]["model_response"]["output_tokens"] = tokens[:-1]; invalid()
    rows[0]["model_response"]["output_tokens"], config["max_new_tokens"] = tokens, 0; invalid()
    config["max_new_tokens"], rows[0]["model_response"]["text"] = 2, []
    invalid(); rows[0]["model_response"]["logprobs"] = [0.0, 0.0]; assert evaluate.diagnose(results, rows)["completions"] == 0
def test_scoreboard_projection_is_honest_and_transport_is_gzip_idempotent(monkeypatch):
    results, rows = artifacts()
    def rejected(match):
        with pytest.raises(ValueError, match=match): evaluate.publication_payload(results, rows)
    rejected("multi-completion"); rows[0]["model_response"] = {"text": ["2"], "output_tokens": [[1]]}; results["results"]["gsm8k|0"]["other"] = 1.0
    rejected("multi-metric"); del results["results"]["gsm8k|0"]["other"]; rejected("lack the identity")
    sent = {}; monkeypatch.setattr(evaluate, "urlopen", lambda request, timeout:
        (sent.update(request=request, timeout=timeout), SimpleNamespace(read=lambda: b"ok"))[1])
    payload = {"manifest": {"digest": "a" * 64}, "non_official": True}
    assert evaluate.publish(payload, "https://scoreboard", "token", "run") == b"ok"
    request = sent["request"]; assert json.loads(gzip.decompress(request.data)) == payload
    assert request.get_header("Idempotency-key") == f"publish:{'a' * 64}" and sent["timeout"] == 30
    pipeline = SimpleNamespace(evaluate=lambda: None, save_and_push_results=lambda: None,
        show_results=lambda: None, get_results=lambda: {"evaluation": "succeeded"})
    monkeypatch.setattr(evaluate, "build_pipeline", lambda: pipeline); monkeypatch.setattr(evaluate, "read_standard_artifacts", lambda _: artifacts()); monkeypatch.setattr(evaluate, "PUBLISH_SCOREBOARD", True)
    assert evaluate.main() == {"evaluation": "succeeded"}
    pipeline.evaluate = lambda: (_ for _ in ()).throw(RuntimeError("native prerequisite"))
    with pytest.raises(RuntimeError, match="native prerequisite"): evaluate.main()
