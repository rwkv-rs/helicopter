import gzip, importlib.metadata, importlib.util, json
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
from vllm.tokenizers.registry import resolve_tokenizer_args
ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("helicopter_evaluate", ROOT / "src/eval/lighteval/evaluate.py")
evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader; SPEC.loader.exec_module(evaluate)
def artifacts(limit=3):
    return {
        "config_general": {"model_config": {"generation_parameters": {"max_new_tokens": limit}}}, "config_tasks": {"gsm8k|0": {"generation_size": 2}}, "results": {"gsm8k|0": {"exact_match": 0.5}}}, [{
        "doc": {"id": "0", "query": "1+1?", "task_name": "gsm8k|0"},
        "model_response": {"text": ["2", "bad\nUser:"], "output_tokens": [[1, 2, 3], [4, 5]]}, "metric": {"exact_match": 1.0}}]
def test_layout_registry_passthrough_and_generation_contract():
    component = ROOT / "src/eval/lighteval"
    assert list(component.glob("*.py")) == [component / "evaluate.py"]
    assert not (component / "pyproject.toml").exists()
    assert all(text not in (component / "evaluate.py").read_text() for text in ("Question:", "Answer:", "DAPO")) and 'kwargs.setdefault("disable_log_stats", "VLLM_LOG_STATS_INTERVAL" not in os.environ)' in (ROOT / "src/infer/vllm-rwkv/vllm/entrypoints/llm.py").read_text()
    assert Registry(tasks=evaluate.TASKS).load_tasks()
    with pytest.raises(ValueError): Registry(tasks="definitely_unknown_task|0").load_tasks()
    params = evaluate._generation_parameters()
    backend = params.to_vllm_dict()
    keys = ("temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty", "frequency_penalty", "penalty_decay", "max_tokens")
    assert tuple(backend[key] for key in keys) == (0.96, 0.76, 32, 1.0, 0.1, 0.0, 0.988, 2048)
    assert backend["stop"] == ["\nUser:"]
    config = evaluate.RWKVVLLMModelConfig(model_name="model", generation_parameters=params)
    logical = config.model_dump()["generation_parameters"]
    assert (logical["frequency_penalty"], logical["penalty_decay"]) == (0.1, 0.988)
def test_pipeline_receives_tasks_precision_and_candidate_unchanged(monkeypatch):
    captured = {}
    for name in ("EvaluationTracker", "PipelineParameters", "RWKVVLLMModelConfig"):
        monkeypatch.setattr(evaluate, name, lambda **kw: kw)
    monkeypatch.setattr(evaluate, "Pipeline", lambda **kw: captured.update(kw) or kw)
    evaluate.build_pipeline()
    assert captured["tasks"] == evaluate.TASKS
    assert captured["pipeline_parameters"]["launcher_type"] is ParallelismManager.VLLM
    assert captured["model_config"]["max_num_seqs"] in evaluate.CONCURRENCY_CANDIDATES and captured["model_config"]["override_chat_template"] is True
def test_official_vllm_bridge_restores_rwkv_boundaries_and_sampling():
    assert is_package_available("vllm") and not getattr(VLLMModel, "is_dummy", False)
    assert resolve_tokenizer_args(evaluate.MODEL_PATH)[0] == "rwkv"
    assert resolve_tokenizer_args("facebook/opt-125m")[0] == "hf"
    config = evaluate.RWKVVLLMModelConfig(model_name=evaluate.MODEL_PATH, generation_parameters=evaluate._generation_parameters())
    tokenizer = object.__new__(VLLMModel)._create_auto_tokenizer(config)
    assert (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token) == ("<|endoftext|>",) * 3
    backend, captured = object.__new__(LLM), {}
    backend.model_config = SimpleNamespace(runner_type="generate", tokenizer_mode="rwkv", hf_config=SimpleNamespace(model_type="rwkv7"))
    backend._run_completion = MethodType(lambda self, **kwargs: captured.update(kwargs) or [], backend)
    model = object.__new__(VLLMModel)
    model.config = config
    model.data_parallel_size, model.model = 1, backend
    task = next(iter(Registry(tasks="aime24_gpassk|0").load_tasks().values()))
    model._generate(inputs=[[1]], max_new_tokens=13, stop_tokens=[], num_samples=max(task.num_samples))
    params = captured["params"]
    assert captured["prompts"] == [{"prompt_token_ids": [1]}]
    assert (params.stop, params.stop_token_ids, params.ignore_eos) == (["\nUser:"], [0], False)
    assert (params.n, params.max_tokens, params.repetition_penalty, params.frequency_penalty, params.penalty_decay) == (48, 13, 0.1, 0.0, 0.988)
def test_official_task_native_metrics_receive_raw_completions(monkeypatch):
    math = next(iter(Registry(tasks="gsm8k|0").load_tasks().values()))
    doc = math.formatter({"question": "1+1?", "answer": "work #### 2"}, math.name)
    prompt = {}
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: (prompt.update(messages=messages, options=kw), "rendered")[1])
    assert PromptManager(True, tokenizer).prepare_prompt(doc) == "rendered" and prompt["messages"][-1]["content"] == doc.query
    seen, compute = [], math.metrics[0].compute_sample
    monkeypatch.setattr(math.metrics[0], "compute_sample",
                        lambda **kw: seen.append(kw["model_response"].final_text[0]) or compute(**kw))
    raw, truncated = "<think>open\nANSWER: 2", "<think>truncated"
    assert apply_metric([ModelResponse(text=[raw])], [doc], math.metrics)[0]["extractive_match"] == 1
    assert apply_metric([ModelResponse(text=[truncated])], [doc], math.metrics)[0]["extractive_match"] == 0
    assert seen == [raw, truncated]
    results, rows = artifacts(2)
    rows[0]["model_response"] = {"text": [truncated], "output_tokens": [[1, 2]]}
    assert evaluate.diagnose(results, rows)["truncated"] == 1
    arc = next(iter(Registry(tasks="arc:easy|0").load_tasks().values()))
    arc_doc = arc.formatter({"question": "1+1?", "choices": {"text": ["1", "2", "3", "4"],
        "label": ["A", "B", "C", "D"]}, "answerKey": "B"}, arc.name)
    assert apply_metric([ModelResponse(logprobs=[-2, -.1, -3, -4])], [arc_doc], arc.metrics)[0] == {"acc": 1}
    instruction = next(iter(Registry(tasks="ifeval|0").load_tasks().values()))
    if_doc = instruction.formatter({"prompt": "Use at least two words",
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"num_words": 2, "relation": "at least"}]}, instruction.name)
    assert apply_metric([ModelResponse(text=["hello world"])], [if_doc], instruction.metrics)[0] == {
        "prompt_level_strict_acc": 1, "inst_level_strict_acc": [True],
        "prompt_level_loose_acc": 1, "inst_level_loose_acc": [True]}
def test_parser_reads_artifacts_written_by_pinned_evaluation_tracker(tmp_path):
    assert importlib.metadata.version("lighteval") == "0.13.0"
    tasks = Registry(tasks="gsm8k|0").load_tasks()
    name, task = next(iter(tasks.items()))
    task.config.generation_size = 2
    tracker = EvaluationTracker(output_dir=str(tmp_path), save_details=True)
    tracker.task_config_logger.log(tasks); tracker.general_config_logger.log_args_info(1, None, 0)
    tracker.general_config_logger.log_model_info(evaluate.RWKVVLLMModelConfig(
        model_name="fixture", generation_parameters=evaluate.RWKVGenerationParameters(max_new_tokens=2)))
    doc = task.formatter({"question": "1+1?", "answer": "work #### 2"}, task.name)
    doc.id, doc.task_name = "0", name
    response = ModelResponse(input=doc.query, text=["ANSWER: 2\nUser:", "short"], output_tokens=[[1, 2], [3]])
    tracker.details_logger.log(name, doc, response, {"extractive_match": 1.0})
    tracker.metrics_logger.metric_aggregated[name] = {"extractive_match": 1.0}; tracker.save()
    decoy = tmp_path / "details/fixture/other/details_decoy_other.parquet"; decoy.parent.mkdir(parents=True); decoy.write_bytes(b"invalid")
    results, rows = evaluate.read_standard_artifacts(tmp_path)
    assert evaluate.diagnose(results, rows) == {
        "samples": 1, "completions": 2, "truncated": 1, "non_truncated": 1,
        "truncation_rate": 0.5, "turn_boundary_violations": 1,
        "turn_boundary_violation_rate": 0.5,
    }
    config = results["config_general"]["model_config"]["generation_parameters"]
    config["max_new_tokens"] = None
    assert evaluate.diagnose(results, rows)["truncated"] == 1
    tokens = rows[0]["model_response"].pop("output_tokens")
    with pytest.raises(ValueError): evaluate.diagnose(results, rows)
    rows[0]["model_response"]["output_tokens"] = tokens[:-1]
    with pytest.raises(ValueError): evaluate.diagnose(results, rows)
    rows[0]["model_response"]["output_tokens"], config["max_new_tokens"] = tokens, 0
    with pytest.raises(ValueError): evaluate.diagnose(results, rows)
    config["max_new_tokens"], rows[0]["model_response"]["text"] = 2, []
    with pytest.raises(ValueError): evaluate.diagnose(results, rows)
    rows[0]["model_response"]["logprobs"] = [0.0, 0.0]
    assert evaluate.diagnose(results, rows)["completions"] == 0
def test_scoreboard_projection_is_honest_and_transport_is_gzip_idempotent(monkeypatch):
    results, rows = artifacts()
    with pytest.raises(ValueError, match="multi-completion"): evaluate.publication_payload(results, rows)
    rows[0]["model_response"] = {"text": ["2"], "output_tokens": [[1]]}
    results["results"]["gsm8k|0"]["other"] = 1.0
    with pytest.raises(ValueError, match="multi-metric"): evaluate.publication_payload(results, rows)
    del results["results"]["gsm8k|0"]["other"]
    with pytest.raises(ValueError, match="lack the identity"): evaluate.publication_payload(results, rows)
    sent = {}
    monkeypatch.setattr(evaluate, "urlopen", lambda request, timeout:
                        (sent.update(request=request, timeout=timeout), SimpleNamespace(read=lambda: b"ok"))[1])
    payload = {"manifest": {"digest": "a" * 64}, "non_official": True}
    assert evaluate.publish(payload, "https://scoreboard", "token", "run") == b"ok"
    request = sent["request"]
    assert json.loads(gzip.decompress(request.data)) == payload
    assert request.get_header("Idempotency-key") == f"publish:{'a' * 64}" and sent["timeout"] == 30
    pipeline = SimpleNamespace(evaluate=lambda: None, save_and_push_results=lambda: None,
        show_results=lambda: None, get_results=lambda: {"evaluation": "succeeded"})
    monkeypatch.setattr(evaluate, "build_pipeline", lambda: pipeline)
    monkeypatch.setattr(evaluate, "read_standard_artifacts", lambda _: artifacts())
    monkeypatch.setattr(evaluate, "PUBLISH_SCOREBOARD", True)
    assert evaluate.main() == {"evaluation": "succeeded"}
    pipeline.evaluate = lambda: (_ for _ in ()).throw(RuntimeError("native prerequisite"))
    with pytest.raises(RuntimeError, match="native prerequisite"): evaluate.main()
