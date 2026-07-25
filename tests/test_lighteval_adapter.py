import collections
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from helicopter_eval import lighteval_adapter


def test_runtime_environment_is_restored_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "before")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    with pytest.raises(RuntimeError, match="stop"):
        with lighteval_adapter._temporary_environment(
            {
                "VLLM_RWKV7_WKV_MODE": "fp16",
                "VLLM_USE_V2_MODEL_RUNNER": "1",
            }
        ):
            assert os.environ["VLLM_RWKV7_WKV_MODE"] == "fp16"
            assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
            raise RuntimeError("stop")

    assert os.environ["VLLM_RWKV7_WKV_MODE"] == "before"
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ


def test_evaluation_scopes_recurrent_total_length_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, str] = {}
    unit = SimpleNamespace(weight=object(), wkv_mode="fp16")
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "before")
    monkeypatch.setattr(
        lighteval_adapter, "verify_weight_identity", lambda _weight: None
    )

    def fake_evaluate_unit(**_kwargs):
        observed["allow_long_max_model_len"] = os.environ[
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
        ]
        return [], []

    monkeypatch.setattr(lighteval_adapter, "_evaluate_unit", fake_evaluate_unit)

    lighteval_adapter.evaluate_unit(
        unit=unit,
        shards=(),
        campaign_dir=Path("/unused"),
    )

    assert observed == {"allow_long_max_model_len": "1"}
    assert os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "before"


def test_task_failure_record_uses_only_exception_type() -> None:
    error = RuntimeError("credential=do-not-record")

    failure_type = lighteval_adapter._exception_type(error)

    assert failure_type == "builtins.RuntimeError"
    assert "do-not-record" not in failure_type


def test_model_length_reserves_checkpoint_context_and_full_output_budget() -> None:
    assert lighteval_adapter.evaluation_max_model_length(8192) == 16384
    assert lighteval_adapter.evaluation_max_model_length(10240) == 18432
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="context length must be positive"):
            lighteval_adapter.evaluation_max_model_length(invalid)


def test_perplexity_windows_score_every_token_once_with_bounded_context() -> None:
    windows = lighteval_adapter._rolling_token_windows(
        list(range(1, 13)),
        prefix_token_id=0,
        checkpoint_context_length=7,
    )

    assert windows == (
        ([0], [1, 2, 3, 4, 5]),
        ([5], [6, 7, 8, 9, 10]),
        ([7, 8, 9, 10], [11, 12]),
    )
    assert [token for _, continuation in windows for token in continuation] == list(
        range(1, 13)
    )
    assert all(
        len(context) + len(continuation) + 1 <= 7 for context, continuation in windows
    )


def test_perplexity_uses_raw_query_and_preserves_token_logprobs() -> None:
    types = lighteval_adapter._runtime_types()
    _, _, Model, _, _ = lighteval_adapter._build_runtime_classes(types)
    generated_inputs: list[list[int]] = []
    model = object.__new__(Model)
    model._checkpoint_context_length = 6
    model._tokenizer = lambda text, add_special_tokens: {
        "input_ids": list(range(1, len(text) + 1))
    }

    def generate(inputs, *, generate):
        assert generate is False
        generated_inputs.extend(inputs)
        outputs = []
        for prompt in inputs:
            outputs.append(
                SimpleNamespace(
                    prompt_token_ids=prompt,
                    prompt_logprobs=[
                        None,
                        *[
                            {
                                token_id: SimpleNamespace(
                                    logprob=-token_id / 10,
                                    rank=1,
                                )
                            }
                            for token_id in prompt[1:]
                        ],
                    ],
                )
            )
        return outputs

    model._generate = generate
    docs = [SimpleNamespace(query="abcdefg")]

    responses = model.loglikelihood_rolling(docs)

    assert generated_inputs == [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]]
    assert len(responses) == 1
    assert responses[0].input == "abcdefg"
    assert responses[0].input_tokens == [1, 2, 3, 4, 5, 6, 7]
    assert responses[0].output_tokens == [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
    ]
    assert responses[0].logprobs == pytest.approx(
        [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
    )
    assert responses[0].argmax_logits_eq_gold == [True] * 7


def test_perplexity_fails_closed_when_vllm_omits_a_token_logprob() -> None:
    types = lighteval_adapter._runtime_types()
    _, _, Model, _, _ = lighteval_adapter._build_runtime_classes(types)
    model = object.__new__(Model)
    model._checkpoint_context_length = 6
    model._tokenizer = lambda text, add_special_tokens: {"input_ids": [1, 2]}
    model._generate = lambda inputs, generate: [
        SimpleNamespace(
            prompt_token_ids=[0, 1, 2],
            prompt_logprobs=[None, {1: SimpleNamespace(logprob=-1, rank=1)}, {}],
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="omitted a perplexity token logprob",
    ):
        model.loglikelihood_rolling([SimpleNamespace(query="document")])


def test_perplexity_bounds_prompt_logprob_output_batches() -> None:
    types = lighteval_adapter._runtime_types()
    _, _, Model, _, _ = lighteval_adapter._build_runtime_classes(types)
    model = object.__new__(Model)
    model._checkpoint_context_length = 6
    model._tokenizer = lambda text, add_special_tokens: {"input_ids": [1]}
    batch_sizes: list[int] = []

    def generate(inputs, *, generate):
        assert generate is False
        batch_sizes.append(len(inputs))
        return [
            SimpleNamespace(
                prompt_token_ids=prompt,
                prompt_logprobs=[
                    None,
                    {1: SimpleNamespace(logprob=-0.1, rank=1)},
                ],
            )
            for prompt in inputs
        ]

    model._generate = generate
    documents = [
        SimpleNamespace(query=f"document-{index}")
        for index in range(lighteval_adapter.PERPLEXITY_WINDOW_BATCH_SIZE + 1)
    ]

    responses = model.loglikelihood_rolling(documents)

    assert batch_sizes == [
        lighteval_adapter.PERPLEXITY_WINDOW_BATCH_SIZE,
        1,
    ]
    assert len(responses) == len(documents)


def test_vllm_model_receives_the_same_recurrent_total_length_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = lighteval_adapter._runtime_types()
    _, _, Model, _, _ = lighteval_adapter._build_runtime_classes(types)
    captured = {}

    class FakeLlm:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("vllm.LLM", FakeLlm)
    model = object.__new__(Model)
    model._max_length = 16384
    config = SimpleNamespace(
        model_name="file:///weights/rwkv7-g1g-7.2b-20260523-ctx8192.pth",
        checkpoint_context_length=8192,
        gpu_memory_utilization=0.9,
        revision="main",
        subfolder=None,
        dtype="float16",
        trust_remote_code=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        swap_space=4,
        seed=0,
        quantization=None,
        load_format=None,
        data_parallel_size=1,
    )

    model._create_auto_model(config)

    assert captured["max_model_len"] == 16384
    assert captured["hf_overrides"] == {"model_max_length": 16384}


def test_generation_contract_maps_logical_penalty_once() -> None:
    types = lighteval_adapter._runtime_types()
    Generation, ModelConfig, _, _, choice_answer = (
        lighteval_adapter._build_runtime_classes(types)
    )
    parameters = Generation(
        temperature=0.96,
        top_p=0.76,
        top_k=32,
        presence_penalty=1.0,
        frequency_penalty=0.1,
        penalty_decay=0.988,
        stop_tokens=["\nUser:"],
        max_new_tokens=8192,
    )
    backend = parameters.to_vllm_dict()
    assert backend["repetition_penalty"] == 0.1
    assert backend["frequency_penalty"] == 0.0
    assert backend["penalty_decay"] == 0.988
    assert backend["stop_token_ids"] == [0]
    assert backend["ignore_eos"] is False
    config = ModelConfig(
        model_name="model",
        wkv_mode="fp16",
        checkpoint_context_length=8192,
        generation_parameters=parameters,
    )
    assert config.max_num_seqs is None
    assert config.max_num_batched_tokens is None

    choices = ["one", "two", "three", "four"]
    assert choice_answer("<think>x</think>Answer: B", [1], choices) == "two"
    assert choice_answer("<think>x", [1], choices) == ""
    assert choice_answer("<think>x</think>Answer: B\nAnswer: C", [1], choices) == ""
    assert (
        choice_answer(
            "<think>x</think>Answer: B",
            [1] * lighteval_adapter.MAX_NEW_TOKENS,
            choices,
        )
        == ""
    )


def test_pipeline_uses_registry_only_cache_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = lighteval_adapter._runtime_types()
    _, _, _, Pipeline, _ = lighteval_adapter._build_runtime_classes(types)
    observed = {}

    def fake_init(self, *args, **kwargs):
        model = kwargs["model"]
        observed["cache"] = model._cache
        model._cache._init_registry(object())

    monkeypatch.setattr(types["Pipeline"], "__init__", fake_init)
    model = SimpleNamespace(_cache=None)
    Pipeline(
        tasks="task|0",
        pipeline_parameters=object(),
        evaluation_tracker=object(),
        model=model,
    )
    assert observed["cache"] is not None
    assert model._cache is None


def test_campaign_owned_model_cleanup_only_runs_at_unit_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = lighteval_adapter._runtime_types()
    _, _, Model, _, _ = lighteval_adapter._build_runtime_classes(types)
    calls = []
    monkeypatch.setattr(
        types["VLLMModel"],
        "cleanup",
        lambda self: calls.append("cleanup"),
    )
    model = object.__new__(Model)
    model._unit_closed = False
    model.cleanup()
    assert calls == []
    model.close_unit()
    assert calls == ["cleanup"]
    model.cleanup()
    model.close_unit()
    assert calls == ["cleanup"]


def test_failed_model_cleanup_never_unregisters_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend:
        @staticmethod
        def close_unit():
            raise RuntimeError("engine still active")

    finished: list[str] = []
    removed: list[str] = []
    campaign_dir = tmp_path / "campaign"
    runtime_dir = campaign_dir / "runtime" / "weight" / "fp16"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(
        lighteval_adapter,
        "_remove_runtime_directory",
        lambda *_args: removed.append("removed"),
    )

    with pytest.raises(
        lighteval_adapter.UnsafeModelCleanupError,
        match="model cleanup failed",
    ) as raised:
        lighteval_adapter._finish_runtime(
            backend=Backend(),
            campaign_dir=campaign_dir,
            runtime_dir=runtime_dir,
            on_runtime_finished=lambda: finished.append("finished"),
        )

    assert "engine still active" not in str(raised.value)
    assert "builtins.RuntimeError" in str(raised.value)
    assert finished == []
    assert removed == []
    assert runtime_dir.is_dir()


def test_failed_model_construction_is_an_unsafe_lifecycle_stop() -> None:
    class Model:
        def __init__(self, _config):
            raise RuntimeError("credential=must-not-be-reported")

    with pytest.raises(
        lighteval_adapter.UnsafeModelCleanupError,
        match="before lifecycle ownership could be proven safe",
    ) as raised:
        lighteval_adapter._construct_backend(Model, object())

    assert "must-not-be-reported" not in str(raised.value)
    assert "builtins.RuntimeError" in str(raised.value)


def test_generation_keeps_doc_stop_while_reusing_chat_prompt_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = lighteval_adapter._runtime_types()
    _, _, Model, _, _ = lighteval_adapter._build_runtime_classes(types)
    observed: list[bool] = []
    monkeypatch.setattr(
        types["VLLMModel"],
        "_greedy_until",
        lambda self, docs: observed.append(self.use_chat_template) or docs,
    )
    model = object.__new__(Model)
    model.use_chat_template = True
    docs = [SimpleNamespace(stop_sequences=[lighteval_adapter.STOP_SEQUENCE])]

    assert model._greedy_until(docs) is docs
    assert observed == [False]
    assert model.use_chat_template is True

    model.use_chat_template = False
    with pytest.raises(RuntimeError, match="requires its chat template"):
        model._greedy_until(docs)


def test_mixed_choice_docs_preserve_native_metric_names_and_aggregator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = lighteval_adapter._runtime_types()
    sampling = types["SamplingMethod"]
    generated_metrics: list[SimpleNamespace] = []

    def sample_metric(**values):
        metric = SimpleNamespace(**values)
        generated_metrics.append(metric)
        return metric

    types["ExactMatches"] = lambda: "strict-exact-match"
    types["SampleLevelMetric"] = sample_metric
    _, _, _, Pipeline, _ = lighteval_adapter._build_runtime_classes(types)
    monkeypatch.setattr(
        types["Pipeline"],
        "_init_tasks_and_requests",
        lambda self, tasks: None,
    )

    eligible = SimpleNamespace(
        choices=["alpha", "beta"],
        gold_index=1,
        query="Choose one.",
        sampling_methods=[sampling.LOGPROBS],
        specific={},
        generation_size=None,
        stop_sequences=None,
    )
    ineligible = SimpleNamespace(
        choices=[str(index) for index in range(27)],
        gold_index=0,
        query="Too many choices.",
        sampling_methods=[sampling.LOGPROBS],
        specific={},
        generation_size=None,
        stop_sequences=None,
    )
    corpus_aggregate = object()
    native_metric = SimpleNamespace(
        metric_name="acc",
        category=sampling.LOGPROBS,
        corpus_level_fn=corpus_aggregate,
        higher_is_better=True,
    )
    config = SimpleNamespace(metrics=(native_metric,))
    task = SimpleNamespace(
        full_name="mixed|0",
        metrics=(native_metric,),
        config=config,
        eval_docs=lambda: [eligible, ineligible],
        sampling_methods=[sampling.LOGPROBS],
    )
    pipeline = object.__new__(Pipeline)
    pipeline.tasks_dict = {"mixed|0": task}
    pipeline.documents_dict = {"mixed|0": [eligible, ineligible]}
    pipeline.sampling_docs = collections.defaultdict(list)
    pipeline.evaluation_tracker = SimpleNamespace(
        task_config_logger=SimpleNamespace(log=lambda tasks: None)
    )

    pipeline._init_tasks_and_requests("mixed|0")

    assert eligible.sampling_methods == [sampling.GENERATIVE]
    assert eligible.specific["rwkv_generative_choice"] is True
    assert ineligible.sampling_methods == [sampling.LOGPROBS]
    assert task.metrics[0] is native_metric
    assert len(generated_metrics) == 1
    assert generated_metrics[0].metric_name == "acc"
    assert generated_metrics[0].category == sampling.GENERATIVE
    assert generated_metrics[0].corpus_level_fn is corpus_aggregate
    assert generated_metrics[0].higher_is_better is True
    assert task.config is not config
    assert task.config.metrics == task.metrics
    assert config.metrics == (native_metric,)
    assert pipeline.sampling_docs[sampling.GENERATIVE] == [eligible]
    assert pipeline.sampling_docs[sampling.LOGPROBS] == [ineligible]
    assert eligible.generation_size == lighteval_adapter.MAX_NEW_TOKENS
    assert eligible.stop_sequences == [lighteval_adapter.STOP_SEQUENCE]


def test_singleton_gold_index_list_is_a_uniquely_resolved_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = lighteval_adapter._runtime_types()
    sampling = types["SamplingMethod"]
    generated_metrics: list[SimpleNamespace] = []
    types["ExactMatches"] = lambda: "strict-exact-match"
    types["SampleLevelMetric"] = lambda **values: (
        generated_metrics.append(SimpleNamespace(**values)) or generated_metrics[-1]
    )
    _, _, _, Pipeline, _ = lighteval_adapter._build_runtime_classes(types)
    monkeypatch.setattr(
        types["Pipeline"],
        "_init_tasks_and_requests",
        lambda self, tasks: None,
    )
    document = SimpleNamespace(
        choices=["alpha", "beta"],
        gold_index=[1],
        query="Choose one.",
        sampling_methods=[sampling.LOGPROBS],
        specific={},
        generation_size=None,
        stop_sequences=None,
    )
    metric = SimpleNamespace(
        metric_name="acc",
        category=sampling.LOGPROBS,
        corpus_level_fn=sum,
        higher_is_better=True,
    )
    task = SimpleNamespace(
        full_name="singleton-gold|0",
        metrics=(metric,),
        config=SimpleNamespace(metrics=(metric,)),
        eval_docs=lambda: [document],
        sampling_methods=[sampling.LOGPROBS],
    )
    pipeline = object.__new__(Pipeline)
    pipeline.tasks_dict = {task.full_name: task}
    pipeline.documents_dict = {task.full_name: [document]}
    pipeline.sampling_docs = collections.defaultdict(list)
    pipeline.evaluation_tracker = SimpleNamespace(
        task_config_logger=SimpleNamespace(log=lambda tasks: None)
    )

    pipeline._init_tasks_and_requests(task.full_name)

    assert document.sampling_methods == [sampling.GENERATIVE]
    assert document.specific["rwkv_generative_choice"] is True
    assert generated_metrics[0].metric_name == "acc"


def test_choice_conversion_does_not_replace_native_generative_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = lighteval_adapter._runtime_types()
    sampling = types["SamplingMethod"]
    types["ExactMatches"] = lambda: "strict-exact-match"
    types["SampleLevelMetric"] = lambda **values: SimpleNamespace(**values)
    _, _, _, Pipeline, _ = lighteval_adapter._build_runtime_classes(types)
    monkeypatch.setattr(
        types["Pipeline"],
        "_init_tasks_and_requests",
        lambda self, tasks: None,
    )
    document = SimpleNamespace(
        choices=["alpha", "beta"],
        gold_index=1,
        query="Keep the native prompt.",
        sampling_methods=[sampling.LOGPROBS, sampling.GENERATIVE],
        specific={},
        generation_size=None,
        stop_sequences=None,
    )
    logprob_metric = SimpleNamespace(
        metric_name="acc",
        category=sampling.LOGPROBS,
        corpus_level_fn=sum,
        higher_is_better=True,
    )
    generative_metric = SimpleNamespace(
        metric_name="native_generation",
        category=sampling.GENERATIVE,
        corpus_level_fn=sum,
        higher_is_better=True,
    )
    config = SimpleNamespace(metrics=(logprob_metric, generative_metric))
    task = SimpleNamespace(
        full_name="hybrid|0",
        metrics=config.metrics,
        config=config,
        eval_docs=lambda: [document],
        sampling_methods=[sampling.LOGPROBS, sampling.GENERATIVE],
    )
    pipeline = object.__new__(Pipeline)
    pipeline.tasks_dict = {task.full_name: task}
    pipeline.documents_dict = {task.full_name: [document]}
    pipeline.sampling_docs = collections.defaultdict(list)
    pipeline.evaluation_tracker = SimpleNamespace(
        task_config_logger=SimpleNamespace(log=lambda tasks: None)
    )

    pipeline._init_tasks_and_requests(task.full_name)

    assert document.query == "Keep the native prompt."
    assert document.sampling_methods == [
        sampling.LOGPROBS,
        sampling.GENERATIVE,
    ]
    assert "rwkv_generative_choice" not in document.specific
    assert task.metrics == (logprob_metric, generative_metric)


def test_choice_conversion_does_not_add_metrics_to_mixed_generative_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = lighteval_adapter._runtime_types()
    sampling = types["SamplingMethod"]
    generated_metrics: list[SimpleNamespace] = []
    types["ExactMatches"] = lambda: "strict-exact-match"
    types["SampleLevelMetric"] = lambda **values: (
        generated_metrics.append(SimpleNamespace(**values)) or generated_metrics[-1]
    )
    _, _, _, Pipeline, _ = lighteval_adapter._build_runtime_classes(types)
    monkeypatch.setattr(
        types["Pipeline"],
        "_init_tasks_and_requests",
        lambda self, tasks: None,
    )
    choice = SimpleNamespace(
        choices=["alpha", "beta"],
        gold_index=1,
        query="Keep this native choice prompt.",
        sampling_methods=[sampling.LOGPROBS],
        specific={},
        generation_size=None,
        stop_sequences=None,
    )
    generative = SimpleNamespace(
        choices=[],
        gold_index=[],
        query="Keep this native generation prompt.",
        sampling_methods=[sampling.GENERATIVE],
        specific={},
        generation_size=128,
        stop_sequences=["native-stop"],
    )
    logprob_metric = SimpleNamespace(
        metric_name="acc",
        category=sampling.LOGPROBS,
        corpus_level_fn=sum,
        higher_is_better=True,
    )
    generative_metric = SimpleNamespace(
        metric_name="native_generation",
        category=sampling.GENERATIVE,
        corpus_level_fn=sum,
        higher_is_better=True,
    )
    task = SimpleNamespace(
        full_name="mixed-generative|0",
        metrics=(logprob_metric, generative_metric),
        config=SimpleNamespace(metrics=(logprob_metric, generative_metric)),
        eval_docs=lambda: [choice, generative],
        sampling_methods=[sampling.LOGPROBS, sampling.GENERATIVE],
    )
    pipeline = object.__new__(Pipeline)
    pipeline.tasks_dict = {task.full_name: task}
    pipeline.documents_dict = {task.full_name: [choice, generative]}
    pipeline.sampling_docs = collections.defaultdict(list)
    pipeline.evaluation_tracker = SimpleNamespace(
        task_config_logger=SimpleNamespace(log=lambda tasks: None)
    )

    pipeline._init_tasks_and_requests(task.full_name)

    assert generated_metrics == []
    assert choice.query == "Keep this native choice prompt."
    assert choice.sampling_methods == [sampling.LOGPROBS]
    assert task.metrics == (logprob_metric, generative_metric)
    assert pipeline.sampling_docs[sampling.LOGPROBS] == [choice]
    assert pipeline.sampling_docs[sampling.GENERATIVE] == [generative]
    assert generative.generation_size == lighteval_adapter.MAX_NEW_TOKENS
    assert generative.stop_sequences == [lighteval_adapter.STOP_SEQUENCE]
