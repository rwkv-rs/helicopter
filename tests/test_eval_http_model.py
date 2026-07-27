from __future__ import annotations

import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from lighteval.tasks.prompt_manager import PromptManager
except ImportError as error:
    raise unittest.SkipTest("LightEval environment is not active") from error

from helicopter_lighteval import http_model
from helicopter_lighteval.http_model import VLLMHttpModel
from helicopter_lighteval.http_pool import Completion


def _document(query: str, num_samples: int):
    return SimpleNamespace(
        query=query,
        instruction=None,
        fewshot_samples=[],
        use_logits=False,
        num_samples=num_samples,
        generation_size=8192,
        stop_sequences=["✿"],
    )


def test_model_splits_samples_into_requests_and_preserves_document_order() -> None:
    calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
    lock = threading.Lock()

    class Pool:
        total_capacity = 4

        def complete(self, messages, parameters):
            with lock:
                calls.append((messages, parameters))
                token = len(calls)
            return Completion(
                text=messages[-1]["content"],
                reasoning=None,
                prompt_token_ids=(1, 2),
                output_token_ids=(token,),
            )

    model = VLLMHttpModel.__new__(VLLMHttpModel)
    model.pool = Pool()
    model.prompt_manager = PromptManager(use_chat_template=True, tokenizer=None)
    model._raw_prompt_template = "\nBot✿"
    model._generation_parameters = {
        "temperature": 0.96,
        "top_p": 0.76,
        "top_k": 32,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.1,
        "repetition_penalty": 1.0,
        "penalty_decay": 0.988,
        "stop_token_ids": [0],
        "ignore_eos": False,
    }

    responses = model.greedy_until([_document("first", 3), _document("second", 1)])

    assert [response.text for response in responses] == [
        ["first", "first", "first"],
        ["second"],
    ]
    assert len(calls) == 4
    for messages, parameters in calls:
        assert messages[-1]["role"] == "user"
        assert parameters == {
            **model._generation_parameters,
            "max_completion_tokens": 8192,
            "stop": ["✿"],
            "chat_template_kwargs": {
                "rwkv_prompt_template": "\nBot✿",
            },
            "return_token_ids": True,
        }


def test_model_initialization_uses_remote_identity_without_local_vllm() -> None:
    closed: list[bool] = []
    served_model_id = (
        "/home/caizus/Weights/RWKV/rwkv7/pth/"
        "rwkv7-g1h-1.5b-20260710-ctx10240.pth"
    )

    class Pool:
        def __init__(self, manifest):
            self.manifest = manifest

        def preflight(self):
            self._model_id = served_model_id
            return self._model_id

        def close(self):
            closed.append(True)

    original = http_model.VLLMHttpPool
    http_model.VLLMHttpPool = Pool
    try:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            model = VLLMHttpModel(
                manifest=SimpleNamespace(max_model_len=10240, global_step=0),
                cache_dir=cache_dir,
                raw_prompt_template="\nBot✿",
                generation_parameters={"temperature": 0.96},
            )
            assert model.config.model_name.startswith("vllm-http-")
            assert model.pool._model_id == served_model_id
            assert model._cache.cache_dir.is_relative_to(cache_dir)
            assert model.max_length == 10240
            model.cleanup()

            next_step = VLLMHttpModel(
                manifest=SimpleNamespace(max_model_len=10240, global_step=1),
                cache_dir=cache_dir,
                raw_prompt_template="\nBot✿",
                generation_parameters={"temperature": 0.96},
            )
            assert next_step.config.model_name != model.config.model_name
            next_step.cleanup()
    finally:
        http_model.VLLMHttpPool = original
    assert closed == [True, True]


if __name__ == "__main__":
    test_model_splits_samples_into_requests_and_preserves_document_order()
    test_model_initialization_uses_remote_identity_without_local_vllm()
