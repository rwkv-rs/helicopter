from __future__ import annotations

from types import SimpleNamespace

import pytest

from helicopter_lighteval.http_pool import PromptLogprobs, TextCompletion
from helicopter_lm_eval.model import RWKVVLLMHttpLM


class Pool:
    total_capacity = 4
    model_id = "rwkv-test"
    manifest = SimpleNamespace(max_model_len=5)

    def tokenize(self, text: str):
        return tuple(ord(character) for character in text)

    def score_tokens(self, token_ids, *, implicit_prefix_token_id=None):
        del implicit_prefix_token_id
        tokens = tuple(token_ids)
        return PromptLogprobs(
            token_ids=tokens,
            token_logprobs=(None,) + (-1.0,) * (len(tokens) - 1),
            top_logprobs=(None,)
            + tuple({f"token_id:{token}": -1.0} for token in tokens[1:]),
        )

    def generate_text(self, prompt_token_ids, parameters):
        tokens = tuple(prompt_token_ids)
        return TextCompletion(
            text=f"answer-{tokens[-1]} STOP ignored",
            prompt_token_ids=tokens,
            output_token_ids=(20, 21),
            finish_reason="stop",
            stop_reason="STOP",
        )


def test_loglikelihood_sums_only_continuation_tokens() -> None:
    model = RWKVVLLMHttpLM(pool=Pool(), eot_token_id=0, batch_size=2)

    result = model._loglikelihood_tokens(
        [(('context', 'target'), [10, 11], [12, 13])]
    )

    assert result == [(-2.0, True)]


def test_public_loglikelihood_supports_multiple_choice_requests() -> None:
    model = RWKVVLLMHttpLM(pool=Pool(), eot_token_id=0, batch_size=2)
    request = SimpleNamespace(args=("ab", "cd"))

    assert model.loglikelihood([request]) == [(-2.0, True)]


def test_tokenization_removes_remote_rwkv_prefix_token() -> None:
    class PrefixPool(Pool):
        def tokenize(self, text: str):
            del text
            return (0, 10, 11)

    model = RWKVVLLMHttpLM(pool=PrefixPool(), eot_token_id=0, batch_size=1)

    assert model.tok_encode("text", add_special_tokens=False) == [10, 11]


def test_rolling_loglikelihood_scores_every_token_once_across_windows() -> None:
    model = RWKVVLLMHttpLM(pool=Pool(), eot_token_id=0, batch_size=2)
    request = SimpleNamespace(args=("abcde",))

    assert model.loglikelihood_rolling([request]) == [-5.0]


def test_rolling_windows_respect_vllm_decoder_prompt_limit() -> None:
    class BoundaryPool(Pool):
        manifest = SimpleNamespace(max_model_len=16_384)

        def __init__(self) -> None:
            self.scored_lengths: list[int] = []

        def tokenize(self, text: str):
            del text
            return tuple(range(17_000))

        def score_tokens(self, token_ids, *, implicit_prefix_token_id=None):
            del implicit_prefix_token_id
            tokens = tuple(token_ids)
            self.scored_lengths.append(len(tokens))
            return PromptLogprobs(
                token_ids=tokens,
                token_logprobs=(None,) + (-1.0,) * (len(tokens) - 1),
                top_logprobs=(None,) + ({},) * (len(tokens) - 1),
            )

    pool = BoundaryPool()
    model = RWKVVLLMHttpLM(pool=pool, eot_token_id=0, batch_size=2)
    request = SimpleNamespace(args=("long text",))

    assert model.loglikelihood_rolling([request]) == [-16_999.0]
    assert max(pool.scored_lengths) == 16_382


def test_generate_until_truncates_context_applies_stops_and_preserves_order() -> None:
    class GenerationPool(Pool):
        manifest = SimpleNamespace(max_model_len=8)

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

        def generate_text(self, prompt_token_ids, parameters):
            tokens = tuple(prompt_token_ids)
            self.calls.append((tokens, dict(parameters)))
            return TextCompletion(
                text=f"answer-{tokens[-1]} STOP ignored",
                prompt_token_ids=tokens,
                output_token_ids=(20, 21),
                finish_reason="stop",
                stop_reason="STOP",
            )

    pool = GenerationPool()
    model = RWKVVLLMHttpLM(
        pool=pool,
        eot_token_id=0,
        batch_size=2,
        max_gen_toks=2,
    )
    requests = [
        SimpleNamespace(
            args=("abcdef", {"until": ["STOP"], "do_sample": False})
        ),
        SimpleNamespace(
            args=("xyz", {"max_gen_toks": 2, "temperature": 0.8})
        ),
    ]

    assert model.generate_until(requests) == ["answer-102 ", "answer-122 STOP ignored"]
    calls = sorted(pool.calls, key=lambda value: value[0][-1])
    assert calls[0] == (
        (99, 100, 101, 102),
        {
            "max_tokens": 2,
            "stop": ["STOP"],
            "temperature": 1.0,
            "top_k": 1,
        },
    )
    assert calls[1][0] == (120, 121, 122)
    assert calls[1][1]["temperature"] == 1.0
    assert calls[1][1]["top_k"] == 1


def test_generate_until_supports_sampling_and_rejects_unknown_kwargs() -> None:
    model = RWKVVLLMHttpLM(
        pool=Pool(), eot_token_id=0, batch_size=1, max_gen_toks=2
    )
    sampled = SimpleNamespace(
        args=(
            "x",
            {
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 20,
                "seed": 7,
            },
        )
    )

    assert model.generate_until([sampled]) == ["answer-120 STOP ignored"]
    with pytest.raises(ValueError, match="unsupported lm-eval generation kwargs"):
        model.generate_until([SimpleNamespace(args=("x", {"typical_p": 0.9}))])
