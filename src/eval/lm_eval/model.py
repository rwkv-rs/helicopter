from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

from lm_eval import utils
from lm_eval.api.model import TemplateLM

from helicopter_lighteval.http_pool import PoolError, PromptLogprobs, VLLMHttpPool


class RWKVVLLMHttpLM(TemplateLM):
    def __init__(
        self,
        *,
        pool: VLLMHttpPool,
        eot_token_id: int,
        batch_size: int,
        max_gen_toks: int = 256,
    ) -> None:
        super().__init__()
        self.pool = pool
        self._eot_token_id = eot_token_id
        self._batch_size = batch_size
        self._max_length = pool.manifest.max_model_len - 2
        self._max_gen_toks = max_gen_toks

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def tokenizer_name(self) -> str:
        return f"{self.pool.model_id}:remote-rwkv-tokenizer"

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
        **_kwargs: object,
    ) -> list[int]:
        if add_special_tokens not in {None, False}:
            raise ValueError("RWKV HTTP tokenization does not add special tokens")
        tokens = list(self.pool.tokenize(string))
        if tokens and tokens[0] == self.eot_token_id:
            tokens = tokens[1:]
        if left_truncate_len is not None:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        del disable_tqdm
        if not requests:
            return []
        workers = min(self.batch_size, self.pool.total_capacity, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self._generate_request, requests))
        return results

    def _generate_request(self, request) -> str:
        context, raw_kwargs = request.args
        if not isinstance(context, str):
            raise TypeError("generate_until context must be text")
        if not isinstance(raw_kwargs, Mapping):
            raise TypeError("generate_until kwargs must be a mapping")
        parameters, until, max_tokens = self._generation_parameters(raw_kwargs)
        context_limit = self.max_length - max_tokens
        if context_limit < 1:
            raise PoolError(
                "max_gen_toks must leave at least one token for the generation prompt"
            )
        prompt = self.tok_encode(
            context,
            left_truncate_len=context_limit,
            add_special_tokens=False,
        )
        if not prompt:
            prompt = [self.prefix_token_id]
        completion = self.pool.generate_text(prompt, parameters)
        generated = self._truncate_at_stop(completion.text, until)
        self.cache_hook.add_partial(
            "generate_until",
            (context, dict(raw_kwargs)),
            generated,
        )
        return generated

    def _generation_parameters(
        self,
        raw_kwargs: Mapping[str, object],
    ) -> tuple[dict[str, object], tuple[str, ...], int]:
        supported = {
            "do_sample",
            "max_gen_toks",
            "max_new_tokens",
            "min_p",
            "num_beams",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "until",
        }
        unknown = sorted(set(raw_kwargs) - supported)
        if unknown:
            raise ValueError(
                "unsupported lm-eval generation kwargs: " + ", ".join(unknown)
            )

        configured_until = raw_kwargs.get("until", ())
        if isinstance(configured_until, str):
            until = (configured_until,)
        elif isinstance(configured_until, (list, tuple)) and all(
            isinstance(value, str) and value for value in configured_until
        ):
            until = tuple(configured_until)
        else:
            raise ValueError("until must be a string or an array of non-empty strings")

        max_gen_toks = raw_kwargs.get("max_gen_toks")
        max_new_tokens = raw_kwargs.get("max_new_tokens")
        if max_gen_toks is not None and max_new_tokens is not None:
            raise ValueError("max_gen_toks and max_new_tokens cannot both be set")
        max_tokens = (
            max_gen_toks
            if max_gen_toks is not None
            else max_new_tokens
            if max_new_tokens is not None
            else self.max_gen_toks
        )
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ValueError("max_gen_toks must be a positive integer")

        do_sample = raw_kwargs.get("do_sample", False)
        if not isinstance(do_sample, bool):
            raise ValueError("do_sample must be a boolean")
        num_beams = raw_kwargs.get("num_beams", 1)
        if num_beams != 1:
            raise ValueError("RWKV-vLLM HTTP generation supports num_beams = 1 only")

        temperature = raw_kwargs.get("temperature", 1.0)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be numeric")

        parameters: dict[str, object] = {
            "max_tokens": max_tokens,
            "stop": list(until),
            "temperature": float(temperature),
        }
        for name in ("top_p", "top_k", "min_p", "seed"):
            if name in raw_kwargs:
                parameters[name] = raw_kwargs[name]
        if not do_sample:
            # RWKV's rapid sampler rejects temperature=0. Top-k 1 is the same
            # argmax decode while remaining valid for both rapid and native paths.
            parameters.update(temperature=1.0, top_k=1)
        return parameters, until, max_tokens

    @staticmethod
    def _truncate_at_stop(text: str, until: Sequence[str]) -> str:
        positions = [text.find(stop) for stop in until]
        positions = [position for position in positions if position >= 0]
        return text[: min(positions)] if positions else text

    def loglikelihood_rolling(
        self,
        requests,
        disable_tqdm: bool = False,
    ) -> list[float]:
        del disable_tqdm
        all_windows: list[tuple[int, tuple[None, list[int], list[int]]]] = []
        window_counts: list[int] = []
        for request_index, request in enumerate(requests):
            (string,) = request.args
            tokens = self.tok_encode(string, add_special_tokens=False)
            if not tokens:
                window_counts.append(0)
                continue
            rolling_windows = map(
                utils.make_disjoint_window,
                utils.get_rolling_token_windows(
                    token_list=tokens,
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length - 1,
                    context_len=1,
                ),
            )
            windows = [
                (None, context, continuation)
                for context, continuation in rolling_windows
            ]
            all_windows.extend((request_index, window) for window in windows)
            window_counts.append(len(windows))

        scored: list[tuple[int, tuple[float, bool]]] = []
        for offset in range(0, len(all_windows), self.batch_size):
            batch = all_windows[offset : offset + self.batch_size]
            batch_scores = self._loglikelihood_tokens([item[1] for item in batch])
            scored.extend(
                (item[0], score) for item, score in zip(batch, batch_scores, strict=True)
            )

        totals = [0.0] * len(requests)
        observed_counts = [0] * len(requests)
        for request_index, (loglikelihood, _is_greedy) in scored:
            totals[request_index] += loglikelihood
            observed_counts[request_index] += 1
        if observed_counts != window_counts:
            raise RuntimeError("rolling loglikelihood windows were not fully scored")
        for request, total in zip(requests, totals, strict=True):
            self.cache_hook.add_partial(
                "loglikelihood_rolling",
                (request.args[0],),
                total,
            )
        return totals

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        **_kwargs: object,
    ) -> list[tuple[float, bool]]:
        if not requests:
            return []
        workers = min(self.batch_size, self.pool.total_capacity, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self._score_request, requests))
        for request, result in zip(requests, results, strict=True):
            cache_key = request[0]
            if cache_key is not None:
                self.cache_hook.add_partial("loglikelihood", cache_key, result)
        return results

    def _score_request(
        self,
        request: tuple[tuple[str, str] | None, list[int], list[int]],
    ) -> tuple[float, bool]:
        _cache_key, context_tokens, continuation_tokens = request
        if not continuation_tokens:
            return 0.0, True
        if len(continuation_tokens) >= self.max_length:
            raise PoolError(
                "continuation is too long for the effective vLLM context length"
            )
        context_limit = self.max_length - len(continuation_tokens)
        context = context_tokens[-context_limit:]
        if not context:
            context = [self.prefix_token_id]
        prompt = context + continuation_tokens
        scored = self.pool.score_tokens(
            prompt,
            implicit_prefix_token_id=self.eot_token_id,
        )
        return self._continuation_result(scored, len(context))

    @staticmethod
    def _continuation_result(
        scored: PromptLogprobs,
        continuation_start: int,
    ) -> tuple[float, bool]:
        selected = scored.token_logprobs[continuation_start:]
        if not selected or any(value is None for value in selected):
            raise PoolError("continuation token logprobs are missing")
        loglikelihood = sum(value for value in selected if value is not None)
        greedy = True
        for value, top in zip(
            selected,
            scored.top_logprobs[continuation_start:],
            strict=True,
        ):
            if top is None or value is None or value < max(top.values(), default=value):
                greedy = False
                break
        return loglikelihood, greedy
