from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from tqdm import tqdm


def _completion_url(base_url: str | None) -> str:
    if not base_url:
        raise RuntimeError("LiteLLM base_url is required for OpenAI-compatible logprob patch")
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/completions"
    return f"{base}/v1/completions"


def _served_model_name(model: str) -> str:
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _choice_token_logprobs(payload: dict[str, Any], *, context_chars: int, prompt_chars: int) -> tuple[float, bool, list[str]]:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"completion logprob response has no choices: {payload!r}")
    logprobs = choices[0].get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    token_logprobs = logprobs.get("token_logprobs") or []
    offsets = logprobs.get("text_offset") or []
    top_logprobs = logprobs.get("top_logprobs") or []
    if not (len(tokens) == len(token_logprobs) == len(offsets)):
        raise RuntimeError("completion logprob response has inconsistent token/logprob/offset lengths")

    selected_logprobs: list[float] = []
    selected_tokens: list[str] = []
    selected_argmax: list[bool] = []
    next_offsets = list(offsets[1:]) + [prompt_chars]
    for token, logprob, offset, next_offset, top in zip(tokens, token_logprobs, offsets, next_offsets, top_logprobs):
        if logprob is None:
            continue
        if next_offset <= context_chars or offset >= prompt_chars:
            continue
        selected_logprobs.append(float(logprob))
        selected_tokens.append(str(token))
        if isinstance(top, dict) and top:
            best_token = max(top.items(), key=lambda item: item[1])[0]
            selected_argmax.append(str(best_token) == str(token))
        else:
            selected_argmax.append(False)
    if not selected_logprobs:
        raise RuntimeError("completion logprob response did not include continuation token logprobs")
    return sum(selected_logprobs), all(selected_argmax), selected_tokens


def patch_litellm_logprobs() -> None:
    from lighteval.data import LoglikelihoodDataset
    from lighteval.models.endpoints.litellm_model import LiteLLMClient
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.requests import SamplingMethod
    from lighteval.utils.cache_management import cached

    def _call_completion_logprobs(self: LiteLLMClient, prompt: str) -> dict[str, Any]:
        url = _completion_url(self.base_url)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": _served_model_name(self.model),
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
        }
        last_error: Exception | None = None
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout or 120)
                response.raise_for_status()
                return response.json()
            except Exception as error:  # noqa: BLE001 - mirror LiteLLM retry behavior
                last_error = error
                wait_time = min(64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))
                time.sleep(wait_time)
        raise RuntimeError(f"completion logprob request failed after retries: {last_error}")

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self: LiteLLMClient, docs: list[Any]) -> list[Any]:
        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results: list[Any] = []
        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            contexts = [self.prompt_manager._prepare_plain_text(doc) for doc in split]
            jobs: list[tuple[int, str, str]] = []
            for doc_index, (context, doc) in enumerate(zip(contexts, split)):
                for choice in doc.choices:
                    jobs.append((doc_index, context, context + choice))

            with ThreadPoolExecutor(self.concurrent_requests) as executor:
                responses = list(
                    tqdm(
                        executor.map(lambda item: _call_completion_logprobs(self, item[2]), jobs),
                        total=len(jobs),
                        desc="Loglikelihoods",
                        position=1,
                        leave=False,
                        disable=self.disable_tqdm,
                    )
                )

            grouped_logprobs: list[list[float]] = [[] for _ in split]
            grouped_argmax: list[list[bool]] = [[] for _ in split]
            grouped_tokens: list[list[list[str]]] = [[] for _ in split]
            for (doc_index, context, full_prompt), response in zip(jobs, responses):
                logprob, argmax, tokens = _choice_token_logprobs(
                    response,
                    context_chars=len(context),
                    prompt_chars=len(full_prompt),
                )
                grouped_logprobs[doc_index].append(logprob)
                grouped_argmax[doc_index].append(argmax)
                grouped_tokens[doc_index].append(tokens)

            for context, logprobs, argmax, tokens in zip(contexts, grouped_logprobs, grouped_argmax, grouped_tokens):
                results.append(
                    ModelResponse(
                        input=context,
                        logprobs=logprobs,
                        argmax_logits_eq_gold=argmax,
                        output_tokens=tokens,
                    )
                )

        return dataset.get_original_order(results)

    LiteLLMClient.loglikelihood = loglikelihood


patch_litellm_logprobs()
