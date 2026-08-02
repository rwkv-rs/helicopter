from __future__ import annotations

import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence, TypeVar
from urllib.parse import urlsplit

import httpx


class PoolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Replica:
    base_url: str
    max_concurrency: int


@dataclass(frozen=True)
class PoolManifest:
    global_step: int
    wkv_mode: str
    vllm_version: str
    max_model_len: int
    replicas: tuple[Replica, ...]
    weight_sha256: str | None = None
    weight_display_name: str | None = None

    @classmethod
    def read(cls, path: Path) -> PoolManifest:
        try:
            with path.open(encoding="utf-8") as stream:
                raw = json.load(stream)
        except FileNotFoundError as error:
            raise PoolError(f"vLLM pool manifest not found: {path}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise PoolError(f"invalid vLLM pool manifest: {error}") from error
        if not isinstance(raw, dict):
            raise PoolError("vLLM pool manifest must be a JSON object")

        allowed = {
            "schema_version",
            "global_step",
            "wkv_mode",
            "vllm_version",
            "max_model_len",
            "replicas",
            "weight_sha256",
            "weight_display_name",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PoolError("unknown vLLM pool manifest fields: " + ", ".join(unknown))
        if raw.get("schema_version") != 1 or isinstance(
            raw.get("schema_version"), bool
        ):
            raise PoolError("vLLM pool manifest schema_version must be 1")

        global_step = raw.get("global_step")
        if (
            not isinstance(global_step, int)
            or isinstance(global_step, bool)
            or global_step < 0
        ):
            raise PoolError("vLLM pool manifest global_step must be non-negative")
        wkv_mode = raw.get("wkv_mode")
        if wkv_mode not in {"fp16", "fp32io16"}:
            raise PoolError("vLLM pool manifest has an unsupported wkv_mode")
        vllm_version = raw.get("vllm_version")
        if (
            not isinstance(vllm_version, str)
            or not vllm_version
            or vllm_version != vllm_version.strip()
        ):
            raise PoolError("vLLM pool manifest vllm_version must be non-empty")
        max_model_len = raw.get("max_model_len")
        if (
            not isinstance(max_model_len, int)
            or isinstance(max_model_len, bool)
            or max_model_len <= 0
        ):
            raise PoolError("vLLM pool manifest max_model_len must be positive")
        weight_sha256 = raw.get("weight_sha256")
        if weight_sha256 is not None and (
            not isinstance(weight_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", weight_sha256) is None
        ):
            raise PoolError("vLLM pool manifest weight_sha256 must be lowercase SHA-256")
        weight_display_name = raw.get("weight_display_name")
        if weight_display_name is not None and (
            not isinstance(weight_display_name, str)
            or not weight_display_name
            or weight_display_name != weight_display_name.strip()
        ):
            raise PoolError(
                "vLLM pool manifest weight_display_name must be non-empty"
            )

        configured_replicas = raw.get("replicas")
        if not isinstance(configured_replicas, list) or not configured_replicas:
            raise PoolError("vLLM pool manifest replicas must be a non-empty array")
        replicas = tuple(
            cls._read_replica(value, index)
            for index, value in enumerate(configured_replicas)
        )
        urls = [replica.base_url for replica in replicas]
        duplicate_urls = sorted(url for url in set(urls) if urls.count(url) > 1)
        if duplicate_urls:
            raise PoolError(
                "duplicate vLLM replica base_url: " + ", ".join(duplicate_urls)
            )
        return cls(
            global_step=global_step,
            wkv_mode=wkv_mode,
            vllm_version=vllm_version,
            max_model_len=max_model_len,
            replicas=replicas,
            weight_sha256=weight_sha256,
            weight_display_name=weight_display_name,
        )

    @staticmethod
    def _read_replica(raw: object, index: int) -> Replica:
        if not isinstance(raw, dict):
            raise PoolError(f"vLLM pool replica {index} must be a JSON object")
        unknown = sorted(set(raw) - {"base_url", "max_concurrency"})
        if unknown:
            raise PoolError(
                f"unknown vLLM pool replica {index} fields: " + ", ".join(unknown)
            )
        base_url = raw.get("base_url")
        if not isinstance(base_url, str) or base_url != base_url.strip():
            raise PoolError(f"vLLM pool replica {index} has an invalid base_url")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise PoolError(
                f"vLLM pool replica {index} base_url must be an HTTP(S) origin"
            )
        max_concurrency = raw.get("max_concurrency")
        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
            or max_concurrency <= 0
        ):
            raise PoolError(
                f"vLLM pool replica {index} max_concurrency must be positive"
            )
        return Replica(
            base_url=base_url.rstrip("/"),
            max_concurrency=max_concurrency,
        )

    @property
    def total_capacity(self) -> int:
        return sum(replica.max_concurrency for replica in self.replicas)


class CapacityScheduler:
    """Reserve replica slots using least normalized in-flight load."""

    def __init__(self, capacities: Sequence[int]) -> None:
        if not capacities or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in capacities
        ):
            raise ValueError("capacities must contain positive integers")
        self._capacities = tuple(capacities)
        self._inflight = [0] * len(capacities)
        self._condition = threading.Condition()

    def acquire(self, excluded: frozenset[int] = frozenset()) -> int:
        eligible = [
            index for index in range(len(self._capacities)) if index not in excluded
        ]
        if not eligible:
            raise PoolError("all vLLM replicas have already failed this request")
        with self._condition:
            while True:
                available = [
                    index
                    for index in eligible
                    if self._inflight[index] < self._capacities[index]
                ]
                if available:
                    selected = min(
                        available,
                        key=lambda index: (
                            self._inflight[index] / self._capacities[index],
                            self._inflight[index],
                            index,
                        ),
                    )
                    self._inflight[selected] += 1
                    return selected
                self._condition.wait()

    def release(self, index: int) -> None:
        with self._condition:
            if not 0 <= index < len(self._inflight):
                raise IndexError("replica index is outside the scheduler")
            if self._inflight[index] <= 0:
                raise RuntimeError("replica does not have an in-flight reservation")
            self._inflight[index] -= 1
            self._condition.notify_all()

    @contextmanager
    def lease(self, excluded: frozenset[int] = frozenset()) -> Iterator[int]:
        index = self.acquire(excluded)
        try:
            yield index
        finally:
            self.release(index)

    @property
    def inflight(self) -> tuple[int, ...]:
        with self._condition:
            return tuple(self._inflight)


@dataclass(frozen=True)
class Completion:
    text: str
    reasoning: str | None
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PromptLogprobs:
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float | None, ...]
    top_logprobs: tuple[Mapping[str, float] | None, ...]


@dataclass(frozen=True)
class TextCompletion:
    text: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    finish_reason: str | None
    stop_reason: int | str | None


_Result = TypeVar("_Result")


class VLLMHttpPool:
    def __init__(self, manifest: PoolManifest) -> None:
        self.manifest = manifest
        self._scheduler = CapacityScheduler(
            [replica.max_concurrency for replica in manifest.replicas]
        )
        self._clients = tuple(
            httpx.Client(
                base_url=replica.base_url,
                timeout=httpx.Timeout(3600.0, connect=30.0),
                limits=httpx.Limits(
                    max_connections=replica.max_concurrency,
                    max_keepalive_connections=replica.max_concurrency,
                ),
                trust_env=False,
            )
            for replica in manifest.replicas
        )
        self._model_id: str | None = None

    def preflight(self) -> str:
        with ThreadPoolExecutor(max_workers=len(self._clients)) as executor:
            model_ids = list(
                executor.map(self._preflight_replica, range(len(self._clients)))
            )
        if len(set(model_ids)) != 1:
            raise PoolError(
                "vLLM replicas do not serve one model: " + ", ".join(model_ids)
            )
        self._model_id = model_ids[0]
        return self._model_id

    def _preflight_replica(self, index: int) -> str:
        client = self._clients[index]
        replica = self.manifest.replicas[index]
        try:
            health = client.get("/health")
            health.raise_for_status()
            response = client.get("/v1/models")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise PoolError(
                f"vLLM replica preflight failed for {replica.base_url}: {error}"
            ) from error
        data = payload.get("data") if isinstance(payload, dict) else None
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
            or not isinstance(data[0].get("id"), str)
            or not data[0]["id"]
        ):
            raise PoolError(
                f"vLLM replica {replica.base_url} returned an invalid model list"
            )
        return data[0]["id"]

    def complete(
        self,
        messages: list[dict[str, str]],
        parameters: Mapping[str, object],
    ) -> Completion:
        return self._with_retry(
            "chat completion",
            lambda index: self._complete_on(index, messages, parameters),
        )

    def generate_text(
        self,
        prompt_token_ids: Sequence[int],
        parameters: Mapping[str, object],
    ) -> TextCompletion:
        tokens = self._token_ids(list(prompt_token_ids), "generation prompt")
        if not tokens:
            raise PoolError("generation prompt must contain at least one token")
        return self._with_retry(
            "text completion",
            lambda index: self._generate_text_on(index, tokens, parameters),
        )

    def tokenize(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text to tokenize must be a string")
        return self._with_retry(
            "tokenization",
            lambda index: self._tokenize_on(index, text),
        )

    def _tokenize_on(self, index: int, text: str) -> tuple[int, ...]:
        response = self._clients[index].post(
            "/tokenize",
            json={
                "model": self.model_id,
                "prompt": text,
                "add_special_tokens": False,
            },
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError("tokenization response must be an object")
        return self._token_ids(raw.get("tokens"), "tokenization")

    def score_tokens(
        self,
        token_ids: Sequence[int],
        *,
        implicit_prefix_token_id: int | None = None,
    ) -> PromptLogprobs:
        tokens = self._token_ids(list(token_ids), "prompt")
        if len(tokens) < 2:
            raise PoolError("prompt scoring requires at least two token ids")
        if len(tokens) > self.manifest.max_model_len - 2:
            raise PoolError(
                "prompt scoring exceeds the effective vLLM context length"
            )
        return self._with_retry(
            "prompt scoring",
            lambda index: self._score_tokens_on(
                index,
                tokens,
                implicit_prefix_token_id=implicit_prefix_token_id,
            ),
        )

    def _score_tokens_on(
        self,
        index: int,
        token_ids: tuple[int, ...],
        *,
        implicit_prefix_token_id: int | None,
    ) -> PromptLogprobs:
        response = self._clients[index].post(
            "/v1/completions",
            json={
                "model": self.model_id,
                "prompt": list(token_ids),
                "echo": True,
                "max_tokens": 0,
                "logprobs": 1,
                "prompt_logprobs": 1,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "add_special_tokens": False,
                "stream": False,
                "n": 1,
            },
        )
        response.raise_for_status()
        raw = response.json()
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("prompt scoring must return exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("prompt scoring choice must be an object")
        returned_tokens = choice.get("prompt_token_ids")
        if returned_tokens is None:
            returned_tokens = raw.get("prompt_token_ids")
        parsed_tokens = self._token_ids(returned_tokens, "prompt scoring")
        has_implicit_prefix = (
            implicit_prefix_token_id is not None
            and parsed_tokens == (implicit_prefix_token_id,) + token_ids
        )
        if parsed_tokens != token_ids and not has_implicit_prefix:
            raise ValueError(
                "prompt scoring returned different token ids: "
                f"expected len={len(token_ids)} head={token_ids[:4]} "
                f"tail={token_ids[-4:]}; returned len={len(parsed_tokens)} "
                f"head={parsed_tokens[:4]} tail={parsed_tokens[-4:]}"
            )
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            raise ValueError("prompt scoring logprobs are missing or invalid")
        raw_token_logprobs = logprobs.get("token_logprobs")
        if (
            not isinstance(raw_token_logprobs, list)
            or len(raw_token_logprobs) != len(parsed_tokens)
        ):
            raise ValueError("prompt scoring token logprobs are misaligned")
        token_logprobs: list[float | None] = []
        for position, value in enumerate(raw_token_logprobs):
            if position == 0:
                if value is not None:
                    raise ValueError(
                        "prompt scoring first token logprob must be null"
                    )
                token_logprobs.append(None)
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError("prompt scoring token logprobs are invalid")
            token_logprobs.append(float(value))

        raw_top_logprobs = logprobs.get("top_logprobs")
        if not isinstance(raw_top_logprobs, list):
            raw_top_logprobs = [None] * len(parsed_tokens)
        if len(raw_top_logprobs) != len(parsed_tokens):
            raise ValueError("prompt scoring top logprobs are misaligned")
        top_logprobs: list[Mapping[str, float] | None] = []
        for value in raw_top_logprobs:
            if value is None:
                top_logprobs.append(None)
                continue
            if not isinstance(value, dict) or any(
                not isinstance(key, str)
                or not isinstance(logprob, (int, float))
                or isinstance(logprob, bool)
                or not math.isfinite(logprob)
                for key, logprob in value.items()
            ):
                raise ValueError("prompt scoring top logprobs are invalid")
            top_logprobs.append(
                {key: float(logprob) for key, logprob in value.items()}
            )
        if has_implicit_prefix:
            token_logprobs = token_logprobs[1:]
            top_logprobs = top_logprobs[1:]
            token_logprobs[0] = None
            top_logprobs[0] = None
            parsed_tokens = token_ids
        return PromptLogprobs(
            token_ids=parsed_tokens,
            token_logprobs=tuple(token_logprobs),
            top_logprobs=tuple(top_logprobs),
        )

    def _with_retry(
        self,
        operation: str,
        request: Callable[[int], _Result],
    ) -> _Result:
        if self._model_id is None:
            raise PoolError("vLLM pool must pass preflight before evaluation")
        attempted: set[int] = set()
        failures: list[str] = []
        max_attempts = min(2, len(self._clients))
        for _ in range(max_attempts):
            with self._scheduler.lease(frozenset(attempted)) as index:
                attempted.add(index)
                try:
                    return request(index)
                except httpx.HTTPStatusError as error:
                    status = error.response.status_code
                    if status < 500 and status != 429:
                        detail = error.response.text[:500]
                        raise PoolError(
                            f"vLLM rejected {operation} with HTTP {status}: {detail}"
                        ) from error
                    failures.append(
                        f"{self.manifest.replicas[index].base_url}: HTTP {status}"
                    )
                except (httpx.RequestError, ValueError, KeyError, TypeError) as error:
                    failures.append(
                        f"{self.manifest.replicas[index].base_url}: "
                        f"{type(error).__name__}: {error}"
                    )
        raise PoolError(f"vLLM {operation} failed: " + "; ".join(failures))

    def _complete_on(
        self,
        index: int,
        messages: list[dict[str, str]],
        parameters: Mapping[str, object],
    ) -> Completion:
        payload = dict(parameters)
        payload.update(model=self._model_id, messages=messages, n=1, stream=False)
        response = self._clients[index].post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        raw = response.json()
        choices = raw["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("chat completion must return exactly one choice")
        choice = choices[0]
        message = choice["message"]
        text = message.get("content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError("chat completion content must be text")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("chat completion reasoning_content must be text")
        prompt_token_ids = self._token_ids(raw.get("prompt_token_ids"), "prompt")
        output_token_ids = self._token_ids(choice.get("token_ids"), "output")
        detokenized = self._clients[index].post(
            "/detokenize",
            json={"model": self._model_id, "tokens": list(prompt_token_ids)},
        )
        detokenized.raise_for_status()
        prompt_text = detokenized.json()["prompt"]
        if not isinstance(prompt_text, str):
            raise ValueError("vLLM detokenized prompt must be text")
        return Completion(
            text=text,
            reasoning=reasoning,
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
        )

    def _generate_text_on(
        self,
        index: int,
        prompt_token_ids: tuple[int, ...],
        parameters: Mapping[str, object],
    ) -> TextCompletion:
        payload = dict(parameters)
        payload.update(
            model=self._model_id,
            prompt=list(prompt_token_ids),
            add_special_tokens=False,
            return_token_ids=True,
            n=1,
            stream=False,
        )
        response = self._clients[index].post("/v1/completions", json=payload)
        response.raise_for_status()
        raw = response.json()
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("text completion must return exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("text completion choice must be an object")
        generated = choice.get("text")
        if not isinstance(generated, str):
            raise ValueError("text completion content must be text")
        returned_prompt = choice.get("prompt_token_ids")
        if returned_prompt is None:
            returned_prompt = raw.get("prompt_token_ids")
        output_token_ids = self._token_ids(choice.get("token_ids"), "output")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError("text completion finish_reason must be text")
        stop_reason = choice.get("stop_reason")
        if stop_reason is not None and (
            not isinstance(stop_reason, (int, str)) or isinstance(stop_reason, bool)
        ):
            raise ValueError("text completion stop_reason must be an integer or text")
        return TextCompletion(
            text=generated,
            prompt_token_ids=self._token_ids(returned_prompt, "prompt"),
            output_token_ids=output_token_ids,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )

    @staticmethod
    def _token_ids(value: object, name: str) -> tuple[int, ...]:
        if not isinstance(value, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in value
        ):
            raise ValueError(f"{name} token ids are missing or invalid")
        return tuple(value)

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            raise PoolError("vLLM pool has not completed preflight")
        return self._model_id

    @property
    def total_capacity(self) -> int:
        return self.manifest.total_capacity

    def close(self) -> None:
        for client in self._clients:
            client.close()
