from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence
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
        if self._model_id is None:
            raise PoolError("vLLM pool must pass preflight before evaluation")
        attempted: set[int] = set()
        failures: list[str] = []
        max_attempts = min(2, len(self._clients))
        for _ in range(max_attempts):
            with self._scheduler.lease(frozenset(attempted)) as index:
                attempted.add(index)
                try:
                    return self._complete_on(index, messages, parameters)
                except httpx.HTTPStatusError as error:
                    status = error.response.status_code
                    if status < 500 and status != 429:
                        detail = error.response.text[:500]
                        raise PoolError(
                            f"vLLM rejected chat completion with HTTP {status}: {detail}"
                        ) from error
                    failures.append(
                        f"{self.manifest.replicas[index].base_url}: HTTP {status}"
                    )
                except (httpx.RequestError, ValueError, KeyError, TypeError) as error:
                    failures.append(
                        f"{self.manifest.replicas[index].base_url}: "
                        f"{type(error).__name__}: {error}"
                    )
        raise PoolError("vLLM chat completion failed: " + "; ".join(failures))

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

    @staticmethod
    def _token_ids(value: object, name: str) -> tuple[int, ...]:
        if not isinstance(value, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in value
        ):
            raise ValueError(f"chat completion {name} token ids are missing or invalid")
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
