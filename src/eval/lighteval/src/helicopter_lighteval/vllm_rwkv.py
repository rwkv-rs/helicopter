"""The smallest vLLM-RWKV compatibility layer required by pinned LightEval.

LightEval's pinned LiteLLM backend discards the terminal and token evidence that
the evaluator contract needs.  This module deliberately owns only that gap: it
uses the official OpenAI client, leaves prompt construction to LightEval, and
returns normal ``ModelResponse`` instances plus a side-channel evidence record.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence, cast

import httpx
from openai import AsyncOpenAI

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.cache_management import SampleCache

from .datasets.math import MathRepairStrategy, repair_completion


STOP_TEXT = "\nUser:"
STOP_TOKEN_ID = 0
LIGHTEVAL_REVISION = "64f4f5ae173626509fad6e477ca4ee56ebb26129"


class TerminalReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"


class ProviderResponseError(ValueError):
    """The OpenAI-compatible endpoint did not provide signed evidence."""


class ProviderTransportError(RuntimeError):
    """The OpenAI-compatible endpoint could not be reached."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ProviderResponseError("usage values must be non-negative integers")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ProviderResponseError("usage.total_tokens is inconsistent")


@dataclass(frozen=True, slots=True)
class GenerationEvidence:
    raw_completion: str
    scored_completion: str
    output_token_ids: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]
    prompt_text: str
    finish_reason: str
    stop_reason: str | int | None
    terminal_reason: TerminalReason
    truncated: bool
    generation_limit: int
    request_id: str
    usage: TokenUsage
    repair_strategy: str
    repair_action: str
    reasoning: str | None = None

    @property
    def output_token_count(self) -> int:
        return len(self.output_token_ids)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terminal_reason"] = self.terminal_reason.value
        payload["output_token_ids"] = list(self.output_token_ids)
        payload["prompt_token_ids"] = list(self.prompt_token_ids)
        return payload


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    server_revision: str
    wkv_mode: str
    precision: str
    gemm_policy: str
    launch_contract: str


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    served_name: str
    checkpoint_sha256: str
    tokenizer_revision: str
    chat_template_revision: str


@dataclass(frozen=True, slots=True)
class ProviderAttestation:
    model: ModelIdentity
    provider: ProviderIdentity
    capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "provider": asdict(self.provider),
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True, slots=True)
class AttestationDecision:
    official: bool
    mismatches: tuple[str, ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def digest_source(value: object) -> str:
    """Digest fixed upstream source without creating a second revision system."""

    if callable(value):
        try:
            value = inspect.getsource(value)
        except (OSError, TypeError):
            value = repr(value)
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def normalize_generation(
    *,
    completion: str,
    output_token_ids: Sequence[int],
    finish_reason: str | None,
    stop_reason: str | int | None,
    generation_limit: int,
    prompt_token_ids: Sequence[int],
    prompt_text: str,
    request_id: str,
    usage: TokenUsage,
    repair_strategy: MathRepairStrategy | str = MathRepairStrategy.A,
    prompt_for_repair: str = "",
    math_task: bool = False,
    reasoning: str | None = None,
) -> GenerationEvidence:
    """Classify exactly the two signed terminal conditions.

    The endpoint is asked to stop on token ``0`` or ``\nUser:``.  A response
    without either signal is accepted only when vLLM reports ``length`` and the
    returned token count equals the requested limit.  No local second request or
    arbitrary text truncation is attempted.
    """

    if (
        isinstance(generation_limit, bool)
        or not isinstance(generation_limit, int)
        or generation_limit <= 0
    ):
        raise ProviderResponseError("generation_limit must be positive")
    if not isinstance(completion, str):
        raise ProviderResponseError("completion must be a string")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise ProviderResponseError("provider response is missing finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise ProviderResponseError(
            f"unsupported provider terminal reason: {finish_reason}"
        )
    if stop_reason is not None and (
        isinstance(stop_reason, bool) or not isinstance(stop_reason, (int, str))
    ):
        raise ProviderResponseError("stop_reason must be a string, integer, or null")

    output_ids = _integer_sequence(output_token_ids, "output_token_ids")
    prompt_ids = _integer_sequence(prompt_token_ids, "prompt_token_ids")
    if usage.prompt_tokens != len(prompt_ids):
        raise ProviderResponseError(
            "usage.prompt_tokens does not match prompt_token_ids"
        )
    if usage.completion_tokens != len(output_ids):
        raise ProviderResponseError(
            "usage.completion_tokens does not match output_token_ids"
        )
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ProviderResponseError("prompt_text must be a non-empty string")
    if not isinstance(request_id, str) or not request_id:
        raise ProviderResponseError("request id must be a non-empty string")

    delimiter = completion.find(STOP_TEXT)
    if delimiter >= 0 and completion[delimiter + len(STOP_TEXT) :]:
        raise ProviderResponseError("provider returned text after the stop delimiter")
    zero_positions = tuple(
        index for index, value in enumerate(output_ids) if value == STOP_TOKEN_ID
    )
    if zero_positions and zero_positions != (len(output_ids) - 1,):
        raise ProviderResponseError("token 0 must be the final generated token")

    stopped_by_text = delimiter >= 0 or stop_reason == STOP_TEXT
    stopped_by_token = bool(zero_positions) or stop_reason in {
        STOP_TOKEN_ID,
        str(STOP_TOKEN_ID),
    }
    if stop_reason in {STOP_TOKEN_ID, str(STOP_TOKEN_ID)} and (
        not zero_positions or zero_positions[-1] != len(output_ids) - 1
    ):
        raise ProviderResponseError("token stop_reason requires a final token 0")
    if finish_reason == "length" and (stopped_by_text or stopped_by_token):
        raise ProviderResponseError(
            "length terminal metadata conflicts with stop evidence"
        )

    if stopped_by_text or stopped_by_token:
        raw = completion[:delimiter] if delimiter >= 0 else completion
        terminal = TerminalReason.STOP
        truncated = False
    elif finish_reason == "length":
        if len(output_ids) != generation_limit:
            raise ProviderResponseError(
                "length termination must contain exactly generation_limit output tokens"
            )
        raw = completion
        terminal = TerminalReason.LENGTH
        truncated = True
    else:
        raise ProviderResponseError(
            "provider response did not terminate with token 0, stop marker, or generation_limit"
        )

    scored = raw
    strategy = MathRepairStrategy(repair_strategy)
    action = "none"
    if math_task:
        repaired = repair_completion(
            prompt=prompt_for_repair,
            raw_completion=raw,
            truncated=truncated,
            strategy=strategy,
        )
        scored = repaired.scored_completion
        action = repaired.action.value

    return GenerationEvidence(
        raw_completion=raw,
        scored_completion=scored,
        output_token_ids=output_ids,
        prompt_token_ids=prompt_ids,
        prompt_text=prompt_text,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        terminal_reason=terminal,
        truncated=truncated,
        generation_limit=generation_limit,
        request_id=request_id,
        usage=usage,
        repair_strategy=strategy.value if math_task else "not-applicable",
        repair_action=action if math_task else "not-applicable",
        reasoning=reasoning,
    )


class VllmRwkvModel(LightevalModel):
    """A LightEval model backed by vLLM-RWKV's OpenAI-compatible endpoint."""

    DATASET_SPLITS = 1
    is_async = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        checkpoint_sha256: str,
        tokenizer_revision: str,
        chat_template_revision: str,
        server_revision: str,
        wkv_mode: str,
        precision: str,
        gemm_policy: str,
        launch_contract: str,
        cot_mode: str,
        math_repair_strategy: MathRepairStrategy | str,
        math_task: bool,
        max_concurrent_requests: int,
        timeout_seconds: float = 3600.0,
    ) -> None:
        if cot_mode not in {"none", "cot"}:
            raise ValueError("cot_mode must be none or cot")
        if (
            isinstance(max_concurrent_requests, bool)
            or not isinstance(max_concurrent_requests, int)
            or max_concurrent_requests <= 0
        ):
            raise ValueError("max_concurrent_requests must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout_seconds = timeout_seconds
        self.cot_mode = cot_mode
        self.math_repair_strategy = MathRepairStrategy(math_repair_strategy)
        self.math_task = math_task
        self.config = ModelConfig(model_name=model)
        # Pipeline requires the upstream cache hook.  Generation is deliberately
        # not decorated with @cached, so terminal evidence cannot be skipped.
        self._cache = SampleCache(self.config)
        self.prompt_manager = PromptManager(use_chat_template=True, tokenizer=None)
        self.model_identity = ModelIdentity(
            served_name=model,
            checkpoint_sha256=checkpoint_sha256,
            tokenizer_revision=tokenizer_revision,
            chat_template_revision=chat_template_revision,
        )
        self.provider_identity = ProviderIdentity(
            server_revision=server_revision,
            wkv_mode=wkv_mode,
            precision=precision,
            gemm_policy=gemm_policy,
            launch_contract=launch_contract,
        )
        self.attestation: ProviderAttestation | None = None
        self.evidence: dict[tuple[str, str], list[GenerationEvidence]] = {}

    @property
    def tokenizer(self):
        return None

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        # Task generation sizes are the signed output limits.  This value only
        # satisfies LightEval's model protocol and is not used for prompt slicing.
        return 32768

    def cleanup(self) -> None:
        # The AsyncOpenAI client is scoped to greedy_until so cancellation and
        # connection cleanup complete before LightEval starts scoring.
        return None

    async def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        if not docs:
            return []

        responses: list[ModelResponse | None] = [None] * len(docs)
        evidence_rows: list[GenerationEvidence | None] = [None] * len(docs)
        next_index = 0

        async with AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            max_retries=0,
        ) as client:

            async def worker() -> None:
                nonlocal next_index
                while next_index < len(docs):
                    index = next_index
                    next_index += 1
                    response, evidence = await self._generate_one(client, docs[index])
                    responses[index] = response
                    evidence_rows[index] = evidence

            workers = [
                asyncio.create_task(worker())
                for _ in range(min(self.max_concurrent_requests, len(docs)))
            ]
            try:
                await asyncio.gather(*workers)
            except BaseException:
                for worker_task in workers:
                    worker_task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                raise

        completed_responses: list[ModelResponse] = []
        for doc, response, evidence in zip(docs, responses, evidence_rows, strict=True):
            if response is None or evidence is None:
                raise RuntimeError("concurrent generation completed without a result")
            self.evidence.setdefault((doc.task_name, str(doc.id)), []).append(evidence)
            completed_responses.append(cast(ModelResponse, response))
        return completed_responses

    async def _generate_one(
        self, client: AsyncOpenAI, doc: Doc
    ) -> tuple[ModelResponse, GenerationEvidence]:
        messages = self.prompt_manager.prepare_prompt_api(doc)
        limit = doc.generation_size
        if not isinstance(limit, int) or limit <= 0:
            raise ProviderResponseError(
                f"task {doc.task_name} does not expose a positive generation_size"
            )
        if doc.num_samples != 1:
            raise ProviderResponseError(
                "vLLM-RWKV adapter currently requires num_samples=1"
            )
        response = await self._complete(client, messages, limit)
        choice = _single_choice(response)
        content = _value(choice, "message.content")
        if not isinstance(content, str):
            raise ProviderResponseError(
                "provider choice message.content must be a string"
            )
        reasoning = _value(choice, "message.reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ProviderResponseError("reasoning_content must be a string or null")
        output_ids = _value(choice, "token_ids")
        if output_ids is None:
            output_ids = _value(response, "token_ids")
        evidence = normalize_generation(
            completion=content,
            output_token_ids=output_ids,
            finish_reason=_value(choice, "finish_reason"),
            stop_reason=_first_present(
                _value(choice, "stop_reason"), _value(response, "stop_reason")
            ),
            generation_limit=limit,
            prompt_token_ids=_value(response, "prompt_token_ids"),
            prompt_text=_value(response, "prompt_text"),
            request_id=_value(response, "id"),
            usage=_usage(_value(response, "usage")),
            repair_strategy=self.math_repair_strategy,
            prompt_for_repair=_prompt_text(messages),
            math_task=self.math_task,
            reasoning=reasoning,
        )
        return (
            ModelResponse(
                input=messages,
                input_tokens=list(evidence.prompt_token_ids),
                text=[evidence.scored_completion],
                output_tokens=[list(evidence.output_token_ids)],
                reasonings=[reasoning],
            ),
            evidence,
        )

    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError("vLLM-RWKV evaluation only supports generative tasks")

    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError("vLLM-RWKV evaluation only supports generative tasks")

    async def _complete(
        self,
        client: AsyncOpenAI,
        messages: list[dict[str, str]],
        generation_limit: int,
    ) -> Any:
        try:
            return await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=generation_limit,
                temperature=0.0,
                stop=[STOP_TEXT],
                extra_body={
                    "stop_token_ids": [STOP_TOKEN_ID],
                    "return_token_ids": True,
                    "return_prompt_text": True,
                    "chat_template_kwargs": {
                        "rwkv_generation_prompt": (
                            "open_think" if self.cot_mode == "cot" else "fake_think"
                        )
                    },
                },
            )
        except Exception as error:  # OpenAI exposes several transport exception types.
            raise ProviderTransportError(
                f"vLLM-RWKV request failed: {error}"
            ) from error


def fetch_attestation(
    *, base_url: str, client: httpx.Client | None = None
) -> ProviderAttestation | None:
    """Read the server's optional attestation without touching persistence."""

    owns_client = client is None
    http_client = client or httpx.Client(timeout=15.0)
    server_url = base_url.rstrip("/")
    if not server_url.endswith("/v1"):
        server_url = f"{server_url}/v1"
    try:
        response = http_client.get(f"{server_url}/helicopter/attestation")
        response.raise_for_status()
        payload = response.json()
        model = ModelIdentity(**payload["model"])
        provider = ProviderIdentity(**payload["provider"])
        capabilities = payload["capabilities"]
        if not isinstance(capabilities, list) or any(
            not isinstance(item, str) for item in capabilities
        ):
            raise TypeError("capabilities must be a string array")
        return ProviderAttestation(model, provider, tuple(capabilities))
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return None
    finally:
        if owns_client:
            http_client.close()


def attest(
    *,
    expected_model: ModelIdentity,
    expected_provider: ProviderIdentity,
    expected_capabilities: Sequence[str],
    actual: ProviderAttestation | None,
    allow_non_comparable: bool,
) -> AttestationDecision:
    expected = {
        "model": asdict(expected_model),
        "provider": asdict(expected_provider),
        "capabilities": sorted(expected_capabilities),
    }
    if actual is None:
        mismatches = ("missing_attestation",)
    else:
        observed = actual.to_dict()
        mismatches_list: list[str] = []
        for section in ("model", "provider"):
            for name, value in expected[section].items():
                if observed[section].get(name) != value:
                    mismatches_list.append(f"{section}.{name}")
        if sorted(observed["capabilities"]) != expected["capabilities"]:
            mismatches_list.append("capabilities")
        mismatches = tuple(mismatches_list)
    if mismatches and not allow_non_comparable:
        raise ProviderResponseError(
            "provider attestation mismatch: " + ", ".join(mismatches)
        )
    return AttestationDecision(official=not mismatches, mismatches=mismatches)


def _single_choice(response: Any) -> Any:
    choices = _value(response, "choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or len(choices) != 1
    ):
        raise ProviderResponseError("provider response must contain exactly one choice")
    return choices[0]


def _value(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            candidate = getattr(current, part, None)
            if candidate is None:
                extra = getattr(current, "model_extra", None)
                if isinstance(extra, Mapping):
                    candidate = extra.get(part)
            current = candidate
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _integer_sequence(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProviderResponseError(f"{field} must be an integer array")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value
    ):
        raise ProviderResponseError(f"{field} must be an integer array")
    return tuple(value)


def _usage(value: Any) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=_required_int(value, "prompt_tokens"),
        completion_tokens=_required_int(value, "completion_tokens"),
        total_tokens=_required_int(value, "total_tokens"),
    )


def _required_int(value: Any, field: str) -> int:
    item = _value(value, field)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise ProviderResponseError(f"usage.{field} must be a non-negative integer")
    return item


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return json.dumps(
        messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
