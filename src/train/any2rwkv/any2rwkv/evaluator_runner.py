from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .artifacts import checkpoint_sha256, write_json
from .artifacts import file_sha256
from .evaluate import (
    P0_REQUIRED,
    QualityMetrics,
    QualityThresholdProfile,
    migration_gate,
    p0_gate,
    paired_bootstrap_ratio_ci,
    quality_gate,
    read_quality_threshold_profile,
)
from .distributed import DistributedContext
from .distill import MIGRATION_BASELINE_STAGES
from .errors import ContractError


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class PairedSampleScore:
    sample_id: str
    teacher: float
    student: float
    group: str


@dataclass(frozen=True)
class EvaluatorConfig:
    split: str
    seed: int
    tokenizer_sha256: str
    dataset_sha256: str
    teacher_sha256: str
    student_sha256: str
    burn_in_tokens: int
    layer_schedule: tuple[int, ...]
    smoke_new_tokens: int = 128
    bootstrap_samples: int = 10_000

    def validate(self) -> None:
        if not self.split:
            raise ValueError("evaluation split is required")
        if self.seed < 0 or self.burn_in_tokens < 0:
            raise ValueError("seed and burn_in_tokens must be non-negative")
        if not self.layer_schedule or len(set(self.layer_schedule)) != len(self.layer_schedule):
            raise ValueError("layer_schedule must contain unique layer indices")
        if self.smoke_new_tokens <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("smoke_new_tokens and bootstrap_samples must be positive")
        for name in ("tokenizer_sha256", "dataset_sha256", "teacher_sha256", "student_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass
class _PairAccumulator:
    squared_error: float = 0.0
    teacher_square: float = 0.0
    dot: float = 0.0
    student_square: float = 0.0
    count: int = 0

    def add(self, student: Tensor, teacher: Tensor) -> None:
        student = student.detach().float().cpu().reshape(-1)
        teacher = teacher.detach().float().cpu().reshape(-1)
        if student.shape != teacher.shape:
            raise ValueError(f"hook tensor shapes differ: {tuple(student.shape)} != {tuple(teacher.shape)}")
        self.squared_error += float(torch.sum((student - teacher).square()))
        self.teacher_square += float(torch.sum(teacher.square()))
        self.dot += float(torch.sum(student * teacher))
        self.student_square += float(torch.sum(student.square()))
        self.count += student.numel()

    def result(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("cannot summarize an empty hook accumulator")
        denominator = max(self.teacher_square, 1e-30)
        cosine_denominator = max(math.sqrt(self.student_square * self.teacher_square), 1e-30)
        return {
            "normalized_mse": self.squared_error / denominator,
            "cosine": self.dot / cosine_denominator,
        }


def _first_tensor(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, Tensor)), None)
    if isinstance(value, Mapping):
        return next((item for item in value.values() if isinstance(item, Tensor)), None)
    return None


def _layers(model: nn.Module) -> list[nn.Module]:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, (nn.ModuleList, list, tuple)):
            return list(candidate)
    raise ValueError("model does not expose text decoder layers")


def _mixer(layer: nn.Module) -> nn.Module:
    for name in ("attn", "linear_attn", "self_attn"):
        candidate = getattr(layer, name, None)
        if isinstance(candidate, nn.Module):
            return candidate
    raise ValueError("decoder layer has no recognized sequence mixer")


class _HookRecorder:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.layers = _layers(model)
        self.intermediate: list[list[Tensor]] = [[] for _ in self.layers]
        self.mixer_output: list[list[Tensor]] = [[] for _ in self.layers]
        self.direct_output: list[list[Tensor]] = [[] for _ in self.layers]
        self.boundary_output: list[list[Tensor]] = [[] for _ in self.layers]
        self.handles: list[Any] = []

    def __enter__(self) -> "_HookRecorder":
        for index, layer in enumerate(self.layers):
            self.handles.append(
                layer.register_forward_pre_hook(self._intermediate_hook(index))
            )
            self.handles.append(_mixer(layer).register_forward_hook(self._mixer_hook(index)))
            self.handles.append(layer.register_forward_hook(self._layer_hook(index)))
        for index in range(len(self.layers) - 1):
            boundary = getattr(self.layers[index + 1], "input_layernorm", None)
            if isinstance(boundary, nn.Module):
                self.handles.append(boundary.register_forward_pre_hook(self._boundary_hook(index)))
        final_norm = getattr(getattr(self.model, "model", None), "norm", None)
        if isinstance(final_norm, nn.Module):
            self.handles.append(final_norm.register_forward_pre_hook(self._boundary_hook(len(self.layers) - 1)))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles:
            handle.remove()

    def _mixer_hook(self, index: int):
        def capture(module, args, output):
            value = _first_tensor(output)
            if value is not None:
                self.mixer_output[index].append(value.detach().cpu())

        return capture

    def _intermediate_hook(self, index: int):
        def capture(module, args):
            value = _first_tensor(args)
            if value is not None:
                self.intermediate[index].append(value.detach().cpu())

        return capture

    def _layer_hook(self, index: int):
        def capture(module, args, output):
            value = _first_tensor(output)
            if value is not None:
                self.direct_output[index].append(value.detach().cpu())

        return capture

    def _boundary_hook(self, index: int):
        def capture(module, args):
            value = _first_tensor(args)
            if value is not None:
                self.boundary_output[index].append(value.detach().cpu())

        return capture

    def signal(self, kind: str, index: int) -> Tensor | None:
        values = {
            "intermediate": self.intermediate[index],
            "mixer": self.mixer_output[index],
            "block": self.direct_output[index] or self.boundary_output[index],
        }[kind]
        if not values:
            return None
        return torch.cat([value.reshape(-1) for value in values])


def _output_logits(output: Any) -> Tensor:
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    if isinstance(output, Mapping) and isinstance(output.get("logits"), Tensor):
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise ValueError("model output does not contain logits")


def _past_key_values(output: Any) -> Any:
    if hasattr(output, "past_key_values"):
        return output.past_key_values
    if isinstance(output, Mapping):
        return output.get("past_key_values")
    return output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None


def _positions(attention_mask: Tensor) -> Tensor:
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 0)


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _forward_window(model: nn.Module, tokens: Tensor, *, burn_in: int, warmed: bool) -> tuple[Tensor, _HookRecorder]:
    tokens = tokens.to(_model_device(model))
    attention_mask = torch.ones_like(tokens)
    position_ids = _positions(attention_mask)
    past = None
    if warmed and burn_in:
        prefix = tokens[:, :burn_in]
        with torch.inference_mode():
            prefix_output = model(
                input_ids=prefix,
                attention_mask=attention_mask[:, :burn_in],
                position_ids=position_ids[:, :burn_in],
                use_cache=True,
            )
        past = _past_key_values(prefix_output)
        if past is None:
            raise ValueError("warmed evaluation requires a model cache")
    window = tokens[:, burn_in:]
    recorder = _HookRecorder(model)
    with recorder, torch.inference_mode():
        output = model(
            input_ids=window,
            attention_mask=attention_mask if warmed and burn_in else attention_mask[:, burn_in:],
            position_ids=position_ids[:, burn_in:],
            past_key_values=past,
            use_cache=False,
        )
    return _output_logits(output).detach().float().cpu(), recorder


def _sample_scores(student_logits: Tensor, teacher_logits: Tensor, tokens: Tensor, burn_in: int) -> dict[str, float | int]:
    labels = tokens[:, burn_in:].cpu()
    if labels.shape[1] < 2:
        raise ValueError("each evaluation window must contain at least two tokens")
    teacher_log_probs = torch.log_softmax(teacher_logits[:, :-1], dim=-1)
    student_log_probs = torch.log_softmax(student_logits[:, :-1], dim=-1)
    targets = labels[:, 1:, None]
    teacher_nll = -teacher_log_probs.gather(-1, targets).squeeze(-1)
    student_nll = -student_log_probs.gather(-1, targets).squeeze(-1)
    teacher_prob = teacher_log_probs.exp()
    kl = torch.sum(teacher_prob * (teacher_log_probs - student_log_probs), dim=-1)
    if not all(torch.isfinite(value).all() for value in (teacher_nll, student_nll, kl)):
        raise ValueError("evaluation produced NaN or Inf")
    return {
        "token_count": int(targets.numel()),
        "teacher_nll_sum": float(teacher_nll.sum()),
        "student_nll_sum": float(student_nll.sum()),
        "token_kl_sum": float(kl.sum()),
        "teacher_mean_nll": float(teacher_nll.mean()),
        "student_mean_nll": float(student_nll.mean()),
        "mean_token_kl": float(kl.mean()),
    }


def _evaluate_mode(
    teacher: nn.Module,
    student: nn.Module,
    samples: Sequence[EvaluationSample],
    *,
    burn_in: int,
    warmed: bool,
    distributed: DistributedContext | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layer_count = len(_layers(teacher))
    if len(_layers(student)) != layer_count:
        raise ValueError("teacher/student layer counts differ")
    accumulators = {
        kind: [_PairAccumulator() for _ in range(layer_count)]
        for kind in ("intermediate", "mixer", "block")
    }
    unavailable: dict[str, set[int]] = {kind: set() for kind in accumulators}
    rows: list[dict[str, Any]] = []
    totals = {"tokens": 0, "teacher_nll": 0.0, "student_nll": 0.0, "kl": 0.0}
    for sample in samples:
        tokens = torch.tensor([sample.input_ids], dtype=torch.long)
        teacher_logits, teacher_hooks = _forward_window(teacher, tokens, burn_in=burn_in, warmed=warmed)
        student_logits, student_hooks = _forward_window(student, tokens, burn_in=burn_in, warmed=warmed)
        if teacher_logits.shape != student_logits.shape:
            raise ValueError("teacher/student logits shapes differ")
        scores = _sample_scores(student_logits, teacher_logits, tokens, burn_in)
        rows.append({"sample_id": sample.sample_id, **scores})
        totals["tokens"] += int(scores["token_count"])
        totals["teacher_nll"] += float(scores["teacher_nll_sum"])
        totals["student_nll"] += float(scores["student_nll_sum"])
        totals["kl"] += float(scores["token_kl_sum"])
        for kind in accumulators:
            for index in range(layer_count):
                teacher_signal = teacher_hooks.signal(kind, index)
                student_signal = student_hooks.signal(kind, index)
                if teacher_signal is None or student_signal is None:
                    unavailable[kind].add(index)
                    continue
                accumulators[kind][index].add(student_signal, teacher_signal)
    if distributed is not None and distributed.world_size > 1:
        shards = distributed.all_gather_objects(
            {
                "totals": totals,
                "rows": rows,
                "unavailable": {
                    kind: sorted(indices) for kind, indices in unavailable.items()
                },
                "accumulators": {
                    kind: [asdict(value) for value in values]
                    for kind, values in accumulators.items()
                },
            }
        )
        totals = {"tokens": 0, "teacher_nll": 0.0, "student_nll": 0.0, "kl": 0.0}
        rows = []
        accumulators = {
            kind: [_PairAccumulator() for _ in range(layer_count)]
            for kind in ("intermediate", "mixer", "block")
        }
        unavailable = {kind: set() for kind in accumulators}
        for shard in shards:
            for name in totals:
                totals[name] += shard["totals"][name]
            rows.extend(shard["rows"])
            for kind in accumulators:
                unavailable[kind].update(shard["unavailable"][kind])
                for index, payload in enumerate(shard["accumulators"][kind]):
                    target = accumulators[kind][index]
                    for field, value in payload.items():
                        setattr(target, field, getattr(target, field) + value)
        rows.sort(key=lambda row: row["sample_id"])
    layer_metrics: dict[str, list[dict[str, Any]]] = {}
    for kind, values in accumulators.items():
        layer_metrics[kind] = []
        for index, accumulator in enumerate(values):
            if index in unavailable[kind] or not accumulator.count:
                layer_metrics[kind].append({"layer_index": index, "status": "not_run", "reason": "teacher/student hook signal unavailable"})
            else:
                layer_metrics[kind].append({"layer_index": index, "status": "run", **accumulator.result()})
    count = totals["tokens"]
    summary = {
        "teacher_ppl": math.exp(totals["teacher_nll"] / count),
        "student_ppl": math.exp(totals["student_nll"] / count),
        "ppl_ratio": math.exp((totals["student_nll"] - totals["teacher_nll"]) / count),
        "mean_token_kl": totals["kl"] / count,
        "token_count": count,
        "layers": layer_metrics,
    }
    return summary, rows


def _smoke(
    student: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    config: EvaluatorConfig,
    *,
    prompt_indices: Sequence[int] | None = None,
    distributed: DistributedContext | None = None,
) -> dict[str, Any]:
    if prompt_indices is None:
        prompt_indices = tuple(range(len(prompts)))
    if distributed is None and len(prompts) != 32:
        raise ValueError("smoke evaluation requires exactly 32 prompts")
    rows: list[dict[str, Any]] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        for index, prompt in zip(prompt_indices, prompts, strict=True):
            row: dict[str, Any] = {"prompt_index": index, "prompt": prompt}
            try:
                encoded = tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
                input_ids = input_ids.to(_model_device(student))
                generated: list[int] = []
                past = None
                attention_mask = torch.ones_like(input_ids)
                current = input_ids
                for _ in range(config.smoke_new_tokens):
                    with torch.inference_mode():
                        output = student(
                            input_ids=current,
                            attention_mask=attention_mask,
                            past_key_values=past,
                            use_cache=True,
                        )
                    logits = _output_logits(output)[:, -1]
                    if not torch.isfinite(logits).all():
                        raise ValueError("generation logits contain NaN or Inf")
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    generated.append(int(next_token.item()))
                    past = _past_key_values(output)
                    if past is None:
                        current = torch.cat(
                            (
                                input_ids,
                                torch.tensor(
                                    [generated],
                                    dtype=input_ids.dtype,
                                    device=input_ids.device,
                                ),
                            ),
                            dim=1,
                        )
                    else:
                        current = next_token
                    attention_mask = torch.ones(
                        1,
                        input_ids.shape[1] + len(generated),
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                row.update({
                    "status": "passed",
                    "generated_token_ids": generated,
                    "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
                })
            except Exception as error:  # Keep all 32 raw outcomes for the smoke rubric.
                row.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "generated_token_ids": []})
            rows.append(row)
    if distributed is not None and distributed.world_size > 1:
        gathered = distributed.all_gather_objects(rows)
        rows = [row for shard in gathered for row in shard]
        rows.sort(key=lambda row: row["prompt_index"])
    if len(rows) != 32 or [row["prompt_index"] for row in rows] != list(range(32)):
        raise ValueError("distributed smoke evaluation must cover 32 prompts exactly once")
    passed = sum(row["status"] == "passed" for row in rows)
    return {
        "status": "run",
        "prompt_count": 32,
        "new_tokens_per_prompt": config.smoke_new_tokens,
        "deterministic_greedy": True,
        "pass_count": passed,
        "pass_rate": passed / 32,
        "raw_outputs": rows,
    }


def _external_suite(
    name: str,
    scores: Sequence[PairedSampleScore] | None,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not scores:
        return {"status": "not_run", "reason": f"no {name} sample scores supplied", "raw_sample_scores": []}
    ids = [score.sample_id for score in scores]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} sample ids must be unique")
    teacher = [score.teacher for score in scores]
    student = [score.student for score in scores]
    if not all(math.isfinite(value) for value in (*teacher, *student)):
        raise ValueError(f"{name} scores contain NaN or Inf")
    lower, upper = paired_bootstrap_ratio_ci(student, teacher, samples=samples, seed=seed)
    groups: dict[str, list[PairedSampleScore]] = {}
    for score in scores:
        groups.setdefault(score.group, []).append(score)
    group_rows = []
    for group, values in sorted(groups.items()):
        teacher_mean = sum(value.teacher for value in values) / len(values)
        student_mean = sum(value.student for value in values) / len(values)
        ratio = student_mean / max(teacher_mean, 1e-30)
        group_rows.append({"group": group, "teacher_mean": teacher_mean, "student_mean": student_mean, "ratio": ratio})
    return {
        "status": "run",
        "bootstrap": {"method": "paired-percentile", "samples": samples, "seed": seed, "ratio_ci": [lower, upper]},
        "groups": group_rows,
        "raw_sample_scores": [asdict(score) for score in scores],
    }


def _gate_payload(name: str, passed: bool, failures: Sequence[str]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "failures": list(failures)}


def run_evaluator(
    *,
    teacher: nn.Module,
    student: nn.Module,
    tokenizer: Any,
    samples: Sequence[EvaluationSample],
    smoke_prompts: Sequence[str],
    config: EvaluatorConfig,
    p0_evidence: Mapping[str, bool],
    p0_evidence_sha256: str | None = None,
    migration_baselines: Mapping[str, float] | None = None,
    ruler_scores: Sequence[PairedSampleScore] | None = None,
    downstream_scores: Sequence[PairedSampleScore] | None = None,
    output_path: Path | None = None,
    quality_thresholds: QualityThresholdProfile | None = None,
    distributed: DistributedContext | None = None,
) -> dict[str, Any]:
    """Run a deterministic, hash-bound evaluator without inventing external scores."""
    config.validate()
    if (
        quality_thresholds is not None
        and config.student_sha256 in quality_thresholds.calibration_student_sha256s
    ):
        raise ValueError(
            "candidate checkpoint participated in threshold calibration; "
            "quality evaluation must use an independent student"
        )
    if not samples or len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("evaluation samples must be non-empty with unique ids")
    if any(len(sample.input_ids) < config.burn_in_tokens + 2 for sample in samples):
        raise ValueError("every sample must contain burn_in_tokens plus two supervised tokens")
    expected_schedule = tuple(range(len(_layers(teacher))))
    if tuple(sorted(config.layer_schedule)) != expected_schedule:
        raise ValueError("layer_schedule must cover every teacher layer exactly once")
    teacher.eval()
    student.eval()
    input_payload = {
        "config": asdict(config),
        "samples": [asdict(sample) for sample in samples],
        "smoke_prompts": list(smoke_prompts),
        "ruler_scores": None if ruler_scores is None else [asdict(score) for score in ruler_scores],
        "downstream_scores": None if downstream_scores is None else [asdict(score) for score in downstream_scores],
        "migration_baselines": migration_baselines,
        "p0_evidence": dict(sorted(p0_evidence.items())),
        "p0_evidence_sha256": p0_evidence_sha256,
    }
    input_sha256 = hashlib.sha256(
        json.dumps(input_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    local_samples = samples
    local_prompts = smoke_prompts
    local_prompt_indices: Sequence[int] | None = None
    if distributed is not None and distributed.world_size > 1:
        if len(samples) < distributed.world_size:
            raise ValueError("distributed evaluation requires at least one sample per rank")
        local_samples = samples[distributed.rank :: distributed.world_size]
        local_prompt_indices = tuple(
            range(distributed.rank, len(smoke_prompts), distributed.world_size)
        )
        local_prompts = tuple(smoke_prompts[index] for index in local_prompt_indices)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        cold, cold_rows = _evaluate_mode(
            teacher,
            student,
            local_samples,
            burn_in=config.burn_in_tokens,
            warmed=False,
            distributed=distributed,
        )
        warmed, warmed_rows = _evaluate_mode(
            teacher,
            student,
            local_samples,
            burn_in=config.burn_in_tokens,
            warmed=True,
            distributed=distributed,
        )
        smoke = _smoke(
            student,
            tokenizer,
            local_prompts,
            config,
            prompt_indices=local_prompt_indices,
            distributed=distributed,
        )
    ruler = _external_suite("RULER", ruler_scores, samples=config.bootstrap_samples, seed=config.seed)
    downstream = _external_suite("downstream", downstream_scores, samples=config.bootstrap_samples, seed=config.seed)
    for boundary in ("intermediate", "mixer", "block"):
        if any(
            row["status"] != "run"
            for row in warmed["layers"][boundary]
        ):
            raise ValueError(
                f"{boundary} hooks are required for every layer"
            )
    block_layers = warmed["layers"]["block"]
    layer_cosines = tuple(float(row["cosine"]) for row in block_layers)
    layer_mse = tuple(float(row["normalized_mse"]) for row in block_layers)
    ruler_lower = None
    ruler_bucket_min = None
    if ruler["status"] == "run":
        ruler_lower = float(ruler["bootstrap"]["ratio_ci"][0])
        ruler_bucket_min = min(float(row["ratio"]) for row in ruler["groups"])
    downstream_lower = None
    downstream_max_drop = None
    if downstream["status"] == "run":
        downstream_lower = float(downstream["bootstrap"]["ratio_ci"][0])
        downstream_max_drop = max(
            (float(row["teacher_mean"]) - float(row["student_mean"])) * 100
            for row in downstream["groups"]
        )
    quality_payload = {
        "ppl_ratio": warmed["ppl_ratio"],
        "mean_token_kl": warmed["mean_token_kl"],
        "layer_cosines": list(layer_cosines),
        "layer_normalized_mse": list(layer_mse),
        "smoke_pass_rate": smoke["pass_rate"],
        "ruler_ci_lower_ratio": ruler_lower,
        "ruler_bucket_min_ratio": ruler_bucket_min,
        "downstream_ci_lower_ratio": downstream_lower,
        "downstream_max_drop_points": downstream_max_drop,
    }
    p0 = p0_gate(p0_evidence)
    p1_metrics = QualityMetrics(
        warmed["ppl_ratio"], warmed["mean_token_kl"], layer_cosines, layer_mse,
        smoke["pass_rate"], 0.0, 0.0, 0.0, 0.0,
    )
    p1 = quality_gate(p1_metrics, level="P1", thresholds=quality_thresholds)
    migration = (
        None
        if migration_baselines is None
        else migration_gate(migration_baselines)
    )
    if migration is not None and not migration.passed:
        p1 = type(p1)(
            p1.name,
            False,
            tuple((*p1.failures, *(f"migration:{item}" for item in migration.failures))),
        )
    if not p0.passed:
        p1 = type(p1)(
            p1.name,
            False,
            tuple((*p1.failures, "P0 prerequisite failed")),
        )
    missing_external = [name for name, suite in (("ruler", ruler), ("downstream", downstream)) if suite["status"] != "run"]
    if missing_external:
        p2_payload = _gate_payload("P2", False, [f"{name}:not_run" for name in missing_external])
    else:
        p2_metrics = QualityMetrics(
            warmed["ppl_ratio"], warmed["mean_token_kl"], layer_cosines, layer_mse,
            smoke["pass_rate"], ruler_lower, ruler_bucket_min, downstream_lower, downstream_max_drop,
        )
        p2 = quality_gate(p2_metrics, level="P2", thresholds=quality_thresholds)
        if not p1.passed:
            p2 = type(p2)(
                p2.name,
                False,
                tuple((*p2.failures, "P1 prerequisite failed")),
            )
        p2_payload = _gate_payload(p2.name, p2.passed, p2.failures)
    result = {
        "schema_version": 1,
        "status": "complete",
        "binding": {
            "evaluator_input_sha256": input_sha256,
            "tokenizer_sha256": config.tokenizer_sha256,
            "dataset_sha256": config.dataset_sha256,
            "teacher_sha256": config.teacher_sha256,
            "student_sha256": config.student_sha256,
        },
        "protocol": {
            "split": config.split,
            "seed": config.seed,
            "burn_in_tokens": config.burn_in_tokens,
            "layer_schedule": list(config.layer_schedule),
            "cold_and_warmed": True,
            "bootstrap_samples": config.bootstrap_samples,
            "quality_threshold_profile": (
                None
                if quality_thresholds is None
                else {
                    "profile_id": quality_thresholds.profile_id,
                    "profile_sha256": quality_thresholds.profile_sha256,
                    "calibration_artifact_sha256": quality_thresholds.calibration_artifact_sha256,
                }
            ),
        },
        "metrics": {"cold": cold, "warmed": warmed, "quality_metrics": quality_payload},
        "raw_sample_metrics": {"cold": cold_rows, "warmed": warmed_rows},
        "smoke": smoke,
        "external_evaluations": {"ruler": ruler, "downstream": downstream},
        "gates": {
            "P0": _gate_payload(p0.name, p0.passed, p0.failures),
            "migration": (
                {"name": "migration", "passed": None, "failures": ["not_run"]}
                if migration is None
                else _gate_payload(migration.name, migration.passed, migration.failures)
            ),
            "P1": _gate_payload(p1.name, p1.passed, p1.failures),
            "P2": p2_payload,
        },
    }
    if output_path is not None and distributed is None:
        write_json(output_path, result)
    if distributed is not None:
        distributed.barrier()
    return result


def combined_tokenizer_sha256(path: Path) -> str:
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and (
            candidate.name.startswith("tokenizer")
            or candidate.name
            in {
                "chat_template.jinja",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            }
        )
    )
    if not files:
        raise ValueError(f"checkpoint has no tokenizer assets: {path}")
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(candidate.name.encode())
        digest.update(file_sha256(candidate).encode())
    return digest.hexdigest()


def _read_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"evaluation file SHA-256 mismatch: {path} expected={expected_sha256} actual={actual}"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_manifest_file(manifest_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def read_evaluation_manifest(
    manifest_path: Path,
) -> tuple[tuple[EvaluationSample, ...], tuple[str, ...], dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("split") != "validation":
        raise ValueError("evaluation manifest must use schema_version=1 and split=validation")
    sample_path = _resolve_manifest_file(manifest_path, payload.get("data_file"))
    rows = _read_jsonl(sample_path, str(payload.get("data_sha256", "")))
    samples = tuple(
        EvaluationSample(str(row["sample_id"]), tuple(int(token) for token in row["input_ids"]))
        for row in rows
    )
    if len(samples) != int(payload.get("row_count", 0)):
        raise ValueError("evaluation manifest row_count mismatch")
    smoke_path = _resolve_manifest_file(manifest_path, payload.get("smoke_file"))
    smoke_rows = _read_jsonl(smoke_path, str(payload.get("smoke_sha256", "")))
    prompts = tuple(str(row["prompt"]) for row in smoke_rows)
    if len(prompts) != 32:
        raise ValueError("evaluation smoke file must contain exactly 32 prompts")
    return samples, prompts, payload


def read_paired_scores(
    path: Path | None,
    *,
    expected_suite: str | None = None,
    teacher_sha256: str | None = None,
    student_sha256: str | None = None,
) -> tuple[PairedSampleScore, ...] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("paired score manifest schema_version must be 1")
    if expected_suite is not None and payload.get("suite") != expected_suite:
        raise ValueError(f"paired score manifest is not the frozen {expected_suite} suite")
    if teacher_sha256 is not None and payload.get("teacher_sha256") != teacher_sha256:
        raise ValueError("paired scores are bound to a different teacher checkpoint")
    if student_sha256 is not None and payload.get("student_sha256") != student_sha256:
        raise ValueError("paired scores are bound to a different student checkpoint")
    revision = str(payload.get("runner_revision", ""))
    if expected_suite is not None and (
        len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("paired scores lack a pinned runner revision")
    data_path = _resolve_manifest_file(path, payload.get("data_file"))
    rows = _read_jsonl(data_path, str(payload.get("sha256", "")))
    scores = tuple(
        PairedSampleScore(
            str(row["sample_id"]),
            float(row["teacher"]),
            float(row["student"]),
            str(row["group"]),
        )
        for row in rows
    )
    if len(scores) != int(payload.get("row_count", 0)):
        raise ValueError("paired score manifest row_count mismatch")
    return scores


def read_p0_evidence(path: Path, *, student_sha256: str) -> dict[str, bool]:
    """Verify hash-bound artifacts instead of accepting self-asserted booleans."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("P0 evidence schema_version must be 1")
    if payload.get("student_sha256") != student_sha256:
        raise ValueError("P0 evidence is bound to a different student checkpoint")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("P0 evidence must contain an evidence object")
    missing = sorted(set(P0_REQUIRED) - evidence.keys())
    extra = sorted(evidence.keys() - set(P0_REQUIRED))
    if missing or extra:
        raise ValueError(f"P0 evidence keys differ missing={missing} extra={extra}")
    verified: dict[str, bool] = {}
    for name in P0_REQUIRED:
        entry = evidence[name]
        if not isinstance(entry, dict) or entry.get("passed") is not True:
            raise ValueError(f"P0 evidence {name} is not an accepted artifact result")
        artifact = _resolve_manifest_file(path, entry.get("path"))
        expected = str(entry.get("sha256", ""))
        if len(expected) != 64 or file_sha256(artifact) != expected:
            raise ValueError(f"P0 evidence artifact SHA-256 mismatch: {name}")
        if entry.get("student_sha256") != student_sha256:
            raise ValueError(f"P0 evidence artifact is not student-bound: {name}")
        artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
        if (
            artifact_payload.get("kind") != name
            or artifact_payload.get("passed") is not True
            or artifact_payload.get("student_sha256") != student_sha256
        ):
            raise ValueError(f"P0 evidence artifact content is invalid: {name}")
        verified[name] = True
    return verified


def read_migration_baselines(
    path: Path,
    *,
    student_sha256: str,
) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3:
        raise ValueError(
            "migration baseline schema_version must be 3 with raw stage evidence"
        )
    if payload.get("student_sha256") != student_sha256:
        raise ValueError("migration baselines are bound to a different student checkpoint")
    binding = payload.get("binding")
    required_binding = {
        "student_sha256",
        "teacher_sha256",
        "tokenizer_sha256",
        "dataset_sha256",
        "split",
        "seed",
        "burn_in_tokens",
        "precision",
        "token_budget",
        "sample_ids_sha256",
    }
    if (
        not isinstance(binding, dict)
        or not required_binding.issubset(binding)
        or binding.get("student_sha256") != student_sha256
    ):
        raise ValueError(
            "migration baselines lack a shared hash-bound evaluation protocol"
        )
    rows = payload.get("baselines")
    if not isinstance(rows, dict):
        raise ValueError("migration baselines must contain a baselines object")
    required = set(MIGRATION_BASELINE_STAGES)
    missing = sorted(required - rows.keys())
    if missing:
        raise ValueError(f"migration baseline matrix is incomplete: {missing}")
    for name in MIGRATION_BASELINE_STAGES:
        row = rows[name]
        if (
            not isinstance(row, dict)
            or int(row.get("token_budget", -1))
            != int(binding["token_budget"])
            or not math.isfinite(float(row.get("mean_token_kl", float("nan"))))
            or not math.isfinite(
                float(row.get("end_to_end_zero_step_nmse", float("nan")))
            )
            or float(row.get("end_to_end_zero_step_nmse", -1)) < 0
            or not isinstance(row.get("incremental_zero_step_nmse"), dict)
            or not row["incremental_zero_step_nmse"]
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in row["incremental_zero_step_nmse"].values()
            )
        ):
            raise ValueError(
                f"migration baseline row violates the shared protocol: {name}"
            )
    fitted = rows["activation_fitted"]
    if (
        fitted.get("solver_invoked") is not True
        or len(str(fitted.get("fit_report_sha256", ""))) != 64
        or len(str(fitted.get("materialization_sha256", ""))) != 64
    ):
        raise ValueError(
            "activation_fitted baseline lacks solver/materialization evidence"
        )
    stage_manifests = payload.get("stage_manifests")
    if (
        not isinstance(stage_manifests, dict)
        or set(stage_manifests) != set(rows)
    ):
        raise ValueError(
            "migration baselines lack one raw stage manifest per matrix row"
        )
    from .core.migration_baselines import read_migration_baseline_stage

    shared_stage_binding = {
        name: binding[name]
        for name in (
            "teacher_sha256",
            "tokenizer_sha256",
            "dataset_sha256",
            "split",
            "seed",
            "burn_in_tokens",
            "precision",
            "token_budget",
            "sample_ids_sha256",
        )
    }
    for name, reference in stage_manifests.items():
        if not isinstance(reference, dict) or reference.get("kind") != "file":
            raise ValueError(f"migration baseline stage reference is invalid: {name}")
        manifest_path = _resolve_manifest_file(path, reference.get("path"))
        expected_sha256 = str(reference.get("sha256", ""))
        if (
            len(expected_sha256) != 64
            or not manifest_path.is_file()
            or file_sha256(manifest_path) != expected_sha256
        ):
            raise ValueError(
                f"migration baseline stage manifest SHA-256 mismatch: {name}"
            )
        try:
            stage, sample_rows, _ = read_migration_baseline_stage(manifest_path)
        except (ContractError, OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"migration baseline raw stage evidence is invalid: {name}: {error}"
            ) from error
        if stage.get("stage") != name or stage.get("binding") != shared_stage_binding:
            raise ValueError(
                f"migration baseline raw stage protocol differs: {name}"
            )
        token_budget = sum(int(row["token_count"]) for row in sample_rows)
        recomputed = max(
            0.0, sum(float(row["token_kl_sum"]) for row in sample_rows)
        ) / token_budget
        raw_incremental_keys = {
            tuple(sorted(row.get("incremental_nmse", {})))
            for row in sample_rows
            if isinstance(row.get("incremental_nmse"), dict)
        }
        if (
            len(raw_incremental_keys) != 1
            or any(
                row.get("end_to_end_zero_step_nmse") is None
                or not isinstance(row.get("incremental_nmse"), dict)
                or not row["incremental_nmse"]
                for row in sample_rows
            )
        ):
            raise ValueError(
                f"migration baseline raw zero-step evidence is incomplete: {name}"
            )
        incremental_keys = next(iter(raw_incremental_keys))
        recomputed_zero_step = sum(
            float(row["end_to_end_zero_step_nmse"]) * int(row["token_count"])
            for row in sample_rows
        ) / token_budget
        recomputed_incremental = {
            component: sum(
                float(row["incremental_nmse"][component])
                * int(row["token_count"])
                for row in sample_rows
            )
            / token_budget
            for component in incremental_keys
        }
        if (
            token_budget != int(rows[name]["token_budget"])
            or not math.isclose(
                recomputed,
                float(rows[name]["mean_token_kl"]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                recomputed_zero_step,
                float(rows[name]["end_to_end_zero_step_nmse"]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            or set(recomputed_incremental)
            != set(rows[name]["incremental_zero_step_nmse"])
            or any(
                not math.isclose(
                    value,
                    float(
                        rows[name]["incremental_zero_step_nmse"][component]
                    ),
                    rel_tol=0,
                    abs_tol=1e-15,
                )
                for component, value in recomputed_incremental.items()
            )
            or stage["candidate"]["sha256"] != rows[name]["candidate_sha256"]
            or stage["sample_metrics"]["sha256"]
            != rows[name]["sample_metrics_sha256"]
        ):
            raise ValueError(
                f"migration baseline aggregate differs from raw stage evidence: {name}"
            )
    return {
        name: float(rows[name]["mean_token_kl"])
        for name in MIGRATION_BASELINE_STAGES
    }


def evaluate_hf_checkpoints(
    *,
    teacher_path: Path,
    student_path: Path,
    manifest_path: Path,
    p0_evidence_path: Path,
    migration_baselines_path: Path,
    quality_threshold_profile_path: Path,
    output_path: Path,
    ruler_scores_path: Path | None = None,
    downstream_scores_path: Path | None = None,
    distributed: DistributedContext,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available() and distributed.world_size != 8:
        raise ValueError(
            "real CUDA evaluation requires torchrun --nproc-per-node=8"
        )
    samples, prompts, manifest = read_evaluation_manifest(manifest_path)
    quality_thresholds = read_quality_threshold_profile(quality_threshold_profile_path)
    tokenizer_sha = combined_tokenizer_sha256(student_path)
    expected_tokenizer_sha = str(manifest.get("tokenizer_sha256", ""))
    if tokenizer_sha != expected_tokenizer_sha:
        raise ValueError(
            f"evaluation tokenizer SHA-256 mismatch: expected={expected_tokenizer_sha} actual={tokenizer_sha}"
        )
    teacher_sha = checkpoint_sha256(teacher_path)
    student_sha = checkpoint_sha256(student_path)
    device = str(distributed.device) if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=dtype, device_map=device
    ).eval()
    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        student_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    p0_evidence = read_p0_evidence(
        p0_evidence_path,
        student_sha256=student_sha,
    )
    migration_baselines = read_migration_baselines(
        migration_baselines_path,
        student_sha256=student_sha,
    )
    config = EvaluatorConfig(
        split="validation",
        seed=int(manifest["seed"]),
        tokenizer_sha256=tokenizer_sha,
        dataset_sha256=str(manifest["data_sha256"]),
        teacher_sha256=teacher_sha,
        student_sha256=student_sha,
        burn_in_tokens=int(manifest["burn_in_tokens"]),
        layer_schedule=tuple(range(len(_layers(teacher)))),
        smoke_new_tokens=int(manifest.get("smoke_new_tokens", 128)),
        bootstrap_samples=int(manifest.get("bootstrap_samples", 10_000)),
    )
    result = run_evaluator(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        samples=samples,
        smoke_prompts=prompts,
        config=config,
        p0_evidence=p0_evidence,
        p0_evidence_sha256=file_sha256(p0_evidence_path),
        migration_baselines=migration_baselines,
        ruler_scores=read_paired_scores(
            ruler_scores_path,
            expected_suite="ruler",
            teacher_sha256=teacher_sha,
            student_sha256=student_sha,
        ),
        downstream_scores=read_paired_scores(
            downstream_scores_path,
            expected_suite="downstream",
            teacher_sha256=teacher_sha,
            student_sha256=student_sha,
        ),
        quality_thresholds=quality_thresholds,
        output_path=output_path,
        distributed=distributed,
    )
    return result


def _migration_stage_evidence(
    path: Path | None, *, stage: str
) -> dict[str, object]:
    if path is None:
        return {"artifacts": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("stage") != stage:
        raise ValueError(
            "migration stage evidence must use schema_version=1 and match the stage"
        )
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("migration stage evidence object is missing")
    artifacts = evidence.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("migration stage evidence artifacts must be an object")
    normalized: dict[str, object] = {**evidence, "artifacts": {}}
    for name, reference in artifacts.items():
        if not isinstance(reference, dict):
            raise ValueError(f"migration stage evidence reference is invalid: {name}")
        artifact = _resolve_manifest_file(path, reference.get("path"))
        kind = str(reference.get("kind", ""))
        expected = str(reference.get("sha256", ""))
        actual = (
            file_sha256(artifact)
            if kind == "file" and artifact.is_file()
            else (
                checkpoint_sha256(artifact)
                if kind == "checkpoint" and artifact.is_dir()
                else ""
            )
        )
        if len(expected) != 64 or actual != expected:
            raise ValueError(
                f"migration stage evidence artifact SHA-256 mismatch: {name}"
            )
        normalized["artifacts"][name] = {
            "path": str(artifact),
            "kind": kind,
            "sha256": actual,
        }
    return normalized


def _merge_zero_step_sample_metrics(
    sample_rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, object],
) -> list[dict[str, Any]]:
    artifacts = evidence.get("artifacts")
    reference = (
        artifacts.get("zero_step_sample_metrics")
        if isinstance(artifacts, Mapping)
        else None
    )
    if not isinstance(reference, Mapping):
        return [dict(row) for row in sample_rows]
    if reference.get("kind") != "file":
        raise ValueError("zero-step sample metrics evidence must be a file")
    overlay_rows = _read_jsonl(
        Path(str(reference["path"])),
        str(reference["sha256"]),
    )
    overlays: dict[str, dict[str, Any]] = {}
    for row in overlay_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in overlays:
            raise ValueError("zero-step sample metrics contain invalid sample IDs")
        if set(row) != {
            "sample_id",
            "end_to_end_zero_step_nmse",
            "incremental_nmse",
        }:
            raise ValueError(
                "zero-step sample metrics must contain only sample_id, "
                "end_to_end_zero_step_nmse, and incremental_nmse"
            )
        overlays[sample_id] = row
    expected_ids = [str(row["sample_id"]) for row in sample_rows]
    if set(overlays) != set(expected_ids) or len(overlays) != len(expected_ids):
        raise ValueError(
            "zero-step sample metrics do not match evaluated sample IDs"
        )
    return [
        {
            **dict(row),
            "end_to_end_zero_step_nmse": overlays[
                str(row["sample_id"])
            ]["end_to_end_zero_step_nmse"],
            "incremental_nmse": overlays[
                str(row["sample_id"])
            ]["incremental_nmse"],
        }
        for row in sample_rows
    ]


def evaluate_hf_migration_stage(
    *,
    teacher_path: Path,
    candidate_path: Path,
    manifest_path: Path,
    stage: str,
    output_path: Path,
    evidence_path: Path | None,
    precision: str,
    distributed: DistributedContext,
) -> dict[str, Any]:
    """Evaluate one matrix row without depending on the completed matrix."""
    from transformers import AutoModelForCausalLM

    from .core.migration_baselines import write_migration_baseline_stage

    if stage not in MIGRATION_BASELINE_STAGES and not re.fullmatch(
        r"corrective_sweep_[0-9]+", stage
    ):
        raise ValueError(f"unknown migration baseline stage: {stage}")
    if torch.cuda.is_available() and distributed.world_size != 8:
        raise ValueError(
            "real CUDA migration baseline evaluation requires 8 ranks"
        )
    samples, _, manifest = read_evaluation_manifest(manifest_path)
    local_samples = (
        samples[distributed.rank :: distributed.world_size]
        if distributed.world_size > 1
        else samples
    )
    if not local_samples:
        raise ValueError(
            "migration baseline evaluation requires at least one sample per rank"
        )
    teacher_sha = checkpoint_sha256(teacher_path)
    candidate_sha = checkpoint_sha256(candidate_path)
    device = str(distributed.device) if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    candidate = AutoModelForCausalLM.from_pretrained(
        candidate_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    summary, sample_rows = _evaluate_mode(
        teacher,
        candidate,
        local_samples,
        burn_in=int(manifest["burn_in_tokens"]),
        warmed=True,
        distributed=distributed,
    )
    evidence = _migration_stage_evidence(evidence_path, stage=stage)
    evidence.update(
        {
            "evaluator_summary": summary,
            "teacher_sha256": teacher_sha,
            "candidate_sha256": candidate_sha,
        }
    )
    sample_rows = _merge_zero_step_sample_metrics(sample_rows, evidence)
    result = None
    error = None
    if distributed.is_primary:
        try:
            result = write_migration_baseline_stage(
                output_path,
                stage=stage,
                protocol={
                    "teacher_sha256": teacher_sha,
                    "tokenizer_sha256": str(manifest["tokenizer_sha256"]),
                    "dataset_sha256": str(manifest["data_sha256"]),
                    "split": "validation",
                    "seed": int(manifest["seed"]),
                    "burn_in_tokens": int(manifest["burn_in_tokens"]),
                    "precision": precision,
                },
                sample_rows=sample_rows,
                candidate_path=candidate_path,
                candidate_kind="checkpoint",
                evidence=evidence,
            )
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
    status = distributed.broadcast_object(
        {"result": result, "error": error} if distributed.is_primary else None
    )
    if status["error"] is not None:
        raise ContractError(
            "migration baseline stage publish failed: " + status["error"]
        )
    distributed.barrier()
    return status["result"]
