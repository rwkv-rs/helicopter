from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os
from typing import Any

from tortoise.transactions import in_transaction

from scoreboard_server.cores.normalize import (
    canonical_completion_status,
    canonical_cot_mode,
    canonical_task_status,
    git_hash,
    iter_stage_indices,
    join_dataset,
    json_key,
    normalize_model_name,
    now_utc_naive,
    parse_datetime,
    parse_model_tags,
    parse_nonneg_int,
    sanitize_json,
    split_dataset,
)
from scoreboard_server.db.connection import init_db
from scoreboard_server.db.models import Benchmark, Checker, Completion, EvalRecord, Score, ScoreModel, Task
from scoreboard_server.db.resume import ResumeContext, TaskLookup
from scoreboard_server.db.settings import DatabaseSettings


_REF_ANSWER_KEYS = (
    "ref_answer",
    "expected_answer",
    "reference_answer",
    "expected_judgement",
    "reference_solution",
    "canonical_solution",
    "solution",
    "output",
    "target",
    "final_answer",
)
_RAW_RECORD_REF_KEYS = (
    "expected_answer",
    "reference_answer",
    "reference_solution",
    "canonical_solution",
    "solution",
    "output",
    "target",
    "final_answer",
    "answer",
    "answers",
    "gold",
    "test_cases",
)


class ScoreboardStore:
    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings or DatabaseSettings.from_env()

    async def _ensure_db(self) -> None:
        await init_db(self.settings)

    async def _benchmark(self, dataset: str, *, num_samples: int | None = None) -> Benchmark:
        await self._ensure_db()
        name, split = split_dataset(dataset)
        defaults = {"url": None, "status": "Todo", "num_samples": int(num_samples or 0)}
        benchmark, _ = await Benchmark.get_or_create(
            benchmark_name=name,
            benchmark_split=split,
            defaults=defaults,
        )
        if num_samples and benchmark.num_samples != int(num_samples):
            benchmark.num_samples = int(num_samples)
            await benchmark.save(update_fields=["num_samples"])
        return benchmark

    async def _model(self, model: str) -> ScoreModel:
        await self._ensure_db()
        normalized = normalize_model_name(model)
        arch, data_version, num_params = parse_model_tags(normalized)
        score_model, _ = await ScoreModel.get_or_create(
            model_name=normalized,
            arch_version=arch,
            data_version=data_version,
            num_params=num_params,
            defaults={},
        )
        return score_model

    async def ensure_benchmark_num_samples(self, *, dataset: str, num_samples: int) -> None:
        if int(num_samples) <= 0:
            return
        await self._benchmark(dataset, num_samples=int(num_samples))

    async def get_resume_context(
        self,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
        job_name: str | None = None,
        sampling_config: dict[str, Any] | None = None,
        config_path: str | None = None,
        force_new_task: bool = False,
    ) -> ResumeContext:
        benchmark = await self._benchmark(dataset)
        score_model = await self._model(model)
        ctx = ResumeContext(benchmark_id=benchmark.benchmark_id, model_id=score_model.model_id)
        if force_new_task:
            return ctx

        sanitized_sampling = sanitize_json(sampling_config) if sampling_config is not None else None
        resolved_config_path = self._task_config_path(config_path)
        task_query = Task.filter(
            evaluator=job_name or "",
            git_hash=git_hash(),
            model_id=score_model.model_id,
            benchmark_id=benchmark.benchmark_id,
            is_tmp=False,
        )
        if resolved_config_path is None:
            task_query = task_query.filter(config_path__isnull=True)
        else:
            task_query = task_query.filter(config_path=resolved_config_path)
        tasks = await task_query.order_by("task_id")
        matches: list[TaskLookup] = []
        completed_ids: list[int] = []
        for task in tasks:
            if json_key(task.sampling_config) != json_key(sanitized_sampling):
                continue
            status = "Completed" if await Score.filter(task_id=task.task_id).exists() else task.status
            lookup = TaskLookup(task_id=task.task_id, status=status)
            matches.append(lookup)
            if status.lower() == "completed":
                completed_ids.append(task.task_id)

        resumable = tuple(task.task_id for task in matches if task.status.lower() in {"running", "failed"})
        running = tuple(task.task_id for task in matches if task.status.lower() == "running")
        if len(resumable) > 1 and len(running) == 1:
            resumable = running
        ctx.matching_tasks = tuple(matches)
        ctx.completed_task_ids = tuple(completed_ids)
        ctx.resumable_task_ids = resumable
        if not completed_ids and len(resumable) == 1:
            ctx.task_id = resumable[0]
            ctx.can_resume = True
            ctx.completed_keys = await self.list_completion_keys(task_id=str(ctx.task_id), status="Completed")
        elif completed_ids:
            ctx.task_id = completed_ids[-1]
        elif resumable:
            ctx.task_id = resumable[-1]
        return ctx

    async def create_task_from_context(
        self,
        *,
        ctx: ResumeContext,
        job_name: str | None,
        dataset: str,
        model: str,
        is_param_search: bool,
        sampling_config: dict[str, Any] | None = None,
        config_path: str | None = None,
    ) -> str:
        if ctx.can_resume and ctx.task_id is not None:
            await Task.filter(task_id=ctx.task_id).update(status="Running")
            await Score.filter(task_id=ctx.task_id).delete()
            return str(ctx.task_id)

        benchmark = await self._benchmark(dataset)
        score_model = await self._model(model)
        task = await Task.create(
            config_path=self._task_config_path(config_path),
            evaluator=job_name or "",
            is_param_search=bool(is_param_search),
            is_tmp=self._task_is_tmp(),
            created_at=now_utc_naive(),
            status="Running",
            git_hash=git_hash(),
            model=score_model,
            benchmark=benchmark,
            description=os.environ.get("RWKV_TASK_DESC"),
            sampling_config=sanitize_json(sampling_config) if sampling_config is not None else None,
            log_path=os.environ.get("RWKV_SKILLS_LOG_PATH", ""),
        )
        return str(task.task_id)

    async def get_or_create_task(
        self,
        *,
        job_name: str | None,
        job_id: str | None,
        dataset: str,
        model: str,
        is_param_search: bool,
        sampling_config: dict[str, Any] | None = None,
        config_path: str | None = None,
        allow_resume: bool = True,
    ) -> str:
        ctx = await self.get_resume_context(
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            job_name=job_name,
            sampling_config=sampling_config,
            config_path=config_path,
            force_new_task=not allow_resume,
        )
        return await self.create_task_from_context(
            ctx=ctx,
            job_name=job_name,
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            sampling_config=sampling_config,
            config_path=config_path,
        )

    async def insert_completion_payloads_batch(self, *, payloads: Sequence[dict[str, Any]], task_id: str) -> int:
        await self._ensure_db()
        if not payloads:
            return 0
        task = await Task.get(task_id=int(task_id))
        count = 0
        async with in_transaction():
            for payload in payloads:
                if str(payload.get("_stage", "answer")).strip().lower() != "answer":
                    continue
                sample_index = parse_nonneg_int(payload.get("sample_index"), "sample_index")
                repeat_index = parse_nonneg_int(payload.get("repeat_index"), "repeat_index")
                pass_index = parse_nonneg_int(payload.get("pass_index", 0), "pass_index")
                context = self._build_completion_context(payload)
                await Completion.update_or_create(
                    task=task,
                    sample_index=sample_index,
                    avg_repeat_index=repeat_index,
                    pass_index=pass_index,
                    defaults={
                        "context": context,
                        "created_at": now_utc_naive(),
                        "status": "Completed",
                    },
                )
                count += 1
        return count

    async def insert_completion_payload(self, *, payload: dict[str, Any], task_id: str) -> None:
        await self.insert_completion_payloads_batch(payloads=[payload], task_id=task_id)

    async def ingest_eval_payloads(self, *, payloads: Iterable[dict[str, Any]], task_id: str) -> int:
        await self._ensure_db()
        mapping = await self._completion_id_map(task_id=task_id, status="Completed")
        existing = {
            row.completion_id
            for row in await EvalRecord.filter(completion__task_id=int(task_id)).only("completion_id")
        }
        inserted = 0
        for payload in payloads:
            key = (
                parse_nonneg_int(payload.get("sample_index"), "sample_index"),
                parse_nonneg_int(payload.get("repeat_index"), "repeat_index"),
                parse_nonneg_int(payload.get("pass_index", 0), "pass_index"),
            )
            completion_id = mapping.get(key)
            if completion_id is None or completion_id in existing:
                continue
            completion = await Completion.get(completions_id=completion_id)
            await EvalRecord.update_or_create(
                completion=completion,
                defaults={
                    "answer": self._bounded_text(payload.get("answer"), 65_536),
                    "ref_answer": self._bounded_text(self._extract_reference_answer(payload), 4_096),
                    "is_passed": bool(payload.get("is_passed", False)),
                    "fail_reason": self._bounded_text(payload.get("fail_reason"), 2_048),
                    "created_at": now_utc_naive(),
                },
            )
            existing.add(completion_id)
            inserted += 1
        return inserted

    async def ingest_checker_payloads(self, *, payloads: Iterable[dict[str, Any]], task_id: str) -> int:
        await self._ensure_db()
        mapping = await self._completion_id_map(task_id=task_id, status="Completed")
        existing = {
            row.completion_id
            for row in await Checker.filter(completion__task_id=int(task_id)).only("completion_id")
        }
        inserted = 0
        for payload in payloads:
            key = (
                parse_nonneg_int(payload.get("sample_index"), "sample_index"),
                parse_nonneg_int(payload.get("repeat_index"), "repeat_index"),
                parse_nonneg_int(payload.get("pass_index", 0), "pass_index"),
            )
            completion_id = mapping.get(key)
            if completion_id is None or completion_id in existing:
                continue
            completion = await Completion.get(completions_id=completion_id)
            await Checker.update_or_create(
                completion=completion,
                defaults={
                    "answer_correct": bool(payload.get("answer_correct", False)),
                    "instruction_following_error": bool(payload.get("instruction_following_error", False)),
                    "world_knowledge_error": bool(payload.get("world_knowledge_error", False)),
                    "math_error": bool(payload.get("math_error", False)),
                    "reasoning_logic_error": bool(payload.get("reasoning_logic_error", False)),
                    "thought_contains_correct_answer": bool(payload.get("thought_contains_correct_answer", False)),
                    "needs_human_review": bool(payload.get("needs_human_review", False)),
                    "reason": str(payload.get("reason") or ""),
                    "created_at": now_utc_naive(),
                },
            )
            existing.add(completion_id)
            inserted += 1
        return inserted

    async def ingest_eval_payload_groups(
        self,
        *,
        task_id: str,
        completion_payloads: Sequence[dict[str, Any]],
        payloads_by_group: Mapping[str, Sequence[dict[str, Any]]],
        primary_group: str,
    ) -> dict[str, int]:
        parent_task_id = int(task_id)
        task_ids: dict[str, int] = {}

        primary_payloads = list(payloads_by_group.get(primary_group, ()))
        await self.ingest_eval_payloads(payloads=primary_payloads, task_id=str(parent_task_id))
        task_ids[str(primary_group)] = parent_task_id

        for group, payloads in payloads_by_group.items():
            if group == primary_group:
                continue
            strategy_task_id = await self.create_eval_strategy_task(parent_task_id=parent_task_id, strategy=str(group))
            await self.insert_completion_payloads_batch(payloads=completion_payloads, task_id=str(strategy_task_id))
            await self.ingest_eval_payloads(payloads=list(payloads), task_id=str(strategy_task_id))
            await self.update_task_status(task_id=str(strategy_task_id), status="completed")
            task_ids[str(group)] = strategy_task_id
        return task_ids

    async def create_eval_strategy_task(self, *, parent_task_id: int, strategy: str) -> int:
        await self._ensure_db()
        parent = await Task.filter(task_id=int(parent_task_id)).select_related("model", "benchmark").first()
        if parent is None:
            raise RuntimeError(f"parent task not found: {parent_task_id}")

        parent_desc = str(parent.description or "")
        desc_parts = [
            part
            for part in (
                parent_desc,
                f"parent_task_id={parent_task_id}",
                f"eval_strategy={strategy}",
            )
            if part
        ]
        task = await Task.create(
            config_path=parent.config_path,
            evaluator=f"{parent.evaluator or 'eval'}:{strategy}",
            is_param_search=True,
            is_tmp=True,
            created_at=now_utc_naive(),
            status="Running",
            git_hash=parent.git_hash or git_hash(),
            model=parent.model,
            benchmark=parent.benchmark,
            description="; ".join(desc_parts),
            sampling_config=parent.sampling_config if isinstance(parent.sampling_config, dict) else None,
            log_path=parent.log_path or "",
        )
        return int(task.task_id)

    async def record_score_payload(self, *, payload: dict[str, Any], task_id: str) -> None:
        await self._ensure_db()
        task = await Task.get(task_id=int(task_id))
        await Score.update_or_create(
            task=task,
            defaults={
                "cot_mode": canonical_cot_mode(payload),
                "metrics": sanitize_json(payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}),
                "created_at": parse_datetime(payload.get("created_at")),
            },
        )
        task.status = "Completed"
        await task.save(update_fields=["status"])

    async def count_completions(self, *, task_id: str, status: str | None = None) -> int:
        query = Completion.filter(task_id=int(task_id))
        if status:
            query = query.filter(status=canonical_completion_status(status))
        return await query.count()

    async def list_completion_keys(self, *, task_id: str, status: str | None = None) -> set[tuple[int, int, int]]:
        query = Completion.filter(task_id=int(task_id))
        if status:
            query = query.filter(status=canonical_completion_status(status))
        rows = await query.order_by("sample_index", "avg_repeat_index", "pass_index")
        return {(row.sample_index, row.avg_repeat_index, row.pass_index) for row in rows}

    async def list_completion_payloads(self, *, task_id: str, status: str | None = None) -> list[dict[str, Any]]:
        rows = await self._completion_rows(task_id=task_id, status=status)
        payloads: list[dict[str, Any]] = []
        for row in rows:
            context = row.context if isinstance(row.context, dict) else {}
            payload: dict[str, Any] = {
                "sample_index": row.sample_index,
                "repeat_index": row.avg_repeat_index,
                "pass_index": row.pass_index,
                "sampling_config": context.get("sampling_config") if isinstance(context.get("sampling_config"), dict) else {},
                "context": context,
            }
            stages = context.get("stages")
            if isinstance(stages, list):
                for idx, stage in enumerate(stages, start=1):
                    if not isinstance(stage, Mapping):
                        continue
                    payload[f"prompt{idx}"] = stage.get("prompt")
                    payload[f"completion{idx}"] = stage.get("completion")
                    payload[f"stop_reason{idx}"] = stage.get("stop_reason")
            payloads.append(payload)
        return payloads

    async def list_latest_scores_for_space(self, *, include_param_search: bool = False) -> list[dict[str, Any]]:
        rows = await Score.all().select_related("task", "task__model", "task__benchmark")
        grouped: dict[tuple[int, int, str, str], Score] = {}
        for row in rows:
            task = row.task
            if task.is_tmp:
                continue
            if task.is_param_search and not include_param_search:
                continue
            key = (task.model_id, task.benchmark_id, task.evaluator, json_key(task.sampling_config))
            prev = grouped.get(key)
            if prev is None or (row.created_at, row.score_id) > (prev.created_at, prev.score_id):
                grouped[key] = row
        return [self._score_row_for_space(row) for row in sorted(grouped.values(), key=lambda item: item.created_at)]

    async def list_score_history_pairs(self) -> list[dict[str, Any]]:
        scores = await Score.all().select_related("task", "task__model", "task__benchmark")
        pairs = {
            (
                score.task.model.model_name,
                join_dataset(score.task.benchmark.benchmark_name, score.task.benchmark.benchmark_split),
            )
            for score in scores
            if not score.task.is_tmp and not score.task.is_param_search
        }
        return [{"model": model, "dataset": dataset} for model, dataset in sorted(pairs)]

    async def list_score_history(self, *, model: str, dataset: str) -> list[dict[str, Any]]:
        benchmark_name, benchmark_split = split_dataset(dataset)
        rows = await Score.filter(
            task__model__model_name=normalize_model_name(model),
            task__benchmark__benchmark_name=benchmark_name,
            task__benchmark__benchmark_split=benchmark_split,
            task__is_tmp=False,
            task__is_param_search=False,
        ).select_related("task", "task__model", "task__benchmark").order_by("created_at", "score_id")
        return [self._history_row(row) for row in rows]

    async def list_scores_by_dataset(
        self,
        *,
        dataset: str,
        model: str,
        is_param_search: bool,
    ) -> list[dict[str, Any]]:
        benchmark_name, benchmark_split = split_dataset(dataset)
        rows = await Score.filter(
            task__benchmark__benchmark_name=benchmark_name,
            task__benchmark__benchmark_split=benchmark_split,
            task__model__model_name=normalize_model_name(model),
            task__is_param_search=bool(is_param_search),
            task__is_tmp=False,
        ).select_related("task", "task__model", "task__benchmark").order_by("-created_at", "-score_id")
        return [self._score_row_for_space(row) for row in rows]

    async def get_score_history_detail(self, *, task_id: str) -> dict[str, Any] | None:
        score = await Score.filter(task_id=int(task_id)).select_related("task", "task__model", "task__benchmark").first()
        task = await Task.filter(task_id=int(task_id)).select_related("model", "benchmark").first()
        if score is None and task is None:
            return None
        completion = await Completion.filter(task_id=int(task_id)).order_by("sample_index", "avg_repeat_index", "pass_index").first()
        return {
            "score": self._history_row(score) if score else None,
            "task": self._task_dict(task) if task else None,
            "context": completion.context if completion else None,
        }

    async def get_score_payload(self, *, task_id: str) -> dict[str, Any] | None:
        score = await Score.filter(task_id=int(task_id)).select_related("task", "task__model", "task__benchmark").first()
        return self._score_row_for_space(score) if score else None

    async def get_latest_task_generation_progress(
        self,
        *,
        evaluator: str,
        model_name: str,
        benchmark_name: str,
        benchmark_split: str,
    ) -> dict[str, Any] | None:
        task = await Task.filter(
            evaluator=evaluator,
            model__model_name=normalize_model_name(model_name),
            benchmark__benchmark_name=benchmark_name,
            benchmark__benchmark_split=benchmark_split,
            is_param_search=False,
            is_tmp=False,
        ).order_by("-task_id").first()
        if task is None:
            return None
        return {
            "task_id": task.task_id,
            "status": task.status,
            "sampling_config": task.sampling_config,
            "completed_completions": await Completion.filter(task_id=task.task_id, status="Completed").count(),
            "total_completions": await Completion.filter(task_id=task.task_id).count(),
            "has_score": await Score.filter(task_id=task.task_id).exists(),
        }

    async def list_eval_records_for_space(
        self,
        *,
        task_id: str,
        only_wrong: bool,
        limit: int | None = None,
        offset: int = 0,
        include_context: bool = True,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        query = EvalRecord.filter(completion__task_id=int(task_id)).select_related("completion")
        if only_wrong:
            query = query.filter(is_passed=False)
        query = query.order_by("completion__sample_index", "completion__avg_repeat_index", "completion__pass_index", "eval_id")
        if offset > 0:
            query = query.offset(offset)
        if limit is not None and limit > 0:
            query = query.limit(limit)
        rows = await query
        payloads: list[dict[str, Any]] = []
        for row in rows:
            context = row.completion.context if isinstance(row.completion.context, dict) else {}
            preview = ""
            stages = context.get("stages")
            if isinstance(stages, list) and stages and isinstance(stages[0], Mapping):
                preview = str(stages[0].get("prompt") or "")[:240]
            elif include_preview:
                preview = str(context)[:240]
            item = {
                "sample_index": row.completion.sample_index,
                "repeat_index": row.completion.avg_repeat_index,
                "pass_index": row.completion.pass_index,
                "is_passed": row.is_passed,
                "answer": row.answer,
                "ref_answer": row.ref_answer,
                "fail_reason": row.fail_reason,
                "context_preview": preview,
            }
            if include_context:
                item["context"] = context
            payloads.append(item)
        return payloads

    async def get_eval_context_for_space(
        self,
        *,
        task_id: str,
        sample_index: int,
        repeat_index: int,
        pass_index: int = 0,
    ) -> Any | None:
        row = await EvalRecord.filter(
            completion__task_id=int(task_id),
            completion__sample_index=int(sample_index),
            completion__avg_repeat_index=int(repeat_index),
            completion__pass_index=int(pass_index),
        ).select_related("completion").order_by("-eval_id").first()
        return row.completion.context if row else None

    async def get_task_bundle(self, *, task_id: str) -> dict[str, Any] | None:
        task = await Task.filter(task_id=int(task_id)).select_related("model", "benchmark").first()
        if not task:
            return None
        return {"task": self._task_dict(task), "model": self._model_dict(task.model), "benchmark": self._benchmark_dict(task.benchmark)}

    async def list_completions_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        rows = await Completion.filter(task_id=int(task_id)).order_by("completions_id")
        return [self._completion_dict(row) for row in rows]

    async def list_eval_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        rows = await EvalRecord.filter(completion__task_id=int(task_id)).select_related("completion").order_by("eval_id")
        return [self._eval_dict(row) for row in rows]

    async def list_checker_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        rows = await Checker.filter(completion__task_id=int(task_id)).select_related("completion").order_by("checker_id")
        return [self._checker_dict(row) for row in rows]

    async def list_checker_keys(self, *, task_id: str) -> set[tuple[int, int, int]]:
        rows = await Checker.filter(completion__task_id=int(task_id)).select_related("completion")
        return {
            (row.completion.sample_index, row.completion.avg_repeat_index, row.completion.pass_index)
            for row in rows
        }

    async def list_scores_rows(self, *, task_id: str) -> list[dict[str, Any]]:
        rows = await Score.filter(task_id=int(task_id)).order_by("-created_at", "-score_id")
        return [self._score_dict(row) for row in rows]

    async def update_task_status(self, *, task_id: str, status: str) -> None:
        await Task.filter(task_id=int(task_id)).update(status=canonical_task_status(status))

    async def _completion_id_map(self, *, task_id: str, status: str | None = None) -> dict[tuple[int, int, int], int]:
        rows = await self._completion_rows(task_id=task_id, status=status)
        return {(row.sample_index, row.avg_repeat_index, row.pass_index): row.completions_id for row in rows}

    async def _completion_rows(self, *, task_id: str, status: str | None = None) -> list[Completion]:
        query = Completion.filter(task_id=int(task_id))
        if status:
            query = query.filter(status=canonical_completion_status(status))
        return await query.order_by("sample_index", "avg_repeat_index", "pass_index")

    @staticmethod
    def _build_completion_context(payload: Mapping[str, Any]) -> dict[str, Any]:
        stages: list[dict[str, Any]] = []
        for idx in iter_stage_indices(payload):
            stages.append(
                {
                    "prompt": payload.get(f"prompt{idx}"),
                    "completion": payload.get(f"completion{idx}"),
                    "stop_reason": payload.get(f"stop_reason{idx}"),
                }
            )
        context = {"stages": stages, "sampling_config": payload.get("sampling_config", {})}
        for key in ("stats", "agent_result", "agent_info", "agent_trace", "task_id", "domain", "instruction"):
            if key in payload:
                context[key] = payload[key]
        sanitized = sanitize_json(context)
        return sanitized if isinstance(sanitized, dict) else {}

    @staticmethod
    def _bounded_text(value: Any, max_chars: int) -> str:
        text = str(value or "").replace("\x00", "")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20].rstrip() + "\n...[truncated]"

    @staticmethod
    def _task_config_path(config_path: str | None = None) -> str | None:
        raw = config_path if config_path is not None else os.environ.get("RWKV_TASK_CONFIG_PATH")
        value = str(raw or "").strip()
        return value or None

    @staticmethod
    def _task_is_tmp() -> bool:
        return os.environ.get("RWKV_TASK_IS_TMP", "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_reference_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        sanitized = sanitize_json(value)
        try:
            return json_key(sanitized)
        except (TypeError, ValueError):
            return str(value).strip() or None

    @classmethod
    def _extract_reference_answer(cls, payload: Mapping[str, Any]) -> str | None:
        for key in _REF_ANSWER_KEYS:
            if key not in payload:
                continue
            normalized = cls._normalize_reference_value(payload.get(key))
            if normalized:
                return normalized
        raw_record = payload.get("raw_record")
        if isinstance(raw_record, Mapping):
            for key in _RAW_RECORD_REF_KEYS:
                if key not in raw_record:
                    continue
                normalized = cls._normalize_reference_value(raw_record.get(key))
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _score_row_for_space(score: Score) -> dict[str, Any]:
        task = score.task
        benchmark = task.benchmark
        model = task.model
        dataset = join_dataset(benchmark.benchmark_name, benchmark.benchmark_split)
        return {
            "score_id": score.score_id,
            "task_id": task.task_id,
            "cot": score.cot_mode != "NoCoT",
            "cot_mode": score.cot_mode,
            "metrics": score.metrics,
            "created_at": score.created_at,
            "is_param_search": task.is_param_search,
            "model": model.model_name,
            "dataset": dataset,
            "samples": benchmark.num_samples,
            "problems": benchmark.num_samples,
            "task": task.evaluator,
            "task_details": None,
            "sampling_config": task.sampling_config,
            "log_path": task.log_path,
        }

    @staticmethod
    def _history_row(score: Score) -> dict[str, Any]:
        row = ScoreboardStore._score_row_for_space(score)
        row["evaluator"] = score.task.evaluator
        row["num_samples"] = score.task.benchmark.num_samples
        return row

    @staticmethod
    def _task_dict(task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "config_path": task.config_path,
            "evaluator": task.evaluator,
            "is_param_search": task.is_param_search,
            "is_tmp": task.is_tmp,
            "created_at": task.created_at,
            "status": task.status,
            "git_hash": task.git_hash,
            "model_id": task.model_id,
            "benchmark_id": task.benchmark_id,
            "desc": task.description,
            "sampling_config": task.sampling_config,
            "log_path": task.log_path,
        }

    @staticmethod
    def _completion_dict(completion: Completion) -> dict[str, Any]:
        return {
            "completions_id": completion.completions_id,
            "task_id": completion.task_id,
            "context": completion.context,
            "sample_index": completion.sample_index,
            "avg_repeat_index": completion.avg_repeat_index,
            "pass_index": completion.pass_index,
            "created_at": completion.created_at,
            "status": completion.status,
        }

    @staticmethod
    def _eval_dict(eval_record: EvalRecord) -> dict[str, Any]:
        completion_id = getattr(eval_record, "completion_id", None)
        if completion_id is None and getattr(eval_record, "completion", None) is not None:
            completion_id = eval_record.completion.completions_id
        return {
            "eval_id": eval_record.eval_id,
            "completions_id": completion_id,
            "answer": eval_record.answer,
            "ref_answer": eval_record.ref_answer,
            "is_passed": eval_record.is_passed,
            "fail_reason": eval_record.fail_reason,
            "created_at": eval_record.created_at,
        }

    @staticmethod
    def _checker_dict(checker: Checker) -> dict[str, Any]:
        completion_id = getattr(checker, "completion_id", None)
        if completion_id is None and getattr(checker, "completion", None) is not None:
            completion_id = checker.completion.completions_id
        return {
            "checker_id": checker.checker_id,
            "completions_id": completion_id,
            "answer_correct": checker.answer_correct,
            "instruction_following_error": checker.instruction_following_error,
            "world_knowledge_error": checker.world_knowledge_error,
            "math_error": checker.math_error,
            "reasoning_logic_error": checker.reasoning_logic_error,
            "thought_contains_correct_answer": checker.thought_contains_correct_answer,
            "needs_human_review": checker.needs_human_review,
            "reason": checker.reason,
            "created_at": checker.created_at,
        }

    @staticmethod
    def _score_dict(score: Score) -> dict[str, Any]:
        return {
            "score_id": score.score_id,
            "task_id": score.task_id,
            "cot_mode": score.cot_mode,
            "metrics": score.metrics,
            "created_at": score.created_at,
        }

    @staticmethod
    def _model_dict(model: ScoreModel) -> dict[str, Any]:
        return {
            "model_id": model.model_id,
            "data_version": model.data_version,
            "arch_version": model.arch_version,
            "num_params": model.num_params,
            "model_name": model.model_name,
        }

    @staticmethod
    def _benchmark_dict(benchmark: Benchmark) -> dict[str, Any]:
        return {
            "benchmark_id": benchmark.benchmark_id,
            "benchmark_name": benchmark.benchmark_name,
            "benchmark_split": benchmark.benchmark_split,
            "url": benchmark.url,
            "status": benchmark.status,
            "num_samples": benchmark.num_samples,
        }
