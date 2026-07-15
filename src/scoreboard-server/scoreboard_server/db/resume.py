from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TaskLookup:
    task_id: int
    status: str


@dataclass(slots=True)
class ResumeContext:
    benchmark_id: int | None = None
    model_id: int | None = None
    task_id: int | None = None
    can_resume: bool = False
    matching_tasks: tuple[TaskLookup, ...] = ()
    completed_task_ids: tuple[int, ...] = ()
    resumable_task_ids: tuple[int, ...] = ()
    completed_keys: set[tuple[int, int, int]] = field(default_factory=set)

    @property
    def is_new_task(self) -> bool:
        return self.task_id is None or not self.can_resume
