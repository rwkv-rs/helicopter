from __future__ import annotations

from tortoise import fields
from tortoise.models import Model

from .task import Task


class Completion(Model):
    completions_id = fields.IntField(primary_key=True)
    task: fields.ForeignKeyRelation[Task] = fields.ForeignKeyField(
        "models.Task",
        related_name="completions",
        on_delete=fields.RESTRICT,
    )
    context = fields.JSONField()
    sample_index = fields.IntField()
    avg_repeat_index = fields.IntField()
    pass_index = fields.IntField(default=0)
    created_at = fields.DatetimeField()
    status = fields.CharField(max_length=255)

    class Meta:
        table = "completions"
        unique_together = (("task", "sample_index", "avg_repeat_index", "pass_index"),)
