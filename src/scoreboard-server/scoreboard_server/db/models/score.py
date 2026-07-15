from __future__ import annotations

from tortoise import fields
from tortoise.models import Model

from .task import Task


class Score(Model):
    score_id = fields.IntField(primary_key=True)
    task: fields.OneToOneRelation[Task] = fields.OneToOneField(
        "models.Task",
        related_name="score",
        on_delete=fields.RESTRICT,
    )
    cot_mode = fields.CharField(max_length=32)
    metrics = fields.JSONField()
    created_at = fields.DatetimeField()

    class Meta:
        table = "scores"
