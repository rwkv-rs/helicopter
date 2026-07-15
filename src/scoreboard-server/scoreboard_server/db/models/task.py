from __future__ import annotations

from tortoise import fields
from tortoise.models import Model

from .benchmark import Benchmark
from .score_model import ScoreModel


class Task(Model):
    task_id = fields.IntField(primary_key=True)
    config_path = fields.CharField(max_length=255, null=True)
    evaluator = fields.CharField(max_length=255)
    is_param_search = fields.BooleanField(default=False)
    is_tmp = fields.BooleanField(default=False)
    created_at = fields.DatetimeField()
    status = fields.CharField(max_length=255)
    git_hash = fields.CharField(max_length=255)
    model: fields.ForeignKeyRelation[ScoreModel] = fields.ForeignKeyField(
        "models.ScoreModel",
        related_name="tasks",
        on_delete=fields.RESTRICT,
    )
    benchmark: fields.ForeignKeyRelation[Benchmark] = fields.ForeignKeyField(
        "models.Benchmark",
        related_name="tasks",
        on_delete=fields.RESTRICT,
    )
    description = fields.TextField(null=True, source_field="desc")
    sampling_config = fields.JSONField(null=True)
    log_path = fields.CharField(max_length=255)

    class Meta:
        table = "task"
