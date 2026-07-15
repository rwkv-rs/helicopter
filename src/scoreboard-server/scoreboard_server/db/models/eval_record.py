from __future__ import annotations

from tortoise import fields
from tortoise.models import Model

from .completion import Completion


class EvalRecord(Model):
    eval_id = fields.IntField(primary_key=True)
    completion: fields.OneToOneRelation[Completion] = fields.OneToOneField(
        "models.Completion",
        related_name="eval_record",
        on_delete=fields.RESTRICT,
        source_field="completions_id",
    )
    answer = fields.TextField()
    ref_answer = fields.TextField()
    is_passed = fields.BooleanField()
    fail_reason = fields.TextField()
    created_at = fields.DatetimeField()

    class Meta:
        table = "eval"
