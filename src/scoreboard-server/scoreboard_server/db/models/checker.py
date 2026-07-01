from __future__ import annotations

from tortoise import fields
from tortoise.models import Model

from .completion import Completion


class Checker(Model):
    checker_id = fields.IntField(primary_key=True)
    completion: fields.OneToOneRelation[Completion] = fields.OneToOneField(
        "models.Completion",
        related_name="checker",
        on_delete=fields.RESTRICT,
        source_field="completions_id",
    )
    answer_correct = fields.BooleanField(default=False)
    instruction_following_error = fields.BooleanField(default=False)
    world_knowledge_error = fields.BooleanField(default=False)
    math_error = fields.BooleanField(default=False)
    reasoning_logic_error = fields.BooleanField(default=False)
    thought_contains_correct_answer = fields.BooleanField(default=False)
    needs_human_review = fields.BooleanField(default=False)
    reason = fields.TextField(default="")
    created_at = fields.DatetimeField()

    class Meta:
        table = "checker"
