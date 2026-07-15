from __future__ import annotations

from tortoise import fields
from tortoise.models import Model

from .task import Task


class EvaluationPublication(Model):
    run_id = fields.CharField(max_length=128, primary_key=True)
    task: fields.OneToOneRelation[Task] = fields.OneToOneField(
        "models.Task",
        related_name="evaluation_publication",
        on_delete=fields.RESTRICT,
    )
    publisher_subject = fields.CharField(max_length=255)
    idempotency_key = fields.CharField(max_length=255)
    request_digest = fields.CharField(max_length=64)
    identity_digest = fields.CharField(max_length=64)
    manifest_digest = fields.CharField(max_length=64)
    terminal_status = fields.CharField(max_length=32)
    identity_payload = fields.JSONField()
    accounting_payload = fields.JSONField()
    rejections_payload = fields.JSONField()
    performance_payload = fields.JSONField()
    created_at = fields.DatetimeField()

    class Meta:
        table = "evaluation_publication"
        unique_together = (("publisher_subject", "idempotency_key"),)
