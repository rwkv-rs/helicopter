from __future__ import annotations

from tortoise import fields
from tortoise.models import Model


class SchedulerLease(Model):
    job_id = fields.CharField(max_length=512, primary_key=True)
    owner_id = fields.TextField()
    node_id = fields.TextField()
    claimed_at = fields.DatetimeField()
    heartbeat_at = fields.DatetimeField()
    lease_until = fields.DatetimeField()
    lease_meta = fields.JSONField(null=True)

    class Meta:
        table = "scheduler_lease"
