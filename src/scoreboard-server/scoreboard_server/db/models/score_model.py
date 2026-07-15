from __future__ import annotations

from tortoise import fields
from tortoise.models import Model


class ScoreModel(Model):
    model_id = fields.IntField(primary_key=True)
    data_version = fields.CharField(max_length=255)
    arch_version = fields.CharField(max_length=255)
    num_params = fields.CharField(max_length=255)
    model_name = fields.CharField(max_length=255)

    class Meta:
        table = "model"
        unique_together = (("arch_version", "data_version", "num_params", "model_name"),)
