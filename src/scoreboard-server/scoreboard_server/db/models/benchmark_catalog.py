from __future__ import annotations

from tortoise import fields
from tortoise.models import Model


class BenchmarkCatalog(Model):
    catalog_id = fields.IntField(primary_key=True)
    benchmark_name = fields.CharField(max_length=255)
    benchmark_split = fields.CharField(max_length=255)
    field = fields.CharField(max_length=64)
    source = fields.CharField(max_length=64)
    source_family = fields.CharField(max_length=255)
    target_kind = fields.CharField(max_length=64)
    run_status = fields.CharField(max_length=64)
    scope = fields.CharField(max_length=128)
    metadata = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "benchmark_catalog"
        unique_together = (("benchmark_name", "benchmark_split", "scope"),)
