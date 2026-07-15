from __future__ import annotations

from tortoise import fields
from tortoise.models import Model


class Benchmark(Model):
    benchmark_id = fields.IntField(primary_key=True)
    benchmark_name = fields.CharField(max_length=255)
    benchmark_split = fields.CharField(max_length=255)
    url = fields.CharField(max_length=255, null=True)
    status = fields.CharField(max_length=32)
    num_samples = fields.IntField()

    class Meta:
        table = "benchmark"
        unique_together = (("benchmark_name", "benchmark_split"),)
