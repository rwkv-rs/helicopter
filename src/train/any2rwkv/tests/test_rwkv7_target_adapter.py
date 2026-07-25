from __future__ import annotations

import pytest

from any2rwkv.adapters.rwkv7 import NativeRWKV7TargetAdapter
from any2rwkv.errors import ContractError


def test_native_rwkv7_training_environment_is_exact_and_fail_closed(monkeypatch) -> None:
    adapter = NativeRWKV7TargetAdapter()
    expected = {**adapter.BASE_TRAINING_ENVIRONMENT, "RWKV_HEAD_SIZE": "128"}
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    assert adapter.validate_training_environment(head_size=128) == expected
    monkeypatch.setenv("WKV_MODE", "fp16")
    with pytest.raises(ContractError, match="WKV_MODE"):
        adapter.validate_training_environment(head_size=128)


def test_empty_rwkv_kernel_must_be_explicit(monkeypatch) -> None:
    adapter = NativeRWKV7TargetAdapter()
    expected = {**adapter.BASE_TRAINING_ENVIRONMENT, "RWKV_HEAD_SIZE": "128"}
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("RWKV_KERNEL")
    with pytest.raises(ContractError, match="RWKV_KERNEL"):
        adapter.validate_training_environment(head_size=128)
