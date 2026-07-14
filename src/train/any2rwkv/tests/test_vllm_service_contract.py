from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_vllm_service.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_vllm_service", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engine_is_forced_to_generic_transformers_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))
    module = _load_module()
    module._build_engine("/checkpoint", 2)

    assert captured == {
        "model": "/checkpoint",
        "tensor_parallel_size": 2,
        "trust_remote_code": True,
        "model_impl": "transformers",
    }
    assert module.LOADER_CONTRACT == "generic-transformers-backend-not-pure-rwkv"
