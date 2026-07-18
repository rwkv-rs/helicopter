"""A small, local-first web research harness for RWKV models."""

from .models import GenerationBackend, GenerationRequest, ModelBackendError, RWKVLocalBackend
from .runner import AgentConfig, AgentRunner, RunResult
from .tools import ToolResult, WebToolkit
from .trace import TraceWriter

__all__ = [
    "AgentConfig",
    "AgentRunner",
    "GenerationBackend",
    "GenerationRequest",
    "ModelBackendError",
    "RWKVLocalBackend",
    "RunResult",
    "ToolResult",
    "TraceWriter",
    "WebToolkit",
]

__version__ = "0.1.0"
