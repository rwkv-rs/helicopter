"""A small, local-first web research harness for RWKV models."""

from .batch import TaskSpec, load_task_specs, summarize_cases, validate_case
from .models import (
    GenerationBackend,
    GenerationRequest,
    GenerationResponse,
    ModelBackendError,
    RWKVLocalBackend,
    ToolCall,
)
from .context import G1H_TURN_DELIMITER, G1H_TURN_DELIMITER_TOKEN_ID
from .runner import AgentConfig, AgentRunner, RunResult
from .tools import ToolResult, WebToolkit
from .trace import TraceWriter

__all__ = [
    "AgentConfig",
    "AgentRunner",
    "TaskSpec",
    "GenerationBackend",
    "GenerationRequest",
    "GenerationResponse",
    "G1H_TURN_DELIMITER",
    "G1H_TURN_DELIMITER_TOKEN_ID",
    "ModelBackendError",
    "RWKVLocalBackend",
    "RunResult",
    "ToolCall",
    "ToolResult",
    "TraceWriter",
    "WebToolkit",
    "load_task_specs",
    "summarize_cases",
    "validate_case",
]

__version__ = "0.1.0"
