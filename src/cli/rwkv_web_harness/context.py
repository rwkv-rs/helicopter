from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


G1H_TURN_DELIMITER = "✿"
G1H_TURN_DELIMITER_TOKEN_ID = 10060


@dataclass
class PromptContext:
    system_prompt: str
    task: str
    turns: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.turns.append(text.strip())

    def render(self, max_chars: int) -> tuple[str, bool]:
        if max_chars < 1000:
            raise ValueError("max_chars must be at least 1000")
        prefix = f"{self.system_prompt}\n\nTask:\n{self.task}\n\nHistory:\n"
        suffix = (
            "\n\nNext output must contain exactly one of:\n"
            "<tool_call>{JSON object}</tool_call>\n"
            "<final_answer>answer text or JSON object</final_answer>"
        )
        budget = max_chars - len(prefix) - len(suffix)
        if budget <= 0:
            raise ValueError("system prompt and task exceed max_chars")

        selected: list[str] = []
        used = 0
        truncated = False
        for turn in reversed(self.turns):
            block = f"\n\n{turn}"
            if used + len(block) > budget:
                truncated = True
                break
            selected.append(block)
            used += len(block)
        selected.reverse()
        if len(selected) < len(self.turns):
            truncated = True
        return prefix + "".join(selected) + suffix, truncated


@dataclass
class ChatContext:
    """OpenAI-compatible chat history for the native vLLM-RWKV interface."""

    system_prompt: str
    task: str
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt.strip()},
            {"role": "user", "content": self.task.strip()},
        ]

    def add_assistant(self, content: str | None, tool_calls: list[dict[str, Any]] | None = None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool(self, *, call_id: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def render_messages(self, max_chars: int) -> tuple[list[dict[str, Any]], bool]:
        if max_chars < 1000:
            raise ValueError("max_chars must be at least 1000")
        base = self.messages[:2]
        if len(base) != 2:
            raise ValueError("chat context must contain system and task messages")
        if _message_chars(base) > max_chars:
            raise ValueError("system prompt and task exceed max_chars")

        selected: list[dict[str, Any]] = []
        truncated = False
        for message in reversed(self.messages[2:]):
            candidate = base + [message] + list(reversed(selected))
            if _message_chars(candidate) > max_chars:
                truncated = True
                break
            selected.append(message)
        if len(selected) < len(self.messages) - 2:
            truncated = True
        return base + list(reversed(selected)), truncated

    def render_text(self, max_chars: int) -> tuple[str, bool]:
        messages, truncated = self.render_messages(max_chars)
        history: list[str] = []
        for message in messages[2:]:
            role = str(message.get("role", "unknown")).upper()
            content = message.get("content") or ""
            if message.get("tool_calls"):
                content = f"{content}\nTool calls: {json.dumps(message['tool_calls'], ensure_ascii=False)}"
            history.append(f"{role}:\n{content}")
        prompt = (
            f"SYSTEM:\n{messages[0]['content']}\n\n"
            f"TASK:\n{messages[1]['content']}\n\n"
            f"HISTORY:\n{chr(10).join(history)}\n\n"
            "Next output must contain exactly one of:\n"
            '<tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>\n'
            "<final_answer>answer text or JSON object</final_answer>"
        )
        return prompt, truncated

    def render_rwkv_json(self, max_chars: int) -> tuple[str, bool]:
        """Render the JSON-call prompt used by g1h function-call checkpoints."""

        messages, truncated = self.render_messages(max_chars)
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).lower()
            content = str(message.get("content") or "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "tool":
                parts.append(f"User: Function output:\n{content}")
            elif role == "assistant" and message.get("tool_calls"):
                calls = message["tool_calls"]
                if isinstance(calls, list) and calls:
                    call = calls[0]
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    arguments = function.get("arguments", {}) if isinstance(function, dict) else {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    compact_call = {
                        "name": function.get("name", "") if isinstance(function, dict) else "",
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    }
                    parts.append(f"Assistant: <think></think>\n```json\n{json.dumps(compact_call, ensure_ascii=False)}\n```")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant: <think></think>\n```json\n{")
        prompt = "\n\n".join(parts)
        if len(prompt) > max_chars:
            raise ValueError("RWKV JSON prompt exceeds max_chars")
        return prompt, truncated

    def render_g1h(self, max_chars: int) -> tuple[str, bool]:
        """Render the g1h User✿/Bot✿ recurrent conversation format."""

        messages, truncated = self.render_messages(max_chars)
        parts: list[str] = [str(messages[0]["content"])]
        for message in messages[1:]:
            role = str(message.get("role", "user")).lower()
            content = str(message.get("content") or "")
            if role == "assistant" and message.get("tool_calls"):
                calls = message["tool_calls"]
                call = calls[0] if isinstance(calls, list) and calls else {}
                function = call.get("function", {}) if isinstance(call, dict) else {}
                arguments = function.get("arguments", {}) if isinstance(function, dict) else {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                content = json.dumps(
                    {
                        "name": function.get("name", "") if isinstance(function, dict) else "",
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                parts.append(f"Bot{G1H_TURN_DELIMITER}<think></think>{content}{G1H_TURN_DELIMITER}")
            elif role == "assistant":
                parts.append(f"Bot{G1H_TURN_DELIMITER}{content}{G1H_TURN_DELIMITER}")
            else:
                parts.append(f"User{G1H_TURN_DELIMITER}{content}{G1H_TURN_DELIMITER}")
        parts.append(f"Bot{G1H_TURN_DELIMITER}<think></think>")
        prompt = "\n".join(parts)
        if len(prompt) > max_chars:
            raise ValueError("g1h prompt exceeds max_chars")
        return prompt, truncated


def _message_chars(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
