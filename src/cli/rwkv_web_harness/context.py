from __future__ import annotations

from dataclasses import dataclass, field


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
