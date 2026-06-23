"""Conversation memory: a bounded recent-turns window plus optional summary.

Stays framework-light (plain message dicts) so it serialises cleanly into the
conversation registry and injects easily into the LCEL prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

DEFAULT_WINDOW = 5


@dataclass
class ConversationMemory:
    """Last-``k``-turns window with an optional rolling summary of older turns."""

    window: int = DEFAULT_WINDOW
    turns: list[tuple[str, str]] = field(default_factory=list)  # (user, assistant)
    summary: str = ""

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))

    def recent(self) -> list[tuple[str, str]]:
        return self.turns[-self.window :]

    def as_messages(self) -> list[BaseMessage]:
        """Render recent turns as alternating Human/AI messages for the prompt."""
        messages: list[BaseMessage] = []
        for user, assistant in self.recent():
            messages.append(HumanMessage(content=user))
            messages.append(AIMessage(content=assistant))
        return messages

    def as_text(self) -> str:
        """Plain-text history (for query rewriting / summarisation prompts)."""
        lines = []
        if self.summary:
            lines.append(f"Summary of earlier conversation: {self.summary}")
        for user, assistant in self.recent():
            lines.append(f"User: {user}")
            lines.append(f"Assistant: {assistant}")
        return "\n".join(lines)

    def summarise_old(self, llm: BaseLanguageModel) -> None:
        """Fold turns older than the window into ``summary`` to bound context."""
        old = self.turns[: -self.window] if len(self.turns) > self.window else []
        if not old:
            return
        body = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in old)
        prompt = f"Summarise this conversation concisely:\n{body}\n\nSummary:"
        result = llm.invoke(prompt)
        self.summary = getattr(result, "content", str(result))

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
