"""
Message — unified conversation message type.

Replaces the raw dicts and flat strings used in Day 4 with a single
structured type that all agents share.
"""

from dataclasses import dataclass, field


@dataclass
class Message:
    """A single message in an agent's conversation history.

    Attributes:
        role: "system" | "user" | "assistant" | "tool"
        content: The message text.
        metadata: Optional extra info (e.g. token count, timestamp).
    """

    role: str
    content: str
    metadata: dict = field(default_factory=dict)

    def to_openai(self) -> dict:
        """Convert to the format OpenAI / DeepSeek APIs expect."""
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"Message(role={self.role!r}, content={preview!r})"
