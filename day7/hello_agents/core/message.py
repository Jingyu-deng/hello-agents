"""
Message — unified conversation message type.

Replaces the raw dicts and flat strings used in Day 4 with a single
structured type that all agents share, with validation enforced by Pydantic.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Restrict role to valid OpenAI API values
MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single message in an agent's conversation history.

    Attributes:
        role: Must be "system", "user", "assistant", or "tool".
        content: The message text.
        metadata: Optional extra info (e.g. token count, timestamp).
    """

    content: str
    role: MessageRole
    metadata: dict = Field(default_factory=dict)

    def to_openai(self) -> dict:
        """Convert to the format OpenAI / DeepSeek APIs expect."""
        return {"role": self.role, "content": self.content}
