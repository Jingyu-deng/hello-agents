"""Message system"""
from typing import Any, Dict, Literal
from datetime import datetime

from pydantic import BaseModel, Field

# Define message role type, restricting its values
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """Message class"""

    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (OpenAI API format)"""
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"

