"""
Agent — abstract base class for all agents in the framework.

Every agent MUST inherit from Agent and implement run().
This is the "contract" that makes the framework work:
  - Framework code depends on Agent, not on concrete implementations.
  - Add a new agent type? Just subclass Agent and implement run().
  - The unified interface means demo.py can call .run() on any agent.

Design pattern: Template Method — the base class provides history
management, subclasses provide the specific reasoning loop.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .message import Message
from .llm import HelloAgentsLLM
from .config import Config


class Agent(ABC):
    """Abstract base for all agents.

    Subclasses MUST implement:
        run(input_text: str, **kwargs) -> str

    Inherited for free:
        - name, llm, system_prompt, config
        - add_message(), get_history(), clear_history()
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config.from_env()
        self._history: list[Message] = []

        # If a system prompt is provided, seed history with it
        if system_prompt:
            self._history.append(Message(role="system", content=system_prompt))

    # ------------------------------------------------------------------
    # Abstract contract — every agent must implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """Run the agent on the given input.

        Args:
            input_text: The user's question / task.
            **kwargs: Agent-specific options (e.g. max_steps override).

        Returns:
            The agent's final response.
        """
        ...

    # ------------------------------------------------------------------
    # History management — shared by all agents
    # ------------------------------------------------------------------

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation history."""
        self._history.append(message)

    def get_history(self) -> list[Message]:
        """Return a copy of the conversation history."""
        return self._history.copy()

    def get_history_as_dicts(self) -> list[dict]:
        """Return history as OpenAI-compatible dicts."""
        return [m.to_dict() for m in self._history]

    def clear_history(self) -> None:
        """Clear the conversation history (keeps system prompt if present)."""
        system_msgs = [m for m in self._history if m.role == "system"]
        self._history = system_msgs

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, llm={self.llm.model!r})"
