"""
Tool — abstract base class for all tools in the framework.

Every tool describes itself (name, description, parameters) and
provides a run() method. This self-description is what lets agents
automatically generate correct tool-calling prompts.

See Also:
    ToolRegistry  — stores and dispatches tools
    ToolParameter — describes a single input parameter
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """Describes one input parameter of a tool.

    Used to auto-generate tool documentation in prompts.
    """

    name: str
    description: str
    required: bool = True
    param_type: str = "string"  # "string" | "number" | "boolean" | "array"


class Tool(ABC):
    """Abstract base for all tools.

    Subclasses MUST implement:
        run(**kwargs) -> str
        get_parameters() -> list[ToolParameter]
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool with the given parameters.

        Returns:
            A string result (shown to the agent as an observation).
        """
        ...

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Return the parameters this tool accepts.

        Used by ToolRegistry to build prompt descriptions and schemas.
        """
        ...

    def get_parameter_string(self) -> str:
        """Human-readable parameter list, e.g. '(query: string, limit: number)'."""
        params = self.get_parameters()
        if not params:
            return ""
        parts = [f"{p.name}: {p.param_type}" for p in params]
        return f"({', '.join(parts)})"

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"
