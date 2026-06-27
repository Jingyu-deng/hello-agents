"""
Structured exception hierarchy for the HelloAgents framework.

Using typed exceptions instead of print() + return None makes error
handling predictable and testable.
"""


class AgentError(Exception):
    """Base exception for all agent-related errors."""
    pass


class LLMError(AgentError):
    """Raised when the LLM call fails (network, auth, API error)."""
    pass


class ToolError(AgentError):
    """Raised when a tool execution fails."""
    pass


class ToolNotFound(ToolError):
    """Raised when the agent requests a tool that isn't registered."""
    pass


class MaxStepsError(AgentError):
    """Raised when an agent exceeds its step / iteration limit."""
    pass


class ParseError(AgentError):
    """Raised when the LLM output cannot be parsed (malformed format)."""
    pass
