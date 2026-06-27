"""
HelloAgents — A lightweight, educational agent framework built from scratch.

Quick start:
    from hello_agents import SimpleAgent, HelloAgentsLLM

    llm = HelloAgentsLLM()
    agent = SimpleAgent(name="Bot", llm=llm, system_prompt="You are helpful.")
    print(agent.run("Hello!"))

More:
    from hello_agents import ReActAgent, PlanAndSolveAgent, ReflectionAgent
    from hello_agents import Tool, ToolParameter, ToolRegistry
    from hello_agents.tools.builtin import CalculatorTool, SimulatedSearchTool
"""

# Core
from hello_agents.core.agent import Agent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.message import Message
from hello_agents.core.config import Config

# Exceptions
from hello_agents.core.exceptions import (
    AgentError,
    LLMError,
    ToolError,
    ToolNotFound,
    MaxStepsError,
    ParseError,
)

# Agents
from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.agents.react_agent import ReActAgent
from hello_agents.agents.plan_solve_agent import PlanAndSolveAgent
from hello_agents.agents.reflection_agent import ReflectionAgent

# Tools
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.registry import ToolRegistry

__all__ = [
    # Core
    "Agent",
    "HelloAgentsLLM",
    "Message",
    "Config",
    # Exceptions
    "AgentError",
    "LLMError",
    "ToolError",
    "ToolNotFound",
    "MaxStepsError",
    "ParseError",
    # Agents
    "SimpleAgent",
    "ReActAgent",
    "PlanAndSolveAgent",
    "ReflectionAgent",
    # Tools
    "Tool",
    "ToolParameter",
    "ToolRegistry",
]
