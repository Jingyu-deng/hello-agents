#!/usr/bin/env python3
"""
demo.py — Run all four agent types with the HelloAgents framework.

This script demonstrates:
  1. SimpleAgent  — basic conversation (no tools)
  2. ReActAgent   — tool-calling loop with search + calculator
  3. PlanAndSolveAgent — math word problem
  4. ReflectionAgent   — code generation with self-critique

Each agent uses the same HelloAgentsLLM instance and the same
.run() interface — that's the power of the Agent base class.
"""

import sys
import os

# Allow running from day7/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hello_agents import (
    HelloAgentsLLM,
    SimpleAgent,
    ReActAgent,
    PlanAndSolveAgent,
    ReflectionAgent,
    ToolRegistry,
)
from hello_agents.tools.builtin import SimulatedSearchTool, CalculatorTool


SEPARATOR = "=" * 60


def demo_simple_agent(llm: HelloAgentsLLM):
    """Basic conversational agent — no tools, just chat."""
    print(f"\n{SEPARATOR}")
    print("DEMO 1: SimpleAgent — Basic Conversation")
    print(SEPARATOR)

    agent = SimpleAgent(
        name="Assistant",
        llm=llm,
        system_prompt="You are a helpful AI assistant. Keep answers concise.",
    )
    response = agent.run("What is the capital of France, and what is it known for?")
    print(f"\n[Response] {response}")


def demo_react_agent(llm: HelloAgentsLLM):
    """ReAct agent — Thought → Action → Observation loop with tools."""
    print(f"\n{SEPARATOR}")
    print("DEMO 2: ReActAgent — Tool Calling")
    print(SEPARATOR)

    tools = ToolRegistry()
    tools.register_tool(SimulatedSearchTool())
    tools.register_tool(CalculatorTool())

    agent = ReActAgent(name="ReAct", llm=llm, tool_registry=tools, max_steps=5)
    try:
        answer = agent.run(
            "What are Huawei's latest phones? List their key selling points."
        )
        print(f"\n[Final Answer]\n{answer}")
    except Exception as e:
        print(f"\n[FAIL] ReActAgent: {e}")


def demo_plan_solve_agent(llm: HelloAgentsLLM):
    """Plan-and-Solve — plan all steps, then execute."""
    print(f"\n{SEPARATOR}")
    print("DEMO 3: PlanAndSolveAgent — Math Word Problem")
    print(SEPARATOR)

    agent = PlanAndSolveAgent(name="Planner", llm=llm)
    try:
        result = agent.run(
            "A fruit stand sold 15 apples on Monday. Tuesday's sales were "
            "double Monday's. Wednesday sold 5 fewer than Tuesday. "
            "How many apples were sold in total over the three days?"
        )
        print(f"\n[Result]\n{result}")
    except Exception as e:
        print(f"\n[FAIL] PlanAndSolveAgent: {e}")


def demo_reflection_agent(llm: HelloAgentsLLM):
    """Reflection — generate code, critique, refine."""
    print(f"\n{SEPARATOR}")
    print("DEMO 4: ReflectionAgent — Code Generation")
    print(SEPARATOR)

    agent = ReflectionAgent(name="Reflector", llm=llm, max_iterations=2)
    try:
        result = agent.run(
            "Write a Python function to find all prime numbers from 1 to n."
        )
        print(f"\n[Final Code]\n{result}")
    except Exception as e:
        print(f"\n[FAIL] ReflectionAgent: {e}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("HelloAgents Framework — Demo Suite")
    print(f"All four agents share the same LLM and the same .run() interface.\n")

    # One LLM instance shared by all agents
    llm = HelloAgentsLLM()

    demo_simple_agent(llm)
    demo_react_agent(llm)
    demo_plan_solve_agent(llm)
    demo_reflection_agent(llm)

    print(f"\n{SEPARATOR}")
    print("All demos complete!")
