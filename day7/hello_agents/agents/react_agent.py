"""
ReActAgent — Reasoning + Acting loop (ported from Day 4, now as a framework Agent).

Core idea: interleave thinking and tool-calling in a loop.
  Thought → Action → Observation → (repeat) → Finish

Upgrades over Day 4's standalone version:
  - Inherits from Agent(ABC) — unified run() interface
  - Uses framework Message for history management
  - Uses ToolRegistry (not raw ToolExecutor dict)
  - Raises typed exceptions instead of print() + return None
"""

import re
from typing import Optional

from hello_agents.core.agent import Agent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.message import Message
from hello_agents.core.config import Config
from hello_agents.core.exceptions import MaxStepsError, ParseError, ToolNotFound
from hello_agents.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Prompt template with {tools}, {question}, {history} placeholders
# ---------------------------------------------------------------------------
REACT_PROMPT = """
You are an intelligent assistant capable of calling external tools.

Available tools:
{tools}

You MUST respond in this format:
Thought: [your reasoning — analyze, plan the next step]
Action: [one of the following]
  - tool_name[tool_input]  — call a tool
  - Finish[final answer]   — end the task

When you have enough information, use Finish[final answer].

Question: {question}
History: {history}
"""


class ReActAgent(Agent):
    """An agent that interleaves thinking and acting in a loop.

    Usage:
        llm = HelloAgentsLLM()
        tools = ToolRegistry()
        tools.register_function("Search", "Search the web", search_fn)
        agent = ReActAgent(name="ReAct", llm=llm, tool_registry=tools, max_steps=5)
        answer = agent.run("What are the latest phones?")
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self._react_history: list[str] = []

    # ------------------------------------------------------------------
    # Core run()
    # ------------------------------------------------------------------

    def run(self, input_text: str, **kwargs) -> str:
        """Run the ReAct loop.

        Args:
            input_text: The user's question.
            **kwargs: Override max_steps via 'max_steps' kwarg.

        Returns:
            The final answer string.

        Raises:
            MaxStepsError: If the agent exceeds max_steps.
            ParseError: If the LLM output cannot be parsed.
        """
        max_steps = kwargs.get("max_steps", self.max_steps)
        self._react_history = []

        for step in range(1, max_steps + 1):
            print(f"\n{'─' * 40}\n--- ReAct Step {step} ---")

            # Build prompt
            prompt = REACT_PROMPT.format(
                tools=self.tool_registry.get_available_tools(),
                question=input_text,
                history="\n".join(self._react_history),
            )

            response = self.llm.think([{"role": "user", "content": prompt}])
            if not response:
                raise ParseError("LLM returned empty response at step {step}.")

            # Parse Thought & Action
            thought = self._extract(response, r"Thought:\s*(.*?)(?=\nAction:|$)")
            action = self._extract(response, r"Action:\s*(.*?)$")

            if thought:
                print(f"[Thought] {thought}")

            if not action:
                print("[WARN] No Action found. Stopping.")
                raise ParseError(f"No Action line found in LLM output at step {step}.")

            # Check for Finish
            if action.startswith("Finish"):
                match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    print(f"[Answer] {answer}")
                    self.add_message(Message(role="user", content=input_text))
                    self.add_message(Message(role="assistant", content=answer))
                    return answer

            # Execute tool
            tool_name, tool_input = self._parse_tool_call(action)
            if not tool_name:
                obs = f"Invalid action format: '{action}'. Use ToolName[input] or Finish[answer]."
            else:
                print(f"[Call] {tool_name}[{tool_input}]")
                obs = self.tool_registry.execute(tool_name, input_str=tool_input)

            print(f"[Obs] {obs}")
            self._react_history.append(f"Action: {action}")
            self._react_history.append(f"Observation: {obs}")

        raise MaxStepsError(f"ReActAgent exceeded max_steps ({max_steps}).")

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(text: str, pattern: str) -> str | None:
        """Return the first capture group of pattern in text, or None."""
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_tool_call(action: str) -> tuple[str | None, str | None]:
        """Parse 'ToolName[input]' into (name, input)."""
        m = re.match(r"(\w+)\[(.*)\]", action, re.DOTALL)
        return (m.group(1), m.group(2)) if m else (None, None)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from hello_agents.core.llm import HelloAgentsLLM
    from hello_agents.tools.registry import ToolRegistry
    from hello_agents.tools.builtin.search import SimulatedSearchTool
    from hello_agents.tools.builtin.calculator import CalculatorTool

    llm = HelloAgentsLLM()
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
        print(f"\n[FAIL] {e}")
