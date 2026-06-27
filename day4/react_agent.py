"""
Paradigm 1: ReAct (Reasoning + Acting)
=======================================
Core idea: "Think and act simultaneously" — a Thought → Action → Observation loop.

Your day1 agent already did this. This version adds:
- Formal ReAct prompt template with {tools}, {question}, {history} placeholders
- History tracking (accumulates Action + Observation pairs)
- ToolExecutor integration instead of raw dict

Best for: tasks needing external knowledge, tool/API interactions, exploration.
"""

import re
from hello_agents_llm import HelloAgentsLLM
from tool_executor import ToolExecutor

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


class ReActAgent:
    """An agent that interleaves thinking and acting in a loop."""

    def __init__(self, llm: HelloAgentsLLM, tools: ToolExecutor, max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: list[str] = []

    def run(self, question: str) -> str | None:
        """Run the ReAct loop until Finish or max steps."""
        self.history = []

        for step in range(1, self.max_steps + 1):
            print(f"\n{'─' * 40}\n--- ReAct Step {step} ---")

            # Build prompt with current context
            prompt = REACT_PROMPT.format(
                tools=self.tools.get_available_tools(),
                question=question,
                history="\n".join(self.history),
            )

            response = self.llm.think([{"role": "user", "content": prompt}])
            if not response:
                print("❌ LLM returned empty response.")
                return None

            # ---- Parse Thought and Action ----
            thought = self._extract(response, r"Thought:\s*(.*?)(?=\nAction:|$)")
            action = self._extract(response, r"Action:\s*(.*?)$")

            if thought:
                print(f"💭 Thought: {thought}")

            if not action:
                print("⚠️  No Action found. Stopping.")
                return None

            # ---- Finish? ----
            if action.startswith("Finish"):
                match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    print(f"🎉 Final Answer: {answer}")
                    return answer

            # ---- Execute tool ----
            tool_name, tool_input = self._parse_tool_call(action)
            if not tool_name:
                obs = f"Invalid action format: {action}"
            else:
                fn = self.tools.get_tool(tool_name)
                if fn:
                    print(f"🎬 Calling: {tool_name}[{tool_input}]")
                    obs = fn(tool_input)
                else:
                    obs = f"Tool '{tool_name}' not found. Available: {list(self.tools.tools.keys())}"

            print(f"👀 Observation: {obs}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {obs}")

        print("⚠️  Max steps reached.")
        return None

    @staticmethod
    def _extract(text: str, pattern: str) -> str | None:
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_tool_call(action: str) -> tuple[str | None, str | None]:
        m = re.match(r"(\w+)\[(.*)\]", action, re.DOTALL)
        return (m.group(1), m.group(2)) if m else (None, None)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from hello_agents_llm import HelloAgentsLLM
    from tool_executor import ToolExecutor, search, calculator

    llm = HelloAgentsLLM()
    tools = ToolExecutor()
    tools.register_tool("Search", "Search the web via Google", search)
    tools.register_tool("Calculator", "Evaluate a math expression", calculator)

    agent = ReActAgent(llm, tools, max_steps=5)
    agent.run("What are Huawei's latest phones? List their key selling points.")
