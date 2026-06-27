"""
PlanAndSolveAgent — Plan, then execute (ported from Day 4, now as a framework Agent).

Core idea: generate a complete plan FIRST, then execute it without deviation.

Two phases:
  1. PLAN:  LLM lists the steps needed (no execution)
  2. SOLVE: LLM follows the plan step-by-step

Upgrades over Day 4's standalone version:
  - Inherits from Agent(ABC) — unified run() interface
  - Uses framework Message for history
  - Raises typed exceptions
"""

from typing import Optional

from hello_agents.core.agent import Agent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.message import Message
from hello_agents.core.config import Config
from hello_agents.core.exceptions import LLMError


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PLAN_PROMPT = """
Please create a detailed step-by-step plan for solving the following problem.

Do NOT execute the plan — just list the steps.
Each step should have clear inputs and outputs that naturally feed into the next.
Format your response as a numbered list.

Problem: {question}
"""

SOLVE_PROMPT = """
Please solve the problem by following the plan EXACTLY, step by step.

Problem: {question}

Plan:
{plan}

Important:
- Follow the plan strictly
- After each step, clearly state the intermediate result
- At the end, provide: FINAL ANSWER: <your answer>
"""


class PlanAndSolveAgent(Agent):
    """An agent that plans first, then executes the plan without deviation.

    Usage:
        llm = HelloAgentsLLM()
        agent = PlanAndSolveAgent(name="Planner", llm=llm)
        result = agent.run("A fruit stand sold 15 apples...")
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        super().__init__(name, llm, system_prompt, config)

    # ------------------------------------------------------------------
    # Core run()
    # ------------------------------------------------------------------

    def run(self, input_text: str, **kwargs) -> str:
        """Plan, then solve.

        Args:
            input_text: The problem to solve.
            **kwargs: Reserved for future use.

        Returns:
            The solution text.

        Raises:
            LLMError: If either phase produces no output.
        """
        # ---- Phase 1: Plan ----
        print(f"\n{'─' * 40}")
        print("[Phase 1] PLAN — Planning")
        print(f"{'─' * 40}")

        plan = self.llm.think([
            {"role": "user", "content": PLAN_PROMPT.format(question=input_text)}
        ])
        if not plan:
            raise LLMError("Planning phase returned empty result.")
        print(f"\n[Plan]\n{plan}")

        # ---- Phase 2: Solve ----
        print(f"\n{'─' * 40}")
        print("[Phase 2] SOLVE — Solving")
        print(f"{'─' * 40}")

        result = self.llm.think([
            {"role": "user", "content": SOLVE_PROMPT.format(
                question=input_text, plan=plan
            )}
        ])
        if not result:
            raise LLMError("Solving phase returned empty result.")

        print(f"\n[Done]")
        self.add_message(Message(role="user", content=input_text))
        self.add_message(Message(role="assistant", content=result))
        return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from hello_agents.core.llm import HelloAgentsLLM

    agent = PlanAndSolveAgent(name="Planner", llm=HelloAgentsLLM())
    try:
        result = agent.run(
            "A fruit stand sold 15 apples on Monday. Tuesday's sales were "
            "double Monday's. Wednesday sold 5 fewer than Tuesday. "
            "How many apples were sold in total over the three days?"
        )
        print(f"\n[Final Result]\n{result}")
    except Exception as e:
        print(f"\n[FAIL] {e}")
