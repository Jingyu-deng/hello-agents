"""
Paradigm 2: Plan-and-Solve
===========================
Core idea: "Think first, then act" — plan EVERYTHING upfront, then execute.

Two phases:
  1. PLAN: LLM generates a numbered list of steps
  2. SOLVE: LLM executes each step in order, no deviation

Unlike ReAct (which interleaves thinking and acting), Plan-and-Solve separates
them completely — the plan is fixed before any execution begins.

Best for: math problems, structured reasoning, tasks with clear logical paths.
"""

from hello_agents_llm import HelloAgentsLLM

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


class PlanAndSolveAgent:
    """An agent that plans first, then executes the plan without deviation."""

    def __init__(self, llm: HelloAgentsLLM):
        self.llm = llm

    def run(self, question: str) -> str | None:
        # ---- Phase 1: Plan ----
        print(f"\n{'─' * 40}")
        print("📋 PHASE 1 — Planning")
        print(f"{'─' * 40}")

        plan = self.llm.think([
            {"role": "user", "content": PLAN_PROMPT.format(question=question)}
        ])
        if not plan:
            print("❌ Planning failed.")
            return None
        print(f"\n📋 Plan:\n{plan}")

        # ---- Phase 2: Solve ----
        print(f"\n{'─' * 40}")
        print("⚙️  PHASE 2 — Solving")
        print(f"{'─' * 40}")

        result = self.llm.think([
            {"role": "user", "content": SOLVE_PROMPT.format(question=question, plan=plan)}
        ])
        if not result:
            print("❌ Solving failed.")
            return None

        print(f"\n🎉 Done!")
        return result
