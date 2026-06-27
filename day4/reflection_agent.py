"""
Paradigm 3: Reflection
=======================
Core idea: "Execute → Reflect → Refine" — iterate through a critic loop.

Three stages per iteration:
  1. EXECUTE:  generate code / content
  2. REFLECT:  a critic reviews it and points out issues
  3. REFINE:   regenerate based on the critique

The loop stops when the critic says "no improvement needed" or max iterations
are reached. Unlike ReAct (which is about tool use) and Plan-and-Solve (which
is about planning), Reflection is about SELF-IMPROVEMENT through critique.

Best for: code generation, technical writing, tasks requiring high quality.
"""

import re
from hello_agents_llm import HelloAgentsLLM

EXECUTE_PROMPT = """
You are an expert Python programmer. Write a clean, correct solution.

Task: {task}

Provide the complete code with comments.
"""

REFLECT_PROMPT = """
You are a strict code reviewer. Analyze the following code for:
- Correctness: does it handle edge cases?
- Efficiency: is the algorithm optimal?
- Readability: is it clear and well-structured?

Task: {task}

Code:
{code}

If there are issues, describe them specifically and suggest improvements.
If the code is already optimal, say "NO_IMPROVEMENT_NEEDED".
"""

REFINE_PROMPT = """
Please improve the code based on the reviewer's feedback.

Task: {task}

Previous code:
{code}

Review feedback:
{feedback}

Provide the improved, complete code:
"""


class ReflectionAgent:
    """An agent that iteratively improves its output via self-critique."""

    def __init__(self, llm: HelloAgentsLLM, max_iterations: int = 3):
        self.llm = llm
        self.max_iterations = max_iterations

    def run(self, task: str) -> str | None:
        print(f"\n{'─' * 40}")
        print(f"🔄 REFLECTION — Task: {task}")

        # ---- Initial attempt ----
        print(f"\n{'─' * 40}\n📝 Round 1 — Initial Attempt")
        code = self.llm.think([
            {"role": "user", "content": EXECUTE_PROMPT.format(task=task)}
        ])
        if not code:
            return None

        # ---- Reflect → Refine loop ----
        for i in range(1, self.max_iterations + 1):
            print(f"\n{'─' * 40}\n🔍 Round {i} — Review")
            feedback = self.llm.think([
                {"role": "user", "content": REFLECT_PROMPT.format(task=task, code=code)}
            ])
            if not feedback:
                return code  # return best effort

            # Stop if critic is satisfied
            if "NO_IMPROVEMENT_NEEDED" in feedback:
                print("\n✅ Reviewer satisfied — no further improvements needed.")
                break

            # Extract the "what to improve" from feedback
            print(f"\n{'─' * 40}\n🔧 Round {i} — Refine")
            code = self.llm.think([
                {"role": "user", "content": REFINE_PROMPT.format(
                    task=task, code=code, feedback=feedback
                )}
            ])
            if not code:
                return None

        print(f"\n{'─' * 40}\n🎉 Reflection complete!")
        return code


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from hello_agents_llm import HelloAgentsLLM

    agent = ReflectionAgent(HelloAgentsLLM(), max_iterations=2)
    result = agent.run("Write a Python function to find all prime numbers from 1 to n.")
    if result:
        print(f"\n{'─' * 40}\n📄 Final Output:\n{result}")
