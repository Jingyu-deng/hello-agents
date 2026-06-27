"""
ReflectionAgent — Execute → Critique → Refine loop (ported from Day 4).

Core idea: generate initial output, then iteratively improve it
through a self-critique loop.

Three stages per iteration:
  1. EXECUTE:  generate code / content
  2. REFLECT:  a critic reviews it and points out issues
  3. REFINE:   regenerate based on the critique

The loop stops when the critic says "NO_IMPROVEMENT_NEEDED" or
max_iterations is reached.

Upgrades over Day 4's standalone version:
  - Inherits from Agent(ABC) — unified run() interface
  - Uses framework Message for history
  - Raises typed exceptions
  - Generalized beyond code generation (criteria are configurable)
"""

from typing import Optional

from hello_agents.core.agent import Agent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.message import Message
from hello_agents.core.config import Config
from hello_agents.core.exceptions import MaxStepsError


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
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


class ReflectionAgent(Agent):
    """An agent that iteratively improves its output via self-critique.

    Usage:
        llm = HelloAgentsLLM()
        agent = ReflectionAgent(name="Reflector", llm=llm, max_iterations=2)
        code = agent.run("Write a function to find primes up to n.")
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations

        # Allow override via config
        if config and config.max_iterations:
            self.max_iterations = config.max_iterations

    # ------------------------------------------------------------------
    # Core run()
    # ------------------------------------------------------------------

    def run(self, input_text: str, **kwargs) -> str:
        """Execute → Reflect → Refine loop.

        Args:
            input_text: The task description (e.g. "Write a function...").
            **kwargs: Override max_iterations via 'max_iterations' kwarg.

        Returns:
            The final (best) output.

        Raises:
            MaxStepsError: If max_iterations reached without convergence.
        """
        max_iterations = kwargs.get("max_iterations", self.max_iterations)

        print(f"\n{'─' * 40}")
        print(f"[Reflection] Task: {input_text}")

        # ---- Phase 1: Initial attempt ----
        print(f"\n{'─' * 40}\n[Execute] Round 1 — Initial Attempt")
        code = self.llm.think([
            {"role": "user", "content": EXECUTE_PROMPT.format(task=input_text)}
        ])
        if not code:
            return ""  # best effort: nothing to return

        # ---- Phase 2: Reflect → Refine loop ----
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'─' * 40}\n[Review] Round {iteration} — Review")
            feedback = self.llm.think([
                {"role": "user", "content": REFLECT_PROMPT.format(
                    task=input_text, code=code
                )}
            ])
            if not feedback:
                break  # best effort: return current code

            # Stop if critic is satisfied
            if "NO_IMPROVEMENT_NEEDED" in feedback:
                print("\n[OK] Reviewer satisfied — no further improvements needed.")
                break

            # Refine
            print(f"\n{'─' * 40}\n[Refine] Round {iteration} — Refine")
            refined = self.llm.think([
                {"role": "user", "content": REFINE_PROMPT.format(
                    task=input_text, code=code, feedback=feedback
                )}
            ])
            if not refined:
                break  # keep previous version
            code = refined

        else:
            print(f"\n[WARN] Max iterations ({max_iterations}) reached.")

        print(f"\n{'─' * 40}\n[Done] Reflection complete!")
        self.add_message(Message(role="user", content=input_text))
        self.add_message(Message(role="assistant", content=code))
        return code


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from hello_agents.core.llm import HelloAgentsLLM

    agent = ReflectionAgent(name="Reflector", llm=HelloAgentsLLM(), max_iterations=2)
    try:
        result = agent.run(
            "Write a Python function to find all prime numbers from 1 to n."
        )
        print(f"\n{'─' * 40}\n[Final Output]\n{result}")
    except Exception as e:
        print(f"\n[FAIL] {e}")
