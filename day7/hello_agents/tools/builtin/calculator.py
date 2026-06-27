"""
CalculatorTool — safe arithmetic expression evaluator.

Wraps the Day 4 calculator() function in a proper Tool subclass
with self-describing parameters.
"""

from hello_agents.tools.base import Tool, ToolParameter


class CalculatorTool(Tool):
    """Evaluate a mathematical expression and return the result.

    Sanitises input to only allow digits, operators, parentheses,
    spaces, percent, and decimal points before calling eval().
    """

    def __init__(self):
        super().__init__(
            name="Calculator",
            description="Evaluate a mathematical expression, e.g. '(123 + 456) * 789 / 12'",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                description="The math expression to evaluate",
                required=True,
                param_type="string",
            ),
        ]

    def run(self, input: str = "", **kwargs) -> str:
        """Evaluate the expression safely.

        Args:
            input: The math expression as a string.
            **kwargs: Also accepts 'expression' as keyword.

        Returns:
            Result string, e.g. "Result: 42.0".
        """
        expression = kwargs.get("expression", input)
        if not expression:
            return "[ERROR] No expression provided."

        # Sanitise: only allow safe characters
        allowed = set("0123456789+-*/().% ^")
        sanitized = "".join(c for c in expression if c in allowed)

        if not sanitized:
            return "[ERROR] Expression is empty after sanitisation."

        try:
            result = eval(sanitized)
            return f"Result: {result}"
        except Exception as e:
            return f"[ERROR] Calculation failed: {e}"
