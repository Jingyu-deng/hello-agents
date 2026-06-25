"""
Tool registry — manages and executes the agent's external tools.

Improvement over day1's AVAILABLE_TOOLS dict:
- Each tool has a description (used in prompts)
- registerTool() adds tools safely
- getAvailableTools() formats tool list for prompts
"""

from typing import Callable


class ToolExecutor:
    """Stores and executes named tools with descriptions."""

    def __init__(self):
        self.tools: dict[str, dict] = {}

    def register_tool(self, name: str, description: str, func: Callable):
        """Add a tool to the registry."""
        if name in self.tools:
            print(f"⚠️  Tool '{name}' already exists, overwriting.")
        self.tools[name] = {"description": description, "func": func}
        print(f"🔧 Tool '{name}' registered.")

    def get_tool(self, name: str) -> Callable | None:
        """Get a tool function by name, or None if not found."""
        entry = self.tools.get(name)
        return entry["func"] if entry else None

    def get_available_tools(self) -> str:
        """Format all tools into a prompt-friendly string."""
        return "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        )


# ---------------------------------------------------------------------------
# Simulated tools (no real API needed — keeps the demo self-contained)
# ---------------------------------------------------------------------------

def search(query: str) -> str:
    """
    Simulated search engine. Returns pre-canned results for known queries.
    In production, replace with Tavily or another real search API.
    """
    print(f"🔍 Searching: {query}")
    knowledge_base = {
        "华为最新手机": (
            "[1] HUAWEI Mate 70 Pro — Kirin 9100 chip, satellite communication 2.0, "
            "XMAGE imaging upgrade, Kunlun glass.\n"
            "[2] HUAWEI Pura 80 Pro+ — First retractable camera, variable aperture, "
            "ultra-light-gathering night vision telephoto, HarmonyOS 4.2."
        ),
        "英伟达最新GPU": (
            "[1] GeForce RTX 50 Series — Powered by NVIDIA Blackwell architecture, "
            "bringing game-changing performance to gamers and creators."
        ),
    }
    for key, result in knowledge_base.items():
        if key in query:
            return result
    return f"No results found for '{query}'."


def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression."""
    print(f"🧮 Calculating: {expression}")
    try:
        # Safe eval: only allow digits, operators, parens, decimals
        sanitized = "".join(c for c in expression if c in "0123456789+-*/().% ")
        result = eval(sanitized)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"
