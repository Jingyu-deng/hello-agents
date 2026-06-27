"""
Tool registry — manages and executes the agent's external tools.

Improvement over day1's AVAILABLE_TOOLS dict:
- Each tool has a description (used in prompts)
- registerTool() adds tools safely
- getAvailableTools() formats tool list for prompts
"""

import os
import requests
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
    Simulated search engine. Uses keyword matching — returns results if ALL
    keywords for an entry appear in the query. Much more flexible than exact
    substring match, handles phrasing variations from the LLM naturally.

    In production, replace with Tavily or another real search API.
    """
    print(f"🔍 Searching: {query}")
    query_lower = query.lower()

    # Each entry: (keywords_list, result_text)
    # Match if ALL keywords in the list appear anywhere in the query.
    knowledge_base: list[tuple[list[str], str]] = [
        (
            ["huawei", "phone"],
            "[1] HUAWEI Mate 70 Pro — Kirin 9100 chip, satellite communication 2.0, "
            "XMAGE imaging upgrade, Kunlun glass.\n"
            "[2] HUAWEI Pura 80 Pro+ — First retractable camera, variable aperture, "
            "ultra-light-gathering night vision telephoto, HarmonyOS 4.2."
        ),
        (
            ["nvidia", "gpu"],
            "[1] GeForce RTX 50 Series — Powered by NVIDIA Blackwell architecture, "
            "bringing game-changing performance to gamers and creators."
        ),
    ]

    for keywords, result in knowledge_base:
        if all(kw.lower() in query_lower for kw in keywords):
            return result

    return f"No results found for '{query}'."


# ---------------------------------------------------------------------------
# Real API tools (require API keys in .env)
# ---------------------------------------------------------------------------

def serpapi_search(query: str) -> str:
    """
    Real Google search via SerpAPI. Requires SERPAPI_API_KEY in .env.

    Free tier: 100 searches/month.
    Get your key at: https://serpapi.com/manage-api-key
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "ERROR: SERPAPI_API_KEY not set in .env file."

    print(f"🔍 Searching (SerpAPI): {query}")
    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": api_key,
                "engine": "google",
                "hl": "zh-cn",
                "gl": "cn",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # ---- Smart parsing: prefer direct answers over link lists ----
        # 1. Answer box — Google's direct answer card
        if "answer_box_list" in data:
            return "\n".join(data["answer_box_list"])

        if "answer_box" in data and "answer" in data["answer_box"]:
            return data["answer_box"]["answer"]

        # 2. Knowledge Graph — structured entity info
        if "knowledge_graph" in data and "description" in data["knowledge_graph"]:
            return data["knowledge_graph"]["description"]

        # 3. Organic results — fallback to top 3 snippets
        results = data.get("organic_results", [])
        if results:
            snippets = [
                f"[{i}] {r.get('title', '')}\n    {r.get('snippet', '').replace(chr(10), ' ')}"
                for i, r in enumerate(results[:3], 1)
            ]
            return "\n\n".join(snippets)

        return f"No results found for '{query}'."

    except requests.RequestException as e:
        return f"Search error: {e}"


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------

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
