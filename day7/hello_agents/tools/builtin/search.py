"""
SimulatedSearchTool — keyword-based simulated web search.

Wraps the Day 4 search() function in a proper Tool subclass.
Uses a hardcoded knowledge base with multi-keyword ALL-match.
No API key needed — great for demos and testing.
"""

from hello_agents.tools.base import Tool, ToolParameter

# ---------------------------------------------------------------------------
# Knowledge base — (keywords, result) pairs
# ALL keywords must appear (case-insensitive) for a match.
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: list[tuple[list[str], str]] = [
    (
        ["huawei", "phone"],
        "[1] HUAWEI Mate 70 Pro — Kirin 9100 chip, satellite "
        "communication 2.0, XMAGE imaging upgrade, Kunlun glass.\n"
        "[2] HUAWEI Pura 80 Pro+ — First retractable camera, "
        "variable aperture, ultra-light-gathering night vision "
        "telephoto, HarmonyOS 4.2.",
    ),
    (
        ["nvidia", "gpu"],
        "[1] GeForce RTX 50 Series — Blackwell architecture, DLSS 4, "
        "up to 2x performance over RTX 40.\n"
        "[2] GeForce RTX 5090 — 32 GB GDDR7, PCIe 5.0, 575 W TDP.",
    ),
    (
        ["python", "release"],
        "[1] Python 3.13 — Improved interactive interpreter, experimental "
        "JIT compiler, better error messages.\n"
        "[2] Python 3.12 — Per-interpreter GIL, comprehension inlining, "
        "new type annotation syntax.",
    ),
    (
        ["climate", "change"],
        "[1] IPCC 2025 Report — Global average temperature has risen 1.3 °C "
        "above pre-industrial levels.\n"
        "[2] Key findings: accelerated ice-sheet melt, more frequent extreme "
        "weather events, 40% emissions cut needed by 2035.",
    ),
    (
        ["tokyo", "population"],
        "[1] Tokyo, Japan — Population: ~37 million (metro area), "
        "the most populous metropolitan area in the world.\n"
        "[2] Tokyo's 23 special wards: ~9.7 million residents.",
    ),
]


class SimulatedSearchTool(Tool):
    """Simulated web search using keyword matching against a fixed knowledge base.

    All keywords must appear in the query (case-insensitive). Returns
    pre-written snippets when a match is found.

    Use this for demos. For real search, swap in SerpAPISearchTool.
    """

    def __init__(self):
        super().__init__(
            name="Search",
            description="Search the web for information. Input: search query string.",
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description="The search query string",
                required=True,
                param_type="string",
            ),
        ]

    def run(self, input: str = "", **kwargs) -> str:
        """Search the knowledge base.

        Args:
            input: The search query string.
            **kwargs: Also accepts 'query' as keyword.

        Returns:
            Matching result text or 'No results found'.
        """
        query: str = kwargs.get("query", input)
        query_lower = query.lower()

        for keywords, result in KNOWLEDGE_BASE:
            if all(kw.lower() in query_lower for kw in keywords):
                return result

        return f"No results found for '{query}'."
