"""
ToolRegistry — stores and dispatches tools.

Two registration styles (pick what fits):

  1. Tool objects (full-featured):
       registry.register_tool(CalculatorTool())

  2. Plain functions (backward-compatible with Day 4):
       registry.register_function("Search", "Search the web", search_fn)

Both styles work together — get_tool() and execute() handle either.
"""

from typing import Callable
from .base import Tool


class ToolRegistry:
    """A collection of named tools that agents can call.

    Usage:
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_function("Search", "Search the web", search_fn)

        # Prompt-ready listing
        print(registry.get_available_tools())

        # Dispatch
        result = registry.execute("Search", query="Huawei phones")
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        """Register a Tool subclass instance."""
        if tool.name in self._tools or tool.name in self._functions:
            print(f"[WARN] Tool '{tool.name}' is being overwritten.")
        self._tools[tool.name] = tool
        print(f"[OK] Tool '{tool.name}' registered.")

    def register_function(self, name: str, description: str, func: Callable) -> None:
        """Register a plain function as a tool (Day 4 compatibility).

        Args:
            name: Tool name shown to the agent.
            description: What the tool does (shown in prompt).
            func: Callable that takes a string and returns a string.
        """
        if name in self._tools or name in self._functions:
            print(f"[WARN] Tool '{name}' is being overwritten.")
        self._functions[name] = {"description": description, "func": func}
        print(f"[OK] Tool '{name}' registered (function).")

    # ------------------------------------------------------------------
    # Lookup & dispatch
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Tool | Callable | None:
        """Look up a tool by name. Returns Tool, Callable, or None."""
        if name in self._tools:
            return self._tools[name]
        if name in self._functions:
            return self._functions[name]["func"]
        return None

    def execute(self, name: str, input_str: str = "", **kwargs) -> str:
        """Execute a tool by name, handling both Tool and function types.

        For function-type tools, passes input_str as the sole argument.
        For Tool-type tools, passes **kwargs to run().

        Returns:
            The tool's result string, or an error message.
        """
        # Try Tool first
        tool = self._tools.get(name)
        if tool:
            try:
                return tool.run(input=input_str, **kwargs)
            except Exception as e:
                return f"[ERROR] Tool '{name}' failed: {e}"

        # Try function
        fn_entry = self._functions.get(name)
        if fn_entry:
            try:
                return fn_entry["func"](input_str)
            except Exception as e:
                return f"[ERROR] Tool '{name}' failed: {e}"

        available = list(self._tools.keys()) + list(self._functions.keys())
        return f"[ERROR] Tool '{name}' not found. Available: {available}"

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def get_available_tools(self) -> str:
        """Return a prompt-ready tool listing.

        Format:
            - ToolName: description  (for functions)
            - ToolName params: description  (for Tool objects)
        """
        lines: list[str] = []

        for name, tool in self._tools.items():
            params = tool.get_parameter_string()
            line = f"- {name}{params}: {tool.description}"
            lines.append(line)

        for name, info in self._functions.items():
            line = f"- {name}: {info['description']}"
            lines.append(line)

        if not lines:
            return "(No tools available)"
        return "\n".join(lines)

    def get_tools_schema(self) -> list[dict]:
        """Return structured tool schemas (for native function-calling APIs).

        Each entry follows the OpenAI function-calling schema format.
        """
        schemas: list[dict] = []

        for name, tool in self._tools.items():
            params = tool.get_parameters()
            properties = {}
            required = []
            for p in params:
                properties[p.name] = {
                    "type": p.param_type,
                    "description": p.description,
                }
                if p.required:
                    required.append(p.name)

            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })

        return schemas

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys()) + list(self._functions.keys())

    def __len__(self) -> int:
        return len(self._tools) + len(self._functions)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"
