"""
SimpleAgent — the most basic agent in the framework.

Unlike ReAct / Plan-and-Solve / Reflection, SimpleAgent is a pure
conversational agent with optional tool calling. It demonstrates the
minimum viable Agent subclass: override run().

Use it for:
  - Quick Q&A
  - Chatbots
  - Testing that your LLM config works
  - A baseline to compare against the paradigm agents

Tool calling: when enable_tool_calling=True and a ToolRegistry is
provided, SimpleAgent looks for [TOOL_CALL:name:params] in the LLM
response and executes tools in a loop (up to max_tool_iterations).
"""

from typing import Optional
import re

from hello_agents.core.agent import Agent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.message import Message
from hello_agents.core.config import Config
from hello_agents.core.exceptions import ToolNotFound, MaxStepsError
from hello_agents.tools.registry import ToolRegistry


class SimpleAgent(Agent):
    """A basic conversational agent with optional tool calling.

    Usage:
        llm = HelloAgentsLLM()
        agent = SimpleAgent(name="Bot", llm=llm, system_prompt="You are helpful.")
        response = agent.run("What is the capital of France?")
    """

    TOOL_CALL_RE = re.compile(r"\[TOOL_CALL:([^:]+):([^\]]+)\]")

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    # ------------------------------------------------------------------
    # Core run()
    # ------------------------------------------------------------------

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """Run the agent on user input.

        If tool calling is enabled and the LLM response contains
        [TOOL_CALL:name:params], execute the tool and feed results
        back for another iteration.

        Args:
            input_text: The user's message.
            max_tool_iterations: Maximum tool-calling rounds.

        Returns:
            The agent's final text response.
        """
        messages = self._build_messages(input_text)

        # Fast path: no tools → single call
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, temperature=self.config.temperature)
            self._record_turn(input_text, response)
            return response

        # Tool-calling loop
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, input_text: str) -> list[dict]:
        """Build the message list for the LLM call."""
        messages: list[dict] = []

        # System prompt with tool descriptions
        enhanced_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_prompt})

        # History
        for msg in self._history:
            if msg.role != "system":  # already handled above
                messages.append(msg.to_openai())

        # Current input
        messages.append({"role": "user", "content": input_text})

        return messages

    def _get_enhanced_system_prompt(self) -> str:
        """System prompt, optionally augmented with tool descriptions."""
        base = self.system_prompt or "You are a helpful AI assistant."

        if self.enable_tool_calling and self.tool_registry:
            tools_desc = self.tool_registry.get_available_tools()
            base += (
                f"\n\nYou have access to the following tools:\n{tools_desc}\n\n"
                f"To use a tool, include this in your response: "
                f"[TOOL_CALL:tool_name:input_text]\n\n"
                f"When done, provide the final answer without any tool call."
            )
        return base

    def _run_with_tools(
        self, messages: list[dict], input_text: str, max_iterations: int, **kwargs
    ) -> str:
        """Execute the tool-calling loop."""
        current_iteration = 0
        final_response = ""

        while current_iteration < max_iterations:
            current_iteration += 1
            response = self.llm.invoke(
                messages, temperature=self.config.temperature, **kwargs
            )

            # Check for tool calls
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                final_response = response
                break

            # Execute each tool call
            print(f"\n[Tool] Found {len(tool_calls)} tool call(s):")
            tool_results = []
            clean_response = response
            for call in tool_calls:
                print(f"  -> {call['tool_name']}[{call['parameters']}]")
                result = self._execute_tool(call["tool_name"], call["parameters"])
                tool_results.append(result)
                clean_response = clean_response.replace(call["original"], "")

            # Inject tool results into messages
            messages.append({"role": "assistant", "content": clean_response.strip()})
            results_text = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": f"Tool results:\n{results_text}\n\nContinue based on these results.",
            })

        # If we exhausted iterations, do one final call
        if current_iteration >= max_iterations and not final_response:
            final_response = self.llm.invoke(messages, temperature=self.config.temperature)

        self._record_turn(input_text, final_response or response)
        return final_response or response

    def _parse_tool_calls(self, text: str) -> list[dict]:
        """Extract [TOOL_CALL:name:params] markers from the LLM response."""
        matches = self.TOOL_CALL_RE.findall(text)
        return [
            {
                "tool_name": m[0].strip(),
                "parameters": m[1].strip(),
                "original": f"[TOOL_CALL:{m[0]}:{m[1]}]",
            }
            for m in matches
        ]

    def _execute_tool(self, name: str, params: str) -> str:
        """Execute a tool and return its result or an error."""
        if not self.tool_registry:
            return "[ERROR] No tool registry configured."

        tool = self.tool_registry.get_tool(name)
        if tool is None:
            raise ToolNotFound(
                f"Tool '{name}' not found. Available: {self.tool_registry.tool_names}"
            )

        try:
            # Tool objects get params as input string
            if isinstance(tool, type) or hasattr(tool, "run"):
                return tool.run(input=params)
            # Plain functions
            return tool(params)
        except Exception as e:
            return f"[ERROR] Tool execution failed: {e}"

    def _record_turn(self, user_input: str, assistant_response: str) -> None:
        """Append user + assistant messages to history."""
        self.add_message(Message(role="user", content=user_input))
        self.add_message(Message(role="assistant", content=assistant_response))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from hello_agents.core.llm import HelloAgentsLLM

    llm = HelloAgentsLLM()
    agent = SimpleAgent(
        name="Assistant",
        llm=llm,
        system_prompt="You are a helpful AI assistant. Answer concisely.",
    )
    print("SimpleAgent demo — type 'quit' to exit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("quit", "exit", "q"):
            break
        if not user:
            continue
        response = agent.run(user)
        print(f"\nAssistant: {response}\n")
