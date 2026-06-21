"""
Hello Agents — Chapter 1: Your First AI Agent
==============================================
An intelligent travel assistant that follows the Thought → Action → Observation loop,
driven by an LLM and equipped with real-world tools (weather lookup + attraction search).

Based on: https://github.com/datawhalechina/hello-agents
"""

import os
import re
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# 1. System Prompt — the "operating manual" for the LLM
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """
You are an intelligent travel assistant. Your task is to analyze user requests and use available tools to solve problems step by step.

# Available Tools:
- `get_weather(city: str)`: Query real-time weather for a specified city.
- `get_attraction(city: str, weather: str)`: Search for recommended tourist attractions based on city and weather.

# Output Format Requirements:
Each response must strictly follow this format, containing one Thought-Action pair:

Thought: [Your thinking process and next step plan]
Action: [The specific action you want to execute]

Action format must be one of the following:
1. Call a tool: function_name(arg_name="arg_value")
2. Finish task: Finish[final answer]

# Important Notes:
- Output only one Thought-Action pair each time
- Action must be on the same line, do not break lines
- When you have collected enough information to answer the user's question, you must use Action: Finish[final answer] format to end

Let's begin!
"""


# ---------------------------------------------------------------------------
# 2. Tools — the agent's "hands" for interacting with the world
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """
    Query real-time weather information by calling the wttr.in API.

    Args:
        city: Name of the city to query (e.g. "Beijing", "London").

    Returns:
        A natural-language description of current weather conditions.
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city} current weather: {weather_desc}, temperature {temp_c} degrees Celsius"

    except requests.exceptions.RequestException as e:
        return f"Error: Network problem encountered when querying weather - {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Failed to parse weather data, city name may be invalid - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    Search for recommended tourist attractions based on city and weather,
    using the Tavily Search API.

    Args:
        city: Name of the city.
        weather: Current weather description (e.g. "Sunny", "Rainy").

    Returns:
        A string with attraction recommendations.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not configured."

    try:
        from tavily import TavilyClient
    except ImportError:
        return "Error: tavily-python package not installed. Run: pip install tavily-python"

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' most worthwhile tourist attractions and reasons in '{weather}' weather"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # Prefer the AI-generated summary answer
        if response.get("answer"):
            return response["answer"]

        # Fall back to raw results
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "Sorry, no relevant tourist attraction recommendations found."

        return "Based on search, found the following information for you:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Error: Problem occurred when executing Tavily search - {e}"


# Tool registry — maps tool names to callables
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


# ---------------------------------------------------------------------------
# 3. LLM Client — the agent's "brain"
# ---------------------------------------------------------------------------

class OpenAICompatibleClient:
    """
    A client for calling any LLM service compatible with the OpenAI interface spec.

    Works with: OpenAI, Azure OpenAI, Ollama, vLLM, LM Studio, and many others.
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Call the LLM API and return the generated response text."""
        print("Calling large language model...")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            answer = response.choices[0].message.content
            print("Large language model responded successfully.")
            return answer
        except Exception as e:
            print(f"Error occurred when calling LLM API: {e}")
            return "Error: Error occurred when calling language model service."


# ---------------------------------------------------------------------------
# 4. Agent Loop — the core "perceive → think → act → observe" cycle
# ---------------------------------------------------------------------------

class TravelAgent:
    """
    An intelligent travel assistant agent that uses the Thought → Action → Observation
    loop to autonomously solve travel-related tasks.
    """

    def __init__(self, llm: OpenAICompatibleClient, max_loops: int = 5):
        self.llm = llm
        self.max_loops = max_loops
        self.prompt_history: list[str] = []

    def run(self, user_prompt: str) -> str:
        """
        Execute the agent loop to process a user request.

        Args:
            user_prompt: The natural-language request from the user.

        Returns:
            The final answer produced by the agent.
        """
        self.prompt_history = [f"User request: {user_prompt}"]
        print(f"User input: {user_prompt}\n{'=' * 40}")

        for i in range(self.max_loops):
            print(f"--- Loop {i + 1} ---\n")

            # ---- Step A: Build the full prompt from history ----
            full_prompt = "\n".join(self.prompt_history)

            # ---- Step B: Generate — let the LLM "think" ----
            llm_output = self.llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

            # Trim any extra Thought-Action pairs the model may have emitted
            llm_output = self._trim_output(llm_output)
            print(f"Model output:\n{llm_output}\n")
            self.prompt_history.append(llm_output)

            # ---- Step C: Parse and execute the action ----
            action_match = re.search(r"Action:\s*(.*)", llm_output, re.DOTALL)
            if not action_match:
                observation = "Error: No action found. Please explicitly use Action: finish(...) or other actions."
                observation_str = f"Observation: {observation}"
                print(f"{observation_str}\n{'=' * 40}")
                self.prompt_history.append(observation_str)
                continue

            action_str = action_match.group(1).strip()

            # ---- Check for task completion ----
            if action_str.startswith("Finish"):
                final_answer = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
                if final_answer:
                    answer = final_answer.group(1).strip()
                    print(f"Task completed, final answer: {answer}")
                    return answer
                # Malformed Finish — feed back as observation
                observation_str = (
                    "Observation: Error: Finish action must use format Finish[answer]"
                )
                print(f"{observation_str}\n{'=' * 40}")
                self.prompt_history.append(observation_str)
                continue

            # ---- Parse tool call ----
            tool_name_match = re.search(r"(\w+)\(", action_str)
            if not tool_name_match:
                observation_str = (
                    f"Observation: Error: Could not parse tool name from action '{action_str}'"
                )
                print(f"{observation_str}\n{'=' * 40}")
                self.prompt_history.append(observation_str)
                continue

            tool_name = tool_name_match.group(1)
            args_str_match = re.search(r"\((.*)\)", action_str, re.DOTALL)
            args_str = args_str_match.group(1) if args_str_match else ""

            # Parse keyword arguments from the tool call string
            kwargs = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_str))

            # ---- Execute the tool ----
            if tool_name in AVAILABLE_TOOLS:
                observation = AVAILABLE_TOOLS[tool_name](**kwargs)
            else:
                observation = f"Error: Undefined tool '{tool_name}'. Available tools: {list(AVAILABLE_TOOLS.keys())}"

            # ---- Step D: Record observation for the next loop ----
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n{'=' * 40}")
            self.prompt_history.append(observation_str)

        return "Agent reached maximum loops without completing the task."

    @staticmethod
    def _trim_output(llm_output: str) -> str:
        """
        Keep only the first Thought-Action pair if the model emits more than one.
        """
        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL,
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                print("Truncated extra Thought-Action pairs")
                return truncated
        return llm_output


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

def main():
    # ---- Configuration (set via environment variables) ----
    # Defaults to DeepSeek. Works with any OpenAI-compatible API.
    api_key = os.environ.get("API_KEY", "YOUR_API_KEY")
    base_url = os.environ.get("BASE_URL", "https://api.deepseek.com/v1")
    model_id = os.environ.get("MODEL_ID", "deepseek-chat")
    tavily_key = os.environ.get("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")

    # Make Tavily key available to the tool function
    os.environ.setdefault("TAVILY_API_KEY", tavily_key)

    # ---- Validate configuration ----
    missing = []
    # Only API_KEY is truly required; BASE_URL and MODEL_ID have DeepSeek defaults
    if not api_key or api_key == "YOUR_API_KEY":
        missing.append("API_KEY")
    if missing:
        print(
            "⚠️  Warning: The following environment variables are not configured:\n"
            + "\n".join(f"  - {v}" for v in missing)
            + "\n\nSet them before running:\n"
            "  export API_KEY=sk-your-deepseek-key\n"
            "\n(Default: BASE_URL=https://api.deepseek.com/v1, MODEL_ID=deepseek-chat)\n"
        )

    # ---- Create the LLM client ----
    llm = OpenAICompatibleClient(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
    )

    # ---- Create and run the agent ----
    agent = TravelAgent(llm=llm, max_loops=5)

    user_prompt = (
        "Hello, please help me check today's weather in Shenzhen, "
        "and then recommend a suitable tourist attraction based on the weather."
    )

    final_answer = agent.run(user_prompt)
    print(f"\n{'=' * 40}\nFinal Answer:\n{final_answer}")


if __name__ == "__main__":
    main()
