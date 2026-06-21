# Day 1 — Your First AI Agent

Building an intelligent travel assistant that follows the **Thought → Action → Observation** loop, driven by an LLM (DeepSeek) and equipped with real-world tools.

Based on [Hello Agents - Chapter 1](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter1/Chapter1-Introduction-to-Agents.md).

## Concepts Covered

- Agent loop: Perception → Thought (Planning + Tool Selection) → Action → Observation
- System prompt engineering
- Tool calling (weather via [wttr.in](https://wttr.in), attraction search via [Tavily](https://tavily.com))
- LLM integration via OpenAI-compatible API
- Regex-based output parsing
- Dynamic function dispatch with `**kwargs`

## Setup

```bash
# Create venv (first time only)
python -m venv .venv

# Activate (Git Bash / WSL)
source .venv/Scripts/activate

# Activate (PowerShell)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Fill in your keys in .env (see .env.example)
# Then run:
python agent.py
```

## Project Structure

```
day1/
├── agent.py           # Main agent implementation
├── requirements.txt   # Python dependencies
├── .env.example       # Config template (no secrets)
├── .env               # Your real keys (gitignored)
└── .gitignore
```

## Test Run

```
User input: Hello, please help me check today's weather in Shenzhen, and then
recommend a suitable tourist attraction based on the weather.
========================================
--- Loop 1 ---

Calling large language model...
Large language model responded successfully.
Model output:
Thought: The user wants me to check the weather in Shenzhen today and then
recommend an attraction based on the weather. I'll start by calling the
get_weather function for Shenzhen.
Action: get_weather(city="Shenzhen")

Observation: Shenzhen current weather: Clear , temperature 29 degrees Celsius
========================================
--- Loop 2 ---

Calling large language model...
Large language model responded successfully.
Truncated extra Thought-Action pairs
Model output:
Thought: Now I know the weather in Shenzhen is Clear with 29°C. Next, I need
to call get_attraction with the city and weather to get a recommended tourist
attraction.
Action: get_attraction(city="Shenzhen", weather="Clear")

Observation: Error: tavily-python package not installed. Run: pip install tavily-python
========================================
--- Loop 3 ---

Calling large language model...
Large language model responded successfully.
Model output:
Thought: The get_attraction function encountered an error because the tavily-python
package is not installed. However, based on the current weather condition (Clear, 29°C)
in Shenzhen, I can still provide a reasonable recommendation...
Action: Finish[Today in Shenzhen, the weather is Clear and 29°C. With such sunny
weather, I recommend visiting **Shenzhen Bay Park** — a beautiful coastal park perfect
for a relaxing walk, cycling, or enjoying the sea view under clear skies. Alternatively,
**Window of the World** is also a great choice for outdoor sightseeing with replicas of
global landmarks. Enjoy your day!]

Task completed, final answer: Today in Shenzhen, the weather is Clear and 29°C.
With such sunny weather, I recommend visiting **Shenzhen Bay Park**...
========================================
```

## Key Takeaways

1. The LLM autonomously decided the step order: weather first → then attractions
2. When the tool failed (Tavily not installed), the LLM adapted and used its own knowledge
3. The agent loop enforces structure but the LLM provides the "intelligence"
4. Dynamic tool dispatch lets you add new tools without changing the loop code
