# Day 4 — Classic Agent Paradigms (Chapter 4)

Three fundamentally different ways to build an LLM agent, implemented from scratch.

Based on [Hello Agents - Chapter 4](https://datawhalechina.github.io/hello-agents/chapter4/Chapter4-Building-Classic-Agent-Paradigms.md).

---

## What You'll Learn

| Paradigm | Core Idea | Best For |
|----------|----------|----------|
| **ReAct** | Think → Act → Observe → repeat | Tool calling, web search, API interaction |
| **Plan-and-Solve** | Plan ALL steps first, then execute | Math problems, structured reasoning |
| **Reflection** | Execute → Critique → Refine → repeat | Code generation, high-quality output |

## Project Structure

```
day4/
├── hello_agents_llm.py     # Reusable LLM client (streaming)
├── tool_executor.py         # Tool registry + simulated tools
├── react_agent.py           # Paradigm 1: ReAct (standalone)
├── plan_solve_agent.py      # Paradigm 2: Plan-and-Solve (standalone)
├── reflection_agent.py      # Paradigm 3: Reflection (standalone)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
# Activate the root-level venv
source .venv/Scripts/activate       # Git Bash
# .venv\Scripts\activate            # PowerShell

# Install deps (one-time, covers both day1 and day4)
pip install -r day1/requirements.txt

# Copy your .env from day1
cp day1/.env day4/.env
```

## Run

Each paradigm runs independently — no main.py needed:

```bash
cd day4
python react_agent.py        # ReAct: Think → Act → Observe loop
python plan_solve_agent.py   # Plan-and-Solve: plan first, then solve
python reflection_agent.py   # Reflection: Execute → Critique → Refine
```

## Tools — Simulated, No Extra API Keys

The `search()` and `calculator()` tools in [tool_executor.py](tool_executor.py) are **simulated** — no SerpAPI or Tavily key needed. The original doc uses real APIs, but this repo replaces them with pre-canned results so you can focus on learning the agent paradigms themselves.

To swap in a real search (e.g., Tavily from day1), just register it in the `if __name__ == "__main__"` block:

```python
# Replace simulated search with real Tavily:
from day1.agent import get_attraction  # or write a real search wrapper
tools.register_tool("Search", "Search the web", real_search_function)
```

## Paradigm Comparison

```
ReAct:                  Plan-and-Solve:         Reflection:

  Think                   Plan ──► Step 1         Execute
    ↓                              Step 2           ↓
  Act                      ↓       Step 3         Reflect
    ↓                      ↓         ↓              ↓
  Observe                 Solve ──► Execute all    Refine
    ↓                              at once          ↓
  (loop)                                         (loop until good)
```

## Key Differences from Day 1

| Day 1 | Day 4 |
|-------|-------|
| Raw `AVAILABLE_TOOLS` dict | `ToolExecutor` class with descriptions |
| `OpenAICompatibleClient` (non-streaming) | `HelloAgentsLLM` (streaming) |
| One paradigm (basic ReAct) | Three paradigms with formal structure |
| Inline everything | Modular: LLM / tools / agents are separate files |

## Test Run

### ReAct
```
--- ReAct Step 1 ---
💭 Thought: To answer about Huawei's latest phones, I need to search.
Action: Search[What are Huawei's latest phones?]
🔍 Searching: What are Huawei's latest phones?
👀 Observation: [1] HUAWEI Mate 70 Pro — Kirin 9100 chip...
                 [2] HUAWEI Pura 80 Pro+ — First retractable camera...

--- ReAct Step 2 ---
💭 Thought: I now have the search results. Let me summarize.
Action: Finish[Huawei's latest phones: Mate 70 Pro (Kirin 9100...) and Pura 80 Pro+ (...)]
🎉 Final Answer: ...
```

### Plan-and-Solve
```
📋 PHASE 1 — Planning
✅ 1. Monday: 15 apples
   2. Tuesday: 15 × 2 = 30 apples
   3. Wednesday: 30 - 5 = 25 apples
   4. Sum: 15 + 30 + 25 = 70 apples

⚙️  PHASE 2 — Solving
✅ Step 1: 15 | Step 2: 30 | Step 3: 25 | FINAL ANSWER: 70 apples
```

### Reflection
```
📝 Round 1 — Initial Attempt
✅ def find_primes(n): ...  # O(n√n) trial division

🔍 Round 1 — Review
✅ Issues: O(n√n) is suboptimal. Recommend Sieve of Eratosthenes.

🔧 Round 1 — Refine
✅ def find_primes(n): ...  # O(n log log n) sieve

🔍 Round 2 — Review
✅ NO_IMPROVEMENT_NEEDED
```
