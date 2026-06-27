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

# Copy .env.example and fill in your keys
cp .env.example .env
```

## Run

Each paradigm runs independently:

```bash
cd day4
python react_agent.py        # ReAct: Think → Act → Observe loop
python plan_solve_agent.py   # Plan-and-Solve: plan first, then solve
python reflection_agent.py   # Reflection: Execute → Critique → Refine
```

## Tools

| Tool | Function | Requires |
|------|----------|----------|
| `search()` | Simulated search with pre-canned results | Nothing |
| `serpapi_search()` | Real Google search via [SerpAPI](https://serpapi.com) | `SERPAPI_API_KEY` in `.env` |
| `calculator()` | Safe expression evaluator | Nothing |

The simulated `search()` keeps the demo self-contained. For real web search, switch to `serpapi_search` in the `if __name__ == "__main__"` block:

```python
# Use real SerpAPI search:
from tool_executor import serpapi_search
tools.register_tool("Search", "Search the web via Google", serpapi_search)

# Or use simulated search (no API key needed):
from tool_executor import search
tools.register_tool("Search", "Search the web for information", search)
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
Tool 'Search' registered.
Tool 'Calculator' registered.

────────────────────────────────────────
--- ReAct Step 1 ---
[LLM] Calling deepseek-chat...
[Thought] I need to find the latest Huawei phones and their key selling
  points. I'll search the web for this information.
Action: Search[Huawei latest phones 2025 key selling points]
[Call] Search[Huawei latest phones 2025 key selling points]
Searching: Huawei latest phones 2025 key selling points
[Obs] [1] HUAWEI Mate 70 Pro — Kirin 9100 chip, satellite
       communication 2.0, XMAGE imaging upgrade, Kunlun glass.
       [2] HUAWEI Pura 80 Pro+ — First retractable camera,
       variable aperture, ultra-light-gathering night vision
       telephoto, HarmonyOS 4.2.

────────────────────────────────────────
--- ReAct Step 2 ---
[LLM] Calling deepseek-chat...
[Thought] The search results provide information on two of Huawei's
  latest phones. I can now compile the final answer.
Action: Finish[Huawei's Latest Phones:
  1. HUAWEI Mate 70 Pro — Kirin 9100 chip, Satellite communication 2.0,
     XMAGE imaging upgrade, Kunlun glass.
  2. HUAWEI Pura 80 Pro+ — First retractable camera, Variable aperture,
     Ultra-light-gathering night vision telephoto, HarmonyOS 4.2.]
[Answer] ...
```

### Plan-and-Solve
```
[Phase 1] PLAN — Planning
[LLM] Calling deepseek-chat...
1. Identify Monday's sales. → Monday = 15 apples.
2. Calculate Tuesday's sales. → 15 × 2 = 30 apples.
3. Calculate Wednesday's sales. → 30 - 5 = 25 apples.
4. Calculate total. → 15 + 30 + 25 = 70 apples.

[Phase 2] SOLVE — Solving
[LLM] Calling deepseek-chat...
Step 1: Monday's sales = 15 apples.
Step 2: Tuesday's sales = 15 × 2 = 30 apples.
Step 3: Wednesday's sales = 30 - 5 = 25 apples.
Step 4: Total = 15 + 30 + 25 = 70 apples.

FINAL ANSWER: 70
```

### Reflection
```
[Reflection] Task: Write a Python function to find all prime numbers
  from 1 to n.

[Execute] Round 1 — Initial Attempt
[LLM] Calling deepseek-chat...
def find_primes_up_to(n):  # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1
    return [i for i in range(2, n + 1) if is_prime[i]]

[Review] Round 1 — Review
[LLM] Calling deepseek-chat...
- Correctness: correct, edge cases handled.
- Efficiency: O(n log log n) — optimal. Minor: skip even numbers
  after p=2.
- Readability: clear docstring, comments, well-structured.

NO_IMPROVEMENT_NEEDED — code is production-quality.

[OK] Reviewer satisfied — no further improvements needed.
```
