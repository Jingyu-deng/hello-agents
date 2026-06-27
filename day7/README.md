# Day 7 — Build Your Own Agent Framework (Chapter 7)

Transform the Day 4 standalone agent scripts into a **reusable Python framework** — `hello_agents/` — with abstract base classes, a unified tool system, and pluggable agent implementations.

**The big idea:** Day 4 proved three paradigms work. Now we extract their common skeleton into a proper package so adding a new agent type means writing one class, not starting from scratch.

Based on [Hello Agents - Chapter 7](https://datawhalechina.github.io/hello-agents/chapter7/Chapter7-Building-Your-Agent-Framework.md).

---

## What You'll Learn

| Concept | How We Apply It |
|---------|-----------------|
| **ABC + @abstractmethod** | `Agent(ABC)` defines the contract — all agents must implement `run()` |
| **Template Method pattern** | Base class provides history management; subclasses provide the reasoning loop |
| **Dependency inversion** | Agents depend on `Agent` and `Tool` abstractions, not concrete implementations |
| **Self-describing tools** | `Tool(ABC)` with `get_parameters()` — tools describe themselves for prompt generation |
| **Clean public API** | One import → all agents: `from hello_agents import SimpleAgent, HelloAgentsLLM` |

## Architecture

```
hello_agents/                        # Pip-installable package
├── __init__.py                      # Public API surface
├── core/                            # Framework foundation
│   ├── agent.py      Agent(ABC)     Abstract base — the contract
│   ├── llm.py        HelloAgentsLLM Streaming + non-streaming, provider auto-detect
│   ├── message.py    Message        Structured conversation turns
│   ├── config.py     Config         Centralized settings with env fallback
│   └── exceptions.py                AgentError, LLMError, ToolError, ...
├── tools/                           # Tool abstraction layer
│   ├── base.py       Tool(ABC)      Every tool describes itself
│   ├── registry.py   ToolRegistry   Register Tool objects OR plain functions
│   └── builtin/
│       ├── calculator.py            Safe arithmetic evaluator
│       └── search.py                Simulated web search (keyword matching)
└── agents/                          # Agent implementations
    ├── simple_agent.py              Basic chat + optional tool calling
    ├── react_agent.py               Thought → Action → Observation loop
    ├── plan_solve_agent.py          Plan ALL steps → then execute
    └── reflection_agent.py          Execute → Critique → Refine
```

## Class Hierarchy

```
Agent (ABC)                          ← core/agent.py
├── SimpleAgent                      ← basic conversation + tool calls
├── ReActAgent                       ← Thought → Action → Observation
├── PlanAndSolveAgent                ← Plan → Solve (two-phase)
└── ReflectionAgent                  ← Execute → Reflect → Refine

Tool (ABC)                           ← tools/base.py
├── CalculatorTool                   ← safe eval()
└── SimulatedSearchTool              ← keyword matching
```

## Setup

```bash
# Activate the root-level venv
source .venv/Scripts/activate       # Git Bash
# .venv\Scripts\activate            # PowerShell

# Install deps (same as day1/day4)
pip install -r day1/requirements.txt

# Copy .env.example and fill in your keys
cp .env.example .env
```

## Run

```bash
cd day7

# Full demo — all 4 agents in sequence
python demo.py

# Individual agents
python -m hello_agents.agents.react_agent
python -m hello_agents.agents.plan_solve_agent
python -m hello_agents.agents.reflection_agent
```

## Usage (30 seconds)

```python
from hello_agents import HelloAgentsLLM, SimpleAgent

llm = HelloAgentsLLM()
agent = SimpleAgent(
    name="Assistant",
    llm=llm,
    system_prompt="You are a helpful AI assistant."
)
response = agent.run("What is the capital of France?")
print(response)
```

With tools:
```python
from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry
from hello_agents.tools.builtin import SimulatedSearchTool, CalculatorTool

llm = HelloAgentsLLM()
tools = ToolRegistry()
tools.register_tool(SimulatedSearchTool())
tools.register_tool(CalculatorTool())

agent = ReActAgent(name="ReAct", llm=llm, tool_registry=tools)
answer = agent.run("What are Huawei's latest phones?")
```

## Key Differences from Day 4

| Day 4 | Day 7 |
|-------|-------|
| Standalone scripts, no shared code | Package with `Agent(ABC)` base class |
| `ToolExecutor` dict wrapper | `Tool(ABC)` + `ToolRegistry` with schemas |
| Raw strings for history | `Message` dataclass with role/content/metadata |
| `print()` for all output | Typed exceptions (`AgentError`, `ToolError`, ...) |
| One LLM method (`think`) | `think()` (streaming) + `invoke()` (non-streaming) |
| Hardcoded provider | Auto-detect from base_url |
| Agents have different constructors | All agents share `run(input_text) -> str` |
| 3 agents | 4 agents (+ SimpleAgent) |

## Design Principles

1. **Everything is a Tool.** Calculator, search, memory, RAG — all abstracted as `Tool(ABC)`. No special-casing.

2. **The framework depends on abstractions, not implementations.** `Agent.__init__()` takes `HelloAgentsLLM`, but subclasses only call `self.llm.think()` — swap the LLM without touching agent code.

3. **Self-describing tools.** `Tool.get_parameters()` enables prompt auto-generation and native function-calling schemas from a single method.

4. **Lightweight.** Zero dependencies beyond `openai` and `python-dotenv`. No Pydantic, no LangChain. The whole framework is ~800 lines you can read in one sitting.

## Test Run (actual output)

### SimpleAgent — Basic Conversation
```
[Response] The capital of France is Paris. It is known for its iconic landmarks
like the Eiffel Tower, the Louvre Museum (home to the Mona Lisa), Notre-Dame
Cathedral, and its rich history in art, fashion, cuisine, and culture.
```

### ReActAgent — Tool Calling (2 steps)
```
--- ReAct Step 1 ---
[LLM] Calling deepseek-chat (deepseek)...
Thought: I need to search the web for current information about Huawei's
  latest smartphone models.
Action: Search[Huawei latest phones 2025 key selling points]
[Call] Search[Huawei latest phones 2025 key selling points]
[Obs] [1] HUAWEI Mate 70 Pro — Kirin 9100 chip, satellite communication 2.0,
       XMAGE imaging upgrade, Kunlun glass.
       [2] HUAWEI Pura 80 Pro+ — First retractable camera, variable aperture,
       ultra-light-gathering night vision telephoto, HarmonyOS 4.2.

--- ReAct Step 2 ---
Thought: I can now compile the final answer.
Action: Finish[Huawei's latest phones include:
  1. HUAWEI Mate 70 Pro — Kirin 9100 chip, Satellite communication 2.0, ...
  2. HUAWEI Pura 80 Pro+ — First retractable camera, Variable aperture, ...]
```

### PlanAndSolveAgent — Math Word Problem
```
[Phase 1] PLAN — Planning
1. Monday's sales = 15 apples.
2. Tuesday's sales = 15 × 2 = 30 apples.
3. Wednesday's sales = 30 - 5 = 25 apples.
4. Total = 15 + 30 + 25 = 70 apples.

[Phase 2] SOLVE — Solving
Step 1: Monday's sales = 15 apples.
Step 2: Tuesday's sales = 15 × 2 = 30 apples.
Step 3: Wednesday's sales = 30 - 5 = 25 apples.
Step 4: Total = 15 + 30 + 25 = 70 apples.

FINAL ANSWER: 70
```

### ReflectionAgent — Code Generation
```
[Execute] Round 1 — Generated Sieve of Eratosthenes implementation.

[Review] Round 1 — Review
- Correctness: correct, edge cases handled (n < 2 returns [])
- Efficiency: O(n log log n) — optimal. Starts marking from p².
- Readability: clear docstring, well-commented.

NO_IMPROVEMENT_NEEDED — code is production-quality.

[OK] Reviewer satisfied — no further improvements needed.
```

## What's Next

This framework is the foundation. From here you can:
- Add native OpenAI function calling (Chapter 7.6 — Extension Patterns)
- Add async support (`arun()` methods)
- Add memory/RAG as Tool subclasses (Chapter 8)
- Add MCP/A2A protocol support (Chapter 10)
- Build the capstone travel assistant or deep-research agent (Chapter 13-14)
