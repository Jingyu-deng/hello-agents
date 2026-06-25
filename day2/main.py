"""
Chapter 4 — Classic Agent Paradigms: Demo Runner
=================================================
Runs all three paradigms with example tasks.
"""

from hello_agents_llm import HelloAgentsLLM
from tool_executor import ToolExecutor, search, calculator
from react_agent import ReActAgent
from plan_solve_agent import PlanAndSolveAgent
from reflection_agent import ReflectionAgent


def demo_react(llm: HelloAgentsLLM):
    """ReAct: research Huawei's latest phones via tool calling."""
    print("\n" + "=" * 60)
    print("🔁 PARADIGM 1: ReAct (Reasoning + Acting)")
    print("=" * 60)

    tools = ToolExecutor()
    tools.register_tool("Search", "Search the web for information", search)
    tools.register_tool("Calculator", "Evaluate a math expression", calculator)

    agent = ReActAgent(llm, tools, max_steps=5)
    agent.run("华为最新发布的手机有哪些？列举其主要卖点。")


def demo_plan_solve(llm: HelloAgentsLLM):
    """Plan-and-Solve: solve a multi-step math word problem."""
    print("\n" + "=" * 60)
    print("📋 PARADIGM 2: Plan-and-Solve")
    print("=" * 60)

    agent = PlanAndSolveAgent(llm)
    agent.run(
        "一家水果店周一卖了15个苹果，周二的销量是周一的2倍，周三比周二少卖了5个。"
        "请问这三天一共卖了多少个苹果？"
    )


def demo_reflection(llm: HelloAgentsLLM):
    """Reflection: generate an optimal prime-finding function."""
    print("\n" + "=" * 60)
    print("🔄 PARADIGM 3: Reflection (Execute → Reflect → Refine)")
    print("=" * 60)

    agent = ReflectionAgent(llm, max_iterations=2)
    result = agent.run("Write a Python function to find all prime numbers from 1 to n.")
    if result:
        print(f"\n{'─' * 40}\n📄 Final Output:\n{result}")


def main():
    llm = HelloAgentsLLM()

    demo_react(llm)
    demo_plan_solve(llm)
    demo_reflection(llm)

    print("\n" + "=" * 60)
    print("✅ All three paradigms completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
