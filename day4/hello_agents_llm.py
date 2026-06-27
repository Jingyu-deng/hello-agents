"""
Reusable LLM client — the "brain" for all three agent paradigms.

Improvements over day1's OpenAICompatibleClient:
- Streaming: displays tokens as they arrive (better UX for long responses)
- Config validation: fails early with clear error messages
- Single generate() method shared by all agents
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Fix UnicodeEncodeError on Windows GBK terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


class HelloAgentsLLM:
    """LLM client compatible with any OpenAI-format API (DeepSeek, OpenAI, etc.)."""

    def __init__(self, model=None, api_key=None, base_url=None, timeout=None):
        self.model = model or os.getenv("LLM_MODEL_ID") or os.getenv("MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, api_key, base_url]):
            raise ValueError(
                "Missing required config. Set in .env:\n"
                "  LLM_MODEL_ID (or MODEL_ID)\n"
                "  LLM_API_KEY  (or API_KEY)\n"
                "  LLM_BASE_URL (or BASE_URL)"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, messages: list[dict], temperature: float = 0) -> str:
        """Call the LLM and stream the response token by token."""
        print(f"[LLM] Calling {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            print("", end="", flush=True)
            collected = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected.append(content)
            print()
            return "".join(collected)
        except Exception as e:
            print(f"\n[FAIL] LLM error: {e}")
            return ""
