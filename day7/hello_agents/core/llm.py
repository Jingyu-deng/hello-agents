"""
HelloAgentsLLM — unified LLM client for the entire framework.

Upgrades over Day 4's version:
  - think()  : streaming chat (same as Day 4, great for interactive CLIs)
  - invoke() : non-streaming chat (for agents that want clean return strings)
  - provider : auto-detection from base_url patterns
  - Config   : centralized settings instead of scattered os.getenv()
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

from .config import Config
from .exceptions import LLMError

# Fix UnicodeEncodeError on Windows GBK terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


class HelloAgentsLLM:
    """LLM client compatible with any OpenAI-format API.

    Works with: DeepSeek, OpenAI, Ollama, vLLM, and any /v1/chat/completions endpoint.

    Usage:
        llm = HelloAgentsLLM()                         # auto-detect from env
        llm = HelloAgentsLLM(provider="deepseek")       # explicit provider
        llm = HelloAgentsLLM(model="gpt-4o", ...)       # full control
    """

    # Known provider patterns — matched against base_url
    PROVIDER_PATTERNS = {
        "deepseek":  ["deepseek"],
        "openai":    ["api.openai.com"],
        "ollama":    ["localhost:11434", "127.0.0.1:11434"],
        "vllm":      [":8000"],
        "modelscope": ["api-inference.modelscope"],
        "zhipu":     ["open.bigmodel"],
    }

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        config: Config | None = None,
        timeout: int | None = None,
    ):
        # Merge config if provided
        cfg = config or Config.from_env()

        self.model = model or cfg.model
        self.api_key = api_key or cfg.api_key
        self.base_url = base_url or cfg.base_url
        self.timeout = timeout or cfg.timeout
        self.provider = provider or cfg.provider

        # Auto-detect provider if not set
        if self.provider == "auto":
            self.provider = self._detect_provider()

        # Final validation
        if not all([self.model, self.api_key, self.base_url]):
            missing = []
            if not self.model:
                missing.append("model")
            if not self.api_key:
                missing.append("api_key")
            if not self.base_url:
                missing.append("base_url")
            raise LLMError(
                f"Missing required LLM config: {', '.join(missing)}. "
                f"Set in .env (LLM_MODEL_ID, LLM_API_KEY, LLM_BASE_URL) or pass as arguments."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------
    # Provider detection
    # ------------------------------------------------------------------

    def _detect_provider(self) -> str:
        """Guess the provider from base_url patterns."""
        url = (self.base_url or "").lower()
        for name, patterns in self.PROVIDER_PATTERNS.items():
            if any(p in url for p in patterns):
                return name
        return "generic"

    # ------------------------------------------------------------------
    # Streaming API (for interactive CLIs)
    # ------------------------------------------------------------------

    def think(self, messages: list[dict], temperature: float = 0) -> str:
        """Stream a chat completion, printing tokens as they arrive.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: 0 = deterministic.

        Returns:
            The full response text (empty string on error).
        """
        print(f"[LLM] Calling {self.model} ({self.provider})...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            collected: list[str] = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected.append(content)
            print()  # final newline
            return "".join(collected)
        except Exception as e:
            print(f"\n[FAIL] LLM error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Non-streaming API (for agents that want clean returns)
    # ------------------------------------------------------------------

    def invoke(self, messages: list[dict], temperature: float = 0, **kwargs) -> str:
        """Non-streaming chat — returns the full response as a string.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: 0 = deterministic.

        Returns:
            The assistant's response text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=False,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}") from e

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HelloAgentsLLM(model={self.model!r}, provider={self.provider!r}, "
            f"base_url={self.base_url!r})"
        )
