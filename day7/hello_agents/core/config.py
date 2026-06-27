"""
Config — centralized configuration with environment-variable fallback.

All agents and the LLM client share one Config instance (or a subclass),
avoiding scattered os.getenv() calls.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Framework-wide settings.  Every field can be overridden via constructor
    or falls back to an environment variable."""

    # ---- LLM settings ----
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 60
    temperature: float = 0.0

    # ---- Agent settings ----
    max_steps: int = 5
    max_iterations: int = 3

    # ---- Provider detection ----
    provider: str = "auto"

    def __post_init__(self):
        """Fill any None fields from environment variables."""
        self.model = self.model or os.getenv("LLM_MODEL_ID") or os.getenv("MODEL_ID")
        self.api_key = self.api_key or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
        self.base_url = self.base_url or os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL")
        self.provider = self.provider or os.getenv("LLM_PROVIDER", "auto")

        # Numeric env vars
        for attr, env_name in [
            ("timeout", "LLM_TIMEOUT"),
            ("temperature", "LLM_TEMPERATURE"),
            ("max_steps", "AGENT_MAX_STEPS"),
            ("max_iterations", "AGENT_MAX_ITERATIONS"),
        ]:
            val = os.getenv(env_name)
            if val is not None:
                try:
                    setattr(self, attr, type(getattr(self, attr))(val))
                except (ValueError, TypeError):
                    pass

    def validate(self) -> list[str]:
        """Return a list of missing required fields (empty = valid)."""
        missing = []
        if not self.model:
            missing.append("model (LLM_MODEL_ID)")
        if not self.api_key:
            missing.append("api_key (LLM_API_KEY)")
        if not self.base_url:
            missing.append("base_url (LLM_BASE_URL)")
        return missing
