"""
Configuration management — centralised settings with env-var loading.

Uses Pydantic for validation and follows the pattern from the teaching
material: sensible defaults on the model, explicit from_env() factory
for environment overrides, and to_dict() for serialisation.
"""

import os
from pydantic import BaseModel


class Config(BaseModel):
    """HelloAgents configuration class.

    All fields have sensible defaults.  Call Config.from_env() to
    override defaults from environment variables.
    """

    # ---- LLM configuration ----
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    provider: str = "auto"
    temperature: float = 0.0
    timeout: int = 60

    # ---- Agent configuration ----
    max_steps: int = 5
    max_iterations: int = 3

    # ---- System configuration ----
    debug: bool = False
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Config":
        """Create a Config instance, overriding defaults from environment variables."""
        return cls(
            # LLM settings
            model=os.getenv("LLM_MODEL_ID", cls.model_fields["model"].default),
            api_key=os.getenv("LLM_API_KEY", os.getenv("API_KEY", cls.model_fields["api_key"].default)),
            base_url=os.getenv("LLM_BASE_URL", os.getenv("BASE_URL", cls.model_fields["base_url"].default)),
            provider=os.getenv("LLM_PROVIDER", cls.model_fields["provider"].default),
            temperature=float(os.getenv("LLM_TEMPERATURE", cls.model_fields["temperature"].default)),
            timeout=int(os.getenv("LLM_TIMEOUT", cls.model_fields["timeout"].default)),
            # Agent settings
            max_steps=int(os.getenv("AGENT_MAX_STEPS", cls.model_fields["max_steps"].default)),
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", cls.model_fields["max_iterations"].default)),
            # System
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", cls.model_fields["log_level"].default),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert configuration to a plain dictionary."""
        return self.model_dump()
