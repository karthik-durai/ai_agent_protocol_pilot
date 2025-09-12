"""Central configuration for LLM settings and retries.

Reads from environment with safe defaults and exposes helpers that other
modules can import without duplicating env parsing logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
try:
    # Load .env early so os.getenv sees configured values
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # no-op if .env not present
except Exception:
    pass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    temperature: float
    timeout: int
    max_retries: int
    retry_backoff: float

    @staticmethod
    def from_env() -> "LLMConfig":
        return LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://v65r66mi87vb67-11434.proxy.runpod.net"),
            model=os.getenv("LLM_MODEL", "llama3.1:latest"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            retry_backoff=float(os.getenv("LLM_RETRY_BACKOFF", "0.5")),
        )


def get_llm_config() -> LLMConfig:
    """Return a cached LLM configuration object."""
    # No heavy computation; just read once.
    global _CFG
    try:
        return _CFG  # type: ignore[name-defined]
    except NameError:
        _cfg = LLMConfig.from_env()
        globals()["_CFG"] = _cfg
        return _cfg
