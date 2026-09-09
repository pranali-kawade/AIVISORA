"""Centralized configuration for AEO Radar.

Reads all configuration exclusively from environment variables (loaded via
.env in local development). Nothing here ever hardcodes a credential, and
nothing here prints or logs secret values. Missing API keys are expected
and must not cause the application to crash -- callers should check the
`is_configured` flags before attempting to use a given provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    # Loading .env is optional: the app must still run if python-dotenv
    # is unavailable or no .env file exists (e.g. in a deployed environment
    # where variables are injected directly).
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_bool(name: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean, tolerant of common
    truthy/falsy string representations.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    """Parse an environment variable as an int, falling back to a default
    on missing or malformed values rather than raising at import time.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of application configuration."""

    gemini_api_key: str | None
    openrouter_api_key: str | None
    gemini_model: str
    openrouter_model: str
    request_timeout: int
    mock_mode: bool
    # Optional so existing callers that build Settings without it keep working;
    # populated from TAVILY_API_KEY by load_settings().
    tavily_api_key: str | None = None

    @property
    def gemini_configured(self) -> bool:
        """Whether a Gemini API key is present (does not verify validity)."""
        return bool(self.gemini_api_key)

    @property
    def openrouter_configured(self) -> bool:
        """Whether an OpenRouter API key is present (does not verify validity)."""
        return bool(self.openrouter_api_key)

    @property
    def tavily_configured(self) -> bool:
        """Whether a Tavily API key is present (does not verify validity)."""
        return bool(self.tavily_api_key)


def load_settings() -> Settings:
    """Load configuration fresh from the current environment.

    Called explicitly (rather than evaluated once at import time) so that
    tests can mutate environment variables and reload settings without
    reimporting the module.
    """
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
        request_timeout=_get_int("REQUEST_TIMEOUT", 30),
        mock_mode=_get_bool("MOCK_MODE", default=True),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
    )


settings = load_settings()
