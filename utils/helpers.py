"""Generic, provider-agnostic helper utilities.

Only genuinely reusable helpers belong here. Provider-specific logic
(Gemini/OpenRouter/crawling/etc.) must never be added to this module --
it belongs inside the relevant subsystem package instead.
"""

from __future__ import annotations

import logging

from config import Settings


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the application's root logger.

    Safe to call multiple times; avoids adding duplicate handlers.
    """
    logger = logging.getLogger("aeo_radar")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def configuration_status(settings: Settings) -> dict[str, str]:
    """Produce a display-safe summary of configuration state.

    Never includes actual key values -- only whether each provider is
    configured, and whether mock mode is active. Intended for use in the
    Streamlit status panel and in logs.
    """
    return {
        "gemini_api": "Configured" if settings.gemini_configured else "Not configured",
        "openrouter_api": "Configured" if settings.openrouter_configured else "Not configured",
        "mock_mode": "Enabled" if settings.mock_mode else "Disabled",
    }
