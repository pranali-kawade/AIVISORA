"""Deterministic prompt templates for AEO Radar.

This module defines small, fixed sets of natural-language question
templates for each supported intent category. Templates are plain
strings with placeholders (e.g. ``{brand}``, ``{category}``) that
``prompts/generator.py`` will fill in later.

No randomness, no timestamps, no IDs, no API calls, and no LLM usage --
this module is pure static data plus a couple of trivial lookup helpers.
"""

from __future__ import annotations

# Supported intent categories, in a fixed, deterministic order.
INTENTS: list[str] = [
    "informational",
    "recommendation",
    "comparison",
    "use_case",
    "commercial",
]

# Placeholders that templates below are allowed to use. Kept here as a
# reference for prompts/generator.py -- not enforced at import time.
SUPPORTED_PLACEHOLDERS: list[str] = [
    "brand",
    "category",
    "product",
    "competitor",
    "use_case",
    "requirement",
]

# Fixed, ordered template sets. Order matters for determinism: given the
# same inputs, generator.py must always produce the same output.
#
# Several genuinely different question structures per intent, so a run
# probes different AI-search information needs rather than one question
# reworded. Templates that need only {category} are category-level
# questions (no brand named) -- they let a run test whether the brand is
# surfaced organically; {brand}/{competitor} templates are used where the
# intent requires them. A template whose placeholder has no supplied value
# is skipped by generator.py, never filled with a guess.
TEMPLATES: dict[str, list[str]] = {
    "informational": [
        "What are the main challenges people face with {category}?",
        "What should someone understand before choosing {category}?",
        "What factors most affect the quality of {category}?",
        "How do people typically evaluate {category}?",
    ],
    "recommendation": [
        "Which providers are most recommended for {category}?",
        "What are the top options to consider for {category}?",
        "Who is regarded as a leader in {category}?",
        "Which {category} options work best for {use_case}?",
        "Which {category} would you recommend for someone who needs {requirement}?",
    ],
    "comparison": [
        "How does {brand} compare with {competitor} for {category}?",
        "What are the key differences between {brand} and {competitor} in {category}?",
        "Between {brand} and {competitor}, which is better suited for {category}?",
    ],
    "use_case": [
        "What {category} do people rely on for demanding situations?",
        "Which {category} is considered dependable for everyday needs?",
        "What is a practical {category} choice for getting started?",
        "What is the best {category} option for {use_case}?",
        "Which {category} suits someone who needs {requirement}?",
    ],
    "commercial": [
        "What should someone weigh up before paying for {category}?",
        "Is it worth investing in premium {category}?",
        "What does {brand} offer for {category}?",
        "How can someone get started with {brand} for {category}?",
    ],
}


def get_templates(intent: str) -> list[str]:
    """Return the fixed template list for a given intent.

    Raises ``KeyError`` for an unsupported intent rather than silently
    returning an empty list, so misconfiguration surfaces immediately.
    """
    return TEMPLATES[intent]
