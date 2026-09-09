"""Deterministic prompt generation for AEO Radar.

Fills the fixed templates from `prompts/templates.py` with caller-supplied
values and returns a list of `schemas.models.PromptItem` objects.

No randomness, no UUIDs, no timestamps, no network/API calls, and no LLM
usage -- this is pure, deterministic string formatting plus ordering
logic. Given the same inputs, `generate_prompts` always returns the same
prompts, in the same order, with the same sequential IDs (P001, P002, ...).

Only values the caller actually supplies are ever used to fill a
template. If a template needs a placeholder the caller did not supply
(e.g. a comparison template needs a competitor but none were given), that
template is skipped entirely rather than inventing a value for it.
"""

from __future__ import annotations

import re
from itertools import product as cartesian_product

from prompts.templates import INTENTS, get_templates
from schemas.models import PromptItem

# Placeholders that take a single supplied value (brand/category are
# always required by the caller; product is optional).
_SINGLE_PLACEHOLDERS = {"brand", "category", "product"}

# Placeholders that are filled from a list of supplied values, producing
# one prompt per distinct value.
_MULTI_PLACEHOLDERS = {"competitor", "use_case", "requirement"}

_PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")


def _clean_list(values: list[str] | None) -> list[str]:
    """Strip, drop empties, and de-duplicate while preserving order."""
    if not values:
        return []
    cleaned = (v.strip() for v in values if v and v.strip())
    return list(dict.fromkeys(cleaned))


def _fill_template(
    template: str,
    single_values: dict[str, str],
    multi_values: dict[str, list[str]],
) -> list[str]:
    """Return every distinct filled-in string for one template.

    Returns an empty list if the template requires a placeholder that has
    no supplied value (the template is effectively skipped).
    """
    placeholders = set(_PLACEHOLDER_PATTERN.findall(template))

    # Defensive check: a placeholder outside the known, supported set
    # means the template can't be safely filled -- skip it rather than
    # guess.
    unknown = placeholders - _SINGLE_PLACEHOLDERS - _MULTI_PLACEHOLDERS
    if unknown:
        return []

    for name in placeholders & _SINGLE_PLACEHOLDERS:
        if name not in single_values:
            return []  # required single value (e.g. product) not supplied

    multi_needed = [name for name in ("competitor", "use_case", "requirement") if name in placeholders]
    for name in multi_needed:
        if not multi_values.get(name):
            return []  # required list placeholder has no supplied values

    if not multi_needed:
        return [template.format(**single_values)]

    value_lists = [multi_values[name] for name in multi_needed]
    filled: list[str] = []
    for combo in cartesian_product(*value_lists):
        fill = dict(single_values)
        fill.update(zip(multi_needed, combo))
        filled.append(template.format(**fill))
    return filled


def generate_prompts(
    brand: str,
    category: str,
    competitors: list[str] | None = None,
    product: str | None = None,
    use_cases: list[str] | None = None,
    requirements: list[str] | None = None,
    prompt_count: int = 10,
) -> list[PromptItem]:
    """Deterministically generate up to `prompt_count` PromptItem objects.

    Distribution rule: candidate prompts are generated per intent (in the
    fixed order informational -> recommendation -> comparison -> use_case
    -> commercial, interleaving each intent's templates so different
    question structures come first). Selection then proceeds in
    round-robin fashion across intents -- one
    candidate is taken from each intent in turn, repeating until
    `prompt_count` prompts are selected or every intent's candidates are
    exhausted. This spreads prompts as evenly as possible across intents
    (e.g. prompt_count=10 with all placeholders available yields exactly
    2 per intent) while staying fully deterministic: the same inputs
    always produce the same picks in the same order.

    Raises ValueError for invalid inputs, and if the supplied values do
    not yield enough unique prompts to satisfy `prompt_count`.
    """
    if not brand or not brand.strip():
        raise ValueError("brand must not be empty or whitespace-only")
    if not category or not category.strip():
        raise ValueError("category must not be empty or whitespace-only")
    if prompt_count <= 0:
        raise ValueError("prompt_count must be a positive integer greater than zero")

    single_values: dict[str, str] = {
        "brand": brand.strip(),
        "category": category.strip(),
    }
    if product and product.strip():
        single_values["product"] = product.strip()

    multi_values: dict[str, list[str]] = {
        "competitor": _clean_list(competitors),
        "use_case": _clean_list(use_cases),
        "requirement": _clean_list(requirements),
    }

    # Build deduplicated candidate prompt text per intent, in fixed order.
    # Templates within an intent are interleaved column-by-column so
    # selection rotates across different question structures first (for
    # diversity), then across the supplied placeholder values.
    candidates_by_intent: dict[str, list[str]] = {intent: [] for intent in INTENTS}
    seen_text: set[str] = set()
    for intent in INTENTS:
        expansions = [
            _fill_template(template, single_values, multi_values)
            for template in get_templates(intent)
        ]
        for column in range(max((len(e) for e in expansions), default=0)):
            for expansion in expansions:
                if column < len(expansion):
                    text = expansion[column]
                    if text not in seen_text:
                        seen_text.add(text)
                        candidates_by_intent[intent].append(text)

    # Round-robin selection across intents for even distribution.
    selected: list[tuple[str, str]] = []
    cursor = {intent: 0 for intent in INTENTS}
    made_progress = True
    while len(selected) < prompt_count and made_progress:
        made_progress = False
        for intent in INTENTS:
            if len(selected) >= prompt_count:
                break
            i = cursor[intent]
            pool = candidates_by_intent[intent]
            if i < len(pool):
                selected.append((intent, pool[i]))
                cursor[intent] = i + 1
                made_progress = True

    if len(selected) < prompt_count:
        raise ValueError(
            f"Not enough valid unique prompts could be generated from the supplied "
            f"inputs: requested {prompt_count}, only {len(selected)} available. "
            "Supply additional competitors, use_cases, requirements, and/or a "
            "product to unlock more templates."
        )

    return [
        PromptItem(prompt_id=f"P{i:03d}", prompt=text, intent=intent)
        for i, (intent, text) in enumerate(selected, start=1)
    ]
