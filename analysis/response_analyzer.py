"""Phase 9 -- populate ``AIResponse.brand_mentions`` from a collected answer.

Deterministic, exact-token matching for the explicitly supplied target and
competitor names only. No LLM, no network, no fuzzy/semantic matching, no
config. ``position`` and ``recommended`` are set only on clear structural /
lexical signals, otherwise ``None`` (never ``mentioned == recommended``).
"""

from __future__ import annotations

import re
from typing import Iterable

from analysis.attribution import response_is_usable
from schemas.models import AIResponse, BrandMention
from utils.normalization import normalize_whitespace

_RECOMMEND_RE = re.compile(
    r"\brecommend(?:s|ed|ing)?\b"
    r"|\b(?:best|top|great|good|ideal|solid|excellent|perfect)\s+(?:choice|option|pick|bet)\b",
    re.IGNORECASE,
)
# A numbered list marker ("1." / "2)") captures its rank; bullet markers do not.
_LIST_ITEM_RE = re.compile(r"^\s*(?:(\d+)[.)]|[-*•])\s+(.*)$")


def _norm(text: str) -> str:
    """Lower-cased, punctuation-stripped, whitespace-collapsed text."""
    return normalize_whitespace(re.sub(r"[^\w\s]", " ", text or "")).casefold()


def _numbered_items(answer: str) -> list[tuple[int, str]]:
    """(rank, normalized line text) for each numbered list line, but only when
    the answer has >= 2 numbered items (a single "1." is not a ranked list).
    """
    items = [
        (int(m.group(1)), _norm(m.group(2)))
        for line in answer.splitlines()
        if (m := _LIST_ITEM_RE.match(line)) and m.group(1)
    ]
    return items if len(items) >= 2 else []


def analyze_response(
    response: AIResponse,
    *,
    target_brand: str,
    competitors: Iterable[str] = (),
) -> AIResponse:
    """Return ``response`` with ``brand_mentions`` populated for every supplied
    brand found in the answer. Failed / empty responses are returned unchanged.
    """
    if not response_is_usable(response):
        return response

    answer = response.answer
    normalized = _norm(answer)
    chunks = [c for c in re.split(r"(?<=[.!?])\s+|\n+", answer) if c.strip()]
    numbered = _numbered_items(answer)

    seen: set[str] = set()
    mentions: list[BrandMention] = []
    for label in (target_brand, *competitors):
        key = _norm(label or "")
        if not key or key in seen:
            continue
        seen.add(key)
        pattern = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)")
        if not pattern.search(normalized):
            continue

        ranks = [rank for rank, line in numbered if pattern.search(line)]
        hits = [c for c in chunks if pattern.search(_norm(c))]
        mentions.append(
            BrandMention(
                brand_name=label,
                mentioned=True,
                position=min(ranks) if ranks else None,
                recommended=True if any(_RECOMMEND_RE.search(c) for c in hits) else None,
                context_snippet=normalize_whitespace(hits[0])[:200] if hits else None,
            )
        )

    return response.model_copy(update={"brand_mentions": mentions})
