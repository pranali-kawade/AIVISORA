"""Deterministic JSON-LD / schema.org detector (Phase 5B).

Input: the JSON-LD blocks already extracted by ``crawler.parser`` (any
object exposing ``.valid`` and ``.data``). Output: a small record of which
schema.org ``@type`` values appear and which analysis-relevant properties
are present. It only *identifies* structured-data evidence -- it never
scores, judges quality, or generates recommendations. No network, no LLM,
no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# Types called out by the Phase 5B contract. Detection is NOT limited to
# these -- any @type found is reported -- but these are the ones the later
# analysis phases are expected to care about first.
KNOWN_TYPES = frozenset(
    {
        "Organization",
        "Product",
        "FAQPage",
        "Article",
        "WebPage",
        "BreadcrumbList",
        "LocalBusiness",
        "Service",
    }
)

# Properties worth recording when explicitly present on a node.
INTERESTING_PROPERTIES = frozenset(
    {
        "name",
        "description",
        "url",
        "image",
        "brand",
        "offers",
        "aggregateRating",
        "review",
        "mainEntity",
        "itemListElement",
        "question",
        "acceptedAnswer",
    }
)


@dataclass(frozen=True)
class SchemaEntity:
    """One JSON-LD node that carries a schema.org ``@type``."""

    type: str
    properties: tuple[str, ...]


@dataclass(frozen=True)
class SchemaEvidence:
    """Structured-data evidence found across a page's JSON-LD blocks."""

    types: tuple[str, ...]  # distinct @type values, first-seen order
    entities: tuple[SchemaEntity, ...]
    has_faq: bool


def _type_names(value: Any) -> list[str]:
    """Normalize an ``@type`` value (str or list, possibly a full URL)."""
    values = value if isinstance(value, list) else [value]
    names: list[str] = []
    for item in values:
        if isinstance(item, str) and item.strip():
            names.append(item.strip().split("/")[-1].split("#")[-1])
    return names


def detect_schema(blocks: Iterable[Any]) -> SchemaEvidence:
    types_order: list[str] = []
    entities: list[SchemaEntity] = []
    has_faq = False

    def visit(node: Any) -> None:
        nonlocal has_faq
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        node_types = _type_names(node.get("@type"))
        if node_types:
            props = tuple(k for k in INTERESTING_PROPERTIES if k in node)
            for name in node_types:
                if name not in types_order:
                    types_order.append(name)
                entities.append(SchemaEntity(type=name, properties=props))
                if name in ("FAQPage", "Question"):
                    has_faq = True
        if "acceptedAnswer" in node:
            has_faq = True

        for value in node.values():
            visit(value)

    for block in blocks:
        if getattr(block, "valid", False):
            visit(getattr(block, "data", None))

    return SchemaEvidence(types=tuple(types_order), entities=tuple(entities), has_faq=has_faq)
