"""Text and citation normalization utilities.

Phase 1 kept this minimal: only generic, provider-agnostic string helpers.
Phase 3A adds the shared normalization layer needed to turn raw,
provider-shaped data (whatever a Gemini or OpenRouter payload happens to
look like) into the existing `schemas.models.Citation` / `AIResponse`
types, WITHOUT knowing anything about a specific provider's response
format yet.

Everything in this module is:
- pure Python (no network calls, no LLM calls);
- provider-agnostic (no Gemini/OpenRouter-specific parsing);
- non-fabricating: if a piece of data was not actually present, this
  module represents that as "unavailable" rather than inventing a
  plausible-looking value or silently treating it as "empty/none".

Provider-specific response parsing (turning an actual Gemini/OpenRouter
JSON payload into the raw dicts this module accepts) is out of scope
until those collectors are implemented in a later phase.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import ValidationError

from schemas.models import Citation


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing space.

    A small, genuinely generic helper that will be reused by every
    future collector and by the crawler, so it is safe to define now.
    """
    return " ".join(text.split())


def normalize_answer_text(answer: str | None) -> str | None:
    """Whitespace-normalize an AI answer while preserving its meaning.

    `None` stays `None` (an answer that was never returned is not the
    same thing as an empty string). An answer that is present but
    entirely whitespace normalizes to an empty string rather than being
    silently converted to `None`, since the caller may still want to
    distinguish "provider returned nothing" from "provider returned
    whitespace-only text".
    """
    if answer is None:
        return None
    return normalize_whitespace(answer)


def derive_domain_from_url(url: str) -> str | None:
    """Best-effort extraction of a bare domain (e.g. "example.com") from a
    URL string.

    Returns `None` rather than guessing if the URL can't be parsed or has
    no discernible network location -- this is a convenience derivation,
    never a fabrication.
    """
    try:
        netloc = urlparse(url).netloc
    except ValueError:
        return None
    if not netloc:
        return None
    # Strip a leading "www." for a slightly cleaner display domain, but
    # never invent a domain that wasn't in the URL.
    return netloc[4:] if netloc.startswith("www.") else netloc


def normalize_citation(raw: dict[str, str | None] | None) -> Citation | None:
    """Convert a raw, provider-agnostic citation mapping into a `Citation`.

    `raw` is expected to be a plain dict with any subset of the keys
    "url", "title", "domain" -- the shape every future collector's
    citation-extraction step is expected to produce before handing data
    to this layer.

    Returns `None` (not an empty `Citation`) when:
    - `raw` itself is `None` or empty, or
    - every one of "url"/"title"/"domain" is missing/empty.

    This preserves the schema's rule that absence of citation data must
    never be represented as a citation with empty values -- callers use
    the `None` return to know no usable citation was actually present.

    Never invents a title or domain that wasn't supplied. If a `domain`
    is missing but a usable `url` is present, the domain is derived from
    the URL (a derivation, not a fabrication). If the URL itself is
    malformed, the URL is dropped rather than raising, since a citation
    with a bad link is still better information than discarding the
    whole citation.
    """
    if not raw:
        return None

    url = (raw.get("url") or "").strip() or None
    title = (raw.get("title") or "").strip() or None
    domain = (raw.get("domain") or "").strip() or None

    if url is None and title is None and domain is None:
        return None

    if url and not domain:
        domain = derive_domain_from_url(url)

    try:
        return Citation(url=url, title=title, domain=domain)
    except ValidationError:
        # Most likely cause: `url` failed HttpUrl validation (malformed
        # link). Don't fabricate a corrected URL -- drop just the URL
        # field and keep whatever legitimate title/domain data remains.
        return Citation(url=None, title=title, domain=domain)


# Sentinel-free "unavailable" signal: `None` is used deliberately (rather
# than a custom sentinel object) so this stays simple Pydantic-friendly
# data. Documented explicitly here and on `normalize_citations` because
# it is easy to confuse with "no citations found".
CITATIONS_UNAVAILABLE = None


def normalize_citations(
    raw_citations: list[dict[str, str | None]] | None,
) -> list[Citation] | None:
    """Normalize a list of raw citation mappings into `Citation` objects.

    Distinguishes two different situations that must never be conflated:

    - `raw_citations is None` -> citation extraction was never attempted
      or the provider gives no way to know whether citations exist at
      all. Returns `None` ("unavailable"). Callers must NOT treat this
      the same as "this response has no citations".
    - `raw_citations == []` (or a list where every entry normalizes to
      nothing usable) -> extraction *was* attempted and found nothing.
      Returns `[]` ("no citations"), which matches `AIResponse.citations`
      default_factory of an empty list.

    Individual malformed/empty entries within a non-empty list are
    dropped (via `normalize_citation`) rather than fabricated.
    """
    if raw_citations is None:
        return CITATIONS_UNAVAILABLE

    normalized: list[Citation] = []
    for raw in raw_citations:
        citation = normalize_citation(raw)
        if citation is not None:
            normalized.append(citation)
    return normalized
