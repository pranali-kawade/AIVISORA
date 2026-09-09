"""Deterministic AI Search Visibility metrics (Phase 6).

Pure Python analytics over normalized ``schemas.models.AIResponse`` records.
No LLM, no API/network calls, no crawling, no randomness, timestamps, env
reads, filesystem access, or hidden global state -- the same input always
produces an identical :class:`VisibilityMetrics`.

Rate convention
---------------
Every rate is a fraction in ``[0.0, 1.0]`` (multiply by 100 for a percent).
A rate is ``None`` when its denominator is 0 -- meaning "not measurable from
this benchmark set", which is deliberately distinct from a measured ``0.0``.

Scope
-----
These numbers describe the *selected benchmark responses only*; they are not
a claim about universal AI visibility. ``GOOGLE_AIO_OBSERVED`` results stay
flagged as observed data (``SourceVisibility.is_observed_data``), and
``OPENROUTER_FREE`` results are just that -- free open models, not
ChatGPT/Claude/Perplexity.

Input
-----
``compute_visibility`` accepts ``AIResponse`` objects, or
:class:`ResponseRecord` wrappers when the caller also knows whether citation
availability was assessed for a response (Phase 4 exposes this as
``AIOIngestResult.citations_status``). ``AIResponse.citations`` itself can
only be a list, so:

* ``citations_known=True``  -> availability was assessed (in the Citation
  Rate denominator; counted as observed iff ``len(citations) > 0``)
* ``citations_known=False`` -> availability unknown/unavailable (excluded)
* ``citations_known=None``  -> derived: a non-empty ``citations`` list means
  "known & observed"; an empty list is treated as availability-unknown
  (an empty list is never assumed to mean "checked and none found").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union

from schemas.models import AIResponse, SourceType
from utils.normalization import normalize_whitespace


# ---------------------------------------------------------------------------
# Small project-owned value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ratio:
    """A numerator / denominator and their fraction.

    ``rate`` is ``numerator / denominator`` in ``[0.0, 1.0]``, or ``None``
    when ``denominator == 0`` (not measurable -- never silently 0).
    """

    numerator: int
    denominator: int
    rate: Optional[float]

    @staticmethod
    def of(numerator: int, denominator: int) -> "Ratio":
        return Ratio(numerator, denominator, (numerator / denominator) if denominator else None)


@dataclass(frozen=True)
class AveragePosition:
    """Average ordinal position of the target brand among identified brand
    mentions, over the usable responses where it is mentioned.

    ``value`` is ``None`` when ``sample_size == 0`` (not measurable).
    """

    value: Optional[float]
    sample_size: int


@dataclass(frozen=True)
class SourceVisibility:
    """Target-brand visibility within a single ``SourceType``."""

    source: SourceType
    is_observed_data: bool  # True only for GOOGLE_AIO_OBSERVED
    total_responses: int
    usable_responses: int
    target_mentions: int
    mention_rate: Ratio
    recommendation_rate: Ratio
    citation_rate: Ratio


@dataclass(frozen=True)
class VisibilityMetrics:
    """The Phase 6 output. Rates follow the module's ``[0.0, 1.0]`` / ``None``
    convention.
    """

    target_brand: str
    competitors: tuple[str, ...]
    total_responses: int
    usable_responses: int
    mention_rate: Ratio
    recommendation_rate: Ratio
    average_mention_position: AveragePosition
    citation_rate: Ratio
    share_of_voice: dict[str, Ratio]  # brand label -> SOV; includes target + competitors
    by_source: dict[SourceType, SourceVisibility]


@dataclass(frozen=True)
class ResponseRecord:
    """An ``AIResponse`` plus the one piece of context the schema cannot
    carry: whether citation availability was actually assessed.
    """

    response: AIResponse
    citations_known: Optional[bool] = None


_Input = Union[AIResponse, ResponseRecord]


# ---------------------------------------------------------------------------
# Deterministic helpers (structured data only -- never blind substring search)
# ---------------------------------------------------------------------------
def _norm(name: str) -> str:
    return normalize_whitespace(name or "").casefold()


def _is_usable(response: AIResponse) -> bool:
    """A response with a real, non-empty answer that did not fail."""
    return bool(response.success) and isinstance(response.answer, str) and response.answer.strip() != ""


def _target_positions(response: AIResponse, target: str) -> list[int]:
    """1-based positions of the target brand in one response.

    Uses ``BrandMention.position`` when it is a valid int >= 1, otherwise the
    1-based ordinal of the mention among ``mentioned is True`` brand mentions
    in list order. Malformed rows (no name, ``mentioned`` not True) are
    skipped. The caller uses the smallest value = the earliest mention.
    """
    positions: list[int] = []
    ordinal = 0
    for bm in getattr(response, "brand_mentions", None) or []:
        name = getattr(bm, "brand_name", "") or ""
        if getattr(bm, "mentioned", False) is not True or not name.strip():
            continue
        ordinal += 1
        if _norm(name) == _norm(target):
            pos = getattr(bm, "position", None)
            positions.append(pos if isinstance(pos, int) and pos >= 1 else ordinal)
    return sorted(positions)


def _target_recommended(response: AIResponse, target: str) -> bool:
    """True only when a structured mention explicitly marks the target as
    recommended (``BrandMention.recommended is True``). A plain mention is
    never treated as a recommendation.
    """
    for bm in getattr(response, "brand_mentions", None) or []:
        name = getattr(bm, "brand_name", "") or ""
        if (
            getattr(bm, "mentioned", False) is True
            and name.strip()
            and _norm(name) == _norm(target)
            and getattr(bm, "recommended", None) is True
        ):
            return True
    return False


def _brand_counts(response: AIResponse, tracked: set[str]) -> dict[str, int]:
    """Structured mention counts (rows with ``mentioned is True``) for the
    tracked brands in one response -- the raw occurrence count used by Share
    of Voice.
    """
    counts: dict[str, int] = {}
    for bm in getattr(response, "brand_mentions", None) or []:
        name = getattr(bm, "brand_name", "") or ""
        if getattr(bm, "mentioned", False) is not True or not name.strip():
            continue
        key = _norm(name)
        if key in tracked:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _citation_state(record: ResponseRecord) -> tuple[bool, bool]:
    """Return ``(availability_known, citations_observed)`` for one usable
    response, honoring the Phase 4 None/[]/[...] distinction.
    """
    observed = len(record.response.citations) > 0
    if record.citations_known is True:
        return True, observed
    if record.citations_known is False:
        return False, False
    return (True, True) if observed else (False, False)


def _as_record(item: _Input) -> ResponseRecord:
    return item if isinstance(item, ResponseRecord) else ResponseRecord(item)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_visibility(
    responses: Iterable[_Input],
    *,
    target_brand: str,
    competitors: Iterable[str] = (),
) -> VisibilityMetrics:
    """Compute the six core AI Search Visibility metrics for ``target_brand``
    over ``responses`` (a benchmark set), plus competitor-level Share of
    Voice and a per-source breakdown.
    """
    records = [_as_record(item) for item in responses]
    competitors = tuple(competitors)

    # Tracked brands: target first, then competitors; de-duplicated by
    # normalized name, blanks dropped. Order is preserved and deterministic.
    tracked: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label in (target_brand, *competitors):
        key = _norm(label)
        if key in seen or (not label and label != target_brand):
            continue
        seen.add(key)
        tracked.append((label, key))
    tracked_keys = {key for _, key in tracked}

    total = len(records)
    usable = [r for r in records if _is_usable(r.response)]
    n_usable = len(usable)

    # Mention / recommendation / position over usable responses.
    per_positions = [(_target_positions(r.response, target_brand)) for r in usable]
    mentions = sum(1 for p in per_positions if p)
    recommendations = sum(1 for r in usable if _target_recommended(r.response, target_brand))
    earliest = [p[0] for p in per_positions if p]
    avg_position = AveragePosition(
        value=(sum(earliest) / len(earliest)) if earliest else None,
        sample_size=len(earliest),
    )

    # Citation rate (only usable responses with known availability).
    cit_known = cit_observed = 0
    for r in usable:
        known, observed = _citation_state(r)
        if known:
            cit_known += 1
            cit_observed += int(observed)

    # Share of Voice: raw structured occurrence counts across usable responses.
    sov_counts = {key: 0 for _, key in tracked}
    for r in usable:
        for key, count in _brand_counts(r.response, tracked_keys).items():
            sov_counts[key] += count
    sov_total = sum(sov_counts.values())
    share_of_voice = {label: Ratio.of(sov_counts[key], sov_total) for label, key in tracked}

    # Cross-model visibility: one bucket per SourceType present, in first-seen order.
    by_source: dict[SourceType, SourceVisibility] = {}
    for source in _ordered_sources(records):
        src = [r for r in records if r.response.source_type == source]
        src_usable = [r for r in src if _is_usable(r.response)]
        src_mentions = sum(1 for r in src_usable if _target_positions(r.response, target_brand))
        src_recs = sum(1 for r in src_usable if _target_recommended(r.response, target_brand))
        s_known = s_observed = 0
        for r in src_usable:
            known, observed = _citation_state(r)
            if known:
                s_known += 1
                s_observed += int(observed)
        by_source[source] = SourceVisibility(
            source=source,
            is_observed_data=source == SourceType.GOOGLE_AIO_OBSERVED,
            total_responses=len(src),
            usable_responses=len(src_usable),
            target_mentions=src_mentions,
            mention_rate=Ratio.of(src_mentions, len(src_usable)),
            recommendation_rate=Ratio.of(src_recs, len(src_usable)),
            citation_rate=Ratio.of(s_observed, s_known),
        )

    return VisibilityMetrics(
        target_brand=target_brand,
        competitors=competitors,
        total_responses=total,
        usable_responses=n_usable,
        mention_rate=Ratio.of(mentions, n_usable),
        recommendation_rate=Ratio.of(recommendations, n_usable),
        average_mention_position=avg_position,
        citation_rate=Ratio.of(cit_observed, cit_known),
        share_of_voice=share_of_voice,
        by_source=by_source,
    )


def _ordered_sources(records: list[ResponseRecord]) -> list[SourceType]:
    order: list[SourceType] = []
    for r in records:
        source = r.response.source_type
        if source not in order:
            order.append(source)
    return order
