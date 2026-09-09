"""Deterministic attribution analysis (Phase 7 -- DIAGNOSE).

Compares normalized AI responses against target and competitor website
evidence to describe *observable* attribution patterns only:

* which brands a response mentions -- read from structured
  ``AIResponse.brand_mentions`` (never a blind substring scan of the answer);
* which citation URLs appear, and whether each points at the target domain,
  a competitor domain, or an external/other domain;
* whether citation information for a response is present, checked-but-empty,
  or unavailable.

No LLM, no network, no scoring, no recommendations. The same input always
produces the same output. Per CLAUDE.md section 6 the "citation information
unavailable" state is never collapsed into "no citation observed" -- they
are different :class:`CitationState` values.

This module reports *what was observed*. It never states or implies why a
model produced an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Union

from analysis.visibility import ResponseRecord
from crawler.parser import PageEvidence
from schemas.models import AIResponse, SourceType
from utils.normalization import derive_domain_from_url, normalize_whitespace


class CitationTarget(str, Enum):
    """Where a single citation's domain points."""

    TARGET = "target"
    COMPETITOR = "competitor"
    EXTERNAL = "external"


class CitationState(str, Enum):
    """Citation situation for one response."""

    PRESENT = "present"          # >= 1 citation observed
    NONE_OBSERVED = "none"       # availability known, zero citations observed
    UNAVAILABLE = "unavailable"  # citation availability was not assessed / not known


@dataclass(frozen=True)
class CitationRef:
    url: Optional[str]
    domain: Optional[str]
    points_to: CitationTarget
    competitor_brand: Optional[str] = None  # set only when points_to == COMPETITOR


@dataclass(frozen=True)
class ResponseAttribution:
    prompt_id: str
    source_type: SourceType
    is_observed_data: bool  # True only for GOOGLE_AIO_OBSERVED
    usable: bool
    target_mentioned: bool
    competitors_mentioned: tuple[str, ...]
    citation_state: CitationState
    citations: tuple[CitationRef, ...]
    cites_target: bool
    cites_competitor: bool
    cites_external: bool
    cited_competitor_brands: tuple[str, ...]


@dataclass(frozen=True)
class AttributionSummary:
    target_brand: str
    competitors: tuple[str, ...]
    total_responses: int
    usable_responses: int
    responses: tuple[ResponseAttribution, ...]  # one per input, input order preserved
    # Roll-ups below count USABLE responses only -- failed / no-answer
    # responses must not affect attribution counts (CLAUDE.md section 13).
    target_mention_count: int
    target_citation_count: int
    competitor_citation_count: int
    external_citation_count: int
    citation_present_count: int
    citation_none_count: int
    citation_unavailable_count: int
    cited_competitor_brands: tuple[str, ...]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _norm(text: Optional[str]) -> str:
    return normalize_whitespace(text or "").casefold()


def response_is_usable(response: AIResponse) -> bool:
    """A non-failed response carrying a real, non-empty answer.

    Same rule as ``analysis.visibility`` (CLAUDE.md section 13); re-stated
    here so this module does not depend on another module's private helper.
    """
    return (
        bool(getattr(response, "success", False))
        and isinstance(response.answer, str)
        and response.answer.strip() != ""
    )


def _domain_of(value: Optional[str]) -> Optional[str]:
    """Normalized bare domain for a URL or a bare host string, or None."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    resolved = derive_domain_from_url(text if "://" in text else "https://" + text)
    return resolved.casefold() if resolved else None


def _same_domain(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def _mentions_brand(response: AIResponse, brand: str) -> bool:
    key = _norm(brand)
    if not key:
        return False
    for bm in getattr(response, "brand_mentions", None) or []:
        name = getattr(bm, "brand_name", "") or ""
        if getattr(bm, "mentioned", False) is True and name.strip() and _norm(name) == key:
            return True
    return False


def _as_record(item: Union[AIResponse, ResponseRecord]) -> ResponseRecord:
    return item if isinstance(item, ResponseRecord) else ResponseRecord(item)


def _citation_availability_known(record: ResponseRecord) -> bool:
    """Mirror of the Phase 4 / Phase 6 rule: an explicit flag wins; otherwise
    a non-empty citations list counts as "known", an empty list counts as
    "unknown" (an empty list is never assumed to mean "checked, none found").
    """
    if record.citations_known is True:
        return True
    if record.citations_known is False:
        return False
    return len(record.response.citations) > 0


def _dedup(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def analyze_attribution(
    responses: Iterable[Union[AIResponse, ResponseRecord]],
    *,
    target_brand: str,
    competitors: Iterable[str] = (),
    target_evidence: Optional[PageEvidence] = None,
    competitor_evidence: Optional[Mapping[str, PageEvidence]] = None,
) -> AttributionSummary:
    """Describe the observable attribution pattern of ``responses`` for
    ``target_brand`` against ``competitors``.

    ``target_evidence`` / ``competitor_evidence`` supply the domains used to
    classify citation URLs. A competitor with no evidence entry has an
    unknown domain, so its citations can only be classified as external --
    reported conservatively, never guessed.
    """
    records = [_as_record(r) for r in responses]
    competitors = tuple(competitors)
    competitor_evidence = dict(competitor_evidence or {})

    target_domain = _domain_of(target_evidence.url) if target_evidence is not None else None
    competitor_domains: list[tuple[str, str]] = []
    for brand in competitors:
        ev = competitor_evidence.get(brand)
        dom = _domain_of(ev.url) if ev is not None else None
        if dom:
            competitor_domains.append((brand, dom))

    per: list[ResponseAttribution] = []
    for record in records:
        resp = record.response
        known = _citation_availability_known(record)

        seen: set[str] = set()
        crefs: list[CitationRef] = []
        for cit in resp.citations:
            url = str(cit.url) if getattr(cit, "url", None) is not None else None
            domain = _domain_of(getattr(cit, "domain", None)) or _domain_of(url)
            key = (url or "") + "|" + (domain or "")
            if key in seen:
                continue
            seen.add(key)
            points_to = CitationTarget.EXTERNAL
            competitor_brand: Optional[str] = None
            if _same_domain(domain, target_domain):
                points_to = CitationTarget.TARGET
            else:
                for brand, cdom in competitor_domains:
                    if _same_domain(domain, cdom):
                        points_to, competitor_brand = CitationTarget.COMPETITOR, brand
                        break
            crefs.append(CitationRef(url=url, domain=domain, points_to=points_to, competitor_brand=competitor_brand))

        if not known:
            state = CitationState.UNAVAILABLE
        elif not crefs:
            state = CitationState.NONE_OBSERVED
        else:
            state = CitationState.PRESENT

        per.append(
            ResponseAttribution(
                prompt_id=resp.prompt_id,
                source_type=resp.source_type,
                is_observed_data=resp.source_type is SourceType.GOOGLE_AIO_OBSERVED,
                usable=response_is_usable(resp),
                target_mentioned=_mentions_brand(resp, target_brand),
                competitors_mentioned=tuple(b for b in competitors if _mentions_brand(resp, b)),
                citation_state=state,
                citations=tuple(crefs),
                cites_target=any(c.points_to is CitationTarget.TARGET for c in crefs),
                cites_competitor=any(c.points_to is CitationTarget.COMPETITOR for c in crefs),
                cites_external=any(c.points_to is CitationTarget.EXTERNAL for c in crefs),
                cited_competitor_brands=_dedup(c.competitor_brand for c in crefs if c.competitor_brand),
            )
        )

    usable = [p for p in per if p.usable]
    return AttributionSummary(
        target_brand=target_brand,
        competitors=competitors,
        total_responses=len(records),
        usable_responses=len(usable),
        responses=tuple(per),
        target_mention_count=sum(1 for p in usable if p.target_mentioned),
        target_citation_count=sum(1 for p in usable if p.cites_target),
        competitor_citation_count=sum(1 for p in usable if p.cites_competitor),
        external_citation_count=sum(1 for p in usable if p.cites_external),
        citation_present_count=sum(1 for p in usable if p.citation_state is CitationState.PRESENT),
        citation_none_count=sum(1 for p in usable if p.citation_state is CitationState.NONE_OBSERVED),
        citation_unavailable_count=sum(1 for p in usable if p.citation_state is CitationState.UNAVAILABLE),
        cited_competitor_brands=_dedup(b for p in usable for b in p.cited_competitor_brands),
    )
