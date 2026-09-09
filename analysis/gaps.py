"""Deterministic gap detection (Phase 7 -- DIAGNOSE).

Identifies *probable / observed* gaps by comparing already-acquired data:
normalized AI responses (Phase 3A), target and competitor website
``PageEvidence`` (Phase 5B), the schema detector (Phase 5B) and the
attribution results (``analysis.attribution``).

No LLM, no embeddings, no semantic similarity, no network, no scoring, no
recommendations. The same input always produces the same output.

Wording is deliberately conservative. A missing deterministic match is a
"probable gap" / "not observed in acquired evidence" -- never proof that a
website lacks information (CLAUDE.md section 6). Findings never claim to
know why a model produced an answer, and never make a causal claim.

Term comparison is exact-token only (lower-cased words of length >= 4,
minus a small stop-word list). There is no stemming or synonym handling, so
it over-reports rather than under-reports -- hence "probable".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Union

from analysis.attribution import AttributionSummary, CitationState
from analysis.visibility import ResponseRecord
from crawler.parser import PageEvidence
from crawler.schema_detector import detect_schema
from schemas.models import AIResponse


class GapCategory(str, Enum):
    CONTENT = "CONTENT_GAP"
    PRODUCT_INFORMATION = "PRODUCT_INFORMATION_GAP"
    FAQ = "FAQ_GAP"
    EVIDENCE = "EVIDENCE_GAP"
    CITATION = "CITATION_GAP"
    STRUCTURED_DATA = "STRUCTURED_DATA_GAP"
    COMPETITOR_CONTENT = "COMPETITOR_CONTENT_GAP"


class GapStatus(str, Enum):
    OBSERVED = "OBSERVED"          # a probable gap was observed from the evidence
    NOT_OBSERVED = "NOT_OBSERVED"  # the deterministic check ran and found no gap
    UNAVAILABLE = "UNAVAILABLE"    # the evidence needed to assess was not available


@dataclass(frozen=True)
class GapFinding:
    category: GapCategory
    status: GapStatus
    summary: str
    evidence: tuple[str, ...] = ()
    competitor: Optional[str] = None  # set for COMPETITOR_CONTENT_GAP findings


@dataclass(frozen=True)
class GapReport:
    target_brand: str
    competitors: tuple[str, ...]
    findings: tuple[GapFinding, ...]


# A small, standard English stop-word list.
_STOPWORDS = frozenset(
    "about above after again against also and any are because been before being between "
    "both but can cannot could did does doing down during each few for from further had "
    "has have having here how into its itself more most not now off once only other our "
    "out over own same should some such than that the their them then there these they "
    "this those through under until very was were what when where which while who whom "
    "why will with would you your".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_MAX_EVIDENCE_TERMS = 12

_PRODUCT_TERMS = frozenset(
    "price pricing product products specification specifications specs feature features "
    "model dimensions weight warranty shipping purchase capacity material".split()
)
_PRODUCT_SCHEMA_TYPES = frozenset({"product", "offer", "aggregaterating", "review"})
_FAQ_SCHEMA_TYPES = frozenset({"faqpage", "question"})
_FAQ_HEADING_TERMS = ("faq", "faqs", "frequently asked", "frequently-asked", "questions")
_COMMON_SCHEMA_TYPES = ("Organization", "Product", "FAQPage", "BreadcrumbList", "WebSite", "WebPage")


def _terms(*texts: Optional[str]) -> frozenset[str]:
    out: set[str] = set()
    for text in texts:
        for word in _WORD_RE.findall((text or "").casefold()):
            if len(word) >= 4 and word not in _STOPWORDS:
                out.add(word)
    return frozenset(out)


def _evidence_terms(pe: Optional[PageEvidence]) -> frozenset[str]:
    if pe is None:
        return frozenset()
    h = pe.headings
    return _terms(pe.title, pe.meta_description, *h.h1, *h.h2, *h.h3, *pe.sections)


def _has_evidence(pe: Optional[PageEvidence]) -> bool:
    return pe is not None and pe.content_present


def _schema_types(pe: Optional[PageEvidence]) -> tuple[str, ...]:
    if pe is None:
        return ()
    return tuple(pe.schema_types) if pe.schema_types else detect_schema(pe.json_ld).types


def _has_faq_evidence(pe: Optional[PageEvidence]) -> bool:
    if pe is None:
        return False
    if detect_schema(pe.json_ld).has_faq:
        return True
    if {t.casefold() for t in _schema_types(pe)} & _FAQ_SCHEMA_TYPES:
        return True
    headings = " ".join([*pe.headings.h1, *pe.headings.h2, *pe.headings.h3]).casefold()
    return any(term in headings for term in _FAQ_HEADING_TERMS)


def _has_product_evidence(pe: Optional[PageEvidence]) -> bool:
    if pe is None:
        return False
    if {t.casefold() for t in _schema_types(pe)} & _PRODUCT_SCHEMA_TYPES:
        return True
    return bool(_evidence_terms(pe) & _PRODUCT_TERMS)


def _sample(terms: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(terms)[:_MAX_EVIDENCE_TERMS])


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def analyze_gaps(
    responses: Iterable[Union[AIResponse, ResponseRecord]],
    *,
    attribution: AttributionSummary,
    target_evidence: Optional[PageEvidence] = None,
    competitor_evidence: Optional[Mapping[str, PageEvidence]] = None,
) -> GapReport:
    """Produce one :class:`GapFinding` for each of the seven categories
    (CONTENT, PRODUCT_INFORMATION, FAQ, EVIDENCE, CITATION, STRUCTURED_DATA)
    plus one COMPETITOR_CONTENT finding per competitor, in that fixed order.

    Pass the *same* ``responses`` sequence that was given to
    :func:`analysis.attribution.analyze_attribution` -- findings are aligned
    with ``attribution.responses`` by position.
    """
    records = [r if isinstance(r, ResponseRecord) else ResponseRecord(r) for r in responses]
    competitor_evidence = dict(competitor_evidence or {})

    tgt_ok = _has_evidence(target_evidence)
    tgt_terms = _evidence_terms(target_evidence)

    all_response_terms: set[str] = set()
    target_response_terms: set[str] = set()
    for record, ratt in zip(records, attribution.responses):
        if not ratt.usable:
            continue
        terms = _terms(record.response.answer)
        all_response_terms |= terms
        if ratt.target_mentioned:
            target_response_terms |= terms

    findings: list[GapFinding] = [
        _content_gap(tgt_ok, tgt_terms, all_response_terms, attribution.usable_responses),
        _product_gap(tgt_ok, target_evidence),
        _faq_gap(tgt_ok, target_evidence),
        _evidence_gap(tgt_ok, tgt_terms, target_response_terms, attribution.target_mention_count),
        _citation_gap(attribution),
        _structured_data_gap(tgt_ok, target_evidence),
    ]
    for brand in attribution.competitors:
        findings.append(_competitor_gap(brand, target_evidence, tgt_ok, tgt_terms, competitor_evidence.get(brand)))

    return GapReport(
        target_brand=attribution.target_brand,
        competitors=attribution.competitors,
        findings=tuple(findings),
    )


# ---------------------------------------------------------------------------
# per-category rules
# ---------------------------------------------------------------------------
def _content_gap(tgt_ok, tgt_terms, response_terms, usable_count) -> GapFinding:
    if not tgt_ok:
        return GapFinding(GapCategory.CONTENT, GapStatus.UNAVAILABLE,
                          "Target website evidence was not available for a content comparison.")
    if usable_count == 0:
        return GapFinding(GapCategory.CONTENT, GapStatus.UNAVAILABLE,
                          "No usable AI responses were available for a content comparison.")
    missing = response_terms - tgt_terms
    if missing:
        return GapFinding(GapCategory.CONTENT, GapStatus.OBSERVED,
                          "Terms present in the analyzed AI responses were not observed in the target "
                          "website evidence (probable content gap).", evidence=_sample(missing))
    return GapFinding(GapCategory.CONTENT, GapStatus.NOT_OBSERVED,
                      "All significant terms from the analyzed AI responses were also observed in the "
                      "target website evidence.")


def _product_gap(tgt_ok, target_evidence) -> GapFinding:
    if not tgt_ok:
        return GapFinding(GapCategory.PRODUCT_INFORMATION, GapStatus.UNAVAILABLE,
                          "Target website evidence was not available to check for product information.")
    if _has_product_evidence(target_evidence):
        return GapFinding(GapCategory.PRODUCT_INFORMATION, GapStatus.NOT_OBSERVED,
                          "Product-information signals (product structured data or product-related "
                          "terms) were observed in the target website evidence.")
    return GapFinding(GapCategory.PRODUCT_INFORMATION, GapStatus.OBSERVED,
                      "Product-information signals were not observed in the target website evidence "
                      "(probable product-information gap).")


def _faq_gap(tgt_ok, target_evidence) -> GapFinding:
    if not tgt_ok:
        return GapFinding(GapCategory.FAQ, GapStatus.UNAVAILABLE,
                          "Target website evidence was not available to check for FAQ evidence.")
    if _has_faq_evidence(target_evidence):
        return GapFinding(GapCategory.FAQ, GapStatus.NOT_OBSERVED,
                          "FAQ evidence was observed in the analyzed website evidence.")
    return GapFinding(GapCategory.FAQ, GapStatus.OBSERVED,
                      "FAQ evidence was not observed in the analyzed website evidence.")


def _evidence_gap(tgt_ok, tgt_terms, target_response_terms, target_mention_count) -> GapFinding:
    if not tgt_ok:
        return GapFinding(GapCategory.EVIDENCE, GapStatus.UNAVAILABLE,
                          "Target website evidence was not available to check for supporting evidence.")
    if target_mention_count == 0:
        return GapFinding(GapCategory.EVIDENCE, GapStatus.UNAVAILABLE,
                          "No analyzed AI response mentions the target brand, so a target evidence "
                          "gap could not be assessed.")
    missing = target_response_terms - tgt_terms
    if missing:
        return GapFinding(GapCategory.EVIDENCE, GapStatus.OBSERVED,
                          "AI responses that mention the target contain terms for which corresponding "
                          "supporting evidence was not observed in the target website evidence "
                          "(probable evidence gap).", evidence=_sample(missing))
    return GapFinding(GapCategory.EVIDENCE, GapStatus.NOT_OBSERVED,
                      "For AI responses that mention the target, all significant terms were also "
                      "observed in the target website evidence.")


def _citation_gap(attribution: AttributionSummary) -> GapFinding:
    usable = [p for p in attribution.responses if p.usable]
    if not usable:
        return GapFinding(GapCategory.CITATION, GapStatus.UNAVAILABLE,
                          "No usable AI responses were available to assess citation attribution.")
    if all(p.citation_state is CitationState.UNAVAILABLE for p in usable):
        return GapFinding(GapCategory.CITATION, GapStatus.UNAVAILABLE,
                          "Citation availability was unavailable for all analyzed responses; a "
                          "citation gap could not be assessed.")
    assessable = [p for p in usable if p.citation_state is not CitationState.UNAVAILABLE and p.target_mentioned]
    if not assessable:
        return GapFinding(GapCategory.CITATION, GapStatus.UNAVAILABLE,
                          "The target brand was not mentioned in any response for which citation "
                          "availability was known; a citation gap could not be assessed.")
    no_target_cite = sum(1 for p in assessable if not p.cites_target)
    competitor_cited = sum(1 for p in assessable if p.cites_competitor)
    if no_target_cite or competitor_cited:
        bits: list[str] = []
        if no_target_cite:
            bits.append(f"{no_target_cite} response(s): target mentioned but no target-domain citation observed")
        if competitor_cited:
            bits.append(f"{competitor_cited} response(s): a competitor-domain citation observed while the target was mentioned")
        return GapFinding(GapCategory.CITATION, GapStatus.OBSERVED,
                          "A citation attribution gap was observed in responses where the target was "
                          "mentioned and citation availability was known.", evidence=tuple(bits))
    return GapFinding(GapCategory.CITATION, GapStatus.NOT_OBSERVED,
                      "Where the target was mentioned and citation availability was known, "
                      "target-domain citations were observed and no competitor-only citation pattern "
                      "was observed.")


def _structured_data_gap(tgt_ok, target_evidence) -> GapFinding:
    if not tgt_ok:
        return GapFinding(GapCategory.STRUCTURED_DATA, GapStatus.UNAVAILABLE,
                          "Target website evidence was not available to check for structured data.")
    observed = _schema_types(target_evidence)
    if observed:
        lower = {t.casefold() for t in observed}
        not_seen = tuple(t for t in _COMMON_SCHEMA_TYPES if t.casefold() not in lower)
        evidence = tuple(f"observed: {t}" for t in observed) + tuple(f"not observed: {t}" for t in not_seen)
        return GapFinding(GapCategory.STRUCTURED_DATA, GapStatus.NOT_OBSERVED,
                          "Structured data was observed in the target website evidence.", evidence=evidence)
    return GapFinding(GapCategory.STRUCTURED_DATA, GapStatus.OBSERVED,
                      "No valid structured data (JSON-LD) was observed in the target website evidence "
                      "(probable structured-data gap).")


def _competitor_gap(brand, target_evidence, tgt_ok, tgt_terms, comp_ev) -> GapFinding:
    if comp_ev is None or not comp_ev.content_present:
        return GapFinding(GapCategory.COMPETITOR_CONTENT, GapStatus.UNAVAILABLE,
                          f"Competitor website evidence for {brand} was not available for comparison.",
                          competitor=brand)
    if not tgt_ok:
        return GapFinding(GapCategory.COMPETITOR_CONTENT, GapStatus.UNAVAILABLE,
                          f"Target website evidence was not available to compare against competitor {brand}.",
                          competitor=brand)
    advantages: list[str] = []
    extra_terms = _evidence_terms(comp_ev) - tgt_terms
    if extra_terms:
        advantages.append("topic/heading terms present for the competitor but not observed on the target: "
                          + ", ".join(sorted(extra_terms)[:_MAX_EVIDENCE_TERMS]))
    if _has_faq_evidence(comp_ev) and not _has_faq_evidence(target_evidence):
        advantages.append("FAQ evidence observed for the competitor but not observed on the target")
    if _has_product_evidence(comp_ev) and not _has_product_evidence(target_evidence):
        advantages.append("product-information signals observed for the competitor but not observed on the target")
    tgt_schema = {t.casefold() for t in _schema_types(target_evidence)}
    extra_schema = tuple(t for t in _schema_types(comp_ev) if t.casefold() not in tgt_schema)
    if extra_schema:
        advantages.append("structured-data types observed for the competitor but not observed on the target: "
                          + ", ".join(extra_schema))
    if advantages:
        return GapFinding(GapCategory.COMPETITOR_CONTENT, GapStatus.OBSERVED,
                          f"Competitor evidence advantage observed for {brand}.",
                          evidence=tuple(advantages), competitor=brand)
    return GapFinding(GapCategory.COMPETITOR_CONTENT, GapStatus.NOT_OBSERVED,
                      f"No competitor evidence advantage was observed for {brand} relative to the "
                      "target website evidence.", competitor=brand)
