"""Deterministic action planner (Phase 8 -- OPTIMIZE).

Converts the observable / probable gaps produced by Phase 7
(``analysis.gaps.GapReport``) into a small, prioritized set of optimization
actions. It answers "what should the website owner address based on the
detected gaps?" -- it never claims an action will guarantee AI ranking,
recommendation, citation, or visibility (CLAUDE.md section 1, section 6).

No LLM, no network, no embeddings, no semantic similarity, no scoring
framework. The same ``GapReport`` always yields the same ``ActionPlan``.

Gap-status handling (CLAUDE.md / Phase 7 contract):
* OBSERVED     -> one optimization action.
* NOT_OBSERVED -> no action.
* UNAVAILABLE  -> no optimization action. Optionally (opt-in) a clearly
  labelled evidence-collection action, never an optimization "fix".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from analysis.gaps import GapCategory, GapFinding, GapReport, GapStatus


class Priority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ImpactLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class ActionKind(str, Enum):
    OPTIMIZATION = "OPTIMIZATION"            # address an OBSERVED gap
    EVIDENCE_COLLECTION = "EVIDENCE_COLLECTION"  # gather input an UNAVAILABLE finding lacked


@dataclass(frozen=True)
class OptimizationAction:
    category: GapCategory
    area: str                 # implementation area, e.g. "content", "faq"
    title: str
    description: str
    reason: str               # the Phase 7 finding's own evidence-based wording
    priority: Priority
    expected_impact: ImpactLevel
    kind: ActionKind
    competitor: Optional[str] = None  # set for competitor-content actions


@dataclass(frozen=True)
class ActionPlan:
    target_brand: str
    competitors: tuple[str, ...]
    actions: tuple[OptimizationAction, ...]  # ordered HIGH -> MEDIUM -> LOW, then Phase 7 finding order


# --- deterministic lookup tables -------------------------------------------
# Implementation area per gap category (1:1 with the category, but reported
# explicitly because it names *where the owner acts*, not the gap taxonomy).
_AREA = {
    GapCategory.CONTENT: "content",
    GapCategory.PRODUCT_INFORMATION: "product_information",
    GapCategory.FAQ: "faq",
    GapCategory.EVIDENCE: "evidence",
    GapCategory.CITATION: "citation",
    GapCategory.STRUCTURED_DATA: "structured_data",
    GapCategory.COMPETITOR_CONTENT: "competitor_content",
}

# Priority tiers.  CITATION and COMPETITOR_CONTENT gaps are the ones most
# directly tied to how AI systems attribute and compare sources -> HIGH.
# Concrete on-page information / structure gaps -> MEDIUM.  The content
# term-overlap comparison is the broadest, most over-reporting signal -> LOW.
_PRIORITY = {
    GapCategory.CITATION: Priority.HIGH,
    GapCategory.COMPETITOR_CONTENT: Priority.HIGH,
    GapCategory.PRODUCT_INFORMATION: Priority.MEDIUM,
    GapCategory.FAQ: Priority.MEDIUM,
    GapCategory.EVIDENCE: Priority.MEDIUM,
    GapCategory.STRUCTURED_DATA: Priority.MEDIUM,
    GapCategory.CONTENT: Priority.LOW,
}
_PRIORITY_RANK = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}

# Qualitative expected impact.  "UNKNOWN" where the outcome is outside the
# owner's deterministic control (whether an AI system cites a page).
_IMPACT = {
    GapCategory.CITATION: ImpactLevel.UNKNOWN,
    GapCategory.COMPETITOR_CONTENT: ImpactLevel.MEDIUM,
    GapCategory.PRODUCT_INFORMATION: ImpactLevel.MEDIUM,
    GapCategory.FAQ: ImpactLevel.MEDIUM,
    GapCategory.EVIDENCE: ImpactLevel.MEDIUM,
    GapCategory.STRUCTURED_DATA: ImpactLevel.MEDIUM,
    GapCategory.CONTENT: ImpactLevel.LOW,
}

_NO_GUARANTEE = (
    " It does not guarantee any change in AI ranking, recommendation, citation, or visibility."
)


def _terms_phrase(terms: tuple[str, ...], limit: int = 8) -> str:
    """A short, comma-joined preview of already-observed evidence terms."""
    return ", ".join(list(terms)[:limit])


def _schema_types_for(product_status, faq_status) -> str:
    """schema.org types to recommend, chosen from what the report's own
    sibling findings show the site already has (NOT_OBSERVED product/FAQ
    gap == that content is present). Never invents content.
    """
    base = ["Organization", "WebSite", "BreadcrumbList"]
    ready: list[str] = []
    if product_status is GapStatus.NOT_OBSERVED:
        ready += ["Product", "Offer", "Review", "AggregateRating"]
    if faq_status is GapStatus.NOT_OBSERVED:
        ready.append("FAQPage")
    if ready:
        return ", ".join(base + ready)
    return ", ".join(base) + " (add Product/Offer/Review or FAQPage only once that content exists on the page)"


def _optimization_action(
    finding: GapFinding,
    *,
    observed_terms: tuple[str, ...] = (),
    product_status=None,
    faq_status=None,
) -> OptimizationAction:
    """Turn one OBSERVED finding into a concrete action that answers WHAT /
    WHERE / HOW (``description``) and WHY + what triggered it (``reason``),
    grounded only in the finding's own evidence and the report's sibling
    finding statuses -- no fabricated website facts.
    """
    category = finding.category
    competitor = finding.competitor
    finding_terms = _terms_phrase(finding.evidence)   # CONTENT / EVIDENCE: real observed tokens
    theme_terms = _terms_phrase(observed_terms)       # cross-finding themes (FAQ / product hints)
    ev_sentences = "; ".join(finding.evidence)        # CITATION / COMPETITOR: readable bits

    if category is GapCategory.CONTENT:
        focus = f": {finding_terms}" if finding_terms else ""
        title = (
            f"Add website content covering AI-answer topics: {finding_terms}"
            if finding_terms else "Add website content covering the topics the AI answers raise"
        )
        description = (
            f"WHAT: add or expand on-site content that explicitly addresses the topics that appear "
            f"in the analyzed AI answers but are absent from the acquired site evidence{focus}. "
            f"WHERE: on the most relevant existing category, product, or resource pages, or a new "
            f"topic page where none fits. HOW: give each topic its own clearly worded heading and a "
            f"factual section (definitions, use cases, comparisons, specifications) rather than "
            f"marketing copy, so the wording appears in both headings and body text."
        )
        why = (
            "The analyzed AI answers discuss these topics while the acquired site evidence does not "
            "surface them, leaving less on-site material for AI systems to draw on."
        )
    elif category is GapCategory.PRODUCT_INFORMATION:
        hint = f" Topics raised in the analyzed AI answers that may be worth covering: {theme_terms}." if theme_terms else ""
        title = "Add a structured product-information section to the target site"
        description = (
            "WHAT: add an explicit, structured product-information block covering the attributes "
            "buyers use to choose in this category - intended use case, target user, key features, "
            "technical specifications, differentiators, and selection criteria - for each product or "
            "the product range. WHERE: on product and category/landing pages as a dedicated, clearly "
            "headed section (for example 'Specifications', \"Who it's for\", 'How to choose'). HOW: "
            "present the attributes as labelled fields or a specification table with concrete values, "
            "and include an attribute only where it genuinely applies." + hint
        )
        why = (
            "Product-information signals (product structured data or product-related terms) were not "
            "observed in the acquired site evidence, so AI systems have limited explicit "
            "product-selection information to extract."
        )
    elif category is GapCategory.FAQ:
        hint = f" Question themes suggested by topics seen in the analyzed AI answers: {theme_terms}." if theme_terms else ""
        title = "Add or expand an FAQ section for the audited product or service"
        description = (
            "WHAT: create or expand an FAQ that answers the recurring questions users ask about this "
            "product or service and its use cases - suitability (\"is it right for ...?\"), how to "
            "choose between options, setup and usage, limitations, comparisons, and pricing or "
            "availability. WHERE: a dedicated FAQ page and/or an FAQ block on the most relevant "
            "product and category pages. HOW: use genuine question-and-answer pairs with short, "
            "factual answers, and answer only what the site can substantiate - do not invent facts."
            + hint
        )
        why = (
            "No FAQ evidence was observed in the acquired site evidence, so AI systems have no "
            "structured question-and-answer material from the site for these queries."
        )
    elif category is GapCategory.EVIDENCE:
        focus = f", in particular {finding_terms}" if finding_terms else ""
        title = "Add verifiable supporting evidence for the site's key claims"
        description = (
            f"WHAT: for the important claims the site makes{focus}, add verifiable supporting "
            f"evidence - specifications, test or measurement data, comparisons, certifications, "
            f"standards compliance, or links to authoritative third-party references. WHERE: next to "
            f"each claim on the relevant product, specification, or documentation page. HOW: state "
            f"the claim and then the concrete evidence (numbers, method, source) beside it, and keep "
            f"the wording generic where no specific source is identified."
        )
        why = (
            "AI answers that mention the target reference these topics while corresponding supporting "
            "detail was not observed on the site, so those claims are weakly substantiated in the "
            "material available to AI systems."
        )
    elif category is GapCategory.CITATION:
        title = "Strengthen authoritative, directly citable pages on the target domain"
        description = (
            "WHAT: make sure the key claims and facts about the target live on stable, authoritative "
            "pages that state them plainly - clear canonical URLs, descriptive titles and headings, "
            "self-contained factual statements, and visible publication or update dates. WHERE: the "
            "primary product, specification, comparison, and 'about' pages on the target domain. HOW: "
            "consolidate scattered claims onto a small number of durable reference pages, give each a "
            "stable canonical URL and an unambiguous heading, and make each important statement "
            "quotable on its own. Whether an AI system cites these pages is outside the site owner's "
            "control."
        )
        why = (
            (f"Observed pattern: {ev_sentences}. " if ev_sentences else "")
            + "Where the target is mentioned, AI systems are not attributing to target-domain pages "
            "(and in some responses attribute to a competitor domain instead), which points to "
            "weaker or less quotable source material on the target domain."
        )
    elif category is GapCategory.STRUCTURED_DATA:
        types = _schema_types_for(product_status, faq_status)
        title = "Add schema.org structured data (JSON-LD) matching existing content"
        description = (
            f"WHAT: add valid schema.org JSON-LD for content the site already has: {types}. WHERE: "
            f"in the head or body of the corresponding pages, one primary type per page. HOW: mark "
            f"up only facts that are already visible on the page, and validate with a schema "
            f"validator before publishing. This step recommends the types; it does not generate the "
            f"markup, and structured-data markup alone does not guarantee AI visibility."
        )
        why = (
            "No valid structured data (JSON-LD) was observed in the acquired site evidence, so AI "
            "systems must infer structure from raw text."
        )
    else:  # COMPETITOR_CONTENT
        title = f"Match {competitor}'s content coverage where the target site is thinner"
        description = (
            f"WHAT: close the specific coverage differences observed between {competitor}'s site "
            f"evidence and the target's - {ev_sentences}. WHERE: on the target's equivalent product, "
            f"category, or resource pages. HOW: for each observed difference add the equivalent "
            f"on-site material - a topic section, an FAQ block, explicit product attributes, or "
            f"structured data - using only facts the target can substantiate, and do not copy "
            f"{competitor}'s claims. Only the differences listed here are supported by the "
            f"comparison; treat anything else as unverified."
        )
        why = (
            f"{competitor}'s acquired evidence exposes this coverage while the target's does not, so "
            f"AI systems comparing sources have less to work with for the target on these points."
        )

    return OptimizationAction(
        category=category,
        area=_AREA[category],
        title=title,
        description=description + _NO_GUARANTEE,
        reason=f"{finding.summary} {why}",
        priority=_PRIORITY[category],
        expected_impact=_IMPACT[category],
        kind=ActionKind.OPTIMIZATION,
        competitor=competitor,
    )


def _evidence_collection_action(finding: GapFinding) -> OptimizationAction:
    area = _AREA[finding.category]
    competitor = finding.competitor
    scope = f"competitor {competitor}" if competitor else "the target"
    return OptimizationAction(
        category=finding.category,
        area=area,
        title=f"Collect the missing input evidence for {area} analysis",
        description=(
            f"Acquire or verify the website evidence for {scope} that this finding lacked, so the "
            f"{area} gap can be assessed. This is an evidence-collection step, not an optimization fix."
        ),
        reason=finding.summary,
        priority=Priority.LOW,
        expected_impact=ImpactLevel.UNKNOWN,
        kind=ActionKind.EVIDENCE_COLLECTION,
        competitor=competitor,
    )


def plan_actions(report: GapReport, *, include_evidence_collection: bool = False) -> ActionPlan:
    """Build a prioritized :class:`ActionPlan` from a Phase 7 ``GapReport``.

    Only OBSERVED findings produce optimization actions. UNAVAILABLE findings
    are skipped unless ``include_evidence_collection`` is set, in which case
    each yields one clearly labelled EVIDENCE_COLLECTION action.
    """
    # Deterministic cross-finding context for more specific wording:
    # first-seen status per category, and the real terms already observed in
    # the CONTENT / EVIDENCE findings (used only as themed hints).
    status_by_category: dict[GapCategory, GapStatus] = {}
    for finding in report.findings:
        status_by_category.setdefault(finding.category, finding.status)
    observed_terms = tuple(sorted({
        term
        for finding in report.findings
        if finding.status is GapStatus.OBSERVED
        and finding.category in (GapCategory.CONTENT, GapCategory.EVIDENCE)
        for term in finding.evidence
    }))[:10]
    product_status = status_by_category.get(GapCategory.PRODUCT_INFORMATION)
    faq_status = status_by_category.get(GapCategory.FAQ)

    built: list[tuple[int, OptimizationAction]] = []
    for index, finding in enumerate(report.findings):
        if finding.status is GapStatus.OBSERVED:
            built.append((index, _optimization_action(
                finding,
                observed_terms=observed_terms,
                product_status=product_status,
                faq_status=faq_status,
            )))
        elif finding.status is GapStatus.UNAVAILABLE and include_evidence_collection:
            built.append((index, _evidence_collection_action(finding)))

    # De-duplicate identical actions (frozen dataclass -> hashable), keeping
    # the first occurrence, then order HIGH -> MEDIUM -> LOW, ties by finding order.
    seen: set[OptimizationAction] = set()
    unique: list[tuple[int, OptimizationAction]] = []
    for index, action in built:
        if action not in seen:
            seen.add(action)
            unique.append((index, action))
    unique.sort(key=lambda pair: (_PRIORITY_RANK[pair[1].priority], pair[0]))

    return ActionPlan(
        target_brand=report.target_brand,
        competitors=report.competitors,
        actions=tuple(action for _, action in unique),
    )
