"""Focused deterministic tests for the more specific Optimize wording.

Synthetic GapReports only -- no pipeline, no network, no LLM. Verifies that
every OBSERVED category yields a concrete WHAT/WHERE/HOW action tied to the
finding's own evidence, and that NOT_OBSERVED / UNAVAILABLE behaviour and
the priority / impact / category contract are unchanged.

Run: python3 tests_optimize_specificity_manual.py
"""

from __future__ import annotations

import sys

from analysis.gaps import GapCategory as C
from analysis.gaps import GapFinding, GapReport, GapStatus
from optimizer.action_planner import ActionKind, ImpactLevel, Priority, plan_actions

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, PASS if cond else FAIL, detail))


def rep(findings: list[GapFinding], brand: str = "Acme", competitors=("BrewCo",)) -> GapReport:
    return GapReport(target_brand=brand, competitors=tuple(competitors), findings=tuple(findings))


def obs(cat, summary, evidence=(), competitor=None) -> GapFinding:
    return GapFinding(cat, GapStatus.OBSERVED, summary, tuple(evidence), competitor)


def by_cat(plan, cat, competitor=None):
    for a in plan.actions:
        if a.category is cat and (competitor is None or a.competitor == competitor):
            return a
    return None


def whw(text: str) -> bool:
    return "WHAT:" in text and "WHERE:" in text and "HOW:" in text


NG = "does not guarantee any change in ai ranking"

# A full report with every category OBSERVED (CONTENT + EVIDENCE carry real
# term tokens; CITATION + COMPETITOR carry the finding's readable bits).
FULL = rep([
    obs(C.CONTENT, "Terms in the AI answers were not observed on the site (probable content gap).",
        ("cushioning", "midsole", "terrain")),
    obs(C.PRODUCT_INFORMATION, "Product-information signals were not observed in the site evidence."),
    obs(C.FAQ, "FAQ evidence was not observed in the analyzed website evidence."),
    obs(C.EVIDENCE, "AI responses mentioning the target contain unsupported terms (probable evidence gap).",
        ("warranty", "waterproof")),
    obs(C.CITATION, "A citation attribution gap was observed.",
        ("2 response(s): target mentioned but no target-domain citation observed",)),
    obs(C.STRUCTURED_DATA, "No valid structured data (JSON-LD) was observed in the site evidence."),
    obs(C.COMPETITOR_CONTENT, "Competitor evidence advantage observed for BrewCo.",
        ("FAQ evidence observed for the competitor but not observed on the target",
         "product-information signals observed for the competitor but not observed on the target"),
        competitor="BrewCo"),
])
plan = plan_actions(FULL)

# ---- CONTENT_GAP -------------------------------------------------------
a = by_cat(plan, C.CONTENT)
check("CONTENT: LOW / impact LOW / area 'content', unchanged contract",
      a.priority is Priority.LOW and a.expected_impact is ImpactLevel.LOW
      and a.area == "content" and a.kind is ActionKind.OPTIMIZATION)
check("CONTENT: title + description are specific to the observed terms, WHAT/WHERE/HOW present",
      "cushioning" in a.title and whw(a.description)
      and "cushioning" in a.description and "terrain" in a.description, a.title)
check("CONTENT: reason keeps the finding summary and adds a why-clause",
      FULL.findings[0].summary in a.reason and a.reason != FULL.findings[0].summary
      and "less on-site material" in a.reason)

# ---- PRODUCT_INFORMATION_GAP -----------------------------------------
p = by_cat(plan, C.PRODUCT_INFORMATION)
check("PRODUCT: MEDIUM / area 'product_information'",
      p.priority is Priority.MEDIUM and p.area == "product_information")
check("PRODUCT: recommends explicit attribute categories, WHAT/WHERE/HOW, no 'is missing' claim",
      whw(p.description) and "intended use case" in p.description
      and "technical specifications" in p.description and "selection criteria" in p.description
      and "where it genuinely applies" in p.description and "is missing" not in p.description.lower())
check("PRODUCT: themed hint uses only already-observed terms (from CONTENT/EVIDENCE findings)",
      "may be worth covering" in p.description
      and "cushioning" in p.description and "warranty" in p.description)

# ---- FAQ_GAP ---------------------------------------------------------
fq = by_cat(plan, C.FAQ)
check("FAQ: MEDIUM / area 'faq'", fq.priority is Priority.MEDIUM and fq.area == "faq")
check("FAQ: creates/expands an FAQ around use cases, WHAT/WHERE/HOW, 'do not invent facts'",
      whw(fq.description) and "FAQ" in fq.title
      and "how to choose" in fq.description and "do not invent facts" in fq.description)
check("FAQ: question themes derived from observed AI-answer terms",
      "Question themes suggested by" in fq.description and "midsole" in fq.description)

# ---- EVIDENCE_GAP --------------------------------------------------
e = by_cat(plan, C.EVIDENCE)
check("EVIDENCE: MEDIUM / area 'evidence'", e.priority is Priority.MEDIUM and e.area == "evidence")
check("EVIDENCE: verifiable supporting evidence, tied to the unsupported terms, WHAT/WHERE/HOW",
      whw(e.description) and "verifiable supporting evidence" in e.description
      and "warranty" in e.description and "certifications" in e.description
      and "generic where no specific source is identified" in e.description)

# ---- CITATION_GAP -------------------------------------------------
c = by_cat(plan, C.CITATION)
check("CITATION: HIGH / impact UNKNOWN / area 'citation', no-guarantee wording kept",
      c.priority is Priority.HIGH and c.expected_impact is ImpactLevel.UNKNOWN
      and c.area == "citation" and "outside the site owner's control" in c.description)
check("CITATION: reason surfaces the observed attribution pattern",
      "no target-domain citation observed" in c.reason
      and FULL.findings[4].summary in c.reason and whw(c.description))

# ---- STRUCTURED_DATA_GAP ----------------------------------------
sd = by_cat(plan, C.STRUCTURED_DATA)
check("STRUCTURED_DATA: MEDIUM / area 'structured_data', does not generate markup / no visibility guarantee",
      sd.priority is Priority.MEDIUM and sd.area == "structured_data"
      and "it does not generate the markup" in sd.description
      and "does not guarantee AI visibility" in sd.description and whw(sd.description))
sd_present = by_cat(plan_actions(rep([
    obs(C.STRUCTURED_DATA, "No JSON-LD observed."),
    GapFinding(C.PRODUCT_INFORMATION, GapStatus.NOT_OBSERVED, "product signals observed"),
    GapFinding(C.FAQ, GapStatus.NOT_OBSERVED, "faq observed"),
])), C.STRUCTURED_DATA)
sd_absent = by_cat(plan_actions(rep([
    obs(C.STRUCTURED_DATA, "No JSON-LD observed."),
    obs(C.PRODUCT_INFORMATION, "no product signals"),
    obs(C.FAQ, "no faq"),
])), C.STRUCTURED_DATA)
check("STRUCTURED_DATA: recommended types follow the report's sibling findings",
      "AggregateRating" in sd_present.description and "FAQPage" in sd_present.description
      and "AggregateRating" not in sd_absent.description
      and "only once that content exists" in sd_absent.description)

# ---- COMPETITOR_CONTENT_GAP -----------------------------------
cc = by_cat(plan, C.COMPETITOR_CONTENT, "BrewCo")
check("COMPETITOR: HIGH / area 'competitor_content' / brand in title / comparative + non-fabrication guard",
      cc.priority is Priority.HIGH and cc.area == "competitor_content"
      and "BrewCo" in cc.title and whw(cc.description)
      and "not observed on the target" in cc.description
      and "do not copy" in cc.description and "treat anything else as unverified" in cc.description)
check("COMPETITOR: reason is explicitly comparative and names the competitor",
      "BrewCo" in cc.reason and FULL.findings[6].summary in cc.reason)

# ---- status semantics unchanged -----------------------------
notobs = rep([GapFinding(c_, GapStatus.NOT_OBSERVED, "no gap") for c_ in C])
check("NOT_OBSERVED -> no action (unchanged)", plan_actions(notobs).actions == ())
unavail = rep([GapFinding(C.FAQ, GapStatus.UNAVAILABLE, "evidence not available")])
check("UNAVAILABLE -> no action by default (unchanged)", plan_actions(unavail).actions == ())
optin = plan_actions(unavail, include_evidence_collection=True)
check("UNAVAILABLE + opt-in -> one EVIDENCE_COLLECTION action, LOW/UNKNOWN, 'not an optimization fix'",
      len(optin.actions) == 1 and optin.actions[0].kind is ActionKind.EVIDENCE_COLLECTION
      and optin.actions[0].priority is Priority.LOW
      and optin.actions[0].expected_impact is ImpactLevel.UNKNOWN
      and "not an optimization fix" in optin.actions[0].description)

# ---- plan-level invariants ----------------------------------
check("every OBSERVED category produced exactly one action (7 total)",
      {x.category for x in plan.actions} == set(C) and len(plan.actions) == 7)
ranks = [{"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x.priority.value] for x in plan.actions]
check("actions ordered HIGH -> MEDIUM -> LOW (unchanged)", ranks == sorted(ranks))
check("deterministic: same report -> identical plan", plan_actions(FULL) == plan == plan_actions(FULL))
check("no action promises a guaranteed AI outcome",
      all(NG in x.description.lower() for x in plan.actions)
      and not any(w in (x.title + " " + x.description).lower()
                  for x in plan.actions
                  for w in ("will rank", "guarantees ", "ensures citation", "guaranteed visibility")))
check("every reason keeps its finding summary but is richer than it",
      all(any(f.summary in x.reason and x.reason != f.summary for f in FULL.findings)
          for x in plan.actions))

print("\n=== Optimize specificity ===")
p_ = f_ = 0
for n, s, d in results:
    p_ += s == PASS
    f_ += s == FAIL
    print(f"[{s}] {n}" + (f" -- {d}" if d and s == FAIL else ""))
print(f"\nTOTAL: {p_} passed, {f_} failed")
print("NETWORK / LLM CALLS = 0 (synthetic GapReports only)")
sys.exit(0 if f_ == 0 else 1)
