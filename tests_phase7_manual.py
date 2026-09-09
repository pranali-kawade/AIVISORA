"""Manual Phase 7 verification: Attribution & Gap Analysis (DIAGNOSE).

Not a pytest suite (project manual-test style). Deterministic in-memory
fixtures only -- no network, no API, no LLM. Asserts actual values, then
re-runs the earlier manual phases as a regression check.

Run: python3 tests_phase7_manual.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from schemas.models import AIResponse, BrandMention, Citation, SourceType
from crawler.parser import Headings, JsonLdBlock, PageEvidence
from analysis.visibility import ResponseRecord
from analysis.attribution import (
    AttributionSummary,
    CitationState,
    CitationTarget,
    ResponseAttribution,
    analyze_attribution,
)
from analysis.gaps import GapCategory, GapReport, GapStatus, analyze_gaps

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


# --- fixture builders ----------------------------------------------------
def bm(name, mentioned=True, recommended=None, position=None):
    return BrandMention(brand_name=name, mentioned=mentioned, recommended=recommended, position=position)


def resp(source, mentions=(), *, answer="An answer about camping coffee makers.", success=True, citations=None, prompt_id="p"):
    kw = dict(source_type=source, prompt_id=prompt_id, prompt="q", brand_mentions=list(mentions))
    if success:
        kw.update(success=True, answer=answer)
    else:
        kw.update(success=False, answer=None, error_message="collection failed")
    if citations is not None:
        kw["citations"] = [Citation(url=u) for u in citations]
    return AIResponse(**kw)


def pe(url, *, title=None, meta=None, h1=(), h2=(), h3=(), sections=(), schema_types=(), json_ld=(), content=True):
    return PageEvidence(
        url=url, title=title, meta_description=meta,
        headings=Headings(h1=tuple(h1), h2=tuple(h2), h3=tuple(h3)),
        sections=tuple(sections), internal_links=(), external_links=(),
        canonical=None, robots=None, json_ld=tuple(json_ld), schema_types=tuple(schema_types),
        content_present=content,
    )


G, O, A = SourceType.LIVE_GEMINI, SourceType.OPENROUTER_FREE, SourceType.GOOGLE_AIO_OBSERVED

# website evidence
TARGET_FULL = pe(
    "https://www.Acme.Example/coffee",
    title="Acme Camping Coffee Makers",
    meta="Acme pour over and french press camping coffee makers.",
    h1=("Acme Camping Coffee Makers",),
    h2=("Features", "Frequently asked questions", "Pricing"),
    sections=("Acme makes durable pour over and french press gear.", "Specifications and warranty details.",
              "Price starts at 29.99 with free shipping."),
    schema_types=("Organization", "Product", "Offer", "FAQPage", "BreadcrumbList"),
)
TARGET_THIN = pe("https://acme.example/", h1=("Acme",), sections=("Welcome to Acme.",))
TARGET_EMPTY = pe("https://acme.example/", content=False)

COMP_STRONG = pe(
    "https://brewco.example/gear",
    title="BrewCo Backpacking Coffee",
    h1=("BrewCo Backpacking Coffee",),
    h2=("Product specifications", "FAQ"),
    sections=("BrewCo aeropress and percolator options with titanium build.",
              "Price, warranty and shipping information."),
    schema_types=("Organization", "Product", "FAQPage"),
)
COMP_EQUAL = pe("https://campcup.example/", h1=("Acme",), sections=("Welcome to Acme.",))  # same evidence terms as TARGET_THIN

TARGET_CIT = ["https://acme.example/guide", "https://www.acme.example/specs"]
COMP_CIT = ["https://brewco.example/review"]
EXT_CIT = ["https://wirecutter.example/best-camping-coffee"]

R_TARGET = resp(G, [bm("Acme", position=1)], answer="Acme makes reliable pour over french press camping coffee makers.", citations=TARGET_CIT)
R_COMP = resp(O, [bm("BrewCo", position=1), bm("Acme", position=2)], answer="BrewCo aeropress and Acme percolator gear.", citations=COMP_CIT)
R_EXT = resp(G, [bm("Acme", position=1)], answer="Acme is often listed among camping coffee makers.", citations=EXT_CIT)
R_DUPES = resp(O, [bm("Acme", position=1)], citations=["https://acme.example/guide", "https://acme.example/guide"])
R_NONE_KNOWN = ResponseRecord(resp(O, [bm("Acme", position=1)]), citations_known=True)     # citations == [] checked
R_UNAVAIL = ResponseRecord(resp(A, [bm("Acme", position=1)]), citations_known=False)       # availability unknown
R_FAILED = resp(G, [bm("Acme", position=1)], success=False)
R_NOANSWER = AIResponse(source_type=G, prompt_id="p", prompt="q", success=True, answer=None)
R_NO_TARGET = resp(O, [bm("BrewCo", position=1)], answer="BrewCo makes good gear.")
R_MULTI_COMP = resp(G, [bm("BrewCo", position=1), bm("CampCup", position=2), bm("Acme", position=3)])

CORE = [R_TARGET, R_COMP, R_EXT, R_NONE_KNOWN, R_UNAVAIL, R_FAILED, R_NOANSWER]
COMP_EV = {"BrewCo": COMP_STRONG, "CampCup": COMP_EQUAL}
attr = analyze_attribution(CORE, target_brand="Acme", competitors=["BrewCo", "CampCup"],
                           target_evidence=TARGET_THIN, competitor_evidence=COMP_EV)

# ============================ ATTRIBUTION ================================
check("1. empty response list -> zeroed AttributionSummary, no crash",
      (lambda s: s.total_responses == 0 and s.usable_responses == 0 and s.responses == ()
       and s.target_mention_count == 0 and s.cited_competitor_brands == ())(
          analyze_attribution([], target_brand="Acme")))

check("2. failed response is marked not-usable and excluded from roll-ups",
      attr.responses[5].usable is False and attr.total_responses == 7 and attr.usable_responses == 5)
check("3. success=True but answer=None -> not usable", attr.responses[6].usable is False)

check("4. target mentioned: target_mention_count counts usable target mentions (R_TARGET,R_COMP,R_EXT,R_NONE_KNOWN,R_UNAVAIL)",
      attr.target_mention_count == 5 and attr.responses[0].target_mentioned is True)
check("5. target-not-mentioned response reports target_mentioned False",
      analyze_attribution([R_NO_TARGET], target_brand="Acme").responses[0].target_mentioned is False)
check("6. competitor mention detected from structured BrandMention data",
      attr.responses[1].competitors_mentioned == ("BrewCo",))
check("7. multiple competitors detected, input order preserved",
      analyze_attribution([R_MULTI_COMP], target_brand="Acme", competitors=["BrewCo", "CampCup"])
      .responses[0].competitors_mentioned == ("BrewCo", "CampCup"))

check("8. multiple citations in one response are all classified",
      len(attr.responses[0].citations) == 2
      and all(c.points_to is CitationTarget.TARGET for c in attr.responses[0].citations))
check("9. target citation -> cites_target True, points_to target",
      attr.responses[0].cites_target is True and attr.responses[0].cites_competitor is False
      and attr.target_citation_count == 1)
check("10. competitor citation -> cites_competitor True, competitor_brand resolved, rollup populated",
      attr.responses[1].cites_competitor is True and attr.responses[1].citations[0].competitor_brand == "BrewCo"
      and attr.responses[1].cited_competitor_brands == ("BrewCo",)
      and attr.competitor_citation_count == 1 and attr.cited_competitor_brands == ("BrewCo",))
check("11. external citation -> cites_external True, points_to external",
      attr.responses[2].cites_external is True and attr.responses[2].citations[0].points_to is CitationTarget.EXTERNAL
      and attr.external_citation_count == 1)
check("12. citations == [] with availability known -> NONE_OBSERVED (not UNAVAILABLE)",
      attr.responses[3].citation_state is CitationState.NONE_OBSERVED and attr.citation_none_count == 1)
check("13. citation availability unknown -> UNAVAILABLE, kept distinct from NONE_OBSERVED",
      attr.responses[4].citation_state is CitationState.UNAVAILABLE and attr.citation_unavailable_count == 1)

check("14. relative/normalized domains: www + mixed case still classify as target",
      analyze_attribution([resp(G, [bm("Acme", position=1)], citations=["https://acme.example/x"])],
                          target_brand="Acme", target_evidence=pe("https://www.ACME.example/"))
      .responses[0].cites_target is True)
check("15. duplicate citations in one response are de-duplicated",
      len(analyze_attribution([R_DUPES], target_brand="Acme", target_evidence=TARGET_THIN).responses[0].citations) == 1)

check("16. empty target website evidence -> citations cannot be target-classified",
      analyze_attribution([R_TARGET], target_brand="Acme", target_evidence=None)
      .responses[0].cites_target is False)
check("17. empty competitor evidence -> competitor citation falls back to external, never guessed",
      analyze_attribution([R_COMP], target_brand="Acme", competitors=["BrewCo"], target_evidence=TARGET_THIN)
      .responses[0].cites_competitor is False
      and analyze_attribution([R_COMP], target_brand="Acme", competitors=["BrewCo"], target_evidence=TARGET_THIN)
      .responses[0].cites_external is True)

check("24. attribution is deterministic across repeated execution",
      analyze_attribution(CORE, target_brand="Acme", competitors=["BrewCo", "CampCup"],
                          target_evidence=TARGET_THIN, competitor_evidence=COMP_EV) == attr)
check("output model is a project-owned frozen dataclass (analysis.attribution)",
      type(attr).__module__ == "analysis.attribution" and isinstance(attr, AttributionSummary)
      and isinstance(attr.responses[0], ResponseAttribution))

# AIO stays flagged as observed data; no causal/private-reasoning wording anywhere
check("Google AIO response stays flagged is_observed_data=True",
      attr.responses[4].is_observed_data is True and attr.responses[0].is_observed_data is False)

# ============================== GAPS ===================================
gaps_thin = analyze_gaps(CORE, attribution=attr, target_evidence=TARGET_THIN, competitor_evidence=COMP_EV)
attr_full = analyze_attribution(CORE, target_brand="Acme", competitors=["BrewCo", "CampCup"],
                                target_evidence=TARGET_FULL, competitor_evidence=COMP_EV)
gaps_full = analyze_gaps(CORE, attribution=attr_full, target_evidence=TARGET_FULL, competitor_evidence=COMP_EV)

# a clean single response whose every significant term is present in TARGET_FULL,
# isolating the CONTENT/EVIDENCE "no gap" branch from multi-response term noise
CLEAN = resp(G, [bm("Acme", position=1)], answer="Acme camping coffee makers pour over french press")
CLEAN_ATTR = analyze_attribution([CLEAN], target_brand="Acme", target_evidence=TARGET_FULL)
CLEAN_GAPS = analyze_gaps([CLEAN], attribution=CLEAN_ATTR, target_evidence=TARGET_FULL)


def finding(report: GapReport, category, competitor=None):
    for f in report.findings:
        if f.category is category and f.competitor == competitor:
            return f
    return None


check("report shape: 6 core findings + one per competitor, categories in fixed order",
      [f.category for f in gaps_thin.findings[:6]] == [
          GapCategory.CONTENT, GapCategory.PRODUCT_INFORMATION, GapCategory.FAQ,
          GapCategory.EVIDENCE, GapCategory.CITATION, GapCategory.STRUCTURED_DATA]
      and len(gaps_thin.findings) == 8
      and {f.category for f in gaps_thin.findings[6:]} == {GapCategory.COMPETITOR_CONTENT})
check("all seven gap categories are representable",
      {c for c in GapCategory} == {GapCategory.CONTENT, GapCategory.PRODUCT_INFORMATION, GapCategory.FAQ,
                                   GapCategory.EVIDENCE, GapCategory.CITATION, GapCategory.STRUCTURED_DATA,
                                   GapCategory.COMPETITOR_CONTENT})

# A. CONTENT_GAP
check("CONTENT_GAP OBSERVED when response terms are absent from thin target evidence",
      finding(gaps_thin, GapCategory.CONTENT).status is GapStatus.OBSERVED
      and "camping" in finding(gaps_thin, GapCategory.CONTENT).evidence)
check("CONTENT_GAP NOT_OBSERVED when target evidence covers every response term",
      finding(CLEAN_GAPS, GapCategory.CONTENT).status is GapStatus.NOT_OBSERVED)
check("CONTENT_GAP UNAVAILABLE when target evidence is missing",
      finding(analyze_gaps(CORE, attribution=analyze_attribution(CORE, target_brand="Acme"), target_evidence=None),
              GapCategory.CONTENT).status is GapStatus.UNAVAILABLE)
check("CONTENT_GAP UNAVAILABLE when there are no usable responses",
      finding(analyze_gaps([R_FAILED], attribution=analyze_attribution([R_FAILED], target_brand="Acme"),
                           target_evidence=TARGET_THIN), GapCategory.CONTENT).status is GapStatus.UNAVAILABLE)

# B. PRODUCT_INFORMATION_GAP
check("20. PRODUCT_INFORMATION_GAP OBSERVED when no product signals on target",
      finding(gaps_thin, GapCategory.PRODUCT_INFORMATION).status is GapStatus.OBSERVED)
check("21. PRODUCT_INFORMATION_GAP NOT_OBSERVED when Product schema / product terms present",
      finding(gaps_full, GapCategory.PRODUCT_INFORMATION).status is GapStatus.NOT_OBSERVED)
check("PRODUCT_INFORMATION_GAP UNAVAILABLE with no target evidence",
      finding(analyze_gaps([], attribution=analyze_attribution([], target_brand="Acme"), target_evidence=TARGET_EMPTY),
              GapCategory.PRODUCT_INFORMATION).status is GapStatus.UNAVAILABLE)

# C. FAQ_GAP  (evidence-based wording, never "the AI does not understand")
faq_thin = finding(gaps_thin, GapCategory.FAQ)
faq_full = finding(gaps_full, GapCategory.FAQ)
check("18. FAQ_GAP OBSERVED with the exact conservative phrasing when no FAQ evidence",
      faq_thin.status is GapStatus.OBSERVED
      and faq_thin.summary == "FAQ evidence was not observed in the analyzed website evidence.")
check("19. FAQ_GAP NOT_OBSERVED when FAQ evidence is present",
      faq_full.status is GapStatus.NOT_OBSERVED
      and faq_full.summary == "FAQ evidence was observed in the analyzed website evidence.")
check("FAQ wording never claims the AI misunderstands the site",
      "does not understand" not in faq_thin.summary.lower() and "the ai" not in faq_thin.summary.lower())

# D. EVIDENCE_GAP
check("EVIDENCE_GAP OBSERVED: target-mentioning responses have terms absent from thin target evidence",
      finding(gaps_thin, GapCategory.EVIDENCE).status is GapStatus.OBSERVED)
check("EVIDENCE_GAP NOT_OBSERVED when target evidence supports every target-mention term",
      finding(CLEAN_GAPS, GapCategory.EVIDENCE).status is GapStatus.NOT_OBSERVED)
check("EVIDENCE_GAP UNAVAILABLE when the target is never mentioned",
      finding(analyze_gaps([R_NO_TARGET], attribution=analyze_attribution([R_NO_TARGET], target_brand="Acme"),
                           target_evidence=TARGET_THIN), GapCategory.EVIDENCE).status is GapStatus.UNAVAILABLE)

# E. CITATION_GAP  (uses attribution; unavailable never treated as failure)
cit_thin = finding(gaps_thin, GapCategory.CITATION)
check("CITATION_GAP OBSERVED: target mentioned but no target citation / competitor cited while target mentioned",
      cit_thin.status is GapStatus.OBSERVED and len(cit_thin.evidence) >= 1
      and "because" not in cit_thin.summary.lower())
check("CITATION_GAP NOT_OBSERVED when the target is cited wherever it is mentioned & availability known",
      finding(analyze_gaps([R_TARGET], attribution=analyze_attribution([R_TARGET], target_brand="Acme",
                                                                       target_evidence=TARGET_THIN),
                           target_evidence=TARGET_THIN), GapCategory.CITATION).status is GapStatus.NOT_OBSERVED)
check("CITATION_GAP UNAVAILABLE when citation availability is unavailable for every response",
      finding(analyze_gaps([R_UNAVAIL], attribution=analyze_attribution([R_UNAVAIL], target_brand="Acme"),
                           target_evidence=TARGET_THIN), GapCategory.CITATION).status is GapStatus.UNAVAILABLE)

# F. STRUCTURED_DATA_GAP  (uses schema_detector)
check("STRUCTURED_DATA_GAP OBSERVED when no JSON-LD / schema types on target",
      finding(gaps_thin, GapCategory.STRUCTURED_DATA).status is GapStatus.OBSERVED)
sd_full = finding(gaps_full, GapCategory.STRUCTURED_DATA)
check("STRUCTURED_DATA_GAP NOT_OBSERVED when structured data present; observed types listed as evidence",
      sd_full.status is GapStatus.NOT_OBSERVED and any(e.startswith("observed: Product") for e in sd_full.evidence)
      and any(e.startswith("not observed: WebSite") for e in sd_full.evidence))
check("STRUCTURED_DATA_GAP detects nested @graph types via schema_detector",
      finding(analyze_gaps([], attribution=analyze_attribution([], target_brand="Acme"),
                           target_evidence=pe("https://acme.example/", h1=("x",), json_ld=(
                               JsonLdBlock(valid=True, raw="{}", data={"@graph": [
                                   {"@type": "Organization", "name": "Acme"},
                                   {"@type": "BreadcrumbList", "itemListElement": [{"@type": "ListItem"}]}]}),))),
              GapCategory.STRUCTURED_DATA).status is GapStatus.NOT_OBSERVED)

# G. COMPETITOR_CONTENT_GAP  (framed as advantage, never causal)
cc_brewco = finding(gaps_thin, GapCategory.COMPETITOR_CONTENT, competitor="BrewCo")
cc_campcup = finding(gaps_thin, GapCategory.COMPETITOR_CONTENT, competitor="CampCup")
check("22. COMPETITOR_CONTENT_GAP OBSERVED: BrewCo has FAQ/Product/terms absent from target",
      cc_brewco.status is GapStatus.OBSERVED and cc_brewco.competitor == "BrewCo"
      and cc_brewco.summary == "Competitor evidence advantage observed for BrewCo."
      and any("FAQ evidence" in e for e in cc_brewco.evidence))
check("22. COMPETITOR_CONTENT_GAP wording is non-causal (no 'because'/'ranks'/'prefers')",
      not any(w in " ".join([cc_brewco.summary, *cc_brewco.evidence]).lower() for w in ("because", "ranks", "prefers", "why")))
check("23. COMPETITOR_CONTENT_GAP NOT_OBSERVED when competitor evidence is equivalent to target",
      cc_campcup.status is GapStatus.NOT_OBSERVED and cc_campcup.competitor == "CampCup")
check("COMPETITOR_CONTENT_GAP UNAVAILABLE when a competitor has no evidence",
      finding(analyze_gaps(CORE, attribution=attr, target_evidence=TARGET_THIN, competitor_evidence={"BrewCo": COMP_STRONG}),
              GapCategory.COMPETITOR_CONTENT, competitor="CampCup").status is GapStatus.UNAVAILABLE)
check("COMPETITOR_CONTENT_GAP UNAVAILABLE when target evidence is missing",
      finding(analyze_gaps(CORE, attribution=analyze_attribution(CORE, target_brand="Acme", competitors=["BrewCo"]),
                           target_evidence=None, competitor_evidence={"BrewCo": COMP_STRONG}),
              GapCategory.COMPETITOR_CONTENT, competitor="BrewCo").status is GapStatus.UNAVAILABLE)

# 16/17 empty inputs for gaps
empty_gaps = analyze_gaps([], attribution=analyze_attribution([], target_brand="Acme", competitors=["BrewCo"]),
                          target_evidence=None, competitor_evidence=None)
check("empty everything -> every core finding UNAVAILABLE, still deterministic shape",
      all(f.status is GapStatus.UNAVAILABLE for f in empty_gaps.findings)
      and len(empty_gaps.findings) == 7)

# 24. deterministic gaps
check("gap analysis is deterministic across repeated execution",
      analyze_gaps(CORE, attribution=attr, target_evidence=TARGET_THIN, competitor_evidence=COMP_EV) == gaps_thin
      and isinstance(gaps_thin, GapReport) and type(gaps_thin).__module__ == "analysis.gaps")

# 25/26 no network / no LLM  (structural: inspect import statements)
_FORBIDDEN = {"urllib", "http", "socket", "ssl", "requests", "httpx", "aiohttp",
              "openai", "anthropic", "google", "langchain", "tavily", "selenium", "playwright", "bs4"}
for mod in ("analysis/attribution.py", "analysis/gaps.py"):
    src = Path(mod).read_text(encoding="utf-8")
    roots = {n.split(".")[0] for n in re.findall(r"(?m)^\s*(?:from|import)\s+([\w.]+)", src)}
    check(f"25/26. {mod} imports nothing network / LLM / provider related",
          not (roots & _FORBIDDEN) and "urlopen" not in src, f"roots={sorted(roots)}")

# ============================ REGRESSION ================================
for label, script in (
    ("Phase 6", "tests_phase6_manual.py"),
    ("Phase 5B", "tests_phase5b_manual.py"),
    ("Phase 5A", "tests_phase5a_manual.py"),
    ("Phase 4", "tests_phase4_manual.py"),
    ("Phase 3D", "tests_phase3d_manual.py"),
    ("Phase 3C", "tests_phase3c_manual.py"),
    ("Phase 3B", "tests_phase3b_manual.py"),
    ("Phase 3A", "tests_phase3a_manual.py"),
    ("Phase 2", "tests_phase2_manual.py"),
    ("Phase 1", "tests_phase1_manual.py"),
):
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=480)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"Regression: {label} ({script}) exits 0", False, repr(exc))


# ============================== REPORT =================================
print("\n=== Phase 7 Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK / API / LLM CALLS = 0 (deterministic in-memory diagnosis only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
