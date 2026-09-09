"""Manual Phase 8 verification: Action Optimizer (OPTIMIZE).

Project manual-test style. Deterministic in-memory fixtures only -- no
network, no API, no LLM. Asserts actual values, then re-runs the earlier
manual phases as a regression check.

Run: python3 tests_phase8_manual.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from analysis.gaps import GapCategory, GapFinding, GapReport, GapStatus
from analysis.attribution import analyze_attribution
from analysis.gaps import analyze_gaps
from analysis.visibility import ResponseRecord
from crawler.parser import Headings, PageEvidence
from schemas.models import AIResponse, BrandMention, Citation, SourceType
from optimizer.action_planner import (
    ActionKind,
    ActionPlan,
    ImpactLevel,
    OptimizationAction,
    Priority,
    plan_actions,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


# --- fixture builders ----------------------------------------------------
def obs(category, evidence=(), competitor=None, summary=None):
    return GapFinding(category, GapStatus.OBSERVED,
                      summary or f"{category.value}: probable gap observed in acquired evidence.",
                      tuple(evidence), competitor)


def notobs(category):
    return GapFinding(category, GapStatus.NOT_OBSERVED, f"{category.value}: no gap observed.", (), None)


def unavail(category, competitor=None):
    return GapFinding(category, GapStatus.UNAVAILABLE,
                      f"{category.value}: website evidence was not available.", (), competitor)


def report(findings, brand="Acme", competitors=()):
    return GapReport(target_brand=brand, competitors=tuple(competitors), findings=tuple(findings))


C = GapCategory
FULL = report(
    [
        obs(C.CONTENT, evidence=("camping", "french", "press")),
        obs(C.PRODUCT_INFORMATION),
        obs(C.FAQ),
        obs(C.EVIDENCE, evidence=("warranty", "titanium")),
        obs(C.CITATION, evidence=("2 response(s): target mentioned but no target-domain citation observed",)),
        obs(C.STRUCTURED_DATA),
        obs(C.COMPETITOR_CONTENT, evidence=("FAQ evidence observed for the competitor but not observed on the target",), competitor="BrewCo"),
    ],
    competitors=["BrewCo"],
)
plan = plan_actions(FULL)


def by_cat(p: ActionPlan, category, competitor=None):
    for a in p.actions:
        if a.category is category and a.competitor == competitor:
            return a
    return None


# 1. OBSERVED -> action generated
one = plan_actions(report([obs(C.FAQ)]))
check("1. OBSERVED gap -> exactly one OPTIMIZATION action",
      len(one.actions) == 1 and one.actions[0].category is C.FAQ and one.actions[0].kind is ActionKind.OPTIMIZATION)

# 2. NOT_OBSERVED -> no action
check("2. NOT_OBSERVED gap -> no action", plan_actions(report([notobs(C.FAQ), notobs(C.CITATION)])).actions == ())

# 3. UNAVAILABLE -> no false optimization action (default); opt-in collection action only
u_default = plan_actions(report([unavail(C.FAQ)]))
u_optin = plan_actions(report([unavail(C.FAQ)]), include_evidence_collection=True)
check("3. UNAVAILABLE gap -> no optimization action by default", u_default.actions == ())
check("3. UNAVAILABLE + opt-in -> a labelled EVIDENCE_COLLECTION action, not a fix",
      len(u_optin.actions) == 1 and u_optin.actions[0].kind is ActionKind.EVIDENCE_COLLECTION
      and u_optin.actions[0].priority is Priority.LOW and u_optin.actions[0].expected_impact is ImpactLevel.UNKNOWN
      and "not an optimization fix" in u_optin.actions[0].description)

# 4. all seven Phase 7 categories
check("4. every one of the seven gap categories maps to an action",
      {a.category for a in plan.actions} == set(GapCategory) and len(plan.actions) == 7)

# 5. HIGH / MEDIUM / LOW priority behaviour
check("5. CITATION and COMPETITOR_CONTENT gaps -> HIGH priority",
      by_cat(plan, C.CITATION).priority is Priority.HIGH
      and by_cat(plan, C.COMPETITOR_CONTENT, "BrewCo").priority is Priority.HIGH)
check("5. PRODUCT/FAQ/EVIDENCE/STRUCTURED_DATA gaps -> MEDIUM priority",
      all(by_cat(plan, c).priority is Priority.MEDIUM
          for c in (C.PRODUCT_INFORMATION, C.FAQ, C.EVIDENCE, C.STRUCTURED_DATA)))
check("5. CONTENT gap -> LOW priority", by_cat(plan, C.CONTENT).priority is Priority.LOW)

# 6. expected-impact values (qualitative only)
check("6. CITATION impact is UNKNOWN (outcome outside owner control)",
      by_cat(plan, C.CITATION).expected_impact is ImpactLevel.UNKNOWN)
check("6. CONTENT impact LOW; other on-page gaps MEDIUM; all are ImpactLevel members",
      by_cat(plan, C.CONTENT).expected_impact is ImpactLevel.LOW
      and by_cat(plan, C.FAQ).expected_impact is ImpactLevel.MEDIUM
      and all(a.expected_impact in set(ImpactLevel) for a in plan.actions))

# 7. reason references the detected evidence
check("7. every action.reason is exactly its Phase 7 finding summary",
      all(a.reason == f.summary
          for a, f in zip(sorted(plan.actions, key=lambda a: a.category.value),
                          sorted(FULL.findings, key=lambda f: f.category.value))))
check("7. CONTENT/EVIDENCE/COMPETITOR descriptions surface the finding's evidence terms",
      "camping" in by_cat(plan, C.CONTENT).description
      and "warranty" in by_cat(plan, C.EVIDENCE).description
      and "FAQ evidence observed for the competitor" in by_cat(plan, C.COMPETITOR_CONTENT, "BrewCo").description)

# 8. duplicate actions removed
dup = plan_actions(report([obs(C.CONTENT, evidence=("x", "y")), obs(C.CONTENT, evidence=("x", "y"))]))
check("8. identical findings produce a single de-duplicated action", len(dup.actions) == 1)

# 9. deterministic ordering: HIGH before MEDIUM before LOW; ties keep finding order
ranks = [{"HIGH": 0, "MEDIUM": 1, "LOW": 2}[a.priority.value] for a in plan.actions]
check("9. actions ordered HIGH -> MEDIUM -> LOW", ranks == sorted(ranks))
check("9. HIGH ties keep Phase 7 finding order (CITATION @idx4 before COMPETITOR @idx6)",
      [a.category for a in plan.actions if a.priority is Priority.HIGH] == [C.CITATION, C.COMPETITOR_CONTENT])
check("19. repeated execution is identical (both modes)",
      plan_actions(FULL) == plan and plan_actions(FULL) == plan_actions(FULL)
      and plan_actions(FULL, include_evidence_collection=True) == plan_actions(FULL, include_evidence_collection=True))

# 10. empty GapReport
empty = plan_actions(report([], brand="Acme"))
check("10. empty GapReport -> empty plan, metadata preserved",
      empty.actions == () and empty.target_brand == "Acme" and isinstance(empty, ActionPlan))

# 11. multiple competitors
multi = plan_actions(report(
    [obs(C.COMPETITOR_CONTENT, evidence=("terms",), competitor="BrewCo"),
     obs(C.COMPETITOR_CONTENT, evidence=("schema",), competitor="CampCup")],
    competitors=["BrewCo", "CampCup"]))
check("11. one action per competitor, competitor field set, order preserved",
      [a.competitor for a in multi.actions] == ["BrewCo", "CampCup"]
      and all(a.category is C.COMPETITOR_CONTENT for a in multi.actions))

# 12. competitor-content advantage
cc = by_cat(plan, C.COMPETITOR_CONTENT, "BrewCo")
check("12. competitor-content action: area, HIGH priority, brand in title, advantages in description",
      cc.area == "competitor_content" and cc.priority is Priority.HIGH
      and "BrewCo" in cc.title and "not observed on the target" in cc.description)

# 13. citation-related gap
cit = by_cat(plan, C.CITATION)
check("13. citation action: area 'citation', HIGH, impact UNKNOWN, no-guarantee wording",
      cit.area == "citation" and cit.priority is Priority.HIGH and cit.expected_impact is ImpactLevel.UNKNOWN
      and "outside the site owner's control" in cit.description)

# 14. structured-data gap
sd = by_cat(plan, C.STRUCTURED_DATA)
check("14. structured-data action: area 'structured_data', MEDIUM, does not generate markup",
      sd.area == "structured_data" and sd.priority is Priority.MEDIUM and "does not generate the markup" in sd.description)

# 15. FAQ gap
faq = by_cat(plan, C.FAQ)
check("15. FAQ action matches the specified example wording",
      faq.title == "Add an FAQ section to the target website" and faq.area == "faq"
      and "recurring questions relevant to the analyzed queries" in faq.description)

# 16. product-information gap
pi = by_cat(plan, C.PRODUCT_INFORMATION)
check("16. product-information action: area, specifications wording",
      pi.area == "product_information" and "specifications" in pi.description)

# 17. content / evidence gap
ce = plan_actions(report([obs(C.CONTENT, evidence=("alpha", "beta")), obs(C.EVIDENCE, evidence=("gamma",))]))
check("17. content action LOW + evidence action MEDIUM, missing terms listed",
      by_cat(ce, C.CONTENT).priority is Priority.LOW and by_cat(ce, C.EVIDENCE).priority is Priority.MEDIUM
      and "alpha" in by_cat(ce, C.CONTENT).description and "gamma" in by_cat(ce, C.EVIDENCE).description)

# 18. no network / API / LLM, and no guaranteed-result wording
src = Path("optimizer/action_planner.py").read_text(encoding="utf-8")
roots = {n.split(".")[0] for n in re.findall(r"(?m)^\s*(?:from|import)\s+([\w.]+)", src)}
_FORBIDDEN = {"urllib", "http", "socket", "ssl", "requests", "httpx", "aiohttp",
              "openai", "anthropic", "google", "langchain", "tavily", "selenium", "playwright"}
check("18. action_planner imports nothing network / LLM / provider related",
      not (roots & _FORBIDDEN) and "urlopen" not in src, f"roots={sorted(roots)}")
_ban = ("will increase", "will improve ranking", "guaranteed to", "by 20%", "boosts ranking", "ensures ranking")
big = " ".join(a.title + " " + a.description for a in plan_actions(FULL, include_evidence_collection=True).actions).lower()
check("18. no action claims a guaranteed AI ranking/visibility result",
      not any(p in big for p in _ban)
      and all("does not guarantee any change in ai ranking" in a.description.lower()
              for a in plan.actions))

# integration: real Phase 7 GapReport -> Phase 8 plan, fully traceable
def bm(n, position=None):
    return BrandMention(brand_name=n, mentioned=True, recommended=None, position=position)


def pe(url, **kw):
    return PageEvidence(url=url, title=kw.get("title"), meta_description=kw.get("meta"),
                        headings=Headings(h1=tuple(kw.get("h1", ())), h2=tuple(kw.get("h2", ())), h3=()),
                        sections=tuple(kw.get("sections", ())), internal_links=(), external_links=(),
                        canonical=None, robots=None, json_ld=(), schema_types=tuple(kw.get("schema_types", ())),
                        content_present=kw.get("content", True))


r_resp = AIResponse(source_type=SourceType.LIVE_GEMINI, prompt_id="p", prompt="q", success=True,
                    answer="Acme camping coffee makers pour over french press titanium",
                    brand_mentions=[bm("Acme", 1)], citations=[Citation(url="https://brewco.example/x")])
tgt = pe("https://acme.example/", h1=("Acme",), sections=("Welcome.",))
comp = {"BrewCo": pe("https://brewco.example/", h1=("BrewCo Gear",), h2=("FAQ", "Specifications"),
                     sections=("Titanium build details.",), schema_types=("Product", "FAQPage"))}
i_attr = analyze_attribution([r_resp], target_brand="Acme", competitors=["BrewCo"],
                             target_evidence=tgt, competitor_evidence=comp)
i_gaps = analyze_gaps([r_resp], attribution=i_attr, target_evidence=tgt, competitor_evidence=comp)
i_plan = plan_actions(i_gaps)
observed_cats = {f.category for f in i_gaps.findings if f.status is GapStatus.OBSERVED}
check("integration: every planned action traces to an OBSERVED Phase 7 finding",
      len(i_plan.actions) == len(observed_cats)
      and {a.category for a in i_plan.actions} == observed_cats
      and all(any(a.reason == f.summary for f in i_gaps.findings) for a in i_plan.actions))

# --- regression --------------------------------------------------------
for label, script in (
    ("Phase 7", "tests_phase7_manual.py"),
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
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=600)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"Regression: {label} ({script}) exits 0", False, repr(exc))


# --- report ----------------------------------------------------------
print("\n=== Phase 8 Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK / API / LLM CALLS = 0 (deterministic in-memory action planning only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
