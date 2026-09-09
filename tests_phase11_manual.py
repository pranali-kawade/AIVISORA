"""Manual Phase 11 verification: end-to-end integration.

Deterministic fake collectors + fake website acquisition only -- no
network, no API keys, no LLM. Then re-runs the earlier manual phases.

Run: python3 tests_phase11_manual.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from schemas.models import AIResponse, BrandMention, PromptItem, SourceType
from crawler.parser import Headings, PageEvidence
from analysis.visibility import VisibilityMetrics
from analysis.attribution import AttributionSummary
from analysis.gaps import GapReport, GapStatus
from optimizer.action_planner import ActionPlan
from agents.graph import graph, full_graph, build_full_graph, run_pipeline

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


def finding_status(state, category_value):
    for f in state["gap_results"].findings:
        if f.category.value == category_value:
            return f.status
    return None


# --- deterministic fakes ------------------------------------------------
class FakeCollector:
    def __init__(self, source_type, *, answer="Acme is a good choice.\n1. Acme\n2. BrewCo", fail=False):
        self.source_type, self._answer, self._fail = source_type, answer, fail

    def collect(self, prompt: PromptItem) -> AIResponse:
        if self._fail:
            return AIResponse(source_type=self.source_type, prompt_id=prompt.prompt_id, prompt=prompt.prompt,
                              success=False, answer=None, error_message="fake provider failure")
        return AIResponse(source_type=self.source_type, prompt_id=prompt.prompt_id, prompt=prompt.prompt,
                          success=True, answer=self._answer)


def pe(url, **kw):
    return PageEvidence(url=url, title=kw.get("title"), meta_description=None,
                        headings=Headings(h1=tuple(kw.get("h1", ())), h2=tuple(kw.get("h2", ())), h3=()),
                        sections=tuple(kw.get("sections", ())), internal_links=(), external_links=(),
                        canonical=None, robots=None, json_ld=(), schema_types=tuple(kw.get("schema_types", ())),
                        content_present=True)


EVIDENCE = {
    "https://acme.example": pe("https://acme.example/", h1=("Acme",), sections=("Welcome to Acme.",)),
    "https://brewco.example": pe("https://brewco.example/", h1=("BrewCo Gear",),
                                 h2=("FAQ", "Specifications"), sections=("Titanium build details.",),
                                 schema_types=("Product", "FAQPage")),
}
fake_acquire = EVIDENCE.get  # url -> PageEvidence | None  (None for unknown urls)

AIO_OBS = [{"observation_id": "O1", "query": "camping coffee", "brand": "Acme", "answer_observed": True,
            "answer_text": "Acme is featured in the AI overview for camping coffee.",
            "observed_at": "2026-01-01T00:00:00+00:00"}]

BASE = {
    "target_brand": "Acme", "website": "https://acme.example", "competitors": ["BrewCo"],
    "category": "camping coffee makers", "prompt_count": 4,
    "competitor_sites": {"BrewCo": "https://brewco.example"},
}
PROVIDERS = [FakeCollector(SourceType.LIVE_GEMINI), FakeCollector(SourceType.OPENROUTER_FREE)]

final = run_pipeline(BASE, providers=PROVIDERS, acquire=fake_acquire)

# 1. prompt generation enters the pipeline
check("1. deterministic prompt generation feeds the pipeline",
      len(final["prompts"]) == 4 and all(isinstance(p, PromptItem) for p in final["prompts"]))

# 2 / 3. multiple AI responses, common AIResponse contract
check("2. every generated prompt is collected by every provider (2 x 4 responses)",
      len(final["ai_responses"]) == 8)
check("3. Gemini + OpenRouter responses share the common AIResponse contract",
      all(isinstance(r, AIResponse) for r in final["ai_responses"])
      and {r.source_type for r in final["ai_responses"]} == {SourceType.LIVE_GEMINI, SourceType.OPENROUTER_FREE})

# 4. response analysis populates BrandMention
check("4. response analysis populates target + competitor BrandMention",
      all(isinstance(x, BrandMention) for r in final["analyzed_responses"] for x in r.brand_mentions)
      and any(x.brand_name == "Acme" for r in final["analyzed_responses"] for x in r.brand_mentions)
      and any(x.brand_name == "BrewCo" for r in final["analyzed_responses"] for x in r.brand_mentions))

# 5. visibility metrics
vis = final["visibility_results"]
check("5. visibility metrics are produced by the existing analyzer",
      isinstance(vis, VisibilityMetrics)
      and vis.mention_rate.rate == 1.0 and vis.recommendation_rate.rate == 1.0
      and "Acme" in vis.share_of_voice
      and set(vis.by_source) == {SourceType.LIVE_GEMINI, SourceType.OPENROUTER_FREE})

# 6 / 7. website + competitor evidence reach diagnosis
check("6. target website evidence is acquired + parsed and reaches diagnosis",
      isinstance(final["website_data"], PageEvidence)
      and any(f.status is not GapStatus.UNAVAILABLE for f in final["gap_results"].findings))
check("7. competitor website evidence reaches diagnosis",
      isinstance(final["competitor_data"].get("BrewCo"), PageEvidence)
      and any(f.competitor == "BrewCo" and f.status is GapStatus.OBSERVED for f in final["gap_results"].findings))

# 8 / 9. gap report + action plan
check("8. a GapReport with the seven categories + competitor findings is produced",
      isinstance(final["gap_results"], GapReport) and len(final["gap_results"].findings) == 7)
check("9. an ActionPlan is produced from the gap report",
      isinstance(final["action_plan"], ActionPlan) and len(final["action_plan"].actions) >= 1)

# 10. final state contains all major outputs
check("10. final state exposes the full data-flow contract",
      {"target_brand", "website", "competitors", "prompts", "ai_responses", "analyzed_responses",
       "website_data", "competitor_data", "visibility_results", "attribution_results",
       "gap_results", "action_plan"} <= set(final))

# 11. provider failure does not fabricate success
fail_final = run_pipeline(BASE, providers=[FakeCollector(SourceType.LIVE_GEMINI, fail=True)], acquire=fake_acquire)
failed = [r for r in fail_final["ai_responses"] if r.source_type is SourceType.LIVE_GEMINI]
check("11. a failing provider yields structured failed AIResponses, never fabricated success",
      len(failed) == 4 and all(r.success is False and r.answer is None and r.error_message for r in failed)
      and fail_final["visibility_results"].usable_responses == 0)

# 12. empty response does not crash
empty_final = run_pipeline(BASE, providers=[FakeCollector(SourceType.OPENROUTER_FREE, answer="   ")], acquire=fake_acquire)
check("12. whitespace-only answers do not crash the pipeline and are not treated as usable",
      isinstance(empty_final["action_plan"], ActionPlan)
      and empty_final["visibility_results"].usable_responses == 0)

# 13. website acquisition failure does not fabricate evidence
noweb = run_pipeline({**BASE, "website": "https://unknown.example", "competitor_sites": {}},
                     providers=PROVIDERS, acquire=fake_acquire)
check("13. an acquisition failure leaves website_data None and keeps gaps UNAVAILABLE (no fabricated PageEvidence)",
      noweb["website_data"] is None
      and finding_status(noweb, "CONTENT_GAP") is GapStatus.UNAVAILABLE
      and finding_status(noweb, "STRUCTURED_DATA_GAP") is GapStatus.UNAVAILABLE)

# 14. Google AIO stays observed data, never live
aio_final = run_pipeline(BASE, providers=[], acquire=fake_acquire, aio_observations=AIO_OBS)
aio_responses = [r for r in aio_final["ai_responses"] if r.source_type is SourceType.GOOGLE_AIO_OBSERVED]
check("14. ingested Google AIO observations enter as GOOGLE_AIO_OBSERVED (not a live provider)",
      len(aio_responses) == 1
      and aio_final["visibility_results"].by_source[SourceType.GOOGLE_AIO_OBSERVED].is_observed_data is True)
check("14b. missing AIO observations are simply absent, no fabrication",
      not any(r.source_type is SourceType.GOOGLE_AIO_OBSERVED
              for r in run_pipeline(BASE, providers=[], acquire=fake_acquire)["ai_responses"]))

# 15. deterministic repeated execution (outputs that do not embed response timestamps)
a = run_pipeline(BASE, providers=PROVIDERS, acquire=fake_acquire)
b = run_pipeline(BASE, providers=PROVIDERS, acquire=fake_acquire)
check("15. repeated execution is deterministic",
      a["visibility_results"] == b["visibility_results"]
      and a["attribution_results"] == b["attribution_results"]
      and a["gap_results"] == b["gap_results"]
      and a["action_plan"] == b["action_plan"])

# 16. no network / API / LLM imports in the agents package
_FORBIDDEN = {"requests", "urllib", "http", "socket", "httpx", "aiohttp",
              "openai", "anthropic", "langchain", "tavily", "selenium", "playwright"}
for mod in ("agents/state.py", "agents/nodes.py", "agents/graph.py"):
    src = Path(mod).read_text(encoding="utf-8")
    roots = {n.split(".")[0] for n in re.findall(r"(?m)^\s*(?:from|import)\s+([\w.]+)", src)}
    check(f"16. {mod} imports nothing network / LLM related", not (roots & _FORBIDDEN) and "urlopen" not in src,
          f"roots={sorted(roots)}")
check("16b. nodes.py contains no direct provider/crawler wiring (deps are injected)",
      "collectors" not in Path("agents/nodes.py").read_text() and "crawler" not in Path("agents/nodes.py").read_text())

# 17. Phase 10 graph behavior remains valid
p10_edges = {(e.source, e.target) for e in graph.get_graph().edges}
check("17. the Phase 10 graph is unchanged (4 nodes, same linear flow)",
      p10_edges == {("__start__", "analyze_responses"), ("analyze_responses", "visibility"),
                    ("visibility", "diagnose"), ("diagnose", "optimize"), ("optimize", "__end__")})

# full graph flow shape
full_edges = {(e.source, e.target) for e in full_graph.get_graph().edges}
check("Phase 11 full_graph flow: generate_prompts -> collect_ai_responses -> analyze_responses -> visibility "
      "-> acquire_website_evidence -> diagnose -> optimize",
      full_edges == {("__start__", "generate_prompts"), ("generate_prompts", "collect_ai_responses"),
                     ("collect_ai_responses", "analyze_responses"), ("analyze_responses", "visibility"),
                     ("visibility", "acquire_website_evidence"), ("acquire_website_evidence", "diagnose"),
                     ("diagnose", "optimize"), ("optimize", "__end__")}
      and callable(build_full_graph))


# --- Phase 1-10 regression -------------------------------------------
for label, script in (
    ("Phase 10", "tests_phase10_manual.py"),
    ("Phase 9", "tests_phase9_manual.py"),
    ("Phase 8", "tests_phase8_manual.py"),
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
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=1800)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"18. Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"18. Regression: {label} ({script}) exits 0", False, repr(exc))


# --- report ---------------------------------------------------------
print("\n=== Phase 11 Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("REAL API / NETWORK / LLM CALLS = 0 (fake collectors + fake acquisition only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
