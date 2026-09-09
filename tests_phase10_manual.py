"""Manual Phase 10 verification: LangGraph orchestration.

Deterministic in-memory fixtures only -- no network, no API keys, no LLM.
Then re-runs the earlier manual phases.

Run: python3 tests_phase10_manual.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from schemas.models import AIResponse, BrandMention, SourceType
from crawler.parser import Headings, PageEvidence
from analysis.visibility import VisibilityMetrics
from analysis.attribution import AttributionSummary
from analysis.gaps import GapReport
from optimizer.action_planner import ActionPlan
from agents.state import PipelineState
from agents.graph import build_graph, graph
from agents import nodes

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


def resp(answer, *, success=True):
    return AIResponse(source_type=SourceType.LIVE_GEMINI, prompt_id="p", prompt="q",
                      success=success, answer=answer if success else None,
                      error_message=None if success else "collection failed")


def pe(url, **kw):
    return PageEvidence(url=url, title=kw.get("title"), meta_description=kw.get("meta"),
                        headings=Headings(h1=tuple(kw.get("h1", ())), h2=tuple(kw.get("h2", ())), h3=()),
                        sections=tuple(kw.get("sections", ())), internal_links=(), external_links=(),
                        canonical=None, robots=None, json_ld=(), schema_types=tuple(kw.get("schema_types", ())),
                        content_present=kw.get("content", True))


INPUT: PipelineState = {
    "target_brand": "Acme",
    "website": "https://acme.example",
    "competitors": ["BrewCo"],
    "prompts": [],
    "ai_responses": [
        resp("Top picks:\n1. Acme\n2. BrewCo\nAcme is a good choice for camping coffee."),
        resp("BrewCo makes solid kettles."),
        resp("collection failed", success=False),   # failed
        resp("   "),                                # empty answer
    ],
    "website_data": pe("https://acme.example/", h1=("Acme",), sections=("Welcome to Acme.",)),
    "competitor_data": {"BrewCo": pe("https://brewco.example/", h1=("BrewCo Gear",),
                                     h2=("FAQ", "Specifications"), sections=("Titanium build details.",),
                                     schema_types=("Product", "FAQPage"))},
}

final = graph.invoke(INPUT)

# 1. state can be constructed
s = PipelineState(target_brand="Acme", competitors=["BrewCo"])
check("1. PipelineState can be constructed with existing types", s["target_brand"] == "Acme" and isinstance(s, dict))

# 2. graph imports
check("2. agents.graph imports successfully (graph + build_graph)", graph is not None and callable(build_graph))

# 3. graph compiles
compiled = build_graph()
check("3. build_graph() returns a compiled, invokable graph", hasattr(compiled, "invoke") and hasattr(compiled, "get_graph"))

# 4. expected sequential flow
edges = {(e.source, e.target) for e in graph.get_graph().edges}
expected = {
    ("__start__", "analyze_responses"),
    ("analyze_responses", "visibility"),
    ("visibility", "diagnose"),
    ("diagnose", "optimize"),
    ("optimize", "__end__"),
}
check("4. graph has exactly the START->analyze_responses->visibility->diagnose->optimize->END flow",
      edges == expected, str(sorted(edges)))
check("4b. graph nodes are the four orchestration stages only",
      {n for n in graph.get_graph().nodes if not n.startswith("__")}
      == {"analyze_responses", "visibility", "diagnose", "optimize"})

# 5. analyzed responses contain BrandMention
analyzed = final["analyzed_responses"]
check("5. analyzed responses carry populated BrandMention objects",
      len(analyzed) == 4
      and any(m.brand_mentions and all(isinstance(x, BrandMention) for x in m.brand_mentions) for m in analyzed)
      and any(x.brand_name == "Acme" for m in analyzed for x in m.brand_mentions))

# 6. visibility result
check("6. visibility result is a VisibilityMetrics", isinstance(final["visibility_results"], VisibilityMetrics))

# 7. diagnosis / gap result
check("7. attribution + gap results are produced",
      isinstance(final["attribution_results"], AttributionSummary) and isinstance(final["gap_results"], GapReport))

# 8. action plan
check("8. action plan is an ActionPlan", isinstance(final["action_plan"], ActionPlan))

# 9. target + competitor data flows through the graph
check("9. target/competitor identity flows through to every stage output",
      final["visibility_results"].target_brand == "Acme"
      and final["visibility_results"].competitors == ("BrewCo",)
      and final["gap_results"].competitors == ("BrewCo",)
      and any(f.competitor == "BrewCo" for f in final["gap_results"].findings)
      and any(a.competitor == "BrewCo" for a in final["action_plan"].actions))

# 10. deterministic repeated execution
final2 = graph.invoke(INPUT)
check("10. repeated execution on the same input yields identical pipeline outputs",
      final["analyzed_responses"] == final2["analyzed_responses"]
      and final["visibility_results"] == final2["visibility_results"]
      and final["attribution_results"] == final2["attribution_results"]
      and final["gap_results"] == final2["gap_results"]
      and final["action_plan"] == final2["action_plan"])

# 11. failed / empty response does not crash the graph
check("11. failed & empty responses pass through without crashing the graph",
      analyzed[2].success is False and analyzed[2].brand_mentions == []
      and analyzed[3].brand_mentions == []
      and isinstance(final["action_plan"], ActionPlan))
check("11b. the graph does not mutate the caller's input responses",
      all(r.brand_mentions == [] for r in INPUT["ai_responses"]))

# 12 / 13. no network / provider imports introduced by Phase 10
_FORBIDDEN_ROOTS = {"requests", "urllib", "http", "socket", "httpx", "aiohttp",
                    "openai", "anthropic", "google", "langchain", "tavily", "selenium", "playwright"}
for mod in ("agents/state.py", "agents/nodes.py", "agents/graph.py"):
    src = Path(mod).read_text(encoding="utf-8")
    roots = {n.split(".")[0] for n in re.findall(r"(?m)^\s*(?:from|import)\s+([\w.]+)", src)}
    check(f"12. {mod} imports nothing network / LLM related",
          not (roots & _FORBIDDEN_ROOTS) and "urlopen" not in src, f"roots={sorted(roots)}")
node_src = Path("agents/nodes.py").read_text(encoding="utf-8")
check("13. nodes.py introduces no provider/collector/crawler calls",
      "collectors" not in node_src and "crawler" not in node_src and "def collect" in node_src
      and "provider collection is intentionally not wired" in node_src)

# 14. final state contains the expected pipeline outputs
check("14. final state carries every pipeline output key",
      {"analyzed_responses", "visibility_results", "attribution_results", "gap_results", "action_plan"} <= set(final))

# --- Phase 1-9 regression ------------------------------------------------
for label, script in (
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
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=1200)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"15. Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"15. Regression: {label} ({script}) exits 0", False, repr(exc))


# --- report -----------------------------------------------------------
print("\n=== Phase 10 Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK / API / LLM CALLS = 0 (LangGraph orchestrates existing deterministic functions only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
