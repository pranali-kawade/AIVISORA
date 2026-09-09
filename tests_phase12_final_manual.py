"""Manual Phase 12 (final) verification: the minimal results UI.

Renders the TRACK / DIAGNOSE / OPTIMIZE sections against a hand-built,
in-memory pipeline result -- no Streamlit runtime, no pipeline, no network,
and no Phase 1-11 regression.

Run: python3 tests_phase12_final_manual.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from analysis.gaps import GapCategory, GapFinding, GapReport, GapStatus
from analysis.visibility import (
    AveragePosition, Ratio, SourceVisibility, VisibilityMetrics,
)
from optimizer.action_planner import (
    ActionKind, ActionPlan, ImpactLevel, OptimizationAction, Priority,
)
from schemas.models import SourceType

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


SRC = Path("app.py").read_text(encoding="utf-8")

# 1 / 2. compile + import
proc = subprocess.run([sys.executable, "-m", "py_compile", "app.py", __file__],
                      capture_output=True, text=True)
check("1. app.py + this test compile", proc.returncode == 0, proc.stderr.strip())
try:
    import app
    check("2. app imports", True)
except Exception as exc:  # noqa: BLE001
    check("2. app imports", False, repr(exc))
    raise SystemExit(1)


# --- a recording stand-in for streamlit ---------------------------------
class _Rec:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def _log(self, *args, **_kw) -> None:
        for a in args:
            if isinstance(a, str):
                self.lines.append(a)

    write = caption = markdown = subheader = header = title = text = _log

    def divider(self) -> None:
        pass

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [self] * n

    def container(self, *a, **k):  # st.container(border=True) is a context manager
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _R:  # a minimal object that only carries a source_type, like AIResponse
    def __init__(self, source_type) -> None:
        self.source_type = source_type


def render(state: dict) -> str:
    rec = _Rec()
    saved = app.st
    app.st = rec
    try:
        app.render_results(state)
    finally:
        app.st = saved
    return "\n".join(rec.lines)


# --- a fully populated, in-memory pipeline result ----------------------
VIS = VisibilityMetrics(
    target_brand="Acme",
    competitors=("Globex",),
    total_responses=4,
    usable_responses=4,
    mention_rate=Ratio.of(2, 4),           # 50%
    recommendation_rate=Ratio.of(1, 4),    # 25%
    average_mention_position=AveragePosition(value=2.4, sample_size=2),
    citation_rate=Ratio.of(3, 4),          # 75%
    share_of_voice={"Acme": Ratio.of(3, 10), "Globex": Ratio.of(7, 10)},  # 30% / 70%
    by_source={SourceType.LIVE_GEMINI: SourceVisibility(
        source=SourceType.LIVE_GEMINI, is_observed_data=False, total_responses=4,
        usable_responses=4, target_mentions=2, mention_rate=Ratio.of(2, 4),
        recommendation_rate=Ratio.of(1, 4), citation_rate=Ratio.of(3, 4),
    )},
)
GAPS = GapReport(
    target_brand="Acme",
    competitors=("Globex",),
    findings=(
        GapFinding(GapCategory.CITATION, GapStatus.OBSERVED, "citation summary text",
                   evidence=("proofpoint",)),
        GapFinding(GapCategory.FAQ, GapStatus.NOT_OBSERVED, "faq summary text"),
        GapFinding(GapCategory.COMPETITOR_CONTENT, GapStatus.OBSERVED,
                   "competitor summary text", competitor="Globex"),
    ),
)
PLAN = ActionPlan(
    target_brand="Acme",
    competitors=("Globex",),
    actions=(
        OptimizationAction(GapCategory.CITATION, "citation", "Add citation-ready evidence",
                           "impl guidance text", "citation reason text", Priority.HIGH,
                           ImpactLevel.HIGH, ActionKind.OPTIMIZATION),
        OptimizationAction(GapCategory.FAQ, "faq", "Add FAQ coverage", "faq impl text",
                           "faq reason text", Priority.MEDIUM, ImpactLevel.UNKNOWN,
                           ActionKind.OPTIMIZATION),
    ),
)
FULL = {
    "prompts": [1, 2, 3, 4, 5, 6, 7, 8],
    "ai_responses": [_R(SourceType.LIVE_GEMINI), _R(SourceType.OPENROUTER_FREE)],
    "competitors": ["Globex"],
    "visibility_results": VIS,
    "gap_results": GAPS,
    "action_plan": PLAN,
}

out = render(FULL)

# 3 / 4 / 5. the three sections exist, both in source and when rendered
check("3. TRACK section", "TRACK" in SRC and "TRACK" in out)
check("4. DIAGNOSE section", "DIAGNOSE" in SRC and "DIAGNOSE" in out)
check("5. OPTIMIZE section", "OPTIMIZE" in SRC and "OPTIMIZE" in out)

# 6. visibility values are the pipeline's own numbers
check("6. TRACK renders the pipeline's visibility values",
      "50%" in out and "25%" in out and "2.4" in out and "75%" in out and "30%" in out, out)
check("6b. cross-model + competitor comparison shown from state",
      "Gemini" in out and "70%" in out and "Globex" in out, out)

# 7. gap values come from gap_results
check("7. DIAGNOSE renders pipeline gap category / status / summary",
      "Citation Gap" in out and "OBSERVED" in out and "NOT_OBSERVED" in out
      and "citation summary text" in out and "competitor: Globex" in out, out)

# 8. action-plan values come from action_plan, existing order preserved
check("8. OPTIMIZE renders pipeline actions, HIGH before MEDIUM",
      "Add citation-ready evidence" in out and "citation reason text" in out
      and "Expected impact: HIGH" in out
      and "HIGH" in out and "MEDIUM" in out and out.index("HIGH") < out.index("MEDIUM"), out)

# 9. missing / unavailable values never crash and never fabricate numbers
empty = render({"visibility_results": None, "gap_results": None, "action_plan": None})
check("9. empty state renders with limitations, no invented numbers",
      "%" not in empty and "N/A" not in empty
      and "could not be measured" in empty.lower()
      and "no gap findings" in empty.lower()
      and "no optimization actions" in empty.lower(), empty)
partial = VisibilityMetrics(
    target_brand="Acme", competitors=(), total_responses=0, usable_responses=0,
    mention_rate=Ratio.of(0, 0), recommendation_rate=Ratio.of(0, 0),
    average_mention_position=AveragePosition(value=None, sample_size=0),
    citation_rate=Ratio.of(0, 0), share_of_voice={}, by_source={},
)
partial_out = render({"visibility_results": partial})
check("9b. not-measurable ratios render as N/A, never 0%",
      "N/A" in partial_out and "0%" not in partial_out, partial_out)

# 10. Google AI Overview stays labelled Observed Data, never a live API
aio_out = render({"ai_responses": [_R(SourceType.GOOGLE_AIO_OBSERVED)]})
check("10. Google AIO labelled 'Google AI Overview — Observed Data'",
      "Google AI Overview — Observed Data" in aio_out, aio_out)
check("10b. app.py never calls Google AI Overview a live API",
      "live google ai overview" not in SRC.lower()
      and "google ai overview api" not in SRC.lower())

# 11. no fabricated metrics: app.py aggregates nothing of its own
check("11. no metric aggregation in app.py",
      "share_of_voice[" not in SRC and "sum(" not in SRC
      and "statistics" not in SRC and "mean(" not in SRC and "numpy" not in SRC)
check("11b. no chart / dashboard widgets",
      not any(t in SRC for t in ("st.metric(", "st.bar_chart", "st.line_chart",
                                 "st.area_chart", "st.plotly_chart", "plotly",
                                 "st.dataframe", "st.altair_chart", "altair")))

# 12. no fake progress
check("12. no time.sleep / progress simulation",
      "time.sleep" not in SRC and "import time" not in SRC and "st.progress(" not in SRC)

# 13. no new dependency
req = Path("requirements.txt").read_text(encoding="utf-8")
pkgs = {ln.split("==")[0].strip().lower() for ln in req.splitlines()
        if ln.strip() and not ln.strip().startswith("#")}
check("13. requirements.txt unchanged",
      pkgs == {"pydantic", "streamlit", "python-dotenv", "google-genai", "langgraph"},
      str(sorted(pkgs)))

# 14. the existing Run Audit flow is still intact
check("14. Run Audit flow intact (button -> run_audit -> session_state -> render)",
      'st.button("Run Audit")' in SRC
      and "run_audit(config, use_gemini=True, use_openrouter=True)" in SRC
      and 'st.session_state["audit_result"] = result' in SRC
      and "render_results(_stored_result)" in SRC
      and SRC.index("run_audit(config") < SRC.rindex("render_results("))

# --- report -----------------------------------------------------------
print("\n=== Phase 12 (final) Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and status == FAIL else ""))
print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK / API / LLM CALLS DURING TESTS = 0 (in-memory result only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
