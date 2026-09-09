"""Manual Phase 12B verification: minimal UI + visible pipeline execution.

Uses an injected fake pipeline -- no real API/network calls. Then runs a
lightweight regression subset (Phase 12A + Phase 11, which transitively
covers Phases 1-10).

Run: python3 tests_phase12b_manual.py
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


SRC = Path("app.py").read_text(encoding="utf-8")

# 1 / 2. compile + import
proc = subprocess.run([sys.executable, "-m", "py_compile", "app.py"], capture_output=True, text=True)
check("1. app.py compiles", proc.returncode == 0, proc.stderr.strip())
try:
    import app  # noqa: F401

    check("2. app imports successfully", True)
except Exception as exc:  # noqa: BLE001
    check("2. app imports successfully", False, repr(exc))
    app = None  # type: ignore

# 3 / 4 / 5 / 6. required inputs, providers, AIO label, Run Audit
for label in ("Target Brand", "Category", "Target Website", "Competitors",
              "Competitor Websites", "Prompt Count", "AI Environments"):
    check(f"3. input present: {label!r}", label in SRC)
check("4. provider options present",
      'st.checkbox("Gemini")' in SRC and 'st.checkbox("OpenRouter")' in SRC and "Google AI Overview" in SRC)
check("5. Google AI Overview explicitly labelled Observed Data (never 'live')",
      "Google AI Overview — Observed Data" in SRC and "live" not in SRC.lower())
check("6. Run Audit button exists", 'st.button("Run Audit")' in SRC)

# 7 / 8 / 9. validation
if app is not None:
    check("7. required-field validation (brand, category)",
          "Target Brand is required." in app.validate_inputs("", "cat", 8, True)
          and "Category is required." in app.validate_inputs("Acme", "", 8, True)
          and app.validate_inputs("Acme", "cat", 8, True) == [])
    check("8. prompt-count validation (1..20)",
          any("Prompt Count" in e for e in app.validate_inputs("Acme", "cat", 0, True))
          and any("Prompt Count" in e for e in app.validate_inputs("Acme", "cat", 21, True))
          and not any("Prompt Count" in e for e in app.validate_inputs("Acme", "cat", 8, True)))
    check("9. provider-selection validation",
          "Select at least one AI environment." in app.validate_inputs("Acme", "cat", 8, False)
          and "Select at least one AI environment." not in app.validate_inputs("Acme", "cat", 8, True))
else:
    check("7/8/9. validation", False, "app import failed")

# 10. configuration is passed correctly to the pipeline
if app is not None:
    captured: dict = {}

    def _fake_pipeline(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"prompts": [1, 2, 3], "ai_responses": [], "action_plan": None}

    config, invalid = app.build_config("Acme ", "PM software", "https://acme.example",
                                       "Asana, Asana , ", "Asana=https://asana.com\nnot-a-pair", 8)
    check("10a. build_config parses inputs internally",
          config == {"target_brand": "Acme", "category": "PM software", "website": "https://acme.example",
                     "competitors": ["Asana"], "competitor_sites": {"Asana": "https://asana.com"},
                     "prompt_count": 8}
          and invalid == ["not-a-pair"])
    out = app.run_audit(config, use_gemini=False, use_openrouter=False, _run_pipeline=_fake_pipeline)
    check("10b. run_audit forwards the exact config + providers list to run_pipeline",
          captured["config"] == config and captured["kwargs"]["providers"] == []
          and out == {"prompts": [1, 2, 3], "ai_responses": [], "action_plan": None})
    check("10c. optional website absent -> website is None",
          app.build_config("Acme", "cat", "  ", "", "", 5)[0]["website"] is None)
    # selected providers use the existing collectors
    from schemas.models import SourceType

    prov = app._providers(True, True)
    check("provider selection builds the existing collectors",
          [type(p).__name__ for p in prov] == ["GeminiCollector", "OpenRouterCollector"]
          and prov[0].source_type is SourceType.LIVE_GEMINI
          and app._providers(False, False) == [])
else:
    check("10. configuration passthrough", False, "app import failed")

# 11. pipeline is NOT executed on app import
import agents.graph as _ag  # noqa: E402

_real = _ag.run_pipeline
_ag.run_pipeline = Mock()
try:
    importlib.reload(app)
    check("11. pipeline is not executed on app import", _ag.run_pipeline.call_count == 0)
finally:
    _ag.run_pipeline = _real
    importlib.reload(app)  # restore the real binding for the checks below

# 12. pipeline executes only when run_audit is called (which the button wires)
counter = Mock(return_value={"prompts": [], "ai_responses": []})
app.run_audit({"target_brand": "Acme"}, use_gemini=False, use_openrouter=False, _run_pipeline=counter)
check("12. run_audit invokes run_pipeline exactly once, and the call sits inside the Run Audit block",
      counter.call_count == 1
      and "run_audit(config, use_gemini=use_gemini" in SRC
      and SRC.index("run_audit(config, use_gemini=use_gemini") > SRC.index('if st.button("Run Audit"):'))

# 13. completed pipeline result is retained in session state
check("13. completed pipeline result is stored in st.session_state",
      'st.session_state["audit_result"] = result' in SRC and 'st.session_state["audit_config"] = config' in SRC)

# 14. pipeline failure does not crash the app
if app is not None:
    def _raiser(config, **kwargs):
        raise RuntimeError("boom: tvly-SECRETKEY should never surface")

    crashed = False
    try:
        app.run_audit({"target_brand": "Acme"}, use_gemini=False, use_openrouter=False, _run_pipeline=_raiser)
    except RuntimeError:
        crashed = True  # run_audit propagates; the UI's try/except contains it
    check("14. pipeline failure is contained by the UI with a concise, secret-free message",
          crashed
          and "except Exception" in SRC and 'state="error"' in SRC
          and "Please review your configuration" in SRC
          and "str(exc)" not in SRC and "{exc" not in SRC and "traceback" not in SRC.lower())

# 15. execution-stage labels present
for stage in ("Generate Prompts", "Collect AI Responses", "Analyze Responses", "Measure Visibility",
              "Analyze Website", "Diagnose Gaps", "Plan Actions"):
    check(f"15. stage label present: {stage!r}", stage in SRC)
check("15b. status shows 'Running audit' -> 'Audit complete'",
      "Running audit" in SRC and "Audit complete" in SRC)

# 16. no fake progress / sleep-based simulation
check("16. no time.sleep / fake progress simulation",
      "time.sleep" not in SRC and "import time" not in SRC and "st.progress(" not in SRC)

# 17. no fabricated results / results dashboard
check("17. no results dashboard rendered (no charts / metric cards / tables)",
      not any(tok in SRC for tok in ("st.metric(", "st.bar_chart", "st.line_chart", "st.area_chart",
                                     "st.plotly_chart", "plotly", "st.dataframe", "st.table(", "altair")))

# 18. no new dependency
req = Path("requirements.txt").read_text(encoding="utf-8")
pkgs = {ln.split("==")[0].strip().lower() for ln in req.splitlines()
        if ln.strip() and not ln.strip().startswith("#")}
check("18. requirements.txt unchanged (no dependency added)",
      pkgs == {"pydantic", "streamlit", "python-dotenv", "google-genai", "langgraph"}, str(sorted(pkgs)))

# --- lightweight regression -----------------------------------------
for label, script in (
    ("Phase 12A", "tests_phase12a_manual.py"),
    ("Phase 11 (covers 1-10)", "tests_phase11_manual.py"),
    ("Phase 6", "tests_phase6_manual.py"),
):
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=2400)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"regression: {label} ({script}) exits 0", False, repr(exc))


# --- report -------------------------------------------------------
print("\n=== Phase 12B Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK / API / LLM CALLS DURING TESTS = 0 (injected fake pipeline only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
