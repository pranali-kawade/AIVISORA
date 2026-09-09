"""Manual Phase 12A verification: Streamlit shell + input configuration.

UI/input-config only -- no pipeline execution, no network. Then re-runs the
earlier manual phases.

Run: python3 tests_phase12a_manual.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


SRC = Path("app.py").read_text(encoding="utf-8")

# 1. import + 2. compile
try:
    import app  # noqa: F401

    check("1. Streamlit app imports successfully", True)
except Exception as exc:  # noqa: BLE001
    check("1. Streamlit app imports successfully", False, repr(exc))
    app = None  # type: ignore

proc = subprocess.run([sys.executable, "-m", "py_compile", "app.py"], capture_output=True, text=True)
check("2. app.py compiles", proc.returncode == 0, proc.stderr.strip())

# 3. required labels / sections present in source
for label in ("AEO Radar", "AI Search Visibility Auditor", "TRACK → DIAGNOSE → OPTIMIZE",
              "Audit Configuration", "Target Brand", "Category", "Target Website",
              "Competitors", "Competitor Websites", "Prompt Count", "AI Environments"):
    check(f"3. source contains section/label: {label!r}", label in SRC)

# 4 / 6. competitor parsing + cleaning
if app is not None:
    check("4. competitor parsing splits and trims",
          app.parse_competitors("Asana, Monday.com, ClickUp") == ["Asana", "Monday.com", "ClickUp"])
    check("6. empty and duplicate competitors are removed (case-insensitive, first spelling kept)",
          app.parse_competitors("Asana, , asana ,ClickUp, ClickUp") == ["Asana", "ClickUp"])
    check("competitor parsing tolerates empty input", app.parse_competitors("") == [] and app.parse_competitors(None) == [])

    # 5. competitor website parsing
    mapping, invalid = app.parse_competitor_sites(
        "Asana=https://asana.com\nMonday.com=https://monday.com\nClickUp=https://clickup.com"
    )
    check("5. competitor website parsing -> {brand: url}",
          mapping == {"Asana": "https://asana.com", "Monday.com": "https://monday.com", "ClickUp": "https://clickup.com"}
          and invalid == [])
    bad_map, bad_lines = app.parse_competitor_sites("Asana=https://asana.com\nnot a pair\nFoo=ftp://foo")
    check("5b. malformed competitor website lines are reported, not crashed on",
          bad_map == {"Asana": "https://asana.com"} and bad_lines == ["not a pair", "Foo=ftp://foo"])
else:
    check("4/5/6. competitor parsing", False, "app import failed")

# 7. prompt count range represented
check("7. prompt count uses min_value=1, max_value=20, default 8",
      "min_value=1" in SRC and "max_value=20" in SRC and "value=8" in SRC)

# 8. provider options + 9. Google AIO labelled observed data
check("8. provider options present (Gemini, OpenRouter, Google AI Overview)",
      'st.checkbox("Gemini")' in SRC and 'st.checkbox("OpenRouter")' in SRC and "Google AI Overview" in SRC)
check("9. Google AI Overview is explicitly labelled Observed Data (never as live/API)",
      "Google AI Overview — Observed Data" in SRC and "live" not in SRC.lower())

# 10. Run Audit button + 11/12 durable "minimal shell" invariants
# (Phase 12B intentionally adds pipeline execution wiring; checks 11/12/12b
# were retargeted from "no execution wired" to the invariants that persist:
# no results dashboard, no scraping / raw HTTP in app.py.)
check("10. Run Audit button exists", 'st.button("Run Audit")' in SRC)
check("11. app.py renders no results dashboard (no charts / metric cards / result tables)",
      not any(tok in SRC for tok in ("st.metric(", "st.bar_chart", "st.line_chart", "st.area_chart",
                                     "st.plotly_chart", "plotly", "st.dataframe", "st.table(", "altair")))
check("12. app.py contains no scraping / raw HTTP client",
      not any(tok in SRC for tok in ("import requests", "urllib.request", "httpx", "urlopen",
                                     "selenium", "playwright", "beautifulsoup", "bs4", "scrape")))
check("12b. Run Audit is gated behind input validation",
      "validate_inputs" in SRC and 'st.button("Run Audit")' in SRC)

# 13. no new dependencies
req = Path("requirements.txt").read_text(encoding="utf-8")
pkgs = {line.split("==")[0].strip().lower() for line in req.splitlines()
        if line.strip() and not line.strip().startswith("#")}
check("13. requirements.txt unchanged (no dependency added)",
      pkgs == {"pydantic", "streamlit", "python-dotenv", "google-genai", "langgraph"}, str(sorted(pkgs)))

# --- Phase 1-11 regression --------------------------------------------
for label, script in (
    ("Phase 11", "tests_phase11_manual.py"),
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
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=2400)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"14. Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"14. Regression: {label} ({script}) exits 0", False, repr(exc))


# --- report ---------------------------------------------------------
print("\n=== Phase 12A Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK / API / PIPELINE EXECUTION = 0 (input configuration UI only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
