"""Manual Phase 9 verification: response / brand analysis.

Project manual-test style. Deterministic in-memory fixtures only -- no
network, no API, no LLM. Then re-runs the earlier manual phases.

Run: python3 tests_phase9_manual.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from schemas.models import AIResponse, SourceType
from analysis.response_analyzer import analyze_response
from analysis.visibility import compute_visibility

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


def resp(answer, *, success=True):
    return AIResponse(
        source_type=SourceType.LIVE_GEMINI, prompt_id="p", prompt="q",
        success=success, answer=answer if success else None,
        error_message=None if success else "failed",
    )


def mentions(answer, target="Acme", competitors=("BrewCo", "CampCup")):
    return analyze_response(resp(answer), target_brand=target, competitors=list(competitors)).brand_mentions


def by(bms, name):
    return next((b for b in bms if b.brand_name == name), None)


# 1-2. target present / absent
check("1. target brand mentioned -> one BrandMention, mentioned=True",
      [b.brand_name for b in mentions("Acme makes good gear.")] == ["Acme"]
      and by(mentions("Acme makes good gear."), "Acme").mentioned is True)
check("2. target brand absent -> no BrandMention for it",
      mentions("Nothing relevant here about coffee.") == [])

# 3-4. competitors
check("3. competitor mentioned is detected", [b.brand_name for b in mentions("BrewCo sells kettles.")] == ["BrewCo"])
check("4. multiple competitors detected, emitted in supplied order (target first)",
      [b.brand_name for b in mentions("CampCup and BrewCo and Acme all compete.")] == ["Acme", "BrewCo", "CampCup"])

# 5-6. normalization
check("5. case-insensitive matching", [b.brand_name for b in mentions("ACME and brewco.")] == ["Acme", "BrewCo"])
check("6. punctuation / whitespace normalization ('Acme,' / 'Acme.' / 'Acme's')",
      [b.brand_name for b in mentions("  Acme,   is here. Also Acme's kit. BrewCo!  ")] == ["Acme", "BrewCo"])

# 7-8. position
numbered = mentions("Top picks:\n1. BrewCo\n2. Acme\n3. CampCup")
check("7. numbered recommendation list -> position detected per rank",
      by(numbered, "BrewCo").position == 1 and by(numbered, "Acme").position == 2 and by(numbered, "CampCup").position == 3)
check("8. ordinary prose -> position stays None",
      by(mentions("Acme is popular and BrewCo is also around."), "Acme").position is None)
check("8b. single '1.' line is not treated as a ranked list",
      by(mentions("1. Acme is one option in this paragraph."), "Acme").position is None)
check("8c. bulleted (unordered) list does not yield a position",
      by(mentions("Options:\n- Acme\n- BrewCo"), "Acme").position is None)

# 9-10. recommendation
check("9a. 'I recommend Acme' -> recommended=True", by(mentions("I recommend Acme for teams."), "Acme").recommended is True)
check("9b. 'Acme is a good choice' -> recommended=True", by(mentions("Acme is a good choice for camping."), "Acme").recommended is True)
check("10. ambiguous mention -> recommended=None (mentioned != recommended)",
      by(mentions("Acme is a brand that exists."), "Acme").recommended is None)

# 11. multiple occurrences -> one BrandMention, earliest position, reliable recommendation
multi = mentions("Rankings:\n1. Acme\n2. BrewCo\nLater, Acme is also a good choice. Acme again.")
check("11. repeated brand -> single BrandMention, earliest rank, recommendation preserved",
      [b.brand_name for b in multi].count("Acme") == 1
      and by(multi, "Acme").position == 1 and by(multi, "Acme").recommended is True)

# 5. context is short and taken verbatim from the answer
ctx = by(mentions("Acme is a durable option for backpackers."), "Acme").context_snippet
check("context_snippet is a short verbatim slice of the answer",
      ctx == "Acme is a durable option for backpackers." and len(ctx) <= 200)

# 12-13. empty / failed -> returned unchanged, no exception
r_empty = resp("   ")
r_failed = resp("ignored", success=False)
check("12. empty answer -> response returned unchanged", analyze_response(r_empty, target_brand="Acme") is r_empty)
check("13. failed response -> response returned unchanged (no exception)",
      analyze_response(r_failed, target_brand="Acme") is r_failed)

# 14-15. no brands / target + competitors together
check("14. no brands supplied -> brand_mentions == []",
      analyze_response(resp("Acme BrewCo CampCup."), target_brand="").brand_mentions == [])
check("15. target + competitors together -> all detected, target first",
      [b.brand_name for b in mentions("Acme, BrewCo and CampCup are all mentioned.")] == ["Acme", "BrewCo", "CampCup"])

# 16. determinism
base = resp("1. Acme\n2. BrewCo\nAcme is a good choice.")
r1 = analyze_response(base, target_brand="Acme", competitors=["BrewCo"])
r2 = analyze_response(base, target_brand="Acme", competitors=["BrewCo"])
check("16. repeated execution on the same input produces an identical result", r1 == r2)

# 17. AIResponse contract stays valid
out = analyze_response(base, target_brand="Acme", competitors=["BrewCo"])
check("17. result is a valid AIResponse and round-trips through the schema",
      isinstance(out, AIResponse) and AIResponse.model_validate(out.model_dump()) == out
      and out.timestamp == base.timestamp and out.success is True)

# 18. no network / LLM imports
src = Path("analysis/response_analyzer.py").read_text(encoding="utf-8")
roots = {n.split(".")[0] for n in re.findall(r"(?m)^\s*(?:from|import)\s+([\w.]+)", src)}
_FORBIDDEN = {"urllib", "http", "socket", "requests", "httpx", "aiohttp", "openai", "anthropic",
              "google", "langchain", "langgraph", "tavily", "selenium", "playwright"}
check("18. response_analyzer imports nothing network / LLM related",
      not (roots & _FORBIDDEN) and "urlopen" not in src, f"roots={sorted(roots)}")

# integration: analyzer output feeds Phase 6 visibility
r_a = analyze_response(resp("Top picks:\n1. Acme\n2. BrewCo\nAcme is a good choice."),
                       target_brand="Acme", competitors=["BrewCo"])
r_b = analyze_response(resp("BrewCo is fine. No target here."), target_brand="Acme", competitors=["BrewCo"])
vis = compute_visibility([r_a, r_b], target_brand="Acme", competitors=["BrewCo"])
check("integration: Phase 6 consumes the populated mentions (mention 1/2, rec 1/2, avg pos 1.0, SOV Acme 1/3)",
      vis.mention_rate.numerator == 1 and vis.mention_rate.denominator == 2
      and vis.recommendation_rate.numerator == 1 and vis.recommendation_rate.denominator == 2
      and vis.average_mention_position.value == 1.0 and vis.average_mention_position.sample_size == 1
      and vis.share_of_voice["Acme"].numerator == 1 and vis.share_of_voice["Acme"].denominator == 3,
      f"{vis.mention_rate} {vis.recommendation_rate} {vis.average_mention_position} {vis.share_of_voice['Acme']}")

# --- regression -------------------------------------------------------
for label, script in (
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
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=900)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"Regression: {label} ({script}) exits 0", False, repr(exc))


# --- report ---------------------------------------------------------
print("\n=== Phase 9 Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK / API / LLM CALLS = 0 (deterministic in-memory text analysis only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
