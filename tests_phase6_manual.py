"""Manual Phase 6 verification: deterministic AI Visibility metrics.

Not a pytest suite (same "minimal dependencies" rule as Phases 1-5B). It
exercises ``analysis/visibility.py`` with deterministic in-memory
``AIResponse`` fixtures and asserts exact numeric results, then re-runs the
earlier manual phases as a regression check.

No network, no API, no LLM, no randomness. Run:
    python3 tests_phase6_manual.py
"""

from __future__ import annotations

import subprocess
import sys

from schemas.models import AIResponse, BrandMention, Citation, SourceType
from analysis.visibility import (
    AveragePosition,
    Ratio,
    ResponseRecord,
    SourceVisibility,
    VisibilityMetrics,
    compute_visibility,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


def close(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


# --- fixture builders -------------------------------------------------------
def bm(name, mentioned=True, recommended=None, position=None):
    return BrandMention(brand_name=name, mentioned=mentioned, recommended=recommended, position=position)


def resp(source, mentions=(), *, success=True, answer="An answer.", citations=None, prompt_id="p"):
    kw = dict(source_type=source, prompt_id=prompt_id, prompt="q", brand_mentions=list(mentions))
    if success:
        kw.update(success=True, answer=answer)
    else:
        kw.update(success=False, answer=None, error_message="collection failed")
    if citations is not None:
        kw["citations"] = [Citation(url=u) for u in citations]
    return AIResponse(**kw)


G, O, A, M = (
    SourceType.LIVE_GEMINI,
    SourceType.OPENROUTER_FREE,
    SourceType.GOOGLE_AIO_OBSERVED,
    SourceType.MOCK,
)

# --- core benchmark set (Acme vs BrewCo, CampCup) -------------------------
R1 = resp(G, [bm("Acme", recommended=True, position=1), bm("BrewCo", position=2)], citations=["https://a.example/1"])
R2 = ResponseRecord(resp(G, [bm("BrewCo", position=1), bm("Acme", position=2), bm("CampCup", position=3)]), citations_known=True)  # citations=[] known
R3 = resp(O, [bm("CampCup", position=1), bm("Acme", recommended=False, position=2)])  # citations=[] no signal -> unknown
R4 = resp(O, [bm("BrewCo", position=1)], citations=["https://b.example/1"])  # no target
R5 = resp(A, [bm("Acme", recommended=True, position=1)], citations=["https://c.example/1", "https://c.example/2"])
R6 = resp(G, [bm("Acme", position=1)], success=False)  # failed -> excluded everywhere
R7 = ResponseRecord(resp(O, [], answer="Only competitors here."), citations_known=False)  # usable, no target, citations unavailable

CORE = [R1, R2, R3, R4, R5, R6, R7]
m = compute_visibility(CORE, target_brand="Acme", competitors=["BrewCo", "CampCup"])

# 1. mention rate
check("1. mention_rate = 4 usable-with-target / 6 usable",
      m.mention_rate.numerator == 4 and m.mention_rate.denominator == 6 and close(m.mention_rate.rate, 4 / 6),
      f"{m.mention_rate}")
check("10. failed response excluded from usable denominator", m.total_responses == 7 and m.usable_responses == 6)

# 2. recommendation rate (only explicit recommended is True: R1, R5)
check("2. recommendation_rate = 2 / 6 (mention != recommendation)",
      m.recommendation_rate.numerator == 2 and m.recommendation_rate.denominator == 6 and close(m.recommendation_rate.rate, 2 / 6),
      f"{m.recommendation_rate}")

# 3. average mention position (earliest target pos: R1=1, R2=2, R3=2, R5=1)
check("3. average_mention_position = (1+2+2+1)/4 = 1.5 over 4 samples",
      close(m.average_mention_position.value, 1.5) and m.average_mention_position.sample_size == 4,
      f"{m.average_mention_position}")

# 4/5/6. citation rate: known = R1,R2,R4,R5 ; observed = R1,R4,R5
check("4/5/6. citation_rate = 3 observed / 4 known (None!=[] preserved)",
      m.citation_rate.numerator == 3 and m.citation_rate.denominator == 4 and close(m.citation_rate.rate, 0.75),
      f"{m.citation_rate}")

# 7/8. share of voice (raw occurrence counts across usable: Acme 4, BrewCo 3, CampCup 2; total 9)
check("7. target SOV = 4 / 9",
      m.share_of_voice["Acme"].numerator == 4 and m.share_of_voice["Acme"].denominator == 9 and close(m.share_of_voice["Acme"].rate, 4 / 9))
check("8. competitor SOV exposed: BrewCo 3/9, CampCup 2/9",
      close(m.share_of_voice["BrewCo"].rate, 3 / 9) and close(m.share_of_voice["CampCup"].rate, 2 / 9)
      and set(m.share_of_voice) == {"Acme", "BrewCo", "CampCup"})
check("SOV != mention rate (different denominator: 9 vs 6)",
      m.share_of_voice["Acme"].denominator == 9 and m.mention_rate.denominator == 6)

# 9. cross-model visibility
gem, orr, aio = m.by_source[G], m.by_source[O], m.by_source[A]
check("9. Gemini bucket: 3 total / 2 usable, mention 2/2=1.0, rec 1/2=0.5, citation 1/2=0.5",
      gem.total_responses == 3 and gem.usable_responses == 2 and gem.target_mentions == 2
      and close(gem.mention_rate.rate, 1.0) and close(gem.recommendation_rate.rate, 0.5)
      and close(gem.citation_rate.rate, 0.5) and gem.is_observed_data is False,
      f"{gem}")
check("9. OpenRouter bucket: 3 usable, mention 1/3, rec 0/3=0.0, citation 1/1=1.0",
      orr.usable_responses == 3 and close(orr.mention_rate.rate, 1 / 3)
      and close(orr.recommendation_rate.rate, 0.0) and close(orr.citation_rate.rate, 1.0)
      and orr.is_observed_data is False)
check("9. Google AIO bucket flagged as observed data; mention 1/1, rec 1/1, citation 1/1",
      aio.is_observed_data is True and close(aio.mention_rate.rate, 1.0)
      and close(aio.recommendation_rate.rate, 1.0) and close(aio.citation_rate.rate, 1.0))

# 18. project-owned output model
check("18. output is analysis.visibility.VisibilityMetrics (not Pydantic, not AIResponse)",
      type(m).__module__ == "analysis.visibility" and isinstance(m, VisibilityMetrics)
      and isinstance(m.mention_rate, Ratio) and isinstance(m.average_mention_position, AveragePosition)
      and isinstance(gem, SourceVisibility))
check("18. VisibilityMetrics field set matches the Phase 6 contract",
      set(VisibilityMetrics.__dataclass_fields__) == {
          "target_brand", "competitors", "total_responses", "usable_responses",
          "mention_rate", "recommendation_rate", "average_mention_position",
          "citation_rate", "share_of_voice", "by_source",
      })

# 19. determinism -- repeated + reordered input
check("19. identical input -> identical output (3x)",
      compute_visibility(CORE, target_brand="Acme", competitors=["BrewCo", "CampCup"])
      == compute_visibility(CORE, target_brand="Acme", competitors=["BrewCo", "CampCup"])
      == m)
check("15. input order does not change aggregate metrics",
      compute_visibility(list(reversed(CORE)), target_brand="Acme", competitors=["BrewCo", "CampCup"]) == m)

# 11. zero usable responses (all failed) + empty list -> not measurable, no ZeroDivisionError
allfail = compute_visibility([resp(G, [bm("Acme")], success=False), resp(O, [], success=False)], target_brand="Acme")
empty = compute_visibility([], target_brand="Acme", competitors=["BrewCo"])
for label, z in (("all-failed", allfail), ("empty-list", empty)):
    check(f"11. {label}: usable=0, rates are None (not 0), no crash",
          z.usable_responses == 0 and z.mention_rate.rate is None and z.recommendation_rate.rate is None
          and z.citation_rate.rate is None and z.average_mention_position.value is None
          and z.average_mention_position.sample_size == 0
          and z.share_of_voice["Acme"].rate is None)
check("11. empty list -> by_source is empty; all-failed keeps buckets with None rates",
      empty.by_source == {} and allfail.by_source[G].mention_rate.rate is None)

# 12. no target mentions -> mention_rate measurably 0.0, avg position NOT measurable
notgt = compute_visibility([resp(G, [bm("BrewCo", position=1)]), resp(O, [bm("CampCup", position=1)])],
                           target_brand="Acme", competitors=["BrewCo", "CampCup"])
check("12. no target mentions: mention_rate == 0.0 (measured), avg position value is None (not measurable)",
      close(notgt.mention_rate.rate, 0.0) and notgt.mention_rate.denominator == 2
      and notgt.average_mention_position.value is None
      and close(notgt.share_of_voice["Acme"].rate, 0.0))

# 13. no competitors / target-only benchmark
solo = compute_visibility([resp(G, [bm("Acme", position=1)]), resp(O, [bm("Acme", position=1)])], target_brand="Acme")
check("13. no competitors: SOV has only the target and equals 1.0 when only the target is mentioned",
      set(solo.share_of_voice) == {"Acme"} and close(solo.share_of_voice["Acme"].rate, 1.0)
      and solo.competitors == ())
soloz = compute_visibility([resp(G, [bm("BrewCo", position=1)])], target_brand="Acme")
check("13. target-only benchmark, target never mentioned: SOV total 0 -> rate None (not 0)",
      soloz.share_of_voice["Acme"].numerator == 0 and soloz.share_of_voice["Acme"].denominator == 0
      and soloz.share_of_voice["Acme"].rate is None)

# 14. multiple target mentions in one response
multi = compute_visibility(
    [resp(G, [bm("Acme", position=1), bm("BrewCo", position=2), bm("Acme", position=3)])],
    target_brand="Acme", competitors=["BrewCo"],
)
check("14. multi-mention: mention_rate counts the response once (1/1)",
      multi.mention_rate.numerator == 1 and multi.mention_rate.denominator == 1)
check("14. multi-mention: average position uses the EARLIEST target mention (=1)",
      close(multi.average_mention_position.value, 1.0) and multi.average_mention_position.sample_size == 1)
check("14. multi-mention: SOV counts raw occurrences (Acme 2 / total 3)",
      multi.share_of_voice["Acme"].numerator == 2 and multi.share_of_voice["Acme"].denominator == 3
      and close(multi.share_of_voice["Acme"].rate, 2 / 3))

# 16. unknown / unexpected source type -> own bucket, no crash, still in global metrics
mock = compute_visibility([resp(M, [bm("Acme", position=1)]), resp(G, [bm("BrewCo", position=1)])],
                          target_brand="Acme", competitors=["BrewCo"])
check("16. unexpected SourceType (MOCK) gets its own by_source bucket and feeds global metrics",
      M in mock.by_source and mock.by_source[M].usable_responses == 1
      and close(mock.by_source[M].mention_rate.rate, 1.0)
      and mock.mention_rate.numerator == 1 and mock.mention_rate.denominator == 2)

# 17. no division by zero anywhere across a source with 0 usable responses
zerosrc = compute_visibility([resp(G, [bm("Acme")], success=False)], target_brand="Acme")
check("17. source with 0 usable responses -> Ratio(0,0) with rate None, no exception",
      zerosrc.by_source[G].usable_responses == 0 and zerosrc.by_source[G].citation_rate.rate is None
      and zerosrc.by_source[G].mention_rate.rate is None)

# citation semantics spot-check: [] known vs [] unknown
known_empty = compute_visibility([ResponseRecord(resp(G, [bm("Acme", position=1)]), citations_known=True)], target_brand="Acme")
unknown_empty = compute_visibility([resp(G, [bm("Acme", position=1)])], target_brand="Acme")
check("citation: [] with citations_known=True is in the denominator (0/1)",
      known_empty.citation_rate.numerator == 0 and known_empty.citation_rate.denominator == 1
      and close(known_empty.citation_rate.rate, 0.0))
check("citation: [] with no signal is EXCLUDED (denominator 0 -> None)",
      unknown_empty.citation_rate.denominator == 0 and unknown_empty.citation_rate.rate is None)

# duplicate responses handled deterministically
dupes = compute_visibility([R1, R1, R1], target_brand="Acme", competitors=["BrewCo"])
check("15. duplicate identical responses: mention_rate 3/3, SOV Acme 3 / (3 Acme + 3 BrewCo)",
      dupes.mention_rate.numerator == 3 and dupes.mention_rate.denominator == 3
      and dupes.share_of_voice["Acme"].numerator == 3 and dupes.share_of_voice["Acme"].denominator == 6)

# malformed / incomplete structured mention data does not crash
malformed = compute_visibility(
    [resp(G, [bm("   ", mentioned=True, position=1), bm("Acme", mentioned=False), bm("Acme", mentioned=True)])],
    target_brand="Acme",
)
check("malformed mentions: blank name & mentioned=False rows ignored; the real mention still counts",
      malformed.mention_rate.numerator == 1 and malformed.average_mention_position.sample_size == 1
      and close(malformed.average_mention_position.value, 1.0))

# --- regression ----------------------------------------------------------
for label, script in (
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
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=420)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"Regression: {label} ({script}) exits 0", False, repr(exc))


# --- report ------------------------------------------------------------
print("\n=== Phase 6 Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK / API / LLM CALLS = 0 (deterministic in-memory analytics only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
