"""Manual Phase 3A verification script.

Not a pytest suite (same "minimal dependencies" rule as Phase 1/2) -- a
small deterministic script that exercises `utils/normalization.py` and
its interaction with the existing `schemas.models` types, then re-runs
the Phase 1 and Phase 2 manual scripts as regression checks.

No network calls. No LLM calls. No API keys required.

Run with: python tests_phase3a_manual.py
"""

from __future__ import annotations

import subprocess
import sys

from pydantic import ValidationError

from schemas.models import AIResponse, Citation, SourceType
from utils.normalization import (
    derive_domain_from_url,
    normalize_answer_text,
    normalize_citation,
    normalize_citations,
    normalize_whitespace,
)

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))


# --- 1. Whitespace normalization ---
try:
    messy = "  This   has\n\nirregular\t\twhitespace   "
    cleaned = normalize_whitespace(messy)
    check(
        "normalize_whitespace collapses and strips whitespace",
        cleaned == "This has irregular whitespace",
        cleaned,
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_whitespace collapses and strips whitespace", False, str(exc))

try:
    check(
        "normalize_answer_text preserves None (no answer returned)",
        normalize_answer_text(None) is None,
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_answer_text preserves None (no answer returned)", False, str(exc))

try:
    result = normalize_answer_text("  Some\n\nanswer   text  ")
    check(
        "normalize_answer_text normalizes whitespace while preserving content",
        result == "Some answer text",
        str(result),
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_answer_text normalizes whitespace while preserving content", False, str(exc))

try:
    result = normalize_answer_text("   ")
    check(
        "normalize_answer_text: whitespace-only answer becomes empty string, not None",
        result == "",
        repr(result),
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_answer_text: whitespace-only answer becomes empty string, not None", False, str(exc))


# --- 2. Creation of a successful AIResponse using normalized data ---
try:
    resp = AIResponse(
        source_type=SourceType.MOCK,
        model_name="mock-model-v1",
        prompt_id="p1",
        prompt="best running shoes for beginners",
        answer=normalize_answer_text("  Some   mocked\n\nanswer.  "),
        success=True,
    )
    check(
        "AIResponse can be built from normalized answer text",
        resp.answer == "Some mocked answer.",
        repr(resp.answer),
    )
except Exception as exc:  # noqa: BLE001
    check("AIResponse can be built from normalized answer text", False, str(exc))


# --- 3. Citation handling when citations exist ---
try:
    raw = {"url": "https://www.example.com/shoes", "title": "  Best Shoes  ", "domain": None}
    citation = normalize_citation(raw)
    check(
        "normalize_citation builds a Citation from a raw dict with a URL",
        isinstance(citation, Citation) and citation.title == "Best Shoes",
        str(citation),
    )
    check(
        "normalize_citation derives domain from URL when domain missing (www. stripped)",
        citation is not None and citation.domain == "example.com",
        str(citation.domain if citation else None),
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_citation builds a Citation from a raw dict with a URL", False, str(exc))

try:
    raw_list = [
        {"url": "https://example.com/a", "title": "A"},
        {"title": "No URL, title only"},
    ]
    normalized = normalize_citations(raw_list)
    check(
        "normalize_citations returns a list of Citation objects when data exists",
        isinstance(normalized, list)
        and len(normalized) == 2
        and all(isinstance(c, Citation) for c in normalized),
        str(normalized),
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_citations returns a list of Citation objects when data exists", False, str(exc))

try:
    resp = AIResponse(
        source_type=SourceType.OPENROUTER_FREE,
        prompt_id="p2",
        prompt="best running shoes",
        answer="An answer with a citation.",
        citations=normalize_citations([{"url": "https://example.com/shoes", "title": "Shoes"}]),
        success=True,
    )
    check(
        "AIResponse accepts normalized citations list",
        len(resp.citations) == 1 and resp.citations[0].title == "Shoes",
        str(resp.citations),
    )
except Exception as exc:  # noqa: BLE001
    check("AIResponse accepts normalized citations list", False, str(exc))


# --- 4. Behavior when citations are unavailable vs genuinely empty ---
try:
    unavailable = normalize_citations(None)
    check(
        "normalize_citations(None) signals 'unavailable' (returns None, not [])",
        unavailable is None,
        repr(unavailable),
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_citations(None) signals 'unavailable' (returns None, not [])", False, str(exc))

try:
    empty_but_attempted = normalize_citations([])
    check(
        "normalize_citations([]) signals 'attempted, found none' (returns [], not None)",
        empty_but_attempted == [],
        repr(empty_but_attempted),
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_citations([]) signals 'attempted, found none' (returns [], not None)", False, str(exc))

try:
    # A raw citation dict with no usable fields at all must not become a
    # fabricated empty Citation.
    check(
        "normalize_citation(None) returns None rather than an empty Citation",
        normalize_citation(None) is None,
    )
    check(
        "normalize_citation({}) returns None rather than an empty Citation",
        normalize_citation({}) is None,
    )
    check(
        "normalize_citation with all-empty fields returns None",
        normalize_citation({"url": "", "title": "  ", "domain": None}) is None,
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_citation empty-input handling", False, str(exc))

try:
    # Malformed URL: dropped rather than raising, title/domain preserved.
    citation = normalize_citation({"url": "not a real url", "title": "Still has a title"})
    check(
        "normalize_citation drops a malformed URL without raising, keeps other fields",
        citation is not None and citation.url is None and citation.title == "Still has a title",
        str(citation),
    )
except Exception as exc:  # noqa: BLE001
    check("normalize_citation drops a malformed URL without raising, keeps other fields", False, str(exc))

try:
    check(
        "derive_domain_from_url strips leading www.",
        derive_domain_from_url("https://www.example.com/page") == "example.com",
    )
    check(
        "derive_domain_from_url returns None for an unparseable/empty netloc",
        derive_domain_from_url("not-a-url") is None,
    )
except Exception as exc:  # noqa: BLE001
    check("derive_domain_from_url behavior", False, str(exc))


# --- 5. Failed response requires an error message ---
try:
    AIResponse(
        source_type=SourceType.LIVE_GEMINI,
        prompt_id="p3",
        prompt="best running shoes",
        answer=normalize_answer_text(None),
        success=False,
    )
    check("AIResponse rejects failure without error_message (Phase 3A re-check)", False, "did not raise")
except (ValidationError, ValueError):
    check("AIResponse rejects failure without error_message (Phase 3A re-check)", True)
except Exception as exc:  # noqa: BLE001
    check("AIResponse rejects failure without error_message (Phase 3A re-check)", False, f"wrong exception: {exc!r}")

try:
    resp = AIResponse(
        source_type=SourceType.LIVE_GEMINI,
        prompt_id="p3",
        prompt="best running shoes",
        answer=normalize_answer_text(None),
        success=False,
        error_message="Request timed out after 30s",
    )
    check(
        "AIResponse valid failure case built from normalization helpers",
        resp.success is False and resp.answer is None,
    )
except Exception as exc:  # noqa: BLE001
    check("AIResponse valid failure case built from normalization helpers", False, str(exc))


# --- 6. JSON serialization / deserialization ---
try:
    resp = AIResponse(
        source_type=SourceType.MOCK,
        model_name="mock-model-v1",
        prompt_id="p4",
        prompt="best running shoes",
        answer=normalize_answer_text("  A clean   answer.  "),
        citations=normalize_citations([{"url": "https://example.com/a", "title": "A"}]),
        success=True,
    )
    dumped = resp.model_dump_json()
    reloaded = AIResponse.model_validate_json(dumped)
    check(
        "AIResponse round-trips through JSON serialization",
        reloaded.answer == resp.answer
        and reloaded.source_type == resp.source_type
        and len(reloaded.citations) == 1
        and str(reloaded.citations[0].url).rstrip("/") == str(resp.citations[0].url).rstrip("/"),
    )
except Exception as exc:  # noqa: BLE001
    check("AIResponse round-trips through JSON serialization", False, str(exc))


# --- 7. Phase 1 regression ---
try:
    proc = subprocess.run(
        [sys.executable, "tests_phase1_manual.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    check(
        "Phase 1 regression: tests_phase1_manual.py exits 0 (all Phase 1 checks pass)",
        proc.returncode == 0,
        proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip(),
    )
except Exception as exc:  # noqa: BLE001
    check("Phase 1 regression: tests_phase1_manual.py exits 0 (all Phase 1 checks pass)", False, str(exc))


# --- 8. Phase 2 regression ---
try:
    proc = subprocess.run(
        [sys.executable, "tests_phase2_manual.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    check(
        "Phase 2 regression: tests_phase2_manual.py exits 0 (all Phase 2 checks pass)",
        proc.returncode == 0,
        proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip(),
    )
except Exception as exc:  # noqa: BLE001
    check("Phase 2 regression: tests_phase2_manual.py exits 0 (all Phase 2 checks pass)", False, str(exc))


# --- report ---
print("\n=== Phase 3A Manual Verification Results ===")
all_ok = True
for name, ok, detail in results:
    status_label = PASS if ok else FAIL
    if not ok:
        all_ok = False
    line = f"[{status_label}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)

passed = sum(1 for _, ok, _ in results if ok)
failed = len(results) - passed
print(f"\nTOTAL: {passed} passed, {failed} failed")
print("OVERALL:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
