"""Manual Phase 4 verification script: Google AI Overview observed-data
ingestion & validation.

Not a pytest suite (same "minimal dependencies" rule as Phases 1-3) -- a
small deterministic script that exercises `collectors/google_aio_observed.py`
using in-test sample records only, then re-runs every earlier manual phase
script as a regression check.

Nothing here performs a network request, scraping, or browser automation,
and no API key is required. All sample observation records are defined
inside this file, explicitly as deterministic test data -- none are written
to `data/observations.json`.

Run with: python3 tests_phase4_manual.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from schemas.models import AIResponse, Citation, SourceType
from utils.normalization import normalize_answer_text
from collectors.google_aio_observed import (
    AIOIngestBatch,
    AIOObservation,
    GoogleAIOObservedAdapter,
    load_observations,
)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIPPED"

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))


def check(name: str, condition: bool, detail: str = "") -> None:
    record(name, PASS if condition else FAIL, detail)


ADAPTER = GoogleAIOObservedAdapter()
OBS_SOURCE = SourceType.GOOGLE_AIO_OBSERVED.value

# Messy observed answer: leading/trailing space, tabs, repeated blank lines.
MESSY_ANSWER = "   Camping coffee makers\t\trange from\n\n\n  pour-over  cones to   french presses.   "
EXPECTED_NORMALIZED = normalize_answer_text(MESSY_ANSWER)


def _obs(**overrides) -> dict:
    """A valid baseline observation record; override individual fields per test."""
    base = {
        "observation_id": "AIO-0001",
        "query": "What should someone know about camping coffee makers?",
        "brand": "TrailBrew",
        "category": "camping coffee makers",
        "answer_observed": True,
        "answer_text": "Camping coffee makers range from pour-over cones to french presses.",
        "citations": [
            {"url": "https://www.rei.com/learn/expert-advice/camping-coffee.html", "title": "  Camping Coffee  "}
        ],
        "observed_at": "2026-09-05T14:30:00+00:00",
        "source": OBS_SOURCE,
        "notes": "Observed manually in an incognito window; recorded by hand.",
    }
    base.update(overrides)
    # Allow explicit removal via sentinel.
    return {k: v for k, v in base.items() if v is not _REMOVE}


class _Remove:
    pass


_REMOVE = _Remove()


# ===========================================================================
# VALID DATA
# ===========================================================================
try:
    batch = ADAPTER.ingest_records([_obs()])
    check("Valid: exactly one result, no failures", len(batch.results) == 1 and len(batch.failures) == 0)
    result = batch.results[0]
    resp = result.response
    check("Valid: result.ok is True", result.ok, str(result.errors))
    check("Valid: response is an AIResponse", isinstance(resp, AIResponse), type(resp).__name__)
    check("Valid: source_type is GOOGLE_AIO_OBSERVED", isinstance(resp, AIResponse) and resp.source_type == SourceType.GOOGLE_AIO_OBSERVED)
    check("Valid: success is True and no error_message", isinstance(resp, AIResponse) and resp.success is True and resp.error_message is None)
    check("Valid: query/prompt preserved verbatim", isinstance(resp, AIResponse) and resp.prompt == _obs()["query"])
    check("Valid: prompt_id falls back to observation_id when omitted", isinstance(resp, AIResponse) and resp.prompt_id == "AIO-0001")
    check("Valid: model_name is None (observed data, no model queried)", isinstance(resp, AIResponse) and resp.model_name is None)
    check(
        "Valid: timestamp preserved and parsed as aware datetime",
        isinstance(resp, AIResponse)
        and resp.timestamp == datetime(2026, 9, 5, 14, 30, tzinfo=timezone.utc),
        str(getattr(resp, "timestamp", None)),
    )
    check("Valid: answer_status == 'observed'", result.answer_status == "observed", str(result.answer_status))
except Exception as exc:  # noqa: BLE001
    check("Valid: baseline observation ingests", False, repr(exc))

# Answer normalization
try:
    batch = ADAPTER.ingest_records([_obs(answer_text=MESSY_ANSWER)])
    resp = batch.results[0].response
    check(
        "Valid: observed answer text is normalized via utils.normalization",
        isinstance(resp, AIResponse) and resp.answer == EXPECTED_NORMALIZED,
        repr(getattr(resp, "answer", None)),
    )
    check(
        "Valid: normalized answer has no leading/trailing space or repeated blank lines",
        isinstance(resp, AIResponse)
        and resp.answer == resp.answer.strip()
        and "\n\n" not in resp.answer
        and "  " not in resp.answer,
    )
except Exception as exc:  # noqa: BLE001
    check("Valid: answer normalization", False, repr(exc))

# Citation normalization
try:
    batch = ADAPTER.ingest_records([
        _obs(citations=[{"url": "https://www.example.org/guide", "title": "  Field Guide  "}])
    ])
    resp = batch.results[0].response
    cite_ok = (
        isinstance(resp, AIResponse)
        and len(resp.citations) == 1
        and isinstance(resp.citations[0], Citation)
        and str(resp.citations[0].url).rstrip("/") == "https://www.example.org/guide"
        and resp.citations[0].title == "Field Guide"
        and resp.citations[0].domain == "example.org"
    )
    check("Valid: citations normalized to the common Citation shape (title trimmed, domain derived)", cite_ok, str(getattr(resp, "citations", None)))
except Exception as exc:  # noqa: BLE001
    check("Valid: citation normalization", False, repr(exc))

# prompt_id explicitly provided
try:
    batch = ADAPTER.ingest_records([_obs(prompt_id="P123")])
    resp = batch.results[0].response
    check("Valid: explicit prompt_id is used when provided", isinstance(resp, AIResponse) and resp.prompt_id == "P123")
except Exception as exc:  # noqa: BLE001
    check("Valid: explicit prompt_id", False, repr(exc))


# ===========================================================================
# CITATION SEMANTICS  (the Phase 3A distinction must survive)
# ===========================================================================
try:
    # 1. answer + citations -> populated
    with_c = ADAPTER.ingest_records([_obs(citations=[{"url": "https://example.com/a", "title": "A"}])]).results[0]
    check(
        "Citation semantics: observed answer WITH citations -> citations populated + status 'with_citations'",
        with_c.ok and len(with_c.response.citations) == 1 and with_c.citations_status == "with_citations",
        str(with_c.citations_status),
    )

    # 2. answer + empty citations list -> [] and status 'without_citations'
    zero_c = ADAPTER.ingest_records([_obs(citations=[])]).results[0]
    check(
        "Citation semantics: observed answer with ZERO citations -> citations == [] + status 'without_citations'",
        zero_c.ok and zero_c.response.citations == [] and zero_c.citations_status == "without_citations",
        str(zero_c.citations_status),
    )

    # 3. answer + citations omitted / null -> [] BUT status 'unavailable'
    unavail_omitted = ADAPTER.ingest_records([_obs(citations=_REMOVE)]).results[0]
    unavail_null = ADAPTER.ingest_records([_obs(citations=None)]).results[0]
    check(
        "Citation semantics: citation info UNAVAILABLE (omitted) -> citations == [] but status 'unavailable'",
        unavail_omitted.ok
        and unavail_omitted.response.citations == []
        and unavail_omitted.citations_status == "unavailable",
        str(unavail_omitted.citations_status),
    )
    check(
        "Citation semantics: citation info UNAVAILABLE (explicit null) -> status 'unavailable'",
        unavail_null.ok and unavail_null.citations_status == "unavailable",
        str(unavail_null.citations_status),
    )
    check(
        "Citation semantics: 'unavailable' is DISTINGUISHABLE from 'without_citations'",
        unavail_omitted.citations_status != zero_c.citations_status
        and unavail_omitted.response.citations == zero_c.response.citations == [],
    )

    # 4. no AIO answer observed / unavailable
    none_obs = ADAPTER.ingest_records([
        _obs(answer_observed=False, answer_text=_REMOVE, citations=_REMOVE)
    ]).results[0]
    resp = none_obs.response
    check(
        "Citation semantics: NO AIO answer observed -> valid record, structured non-success AIResponse",
        none_obs.ok
        and isinstance(resp, AIResponse)
        and resp.success is False
        and resp.answer is None
        and bool(resp.error_message)
        and "AI Overview" in resp.error_message,
        (resp.error_message or "") if isinstance(resp, AIResponse) else "",
    )
    check(
        "Citation semantics: NO AIO answer -> answer_status 'not_observed', citations_status 'unavailable'",
        none_obs.answer_status == "not_observed" and none_obs.citations_status == "unavailable",
    )
    check(
        "Citation semantics: NO AIO answer error message never implies a programmatic query",
        isinstance(resp, AIResponse) and "not an API result" in (resp.error_message or ""),
    )
except Exception as exc:  # noqa: BLE001
    check("Citation semantics", False, repr(exc))


# ===========================================================================
# INVALID DATA  -- each must fail with structured errors, no exception escaping
# ===========================================================================
def _expect_invalid(name: str, record_dict: dict, *, field_hint: str | None = None, type_hint: str | None = None) -> None:
    try:
        batch = ADAPTER.ingest_records([record_dict])
        res = batch.results[0]
        ok = (not res.ok) and res.response is None and len(res.errors) >= 1
        detail = "; ".join(f"{e.field}:{e.error_type}:{e.message}" for e in res.errors)[:200]
        if ok and field_hint is not None:
            ok = any((e.field or "").split(".")[0] == field_hint for e in res.errors)
        if ok and type_hint is not None:
            ok = any(e.error_type == type_hint for e in res.errors)
        check(f"Invalid: {name}", ok, detail)
    except Exception as exc:  # noqa: BLE001
        check(f"Invalid: {name}", False, f"raised instead of structured error: {exc!r}")


_expect_invalid("missing query", _obs(query=_REMOVE), field_hint="query", type_hint="missing")
_expect_invalid("empty query string", _obs(query=""), field_hint="query", type_hint="empty")
_expect_invalid("missing answer_text while answer_observed=true", _obs(answer_text=_REMOVE))
_expect_invalid("whitespace-only answer_text while answer_observed=true", _obs(answer_text="    "))
_expect_invalid("missing observation_id", _obs(observation_id=_REMOVE), field_hint="observation_id", type_hint="missing")
_expect_invalid("missing brand", _obs(brand=_REMOVE), field_hint="brand", type_hint="missing")
_expect_invalid("invalid source type (live_gemini)", _obs(source="live_gemini"), field_hint="source", type_hint="source")
_expect_invalid("invalid source type (chatgpt)", _obs(source="chatgpt"), field_hint="source", type_hint="source")
_expect_invalid("malformed citation: bare string entry", _obs(citations=["https://example.com/x"]), field_hint="citations", type_hint="citation")
_expect_invalid("malformed citation: unknown key", _obs(citations=[{"link": "https://example.com/x"}]), field_hint="citations", type_hint="citation")
_expect_invalid("malformed citation: entirely blank entry", _obs(citations=[{"url": "", "title": None, "domain": "  "}]), field_hint="citations", type_hint="citation")
_expect_invalid("invalid timestamp: not a date", _obs(observed_at="not-a-date"), field_hint="observed_at", type_hint="timestamp")
_expect_invalid("invalid timestamp: impossible date", _obs(observed_at="2026-13-40T99:99:99"), field_hint="observed_at", type_hint="timestamp")
_expect_invalid("wrong field type: answer_observed is a string", _obs(answer_observed="maybe"))
_expect_invalid("wrong field type: citations is a string", _obs(citations="https://example.com/x"), field_hint="citations")
_expect_invalid("wrong field type: query is a list", _obs(query=["a", "b"]), field_hint="query")
_expect_invalid("unknown top-level field (extra forbidden)", _obs(scraped_html="<div>...</div>"), type_hint="structure")
_expect_invalid("record is not a JSON object", "just a string", type_hint="structure")  # type: ignore[arg-type]

# Malformed JSON document
try:
    batch = ADAPTER.load_json_text('{"observations": [ {"observation_id": "x",  ]}')
    res = batch.results[0]
    check(
        "Invalid: malformed JSON document -> single structured 'json' failure, no crash",
        (not res.ok) and len(res.errors) == 1 and res.errors[0].error_type == "json",
        res.errors[0].message[:120] if res.errors else "",
    )
except Exception as exc:  # noqa: BLE001
    check("Invalid: malformed JSON document", False, repr(exc))

# Document missing the observations array
try:
    batch = ADAPTER.ingest_document({"note": "no array here"})
    res = batch.results[0]
    check(
        "Invalid: document without an 'observations' array -> structured 'structure' failure",
        (not res.ok) and res.errors[0].error_type == "structure",
        res.errors[0].message[:120] if res.errors else "",
    )
except Exception as exc:  # noqa: BLE001
    check("Invalid: document missing observations array", False, repr(exc))

# Document that is neither object nor array
try:
    batch = ADAPTER.ingest_document(42)
    check("Invalid: non-object/non-array document -> structured failure", not batch.results[0].ok)
except Exception as exc:  # noqa: BLE001
    check("Invalid: non-object/non-array document", False, repr(exc))


# ===========================================================================
# SAFETY / ARCHITECTURE
# ===========================================================================
_MODULE_SRC = Path("collectors/google_aio_observed.py").read_text(encoding="utf-8")

_NETWORK_TOKENS = ("urllib", "http.client", "httpx", "requests", "socket", "aiohttp")
_BROWSER_TOKENS = ("selenium", "playwright", "webdriver", "bs4", "beautifulsoup", "requests_html", "pyppeteer")
_WRITE_TOKENS = ("write_text(", ".write(", "open(", "json.dump(", "w+", '"w"', "'w'")

check(
    "Safety: adapter module imports nothing network-related",
    not any(tok in _MODULE_SRC for tok in _NETWORK_TOKENS),
    next((tok for tok in _NETWORK_TOKENS if tok in _MODULE_SRC), ""),
)
check(
    "Safety: adapter module has no scraping / browser-automation dependency",
    not any(tok in _MODULE_SRC.lower() for tok in _BROWSER_TOKENS),
    next((tok for tok in _BROWSER_TOKENS if tok in _MODULE_SRC.lower()), ""),
)
check(
    "Safety: adapter module never writes to disk",
    not any(tok in _MODULE_SRC for tok in _WRITE_TOKENS),
    next((tok for tok in _WRITE_TOKENS if tok in _MODULE_SRC), ""),
)

# No API key required: strip any provider keys and ingest anyway.
_saved_env = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "OPENROUTER_API_KEY")}
try:
    batch = ADAPTER.ingest_records([_obs()])
    check("Safety: ingestion works with no API keys in the environment", batch.results[0].ok)
finally:
    for k, v in _saved_env.items():
        if v is not None:
            os.environ[k] = v

# Production observations file is untouched and still an empty dataset.
try:
    raw = json.loads(Path("data/observations.json").read_text(encoding="utf-8"))
    check(
        "Safety: data/observations.json still holds ZERO fabricated observations",
        isinstance(raw, dict) and raw.get("observations") == [],
        str(raw.get("observations")),
    )
    prod_batch = load_observations("data/observations.json")
    check(
        "Safety: the empty production observations file loads cleanly (no results, no failures)",
        isinstance(prod_batch, AIOIngestBatch)
        and len(prod_batch.results) == 0
        and len(prod_batch.failures) == 0
        and prod_batch.responses == [],
    )
except Exception as exc:  # noqa: BLE001
    check("Safety: production observations file handling", False, repr(exc))

# Output is compatible with the common downstream representation and coexists
# with the live-provider identities without provider-specific types.
try:
    aio_resp = ADAPTER.ingest_records([_obs()]).responses[0]
    gemini_like = AIResponse(
        source_type=SourceType.LIVE_GEMINI, prompt_id="AIO-0001", prompt="q",
        answer="g", success=True,
    )
    openrouter_like = AIResponse(
        source_type=SourceType.OPENROUTER_FREE, prompt_id="AIO-0001", prompt="q",
        answer="o", success=True,
    )
    mixed = [gemini_like, openrouter_like, aio_resp]
    fieldsets = [set(r.model_dump().keys()) for r in mixed]
    check(
        "Architecture: observed AIO output is an AIResponse with the identical field set as live-provider responses",
        all(isinstance(r, AIResponse) for r in mixed) and fieldsets[0] == fieldsets[1] == fieldsets[2],
    )
    check(
        "Architecture: three distinct environment identities coexist in one plain list",
        {r.source_type for r in mixed} == {SourceType.LIVE_GEMINI, SourceType.OPENROUTER_FREE, SourceType.GOOGLE_AIO_OBSERVED},
    )
    check(
        "Architecture: GOOGLE_AIO_OBSERVED is not equal to any live-provider identity",
        SourceType.GOOGLE_AIO_OBSERVED != SourceType.LIVE_GEMINI
        and SourceType.GOOGLE_AIO_OBSERVED != SourceType.OPENROUTER_FREE,
    )
    # downstream-style uniform pass, no branching on provider
    summary = [(r.source_type.value, r.prompt_id, bool(r.answer or r.error_message), isinstance(r.citations, list)) for r in mixed]
    check("Architecture: downstream code iterates the mixed list with no provider-specific handling", all(s[3] for s in summary), str(summary))
except Exception as exc:  # noqa: BLE001
    check("Architecture: common representation compatibility", False, repr(exc))

# The dedicated observation model does not leak into / modify AIResponse.
check(
    "Architecture: AIOObservation is a separate model, not a second response schema",
    AIOObservation is not AIResponse and "answer_text" in AIOObservation.model_fields and "success" not in AIOObservation.model_fields,
)


# ===========================================================================
# REGRESSION -- every earlier manual phase still passes, unmodified
# ===========================================================================
for phase_label, script in (
    ("Phase 3D", "tests_phase3d_manual.py"),
    ("Phase 3C", "tests_phase3c_manual.py"),
    ("Phase 3B", "tests_phase3b_manual.py"),
    ("Phase 3A", "tests_phase3a_manual.py"),
    ("Phase 2", "tests_phase2_manual.py"),
    ("Phase 1", "tests_phase1_manual.py"),
):
    try:
        proc = subprocess.run(
            [sys.executable, script], capture_output=True, text=True, timeout=240
        )
        last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"Regression: {phase_label} ({script}) exits 0", proc.returncode == 0, last_line)
    except Exception as exc:  # noqa: BLE001
        check(f"Regression: {phase_label} ({script}) exits 0", False, repr(exc))


# ===========================================================================
# Report
# ===========================================================================
print("\n=== Phase 4 Manual Verification Results ===")
passed = failed = skipped = 0
for name, status, detail in results:
    if status == PASS:
        passed += 1
    elif status == SKIP:
        skipped += 1
    else:
        failed += 1
    line = f"[{status}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)

print(f"\nTOTAL: {passed} passed, {failed} failed, {skipped} skipped")
print("Real network / API request made: NO (observed-data adapter performs none)")
print("Fabricated production observations added: NO")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
