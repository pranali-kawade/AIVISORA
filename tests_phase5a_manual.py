"""Manual Phase 5A verification script: Tavily web-acquisition adapter.

Not a pytest suite (same "minimal dependencies" rule as Phases 1-4). It
exercises `crawler/tavily.py` against a faked HTTP transport and then
re-runs every earlier manual phase as a regression check.

MANDATORY: this suite makes ZERO real Tavily requests and never needs a
real API key. The module-level `_perform_request` seam is replaced with a
trip-wire at start-up; each test installs its own deterministic fake over
that seam and restores the trip-wire afterwards, so any code path that
reached the real transport would fail the run.

Run with: python3 tests_phase5a_manual.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
from pathlib import Path

from config import Settings, load_settings
from schemas.models import SourceType
import crawler.tavily as tavily_mod
from crawler.tavily import (
    AcquiredPage,
    TAVILY_CRAWL_URL,
    TavilyWebAcquirer,
    WebAcquisitionError,
    WebAcquisitionResult,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []

# Obvious placeholder, never a real credential -- and a syntactically valid
# tvly- token so the redaction path is genuinely exercised.
FAKE_KEY = "tvly-FAKEKEYFORTESTSONLY0000000000"
START_URL = "https://example.com"


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


# --- global trip-wire: the real transport must never be called ------------
def _tripwire(*_a, **_k):
    raise AssertionError("REAL TAVILY REQUEST ATTEMPTED -- transport seam was not faked")


_REAL_PERFORM = tavily_mod._perform_request
tavily_mod._perform_request = _tripwire  # type: ignore[assignment]


def _settings(tavily_key):
    base = load_settings()
    return Settings(
        gemini_api_key=base.gemini_api_key,
        openrouter_api_key=base.openrouter_api_key,
        gemini_model=base.gemini_model,
        openrouter_model=base.openrouter_model,
        request_timeout=base.request_timeout,
        mock_mode=base.mock_mode,
        tavily_api_key=tavily_key,
    )


def _run(fake, *, key=FAKE_KEY, start_url=START_URL, acquirer_kwargs=None):
    capture = {"calls": 0, "url": None, "payload": None, "headers": None}

    def _wrapped(url, payload, headers, timeout):
        capture["calls"] += 1
        capture.update(url=url, payload=payload, headers=headers)
        return fake(url, payload, headers, timeout)

    original = tavily_mod._perform_request
    tavily_mod._perform_request = _wrapped  # type: ignore[assignment]
    try:
        acquirer = TavilyWebAcquirer(settings=_settings(key), **(acquirer_kwargs or {}))
        return acquirer.crawl(start_url), capture
    finally:
        tavily_mod._perform_request = original  # type: ignore[assignment]


def _ok_fake(items):
    return lambda url, payload, headers, timeout: (200, json.dumps({"results": items}))


def _status_fake(status, body):
    return lambda url, payload, headers, timeout: (status, body)


def _raise_fake(exc):
    def _f(url, payload, headers, timeout):
        raise exc
    return _f


# ===========================================================================
# 1. Successful Tavily acquisition  +  2. content / URL extraction
# ===========================================================================
try:
    items = [
        {"url": "https://example.com/", "raw_content": "  Home page body.  "},
        {"url": "https://example.com/about", "raw_content": "About us."},
    ]
    result, capture = _run(_ok_fake(items))
    check("Success: returns a project-owned WebAcquisitionResult with ok=True",
          isinstance(result, WebAcquisitionResult) and result.ok is True and result.error is None)
    check("Success: exactly one Tavily request was made", capture["calls"] == 1)
    check("Success: start URL is preserved on the result", result.start_url == START_URL)
    check("Success: every acquired item is a project AcquiredPage", all(isinstance(p, AcquiredPage) for p in result.pages))
    check("Extraction: all returned pages captured, order preserved",
          [p.url for p in result.pages] == ["https://example.com/", "https://example.com/about"])
    check("Extraction: raw content captured and outer-trimmed", result.pages[0].content == "Home page body.")
    check("Extraction: second page content captured", result.pages[1].content == "About us.")
except Exception as exc:  # noqa: BLE001
    check("Success / extraction", False, repr(exc))

try:
    r_empty, _ = _run(_ok_fake([{"url": START_URL, "raw_content": "   "}]))
    r_missing, _ = _run(_ok_fake([{"url": START_URL}]))
    check("Extraction: blank raw_content -> content is None (not '')", r_empty.pages[0].content is None)
    check("Extraction: missing raw_content -> content is None", r_missing.pages[0].content is None)
except Exception as exc:  # noqa: BLE001
    check("Extraction: empty content handling", False, repr(exc))

# Contract-level request checks (endpoint, auth, conservative limits, no key leak)
try:
    result, capture = _run(_ok_fake([{"url": START_URL, "raw_content": "x"}]),
                           acquirer_kwargs={"max_depth": 2, "limit": 5})
    payload, headers = capture["payload"], capture["headers"]
    check("Request: hits the Tavily Crawl endpoint", capture["url"] == TAVILY_CRAWL_URL == "https://api.tavily.com/crawl")
    check("Request: Bearer auth is sent in the Authorization header", headers.get("Authorization") == f"Bearer {FAKE_KEY}")
    check("Request: start URL and conservative crawl limits are in the payload",
          payload.get("url") == START_URL and payload.get("max_depth") == 2 and payload.get("limit") == 5
          and "max_breadth" in payload)
    check("Request: default crawl stays on the requested site", payload.get("allow_external") is False)
    check("Request: API key is not in the URL or request body",
          FAKE_KEY not in capture["url"] and FAKE_KEY not in json.dumps(payload))
    check("Request: out-of-range limits are clamped", _run(_ok_fake([]), acquirer_kwargs={"max_depth": 99})[1]["payload"]["max_depth"] == 5)
except Exception as exc:  # noqa: BLE001
    check("Request: construction", False, repr(exc))


# ===========================================================================
# 3. Missing API key
# ===========================================================================
try:
    result, capture = _run(_ok_fake([]), key=None)
    check("Missing key: ok=False with structured error", result.ok is False and isinstance(result.error, WebAcquisitionError))
    check("Missing key: error_type == 'missing_api_key'", result.error.error_type == "missing_api_key")
    check("Missing key: NO network request attempted", capture["calls"] == 0)
    check("Missing key: message names TAVILY_API_KEY, start URL preserved, no pages",
          "TAVILY_API_KEY" in result.error.message and result.start_url == START_URL and result.pages == ())
except Exception as exc:  # noqa: BLE001
    check("Missing key", False, repr(exc))


# ===========================================================================
# 4. Invalid input
# ===========================================================================
for bad in ("not a url", "", "   ", "ftp://example.com", "example.com"):
    try:
        result, capture = _run(_ok_fake([]), start_url=bad)
        check(f"Invalid input {bad!r}: structured invalid_url, no network",
              result.ok is False and result.error.error_type == "invalid_url" and capture["calls"] == 0)
    except Exception as exc:  # noqa: BLE001
        check(f"Invalid input {bad!r}", False, repr(exc))


# ===========================================================================
# 5. Tavily / API failure   +   6. timeout / network failure
# ===========================================================================
_CASES = [
    ("401 auth failure", _status_fake(401, json.dumps({"detail": {"error": "Unauthorized: invalid API key."}})), "http_error", 401),
    ("429 rate limit", _status_fake(429, json.dumps({"detail": {"error": "Rate limit exceeded"}})), "http_error", 429),
    ("500 provider error", _status_fake(500, json.dumps({"detail": {"error": "Internal server error"}})), "http_error", 500),
    ("non-JSON success body", _status_fake(200, "not json <<<"), "malformed_response", 200),
    ("success body without results[]", _status_fake(200, json.dumps({"foo": 1})), "malformed_response", 200),
    ("timeout (TimeoutError)", _raise_fake(TimeoutError("the read operation timed out")), "timeout", None),
    ("timeout (URLError wrapping TimeoutError)", _raise_fake(urllib.error.URLError(TimeoutError("timed out"))), "timeout", None),
    ("network failure (URLError)", _raise_fake(urllib.error.URLError("connection refused")), "network_error", None),
]
for name, fake, expected_type, expected_status in _CASES:
    try:
        result, capture = _run(fake)
        ok = (
            isinstance(result, WebAcquisitionResult)
            and result.ok is False
            and result.error is not None
            and result.error.error_type == expected_type
            and result.error.status_code == expected_status
            and result.pages == ()
            and capture["calls"] == 1  # one attempt, no retry
        )
        check(f"Failure: {name} -> structured {expected_type}", ok,
              f"{result.error.error_type}/{result.error.status_code}" if result.error else "no error")
    except Exception as exc:  # noqa: BLE001
        check(f"Failure: {name}", False, f"raised instead of structured failure: {exc!r}")


# ===========================================================================
# 7. API-key redaction from errors
# ===========================================================================
try:
    r1, _ = _run(_status_fake(401, json.dumps({"detail": {"error": f"bad key {FAKE_KEY}"}})))
    r2, _ = _run(_raise_fake(RuntimeError(f"boom Authorization Bearer {FAKE_KEY}")))
    blob = " ".join([repr(r1), repr(r2), r1.error.message, r2.error.message])
    check("Redaction: the fake key never appears in any error output", FAKE_KEY not in blob, blob[:180])
    check("Redaction: any tvly- token in a provider message is masked", "tvly-" not in blob)
except Exception as exc:  # noqa: BLE001
    check("Redaction", False, repr(exc))


# ===========================================================================
# 8. No real network request  +  architecture / Phase 4 intact
# ===========================================================================
check("Safety: the real transport was never called (still the trip-wire)",
      tavily_mod._perform_request is _tripwire)
_crawler_files = {p.name for p in Path("crawler").glob("*.py")}
check("Architecture: Phase 5A's crawler/tavily.py is present and unchanged in scope",
      {"__init__.py", "tavily.py"} <= _crawler_files
      and "parser" not in Path("crawler/tavily.py").read_text(encoding="utf-8")
      and "schema_detector" not in Path("crawler/tavily.py").read_text(encoding="utf-8"))
check("Architecture: result carries only url + content per page (no fabricated title/status/final_url)",
      {f for f in AcquiredPage.__dataclass_fields__} == {"url", "content"})

try:
    import collectors.google_aio_observed as aio_mod
    raw = json.loads(Path("data/observations.json").read_text(encoding="utf-8"))
    check("Phase 4 intact: Google AIO adapter still imports and identity unchanged",
          aio_mod.GoogleAIOObservedAdapter.source_type == SourceType.GOOGLE_AIO_OBSERVED)
    check("Phase 4 intact: data/observations.json untouched (zero observations)", raw.get("observations") == [])
    check("Tavily is not an AI-answer provider: no Tavily member on SourceType",
          not any("tavily" in e.value.lower() for e in SourceType))
except Exception as exc:  # noqa: BLE001
    check("Phase 4 preservation", False, repr(exc))


# ===========================================================================
# 9. Regression -- every earlier manual phase still passes, unmodified
# ===========================================================================
for label, script in (
    ("Phase 4", "tests_phase4_manual.py"),
    ("Phase 3D", "tests_phase3d_manual.py"),
    ("Phase 3C", "tests_phase3c_manual.py"),
    ("Phase 3B", "tests_phase3b_manual.py"),
    ("Phase 3A", "tests_phase3a_manual.py"),
    ("Phase 2", "tests_phase2_manual.py"),
    ("Phase 1", "tests_phase1_manual.py"),
):
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=300)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"Regression: {label} ({script}) exits 0", False, repr(exc))


# ===========================================================================
# Report
# ===========================================================================
tavily_mod._perform_request = _REAL_PERFORM  # type: ignore[assignment]

print("\n=== Phase 5A Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("REAL TAVILY REQUESTS = 0")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
