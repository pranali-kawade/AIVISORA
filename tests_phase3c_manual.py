"""Manual Phase 3C verification script.

Not a pytest suite (same "minimal dependencies" rule as Phase 1/2/3A/3B) --
a small deterministic script that exercises `collectors/openrouter.py` (the
OpenRouter free-model collector) using local fakes for the HTTP layer, then
re-runs the Phase 3B / 3A / 2 / 1 manual scripts as regression checks.

Checks performed:

A. Import safety -- the collector module imports with no API key set.
B. Missing-key behavior -- with OPENROUTER_API_KEY absent, `collect()` makes
   no network request (enforced by a trip-wire) and returns a failed
   `AIResponse` carrying a clear error message, with no secret exposed.
C. HTTP / response parsing -- deterministic local fakes for: a successful
   response, empty/malformed responses, HTTP errors, and transport
   timeout/error handling. The real OpenRouter API is never called here.
D. Live request -- ONLY when a real OPENROUTER_API_KEY is already
   configured: make exactly ONE small request with the configured model and
   validate the resulting `AIResponse`. Otherwise SKIPPED (never FAILED),
   with no request made.
E. Regression -- tests_phase3b_manual.py, tests_phase3a_manual.py,
   tests_phase2_manual.py and tests_phase1_manual.py still exit 0. Those
   files are not modified.

Run with: python tests_phase3c_manual.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from config import Settings, load_settings
from schemas.models import AIResponse, Citation, PromptItem, SourceType

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIPPED"

# (name, status, detail) where status is one of PASS / FAIL / SKIPPED.
results: list[tuple[str, str, str]] = []

# Records whether an actual outbound OpenRouter request was attempted.
live_request_made = False

MOCK_KEY = "mock-openrouter-key-not-real"


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))


def check(name: str, condition: bool, detail: str = "") -> None:
    record(name, PASS if condition else FAIL, detail)


def _settings_with_key(key: str | None) -> Settings:
    """A Settings snapshot identical to the current config except for the
    OpenRouter API key. Lets tests drive the collector deterministically
    without touching the real environment.
    """
    base = load_settings()
    return Settings(
        gemini_api_key=base.gemini_api_key,
        openrouter_api_key=key,
        gemini_model=base.gemini_model,
        openrouter_model=base.openrouter_model,
        request_timeout=base.request_timeout,
        mock_mode=base.mock_mode,
    )


def _prompt(pid: str = "P-3C", text: str = "What is drip coffee?") -> PromptItem:
    return PromptItem(prompt_id=pid, prompt=text, intent="informational")


# ---------------------------------------------------------------------------
# A. Import safety: importing the collector must not require an API key.
# ---------------------------------------------------------------------------
_key_backup = os.environ.pop("OPENROUTER_API_KEY", None)
try:
    import collectors.openrouter as openrouter_mod
    from collectors.openrouter import OpenRouterCollector, OPENROUTER_CHAT_COMPLETIONS_URL

    check("A. Import safety: collectors.openrouter imports with no OPENROUTER_API_KEY set", True)
    check(
        "A. Import safety: OpenRouterCollector declares source_type OPENROUTER_FREE",
        getattr(OpenRouterCollector, "source_type", None) == SourceType.OPENROUTER_FREE,
        str(getattr(OpenRouterCollector, "source_type", None)),
    )
    check(
        "A. Import safety: endpoint constant is centralized and points at OpenRouter",
        isinstance(OPENROUTER_CHAT_COMPLETIONS_URL, str)
        and OPENROUTER_CHAT_COMPLETIONS_URL.startswith("https://openrouter.ai/"),
        OPENROUTER_CHAT_COMPLETIONS_URL,
    )
except Exception as exc:  # noqa: BLE001
    check("A. Import safety: collectors.openrouter imports with no OPENROUTER_API_KEY set", False, repr(exc))
    openrouter_mod = None  # type: ignore[assignment]
    OpenRouterCollector = None  # type: ignore[assignment]
finally:
    if _key_backup is not None:
        os.environ["OPENROUTER_API_KEY"] = _key_backup


# ---------------------------------------------------------------------------
# Shared trip-wire / fake plumbing for the HTTP layer.
# ---------------------------------------------------------------------------
class _CallCounter:
    def __init__(self) -> None:
        self.count = 0


def _install_fake(fn):
    """Swap collectors.openrouter._perform_request for `fn`; return the
    original so callers can restore it in a finally block.
    """
    original = openrouter_mod._perform_request
    openrouter_mod._perform_request = fn  # type: ignore[attr-defined]
    return original


def _restore(original) -> None:
    openrouter_mod._perform_request = original  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# B. Missing-key behavior: no key -> no network call, structured failure.
# ---------------------------------------------------------------------------
if OpenRouterCollector is not None:
    counter = _CallCounter()

    def _tripwire(*_args, **_kwargs):
        counter.count += 1
        raise AssertionError("network request attempted despite missing OPENROUTER_API_KEY")

    original = _install_fake(_tripwire)
    try:
        collector = OpenRouterCollector(settings=_settings_with_key(None))
        prompt = _prompt("P-3C-NOKEY", "What is a cortado?")
        result = collector.collect(prompt)

        ok = isinstance(result, AIResponse)
        check("B. Missing key: returns an AIResponse", ok, type(result).__name__)
        check("B. Missing key: no network request attempted (trip-wire untouched)", counter.count == 0)
        check("B. Missing key: success is False", ok and result.success is False)
        check(
            "B. Missing key: source_type is OPENROUTER_FREE",
            ok and result.source_type == SourceType.OPENROUTER_FREE,
        )
        check(
            "B. Missing key: prompt and prompt_id preserved",
            ok and result.prompt == prompt.prompt and result.prompt_id == prompt.prompt_id,
        )
        check("B. Missing key: answer is None", ok and result.answer is None)
        check("B. Missing key: citations not fabricated", ok and result.citations == [])
        check(
            "B. Missing key: clear error message present",
            ok and bool(result.error_message) and "OPENROUTER_API_KEY" in (result.error_message or ""),
            (result.error_message or "") if ok else "",
        )
        dumped = result.model_dump_json() if ok else ""
        _env_key = (os.environ.get("OPENROUTER_API_KEY") or _key_backup) or ""
        check(
            "B. Missing key: serialized result exposes no secret",
            "Bearer " not in dumped and not (_env_key and _env_key in dumped),
        )
    except AssertionError as exc:
        check("B. Missing key: no network request attempted (trip-wire untouched)", False, str(exc))
    except Exception as exc:  # noqa: BLE001
        check("B. Missing key: collect() handled cleanly", False, repr(exc))
    finally:
        _restore(original)
else:
    record("B. Missing key: collector unavailable (import failed)", FAIL, "see import-safety checks")


# ---------------------------------------------------------------------------
# C. HTTP / response parsing tests -- deterministic local fakes only.
# ---------------------------------------------------------------------------
def _run_with_fake(fake, prompt: PromptItem, key: str | None = MOCK_KEY):
    """Call collect() once with `_perform_request` replaced by `fake`.
    Returns (result, call_count).
    """
    counter = _CallCounter()

    def _wrapped(*args, **kwargs):
        counter.count += 1
        return fake(*args, **kwargs)

    original = _install_fake(_wrapped)
    try:
        collector = OpenRouterCollector(settings=_settings_with_key(key))
        return collector.collect(prompt), counter.count
    finally:
        _restore(original)


if OpenRouterCollector is not None:
    settings_snapshot = load_settings()
    configured_model = settings_snapshot.openrouter_model

    # C1. Successful normal response.
    def _fake_ok(url, payload, headers, timeout):
        # Sanity: the collector must send the configured free model, one user message.
        assert payload["model"] == configured_model, payload["model"]
        assert payload["messages"][0]["role"] == "user"
        body = json.dumps(
            {
                "id": "gen-123",
                "model": configured_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "  Drip coffee is\n\nbrewed by gravity.  "},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        return 200, body

    try:
        prompt = _prompt("P-3C-OK")
        result, calls = _run_with_fake(_fake_ok, prompt)
        ok = isinstance(result, AIResponse)
        check("C1. Success: result is an AIResponse", ok, type(result).__name__)
        check("C1. Success: exactly one request made", calls == 1, f"calls={calls}")
        check("C1. Success: success is True", ok and result.success is True,
              (result.error_message or "") if ok else "")
        check("C1. Success: source_type is OPENROUTER_FREE",
              ok and result.source_type == SourceType.OPENROUTER_FREE)
        check("C1. Success: model_name is the configured model",
              ok and result.model_name == configured_model, str(ok and result.model_name))
        check("C1. Success: answer passed through Phase 3A normalization",
              ok and result.answer == "Drip coffee is brewed by gravity.", repr(ok and result.answer))
        check("C1. Success: prompt / prompt_id preserved unchanged",
              ok and result.prompt == prompt.prompt and result.prompt_id == prompt.prompt_id)
        check("C1. Success: no citations fabricated when none provided",
              ok and result.citations == [])
    except Exception as exc:  # noqa: BLE001
        check("C1. Success: handled cleanly", False, repr(exc))

    # C2. Genuine citation data present -> preserved via normalization utils.
    def _fake_with_citations(url, payload, headers, timeout):
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Answer with a source.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": "https://www.example.com/guide",
                                        "title": "  Coffee Guide  ",
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        )
        return 200, body

    try:
        result, _ = _run_with_fake(_fake_with_citations, _prompt("P-3C-CITE"))
        ok = isinstance(result, AIResponse)
        cite_ok = (
            ok
            and result.success is True
            and len(result.citations) == 1
            and isinstance(result.citations[0], Citation)
            and result.citations[0].domain == "example.com"
            and result.citations[0].title == "Coffee Guide"
        )
        check("C2. Citations: genuine provider citation is preserved & normalized",
              cite_ok, str(ok and result.citations))
    except Exception as exc:  # noqa: BLE001
        check("C2. Citations: handled cleanly", False, repr(exc))

    # C3. Empty content -> failure, not a successful empty answer.
    def _fake_empty_content(url, payload, headers, timeout):
        return 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": "   "}}]})

    try:
        result, _ = _run_with_fake(_fake_empty_content, _prompt("P-3C-EMPTY"))
        ok = isinstance(result, AIResponse)
        check("C3. Empty content: success is False with an error message",
              ok and result.success is False and bool(result.error_message),
              (result.error_message or "") if ok else "")
        check("C3. Empty content: answer is None (no successful empty answer)",
              ok and result.answer is None)
    except Exception as exc:  # noqa: BLE001
        check("C3. Empty content: handled cleanly", False, repr(exc))

    # C4. No choices at all -> failure.
    def _fake_no_choices(url, payload, headers, timeout):
        return 200, json.dumps({"id": "gen-1", "choices": []})

    try:
        result, _ = _run_with_fake(_fake_no_choices, _prompt("P-3C-NOCHOICE"))
        ok = isinstance(result, AIResponse)
        check("C4. Missing content: success is False", ok and result.success is False,
              (result.error_message or "") if ok else "")
    except Exception as exc:  # noqa: BLE001
        check("C4. Missing content: handled cleanly", False, repr(exc))

    # C5. Malformed JSON body -> failure.
    def _fake_bad_json(url, payload, headers, timeout):
        return 200, "not-json <<< >>>"

    try:
        result, _ = _run_with_fake(_fake_bad_json, _prompt("P-3C-BADJSON"))
        ok = isinstance(result, AIResponse)
        check("C5. Malformed JSON: success is False with an error message",
              ok and result.success is False and "JSON" in (result.error_message or ""),
              (result.error_message or "") if ok else "")
    except Exception as exc:  # noqa: BLE001
        check("C5. Malformed JSON: handled cleanly", False, repr(exc))

    # C6. HTTP error status -> failure, status surfaced.
    def _fake_http_429(url, payload, headers, timeout):
        return 429, json.dumps({"error": {"message": "rate limit exceeded"}})

    try:
        result, calls = _run_with_fake(_fake_http_429, _prompt("P-3C-429"))
        ok = isinstance(result, AIResponse)
        check("C6. HTTP error: success is False and status is surfaced",
              ok and result.success is False and "429" in (result.error_message or ""),
              (result.error_message or "") if ok else "")
        check("C6. HTTP error: still exactly one request (no retry)", calls == 1, f"calls={calls}")
    except Exception as exc:  # noqa: BLE001
        check("C6. HTTP error: handled cleanly", False, repr(exc))

    # C7. Transport timeout -> failure.
    def _fake_timeout(url, payload, headers, timeout):
        raise TimeoutError("the read operation timed out")

    try:
        result, calls = _run_with_fake(_fake_timeout, _prompt("P-3C-TIMEOUT"))
        ok = isinstance(result, AIResponse)
        check("C7. Timeout: success is False with an error message",
              ok and result.success is False and "TimeoutError" in (result.error_message or ""),
              (result.error_message or "") if ok else "")
        check("C7. Timeout: exactly one attempt (no retry loop)", calls == 1, f"calls={calls}")
    except Exception as exc:  # noqa: BLE001
        check("C7. Timeout: handled cleanly", False, repr(exc))

    # C8. Generic network failure -> failure.
    import urllib.error as _urlerr

    def _fake_urlerror(url, payload, headers, timeout):
        raise _urlerr.URLError("connection refused")

    try:
        result, _ = _run_with_fake(_fake_urlerror, _prompt("P-3C-NET"))
        ok = isinstance(result, AIResponse)
        check("C8. Network failure: success is False with an error message",
              ok and result.success is False and bool(result.error_message),
              (result.error_message or "") if ok else "")
    except Exception as exc:  # noqa: BLE001
        check("C8. Network failure: handled cleanly", False, repr(exc))

    # C9. API key must never leak, even if an exception embeds it.
    def _fake_leaky(url, payload, headers, timeout):
        raise RuntimeError(f"upstream rejected token {MOCK_KEY} in header")

    try:
        result, _ = _run_with_fake(_fake_leaky, _prompt("P-3C-LEAK"))
        ok = isinstance(result, AIResponse)
        dumped = result.model_dump_json() if ok else ""
        check(
            "C9. Redaction: API key is scrubbed from the error message / result",
            ok and result.success is False and MOCK_KEY not in (result.error_message or "") and MOCK_KEY not in dumped,
            (result.error_message or "") if ok else "",
        )
    except Exception as exc:  # noqa: BLE001
        check("C9. Redaction: handled cleanly", False, repr(exc))

    # C10. Provider error object with HTTP 200 -> failure.
    def _fake_error_object(url, payload, headers, timeout):
        return 200, json.dumps({"error": {"message": "model unavailable", "code": 503}})

    try:
        result, _ = _run_with_fake(_fake_error_object, _prompt("P-3C-ERROBJ"))
        ok = isinstance(result, AIResponse)
        check("C10. Provider error object: success is False", ok and result.success is False,
              (result.error_message or "") if ok else "")
    except Exception as exc:  # noqa: BLE001
        check("C10. Provider error object: handled cleanly", False, repr(exc))
else:
    record("C. HTTP/parsing tests skipped: collector import failed", FAIL, "see import-safety checks")


# ---------------------------------------------------------------------------
# D. Live request: ONLY when a real OPENROUTER_API_KEY is already configured.
# ---------------------------------------------------------------------------
_live_key = (load_settings().openrouter_api_key or "") or None
if OpenRouterCollector is None:
    record("D. Live OpenRouter request", FAIL, "collector import failed")
elif not _live_key:
    record(
        "D. Live OpenRouter request",
        SKIP,
        "no OPENROUTER_API_KEY configured -- live request not attempted",
    )
else:
    live_request_made = True
    try:
        collector = OpenRouterCollector()  # real settings, configured free model
        prompt = _prompt("P-3C-LIVE", "In one short sentence, what is a pour-over coffee maker?")
        result = collector.collect(prompt)  # exactly one request

        ok = isinstance(result, AIResponse)
        check("D. Live: result is an AIResponse", ok, type(result).__name__)
        check("D. Live: source_type is OPENROUTER_FREE",
              ok and result.source_type == SourceType.OPENROUTER_FREE)
        check("D. Live: configured model was used",
              ok and result.model_name == load_settings().openrouter_model, str(ok and result.model_name))
        check("D. Live: original prompt / prompt_id preserved",
              ok and result.prompt == prompt.prompt and result.prompt_id == prompt.prompt_id)
        if ok and result.success:
            check("D. Live: answer is non-empty",
                  bool(result.answer and result.answer.strip()), (result.answer or "")[:120])
            check("D. Live: API key not present anywhere in the result",
                  _live_key not in result.model_dump_json())
        else:
            check("D. Live: request succeeded", False,
                  (result.error_message if ok else "not an AIResponse") or "unknown error")
    except Exception as exc:  # noqa: BLE001
        check("D. Live: collect() did not raise", False, repr(exc))


# ---------------------------------------------------------------------------
# E. Regression: prior manual phase scripts must still pass. Not modified.
# ---------------------------------------------------------------------------
for phase_label, script in (
    ("Phase 3B", "tests_phase3b_manual.py"),
    ("Phase 3A", "tests_phase3a_manual.py"),
    ("Phase 2", "tests_phase2_manual.py"),
    ("Phase 1", "tests_phase1_manual.py"),
):
    try:
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        last_line = (
            proc.stdout.strip().splitlines()[-1]
            if proc.stdout.strip()
            else proc.stderr.strip()
        )
        check(f"E. {phase_label} regression: {script} exits 0", proc.returncode == 0, last_line)
    except Exception as exc:  # noqa: BLE001
        check(f"E. {phase_label} regression: {script} exits 0", False, repr(exc))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n=== Phase 3C Manual Verification Results ===")
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
print(f"Actual OpenRouter API request made: {'YES' if live_request_made else 'NO'}")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
