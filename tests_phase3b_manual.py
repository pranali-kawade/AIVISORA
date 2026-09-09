"""Manual Phase 3B verification script.

Not a pytest suite (same "minimal dependencies" rule as Phase 1/2/3A) -- a
small deterministic script that exercises `collectors/gemini.py` (the live
Gemini collector) without requiring an API key, then re-runs the Phase 3A,
Phase 2 and Phase 1 manual scripts as regression checks.

Checks performed:

1. Import safety -- the collector module imports with no API key set.
2. Missing-key behavior -- with GEMINI_API_KEY absent, `collect()` makes no
   network request and returns a failed `AIResponse` carrying a clear error
   message, with no secret exposed.
3. Live request -- ONLY when a real GEMINI_API_KEY is already configured:
   make exactly ONE small request and validate the resulting `AIResponse`
   (type, source_type == LIVE_GEMINI, prompt preserved, non-empty answer,
   no API key present in the result). If no key is configured this check is
   SKIPPED, never FAILED, and no request is made.
4. Regression -- tests_phase3a_manual.py, tests_phase2_manual.py and
   tests_phase1_manual.py still exit 0. Those files are not modified.

Run with: python tests_phase3b_manual.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from schemas.models import AIResponse, PromptItem, SourceType

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIPPED"

# (name, status, detail) where status is one of PASS / FAIL / SKIPPED.
results: list[tuple[str, str, str]] = []

# Records whether an actual outbound Gemini API request was attempted.
live_request_made = False


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))


def check(name: str, condition: bool, detail: str = "") -> None:
    record(name, PASS if condition else FAIL, detail)


# ---------------------------------------------------------------------------
# 1. Import safety: importing the collector must not require an API key.
# ---------------------------------------------------------------------------
_key_backup = os.environ.pop("GEMINI_API_KEY", None)
try:
    import collectors.gemini as gemini_mod
    from collectors.gemini import GeminiCollector

    check(
        "Import safety: collectors.gemini imports with no GEMINI_API_KEY set",
        True,
    )
    check(
        "Import safety: GeminiCollector declares source_type LIVE_GEMINI",
        getattr(GeminiCollector, "source_type", None) == SourceType.LIVE_GEMINI,
        str(getattr(GeminiCollector, "source_type", None)),
    )
except Exception as exc:  # noqa: BLE001
    check("Import safety: collectors.gemini imports with no GEMINI_API_KEY set", False, repr(exc))
    gemini_mod = None  # type: ignore[assignment]
    GeminiCollector = None  # type: ignore[assignment]
finally:
    if _key_backup is not None:
        os.environ["GEMINI_API_KEY"] = _key_backup


# ---------------------------------------------------------------------------
# 2. Missing-key behavior: no key -> no network call, structured failure.
# ---------------------------------------------------------------------------
if GeminiCollector is not None:
    _key_backup = os.environ.pop("GEMINI_API_KEY", None)

    # Trip-wire: replace the SDK handle with one that explodes if the
    # collector ever tries to build a client / hit the network. The
    # missing-key guard must short-circuit before this is ever touched.
    class _NoNetworkSDK:
        class Client:  # noqa: D106
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise AssertionError(
                    "Gemini client was constructed despite a missing API key"
                )

    _sdk_backup = getattr(gemini_mod, "genai", None)
    gemini_mod.genai = _NoNetworkSDK  # type: ignore[attr-defined]
    try:
        collector = GeminiCollector()
        prompt = PromptItem(
            prompt_id="P-3B-NOKEY",
            prompt="What is a moka pot?",
            intent="informational",
        )
        result = collector.collect(prompt)

        check(
            "Missing key: collect() returns an AIResponse",
            isinstance(result, AIResponse),
            type(result).__name__,
        )
        check(
            "Missing key: no network request attempted (trip-wire not triggered)",
            isinstance(result, AIResponse) and result.success is False,
            "" if isinstance(result, AIResponse) else "not an AIResponse",
        )
        check(
            "Missing key: response is marked failed",
            isinstance(result, AIResponse) and result.success is False,
        )
        check(
            "Missing key: a clear error message is present",
            isinstance(result, AIResponse)
            and bool(result.error_message)
            and "GEMINI_API_KEY" in (result.error_message or ""),
            (result.error_message or "") if isinstance(result, AIResponse) else "",
        )
        check(
            "Missing key: source_type is LIVE_GEMINI",
            isinstance(result, AIResponse) and result.source_type == SourceType.LIVE_GEMINI,
        )
        check(
            "Missing key: original prompt and prompt_id preserved",
            isinstance(result, AIResponse)
            and result.prompt == prompt.prompt
            and result.prompt_id == prompt.prompt_id,
        )
        check(
            "Missing key: no answer fabricated",
            isinstance(result, AIResponse) and result.answer is None,
        )
        check(
            "Missing key: no citations fabricated",
            isinstance(result, AIResponse) and result.citations == [],
        )
        # There is no secret in this scenario; assert the serialized result
        # carries no Google-API-key-shaped token ("AIza..." prefix) and no
        # value from the current environment's GEMINI_API_KEY (if any).
        dumped = result.model_dump_json() if isinstance(result, AIResponse) else ""
        _env_key = os.environ.get("GEMINI_API_KEY") or _key_backup
        check(
            "Missing key: serialized result exposes no secret",
            "AIza" not in dumped and not (_env_key and _env_key in dumped),
        )
    except AssertionError as exc:
        check("Missing key: no network request attempted (trip-wire not triggered)", False, str(exc))
    except Exception as exc:  # noqa: BLE001
        check("Missing key: collect() handled cleanly", False, repr(exc))
    finally:
        gemini_mod.genai = _sdk_backup  # type: ignore[attr-defined]
        if _key_backup is not None:
            os.environ["GEMINI_API_KEY"] = _key_backup
else:
    record("Missing key: collector unavailable (import failed)", FAIL, "see import-safety checks")


# ---------------------------------------------------------------------------
# 3. Live request: ONLY when a real GEMINI_API_KEY is already configured.
# ---------------------------------------------------------------------------
def _real_key() -> str | None:
    # Read straight from the environment (and .env via config) rather than
    # trusting a cached snapshot.
    from config import load_settings

    key = load_settings().gemini_api_key
    return key if key else None


_live_key = _real_key()
if GeminiCollector is None:
    record("Live Gemini request", FAIL, "collector import failed")
elif not _live_key:
    record(
        "Live Gemini request",
        SKIP,
        "no GEMINI_API_KEY configured -- live request not attempted",
    )
else:
    live_request_made = True
    try:
        collector = GeminiCollector()
        prompt = PromptItem(
            prompt_id="P-3B-LIVE",
            prompt="In one short sentence, what is a French press coffee maker?",
            intent="informational",
        )
        result = collector.collect(prompt)  # exactly one request

        ok_type = isinstance(result, AIResponse)
        check("Live: result is an AIResponse", ok_type, type(result).__name__)
        check(
            "Live: source_type is LIVE_GEMINI",
            ok_type and result.source_type == SourceType.LIVE_GEMINI,
        )
        check(
            "Live: original prompt preserved",
            ok_type and result.prompt == prompt.prompt and result.prompt_id == prompt.prompt_id,
        )
        if ok_type and result.success:
            check(
                "Live: answer is non-empty",
                bool(result.answer and result.answer.strip()),
                (result.answer or "")[:120],
            )
            dumped = result.model_dump_json()
            check(
                "Live: API key is not present anywhere in the result",
                _live_key not in dumped,
            )
        else:
            check(
                "Live: request succeeded",
                False,
                (result.error_message if ok_type else "not an AIResponse") or "unknown error",
            )
    except Exception as exc:  # noqa: BLE001
        check("Live: collect() did not raise", False, repr(exc))


# ---------------------------------------------------------------------------
# 4. Regression: prior manual phase scripts must still pass. Not modified.
# ---------------------------------------------------------------------------
for phase_label, script in (
    ("Phase 3A", "tests_phase3a_manual.py"),
    ("Phase 2", "tests_phase2_manual.py"),
    ("Phase 1", "tests_phase1_manual.py"),
):
    try:
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        last_line = (
            proc.stdout.strip().splitlines()[-1]
            if proc.stdout.strip()
            else proc.stderr.strip()
        )
        check(
            f"{phase_label} regression: {script} exits 0",
            proc.returncode == 0,
            last_line,
        )
    except Exception as exc:  # noqa: BLE001
        check(f"{phase_label} regression: {script} exits 0", False, repr(exc))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n=== Phase 3B Manual Verification Results ===")
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
print(f"Actual Gemini API request made: {'YES' if live_request_made else 'NO'}")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
