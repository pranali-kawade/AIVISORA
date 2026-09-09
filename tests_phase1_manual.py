"""Manual Phase 1 verification script.

Not a pytest suite (no test framework dependency added yet, per the
"minimal dependencies" rule) -- a small deterministic script that
exercises the schemas, config, and helper layers and reports pass/fail.
Run with: python tests_phase1_manual.py
"""

from __future__ import annotations

import os
import sys

from pydantic import ValidationError

from schemas.models import AIResponse, Citation, PromptItem, SourceType

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))


# --- PromptItem: valid ---
try:
    p = PromptItem(prompt_id="p1", prompt="best running shoes for beginners", intent="recommendation")
    check("PromptItem valid construction", p.prompt_id == "p1")
except Exception as exc:  # noqa: BLE001 - test script, want to see any failure
    check("PromptItem valid construction", False, str(exc))

# --- PromptItem: invalid (empty prompt) ---
try:
    PromptItem(prompt_id="p2", prompt="", intent="recommendation")
    check("PromptItem rejects empty prompt", False, "did not raise")
except ValidationError:
    check("PromptItem rejects empty prompt", True)

# --- PromptItem: invalid (missing required field) ---
try:
    PromptItem(prompt="missing id")  # type: ignore[call-arg]
    check("PromptItem rejects missing prompt_id", False, "did not raise")
except ValidationError:
    check("PromptItem rejects missing prompt_id", True)

# --- SourceType enum validation ---
try:
    AIResponse(
        source_type="chatgpt",  # type: ignore[arg-type]
        prompt_id="p1",
        prompt="x",
        success=True,
        answer="irrelevant, should fail before this matters",
    )
    check("SourceType rejects invalid value ('chatgpt')", False, "did not raise")
except ValidationError:
    check("SourceType rejects invalid value ('chatgpt')", True)

# --- AIResponse: valid success case ---
try:
    resp = AIResponse(
        source_type=SourceType.MOCK,
        model_name="mock-model-v1",
        prompt_id="p1",
        prompt="best running shoes for beginners",
        answer="Some mocked answer text.",
        citations=[Citation(url="https://example.com/shoes", title="Best Shoes", domain="example.com")],
        success=True,
    )
    check("AIResponse valid success case", resp.success is True and len(resp.citations) == 1)
except Exception as exc:  # noqa: BLE001
    check("AIResponse valid success case", False, str(exc))

# --- AIResponse: valid failure case (no answer, has error_message) ---
try:
    resp = AIResponse(
        source_type=SourceType.LIVE_GEMINI,
        prompt_id="p3",
        prompt="best running shoes",
        answer=None,
        success=False,
        error_message="Request timed out after 30s",
    )
    check("AIResponse valid failure case", resp.success is False and resp.answer is None)
except Exception as exc:  # noqa: BLE001
    check("AIResponse valid failure case", False, str(exc))

# --- AIResponse: failure without error_message must be rejected ---
try:
    AIResponse(
        source_type=SourceType.LIVE_GEMINI,
        prompt_id="p4",
        prompt="best running shoes",
        success=False,
    )
    check("AIResponse rejects failure without error_message", False, "did not raise")
except (ValidationError, ValueError):
    check("AIResponse rejects failure without error_message", True)

# --- AIResponse: citations optional / defaults to empty list ---
try:
    resp = AIResponse(
        source_type=SourceType.GOOGLE_AIO_OBSERVED,
        prompt_id="p5",
        prompt="best running shoes",
        answer="Observed sample answer.",
        success=True,
    )
    check("AIResponse citations default to empty list", resp.citations == [])
except Exception as exc:  # noqa: BLE001
    check("AIResponse citations default to empty list", False, str(exc))

# --- Citation: all fields optional ---
try:
    c = Citation()
    check("Citation allows all-empty construction", c.url is None and c.title is None and c.domain is None)
except Exception as exc:  # noqa: BLE001
    check("Citation allows all-empty construction", False, str(exc))

# --- Config: no API keys present ---
for key in ("GEMINI_API_KEY", "OPENROUTER_API_KEY"):
    os.environ.pop(key, None)
import config as config_module  # noqa: E402

settings_no_keys = config_module.load_settings()
check(
    "Config: reports not configured when keys absent",
    settings_no_keys.gemini_configured is False and settings_no_keys.openrouter_configured is False,
)

# --- Config: API keys present ---
os.environ["GEMINI_API_KEY"] = "fake-key-for-test-only"
os.environ["OPENROUTER_API_KEY"] = "fake-key-for-test-only"
settings_with_keys = config_module.load_settings()
check(
    "Config: reports configured when keys present",
    settings_with_keys.gemini_configured is True and settings_with_keys.openrouter_configured is True,
)
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)

# --- Config: mock mode enabled/disabled ---
os.environ["MOCK_MODE"] = "true"
check("Config: mock mode enabled parses true", config_module.load_settings().mock_mode is True)
os.environ["MOCK_MODE"] = "false"
check("Config: mock mode disabled parses false", config_module.load_settings().mock_mode is False)
os.environ.pop("MOCK_MODE", None)

# --- Config: helpers never leak secret values in status dict ---
from utils.helpers import configuration_status  # noqa: E402

os.environ["GEMINI_API_KEY"] = "super-secret-value-should-not-appear"
status = configuration_status(config_module.load_settings())
leaked = any("super-secret-value-should-not-appear" in str(v) for v in status.values())
check("configuration_status never leaks key values", not leaked)
os.environ.pop("GEMINI_API_KEY", None)

# --- report ---
print("\n=== Phase 1 Manual Verification Results ===")
all_ok = True
for name, ok, detail in results:
    status_label = PASS if ok else FAIL
    if not ok:
        all_ok = False
    line = f"[{status_label}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)

print("\nOVERALL:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
