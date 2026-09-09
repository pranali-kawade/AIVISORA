"""Manual Phase 3D verification script: provider integration & common
response contract.

Not a pytest suite (same "minimal dependencies" rule as Phase 1/2/3A/3B/3C)
-- a small deterministic script that proves the two implemented collectors
(`GeminiCollector`, `OpenRouterCollector`) accept the SAME `PromptItem` and
produce the SAME common `schemas.models.AIResponse` contract, so downstream
analysis can treat them provider-agnostically.

This phase adds NO product functionality. It is integration verification
only. Every provider interaction below is driven by a deterministic local
fake -- no real Gemini or OpenRouter inference request is made and no API
key is required.

Run with: python3 tests_phase3d_manual.py
"""

from __future__ import annotations

import json
import subprocess
import sys

import collectors.gemini as gemini_mod
import collectors.openrouter as openrouter_mod
from collectors.gemini import GeminiCollector
from collectors.openrouter import OpenRouterCollector
from config import Settings, load_settings
from schemas.models import AIResponse, Citation, PromptItem, SourceType
from utils.normalization import normalize_answer_text

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIPPED"

results: list[tuple[str, str, str]] = []

# Sentinel keys -- never real, never sent anywhere (the transport layer is
# always faked). Used to prove the collectors redact the key on failure.
FAKE_GEMINI_KEY = "sk-fake-gemini-key-DO-NOT-USE"
FAKE_OPENROUTER_KEY = "sk-fake-openrouter-key-DO-NOT-USE"

# A deliberately messy raw answer: leading/trailing space, irregular inner
# spacing, tabs, repeated blank lines. Both providers get the exact same
# raw string so the normalized output can be compared for consistency.
MESSY_ANSWER = "   Espresso\t\tis\n\n\n  a   concentrated  coffee\tbrew.   "
EXPECTED_NORMALIZED = normalize_answer_text(MESSY_ANSWER)

# Tracks whether a real inference request was ever attempted (must stay False).
real_request_attempted = False


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))


def check(name: str, condition: bool, detail: str = "") -> None:
    record(name, PASS if condition else FAIL, detail)


def _settings(
    gemini_key: str | None = FAKE_GEMINI_KEY,
    openrouter_key: str | None = FAKE_OPENROUTER_KEY,
) -> Settings:
    """A Settings snapshot identical to the live config except for the two
    API keys, so collectors can be driven deterministically without the
    real environment.
    """
    base = load_settings()
    return Settings(
        gemini_api_key=gemini_key,
        openrouter_api_key=openrouter_key,
        gemini_model=base.gemini_model,
        openrouter_model=base.openrouter_model,
        request_timeout=base.request_timeout,
        mock_mode=base.mock_mode,
    )


# ---------------------------------------------------------------------------
# Deterministic local fakes -- Gemini SDK surface
# ---------------------------------------------------------------------------
class _FakeWeb:
    def __init__(self, uri: str | None, title: str | None) -> None:
        self.uri = uri
        self.title = title


class _FakeChunk:
    def __init__(self, uri: str | None, title: str | None) -> None:
        self.web = _FakeWeb(uri, title)


class _FakeGroundingMetadata:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self.grounding_chunks = chunks


class _FakeCandidate:
    def __init__(self, chunks: list[_FakeChunk] | None) -> None:
        self.content = None
        self.grounding_metadata = _FakeGroundingMetadata(chunks) if chunks is not None else None


class _FakeGeminiResponse:
    def __init__(self, text: str | None, candidates: list[_FakeCandidate]) -> None:
        self.text = text
        self.candidates = candidates


class _FakeGeminiModels:
    def __init__(self, response: _FakeGeminiResponse | None, exc: BaseException | None) -> None:
        self._response = response
        self._exc = exc
        self.calls = 0
        self.last_model: str | None = None
        self.last_contents: str | None = None

    def generate_content(self, model: str, contents: str):
        self.calls += 1
        self.last_model = model
        self.last_contents = contents
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeGeminiClient:
    def __init__(self, models: _FakeGeminiModels) -> None:
        self.models = models


class _FakeGenaiModule:
    """Stands in for the imported `google.genai` module handle."""

    def __init__(self, models: _FakeGeminiModels) -> None:
        self._models = models
        self.client_kwargs: dict | None = None

    def Client(self, **kwargs):  # noqa: N802 - mirrors the real SDK API
        self.client_kwargs = kwargs
        return _FakeGeminiClient(self._models)


def _run_gemini(
    *,
    prompt: PromptItem,
    text: str | None = None,
    exc: BaseException | None = None,
    chunks: list[_FakeChunk] | None = None,
    settings: Settings | None = None,
    tripwire: bool = False,
):
    """Drive GeminiCollector.collect() against a deterministic fake.
    Returns (result, fake_models). `tripwire=True` installs a genai handle
    that explodes if constructed (to prove no network attempt).
    """
    candidates = [_FakeCandidate(chunks)] if chunks is not None else []
    models = _FakeGeminiModels(_FakeGeminiResponse(text, candidates), exc)

    if tripwire:
        class _Boom:
            def Client(self, **_kwargs):  # noqa: N802
                raise AssertionError("Gemini client constructed despite missing key")

        fake_handle: object = _Boom()
    else:
        fake_handle = _FakeGenaiModule(models)

    original = gemini_mod.genai
    gemini_mod.genai = fake_handle  # type: ignore[assignment]
    try:
        collector = GeminiCollector(settings=settings or _settings())
        return collector.collect(prompt), models
    finally:
        gemini_mod.genai = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Deterministic local fakes -- OpenRouter HTTP surface
# ---------------------------------------------------------------------------
def _run_openrouter(
    *,
    prompt: PromptItem,
    body: str | None = None,
    status: int = 200,
    exc: BaseException | None = None,
    settings: Settings | None = None,
    tripwire: bool = False,
):
    """Drive OpenRouterCollector.collect() against a deterministic fake
    `_perform_request`. Returns (result, capture) where capture holds the
    call count and the payload/headers that would have been sent.
    """
    capture: dict = {"calls": 0, "payload": None, "headers": None, "url": None}

    def _fake(url, payload, headers, timeout):
        capture["calls"] += 1
        capture["url"] = url
        capture["payload"] = payload
        capture["headers"] = headers
        if tripwire:
            raise AssertionError("OpenRouter request attempted despite missing key")
        if exc is not None:
            raise exc
        return status, body

    original = openrouter_mod._perform_request
    openrouter_mod._perform_request = _fake  # type: ignore[assignment]
    try:
        collector = OpenRouterCollector(settings=settings or _settings())
        return collector.collect(prompt), capture
    finally:
        openrouter_mod._perform_request = original  # type: ignore[assignment]


def _openrouter_body(content: str | None, annotations: list | None = None) -> str:
    message: dict = {"role": "assistant", "content": content}
    if annotations is not None:
        message["annotations"] = annotations
    return json.dumps({"id": "gen-3d", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})


# ===========================================================================
# A. Common PromptItem accepted by both collectors
# ===========================================================================
PROMPT = PromptItem(
    prompt_id="P-3D-001",
    prompt="What should someone know about camping coffee makers?",
    intent="informational",
)

try:
    g_result, g_models = _run_gemini(prompt=PROMPT, text="Gemini says things about camping coffee makers.")
    o_result, o_capture = _run_openrouter(
        prompt=PROMPT, body=_openrouter_body("OpenRouter says other things about camping coffee makers.")
    )

    check("A. Gemini accepts the shared PromptItem (returns AIResponse)", isinstance(g_result, AIResponse))
    check("A. OpenRouter accepts the shared PromptItem (returns AIResponse)", isinstance(o_result, AIResponse))

    gemini_sent = g_models.last_contents
    openrouter_sent = (o_capture["payload"] or {}).get("messages", [{}])[0].get("content")
    check(
        "A. Identical prompt text is sent to both providers",
        gemini_sent == PROMPT.prompt and openrouter_sent == PROMPT.prompt,
        f"gemini={gemini_sent!r} openrouter={openrouter_sent!r}",
    )
    check(
        "A. prompt_id is preserved unchanged by both",
        g_result.prompt_id == PROMPT.prompt_id and o_result.prompt_id == PROMPT.prompt_id,
    )
    check(
        "A. prompt text is preserved unchanged on both AIResponses",
        g_result.prompt == PROMPT.prompt and o_result.prompt == PROMPT.prompt,
    )
    check(
        "A. OpenRouter used exactly one request; Gemini exactly one generate_content call",
        o_capture["calls"] == 1 and g_models.calls == 1,
        f"openrouter={o_capture['calls']} gemini={g_models.calls}",
    )
except Exception as exc:  # noqa: BLE001
    check("A. Common PromptItem handling", False, repr(exc))


# ===========================================================================
# B. Gemini -> AIResponse (deterministic fake, no real request)
# ===========================================================================
try:
    result, models = _run_gemini(prompt=PROMPT, text="  A tidy Gemini answer.  ")
    ok = isinstance(result, AIResponse)
    check("B. Gemini returns an AIResponse", ok, type(result).__name__)
    check("B. Gemini source_type == LIVE_GEMINI", ok and result.source_type == SourceType.LIVE_GEMINI)
    check("B. Gemini prompt_id correct", ok and result.prompt_id == PROMPT.prompt_id)
    check("B. Gemini prompt correct", ok and result.prompt == PROMPT.prompt)
    check(
        "B. Gemini answer is normalized",
        ok and result.answer == "A tidy Gemini answer.",
        repr(ok and result.answer),
    )
    check("B. Gemini citations is a valid list", ok and isinstance(result.citations, list))
    check("B. Gemini success is True", ok and result.success is True)
    check("B. Gemini has no error_message", ok and result.error_message is None)
    check("B. Gemini model_name is the configured model", ok and result.model_name == load_settings().gemini_model)
    check("B. Gemini made no real request (fake generate_content used once)", models.calls == 1)
except Exception as exc:  # noqa: BLE001
    check("B. Gemini -> AIResponse", False, repr(exc))


# ===========================================================================
# C. OpenRouter -> AIResponse (deterministic fake, no real request)
# ===========================================================================
try:
    result, capture = _run_openrouter(prompt=PROMPT, body=_openrouter_body("  A tidy OpenRouter answer.  "))
    ok = isinstance(result, AIResponse)
    check("C. OpenRouter returns an AIResponse", ok, type(result).__name__)
    check("C. OpenRouter source_type == OPENROUTER_FREE", ok and result.source_type == SourceType.OPENROUTER_FREE)
    check("C. OpenRouter prompt_id correct", ok and result.prompt_id == PROMPT.prompt_id)
    check("C. OpenRouter prompt correct", ok and result.prompt == PROMPT.prompt)
    check(
        "C. OpenRouter answer is normalized",
        ok and result.answer == "A tidy OpenRouter answer.",
        repr(ok and result.answer),
    )
    check("C. OpenRouter citations is a valid list", ok and isinstance(result.citations, list))
    check("C. OpenRouter success is True", ok and result.success is True)
    check("C. OpenRouter has no error_message", ok and result.error_message is None)
    check(
        "C. OpenRouter model_name is the configured model",
        ok and result.model_name == load_settings().openrouter_model,
    )
    check("C. OpenRouter made no real request (fake _perform_request used once)", capture["calls"] == 1)
except Exception as exc:  # noqa: BLE001
    check("C. OpenRouter -> AIResponse", False, repr(exc))


# ===========================================================================
# D. Common response contract -- both expose the same AIResponse field set
# ===========================================================================
try:
    gemini_response, _ = _run_gemini(prompt=PROMPT, text="Gemini content here.")
    openrouter_response, _ = _run_openrouter(prompt=PROMPT, body=_openrouter_body("OpenRouter content here."))

    responses = [gemini_response, openrouter_response]
    required_fields = {
        "source_type",
        "model_name",
        "prompt_id",
        "prompt",
        "answer",
        "citations",
        "brand_mentions",
        "timestamp",
        "success",
        "error_message",
    }

    uniform = True
    detail = ""
    field_sets = []
    for response in responses:
        if not isinstance(response, AIResponse):
            uniform = False
            detail = "not an AIResponse"
            break
        keys = set(response.model_dump().keys())
        field_sets.append(keys)
        if not required_fields.issubset(keys):
            uniform = False
            detail = f"missing {required_fields - keys}"
        if not isinstance(response.answer, str) or not response.answer:
            uniform = False
            detail = "answer not a non-empty str"
        if not isinstance(response.citations, list):
            uniform = False
            detail = "citations not a list"
        if response.success is not True:
            uniform = False
            detail = "success not True"

    check("D. Both providers expose identical AIResponse field sets", uniform and field_sets[0] == field_sets[1], detail)
    check(
        "D. Downstream code can iterate both responses uniformly (no provider branching)",
        uniform
        and all(
            isinstance(r, AIResponse) and r.success and isinstance(r.answer, str) and isinstance(r.citations, list)
            for r in responses
        ),
    )
except Exception as exc:  # noqa: BLE001
    check("D. Common response contract", False, repr(exc))


# ===========================================================================
# E. Provider identity -- LIVE_GEMINI vs OPENROUTER_FREE stays intact
# ===========================================================================
try:
    gemini_response, _ = _run_gemini(prompt=PROMPT, text="g")
    openrouter_response, _ = _run_openrouter(prompt=PROMPT, body=_openrouter_body("o"))
    check("E. Gemini identity -> SourceType.LIVE_GEMINI", gemini_response.source_type == SourceType.LIVE_GEMINI)
    check(
        "E. OpenRouter identity -> SourceType.OPENROUTER_FREE",
        openrouter_response.source_type == SourceType.OPENROUTER_FREE,
    )
    check(
        "E. The two provider identities are distinct",
        gemini_response.source_type != openrouter_response.source_type,
    )
except Exception as exc:  # noqa: BLE001
    check("E. Provider identity", False, repr(exc))


# ===========================================================================
# F. Failure contract -- both use the existing AIResponse failure structure
#    and redact the API key.
# ===========================================================================
try:
    g_fail, _ = _run_gemini(prompt=PROMPT, exc=RuntimeError(f"upstream blew up with token {FAKE_GEMINI_KEY}"))
    o_fail, _ = _run_openrouter(
        prompt=PROMPT, exc=RuntimeError(f"upstream blew up with token {FAKE_OPENROUTER_KEY}")
    )

    for label, res, key, src in (
        ("Gemini", g_fail, FAKE_GEMINI_KEY, SourceType.LIVE_GEMINI),
        ("OpenRouter", o_fail, FAKE_OPENROUTER_KEY, SourceType.OPENROUTER_FREE),
    ):
        ok = isinstance(res, AIResponse)
        check(f"F. {label} failure is an AIResponse (no new error schema)", ok, type(res).__name__)
        check(f"F. {label} failure: success is False", ok and res.success is False)
        check(f"F. {label} failure: error_message is present", ok and bool(res.error_message))
        check(f"F. {label} failure: source_type unchanged ({src.value})", ok and res.source_type == src)
        check(f"F. {label} failure: prompt_id preserved", ok and res.prompt_id == PROMPT.prompt_id)
        check(f"F. {label} failure: prompt preserved", ok and res.prompt == PROMPT.prompt)
        check(f"F. {label} failure: answer is None", ok and res.answer is None)
        dumped = res.model_dump_json() if ok else ""
        check(
            f"F. {label} failure: API key is not exposed anywhere in the result",
            ok and key not in (res.error_message or "") and key not in dumped,
            (res.error_message or "")[:160] if ok else "",
        )
except Exception as exc:  # noqa: BLE001
    check("F. Failure contract", False, repr(exc))


# ===========================================================================
# G. Missing-key behavior -- structured failure, NO network attempt.
# ===========================================================================
try:
    g_nokey, g_models = _run_gemini(
        prompt=PROMPT, settings=_settings(gemini_key=None), tripwire=True
    )
    ok = isinstance(g_nokey, AIResponse)
    check("G. Gemini missing key: returns structured AIResponse failure", ok and g_nokey.success is False)
    check("G. Gemini missing key: source_type == LIVE_GEMINI", ok and g_nokey.source_type == SourceType.LIVE_GEMINI)
    check("G. Gemini missing key: error_message present", ok and bool(g_nokey.error_message))
    check("G. Gemini missing key: prompt / prompt_id preserved",
          ok and g_nokey.prompt == PROMPT.prompt and g_nokey.prompt_id == PROMPT.prompt_id)
    check("G. Gemini missing key: no network attempt (generate_content never called)", g_models.calls == 0)
except AssertionError as exc:
    check("G. Gemini missing key: no network attempt", False, str(exc))
except Exception as exc:  # noqa: BLE001
    check("G. Gemini missing-key behavior", False, repr(exc))

try:
    o_nokey, o_capture = _run_openrouter(
        prompt=PROMPT, settings=_settings(openrouter_key=None), tripwire=True
    )
    ok = isinstance(o_nokey, AIResponse)
    check("G. OpenRouter missing key: returns structured AIResponse failure", ok and o_nokey.success is False)
    check(
        "G. OpenRouter missing key: source_type == OPENROUTER_FREE",
        ok and o_nokey.source_type == SourceType.OPENROUTER_FREE,
    )
    check("G. OpenRouter missing key: error_message present", ok and bool(o_nokey.error_message))
    check("G. OpenRouter missing key: prompt / prompt_id preserved",
          ok and o_nokey.prompt == PROMPT.prompt and o_nokey.prompt_id == PROMPT.prompt_id)
    check("G. OpenRouter missing key: no network attempt (_perform_request never called)", o_capture["calls"] == 0)
except AssertionError as exc:
    check("G. OpenRouter missing key: no network attempt", False, str(exc))
except Exception as exc:  # noqa: BLE001
    check("G. OpenRouter missing-key behavior", False, repr(exc))


# ===========================================================================
# H. Normalization consistency -- same messy input -> same normalized output
#    via the existing utils.normalization implementation.
# ===========================================================================
try:
    g_messy, _ = _run_gemini(prompt=PROMPT, text=MESSY_ANSWER)
    o_messy, _ = _run_openrouter(prompt=PROMPT, body=_openrouter_body(MESSY_ANSWER))

    check(
        "H. Gemini normalizes messy answer per utils.normalization",
        isinstance(g_messy, AIResponse) and g_messy.answer == EXPECTED_NORMALIZED,
        repr(getattr(g_messy, "answer", None)),
    )
    check(
        "H. OpenRouter normalizes messy answer per utils.normalization",
        isinstance(o_messy, AIResponse) and o_messy.answer == EXPECTED_NORMALIZED,
        repr(getattr(o_messy, "answer", None)),
    )
    check(
        "H. Both providers produce the SAME normalized text from the SAME raw input",
        isinstance(g_messy, AIResponse)
        and isinstance(o_messy, AIResponse)
        and g_messy.answer == o_messy.answer == EXPECTED_NORMALIZED,
    )
    check(
        "H. Normalized answer has no leading/trailing space and no repeated blank lines",
        isinstance(g_messy, AIResponse)
        and g_messy.answer == g_messy.answer.strip()
        and "\n\n" not in g_messy.answer
        and "  " not in g_messy.answer,
    )
except Exception as exc:  # noqa: BLE001
    check("H. Normalization consistency", False, repr(exc))


# ===========================================================================
# I. Citation contract -- both return the existing common Citation shape;
#    a response with no citation structure must not crash.
# ===========================================================================
try:
    citation_url = "https://www.example.org/camping-coffee-guide"
    g_cited, _ = _run_gemini(
        prompt=PROMPT,
        text="Gemini answer citing a source.",
        chunks=[_FakeChunk(citation_url, "  Camping Coffee Guide  ")],
    )
    o_cited, _ = _run_openrouter(
        prompt=PROMPT,
        body=_openrouter_body(
            "OpenRouter answer citing a source.",
            annotations=[{"type": "url_citation", "url_citation": {"url": citation_url, "title": "  Camping Coffee Guide  "}}],
        ),
    )

    def _one_citation_ok(res) -> bool:
        return (
            isinstance(res, AIResponse)
            and res.success is True
            and isinstance(res.citations, list)
            and len(res.citations) == 1
            and isinstance(res.citations[0], Citation)
            and str(res.citations[0].url).rstrip("/") == citation_url
            and res.citations[0].title == "Camping Coffee Guide"
            and res.citations[0].domain == "example.org"
        )

    check("I. Gemini returns a citation in the common Citation representation", _one_citation_ok(g_cited),
          str(getattr(g_cited, "citations", None)))
    check("I. OpenRouter returns a citation in the common Citation representation", _one_citation_ok(o_cited),
          str(getattr(o_cited, "citations", None)))
    check(
        "I. Both providers' citations use the identical Citation type/fields",
        isinstance(g_cited, AIResponse)
        and isinstance(o_cited, AIResponse)
        and type(g_cited.citations[0]) is type(o_cited.citations[0])
        and g_cited.citations[0].model_dump().keys() == o_cited.citations[0].model_dump().keys(),
    )

    # No citation structure at all -> no crash, citations == [].
    g_plain, _ = _run_gemini(prompt=PROMPT, text="Gemini answer with no sources.")
    o_plain, _ = _run_openrouter(prompt=PROMPT, body=_openrouter_body("OpenRouter answer with no sources."))
    check(
        "I. Response with no citation structure does not crash (Gemini)",
        isinstance(g_plain, AIResponse) and g_plain.success is True and g_plain.citations == [],
    )
    check(
        "I. Response with no citation structure does not crash (OpenRouter)",
        isinstance(o_plain, AIResponse) and o_plain.success is True and o_plain.citations == [],
    )
except Exception as exc:  # noqa: BLE001
    check("I. Citation contract", False, repr(exc))


# ===========================================================================
# J. Mixed provider collection -- provider-agnostic downstream handling.
# ===========================================================================
try:
    gemini_response, _ = _run_gemini(prompt=PROMPT, text="Gemini mixed-collection answer.")
    openrouter_response, _ = _run_openrouter(prompt=PROMPT, body=_openrouter_body("OpenRouter mixed-collection answer."))

    responses = [gemini_response, openrouter_response]

    check("J. Every item in the mixed collection is an AIResponse", all(isinstance(r, AIResponse) for r in responses))
    check(
        "J. The collection holds two distinct provider identities",
        len({r.source_type for r in responses}) == 2
        and {r.source_type for r in responses} == {SourceType.LIVE_GEMINI, SourceType.OPENROUTER_FREE},
    )
    check(
        "J. Both responses retain the same prompt_id",
        len({r.prompt_id for r in responses}) == 1 and responses[0].prompt_id == PROMPT.prompt_id,
    )
    check(
        "J. Both responses retain the same prompt text",
        len({r.prompt for r in responses}) == 1 and responses[0].prompt == PROMPT.prompt,
    )
    # Downstream-style pass: iterate uniformly, no provider-specific types.
    summary = []
    for r in responses:
        assert isinstance(r, AIResponse)
        summary.append((r.source_type.value, r.prompt_id, bool(r.answer), isinstance(r.citations, list)))
    check(
        "J. Downstream code processes the mixed collection with no provider-specific types",
        summary
        == [
            ("live_gemini", "P-3D-001", True, True),
            ("openrouter_free", "P-3D-001", True, True),
        ],
        str(summary),
    )
except Exception as exc:  # noqa: BLE001
    check("J. Mixed provider collection", False, repr(exc))


# ===========================================================================
# Regression: every earlier manual phase script still passes. Not modified.
# ===========================================================================
for phase_label, script in (
    ("Phase 3C", "tests_phase3c_manual.py"),
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
            proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        )
        check(f"Regression: {phase_label} ({script}) exits 0", proc.returncode == 0, last_line)
    except Exception as exc:  # noqa: BLE001
        check(f"Regression: {phase_label} ({script}) exits 0", False, repr(exc))


# ===========================================================================
# Report
# ===========================================================================
print("\n=== Phase 3D Manual Verification Results ===")
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
print(f"Real Gemini inference request made: {'YES' if real_request_attempted else 'NO'}")
print(f"Real OpenRouter inference request made: {'YES' if real_request_attempted else 'NO'}")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
