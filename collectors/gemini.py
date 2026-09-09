"""Live Gemini collector (Phase 3B).

Turns a single prompt into exactly one call to the Google Gemini API using
the current official Google Gen AI SDK (`from google import genai`), then
funnels the raw provider response through the shared Phase 3A normalization
layer to produce the existing `schemas.models.AIResponse`. No new response
schema is introduced.

Flow:

    Gemini API response  ->  utils.normalization  ->  AIResponse

Zero-budget / safety constraints honoured here:

- one request per `collect()` call, no retry loop;
- free-tier model + endpoint only (whatever ``GEMINI_MODEL`` is configured
  to -- the collector does not override it);
- if ``GEMINI_API_KEY`` is absent the collector never touches the network
  and returns a structured ``AIResponse(success=False, ...)``;
- the API key is read only from configuration, is never logged, never put
  into an exception message, and never stored on the returned object.
  Error strings are additionally scrubbed of the key as a defensive
  backstop.

The module is import-safe: it loads even when neither ``GEMINI_API_KEY``
nor the SDK is available. Those conditions are reported as failed
responses at ``collect()`` time, not as import errors.
"""

from __future__ import annotations

from typing import Any, Optional

from collectors.base import BaseCollector
from config import Settings, load_settings
from schemas.models import AIResponse, PromptItem, SourceType
from utils.normalization import normalize_answer_text, normalize_citations

# --- Optional SDK import -------------------------------------------------------
# The SDK is treated as optional so the project stays importable without it.
try:
    from google import genai  # type: ignore[import-not-found]

    _SDK_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 - any import failure must be survivable
    genai = None  # type: ignore[assignment]
    _SDK_IMPORT_ERROR = (
        "google-genai SDK is not importable "
        f"({type(exc).__name__}); install it to enable live Gemini requests"
    )

try:
    from google.genai import types as genai_types  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 - types submodule is a nice-to-have (timeout)
    genai_types = None  # type: ignore[assignment]


_MISSING_KEY_MESSAGE = (
    "GEMINI_API_KEY is not configured; the Gemini collector cannot make a "
    "live request. Set GEMINI_API_KEY in the environment or .env to enable it."
)


class GeminiCollector(BaseCollector):
    """Collects a single live answer from the Gemini API for one prompt.

    Always returns a validated :class:`AIResponse` -- ``success=True`` with a
    normalized answer on success, or ``success=False`` with an
    ``error_message`` on any failure (missing key, missing SDK, API error,
    network/timeout error, or a malformed provider response). It never
    raises for those expected failure modes and never turns a failure into
    a successful empty response.
    """

    source_type: SourceType = SourceType.LIVE_GEMINI

    def __init__(self, settings: Optional[Settings] = None) -> None:
        # Snapshot configuration once per instance (not per call) so tests
        # can mutate the environment and then build a fresh collector.
        self._settings: Settings = settings if settings is not None else load_settings()

    # -- Public API ----------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The configured Gemini model id (never a secret)."""
        return self._settings.gemini_model

    def collect(self, prompt: PromptItem) -> AIResponse:
        # 1. No API key -> no network, structured failure.
        if not self._settings.gemini_configured:
            return self._failure(prompt, _MISSING_KEY_MESSAGE)

        # 2. SDK unavailable -> still no network, structured failure.
        if genai is None or _SDK_IMPORT_ERROR is not None:
            return self._failure(prompt, _SDK_IMPORT_ERROR or "google-genai SDK unavailable")

        # 3. Exactly one request. No retry loop.
        try:
            client = genai.Client(**self._client_kwargs())
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt.prompt,
            )
        except Exception as exc:  # noqa: BLE001 - translate every failure
            return self._failure(prompt, self._describe_error(exc))

        # 4. Parse the provider payload.
        try:
            answer_raw = self._extract_answer_text(response)
            raw_citations = self._extract_raw_citations(response)
        except Exception as exc:  # noqa: BLE001
            return self._failure(
                prompt,
                "Gemini returned a response that could not be parsed: "
                + self._describe_error(exc),
            )

        # 5. Normalize (Phase 3A layer).
        answer = normalize_answer_text(answer_raw)
        if not answer:
            return self._failure(
                prompt,
                "Gemini returned an empty or content-free response.",
            )

        # `normalize_citations` returns None ("unavailable") when Gemini gave
        # us no grounding data at all; AIResponse.citations can only hold a
        # list, so an unavailable signal collapses to an empty list here.
        citations = normalize_citations(raw_citations) or []

        # 6. Return the existing schema. Original prompt preserved verbatim.
        return AIResponse(
            source_type=SourceType.LIVE_GEMINI,
            model_name=self.model_name,
            prompt_id=prompt.prompt_id,
            prompt=prompt.prompt,
            answer=answer,
            citations=citations,
            success=True,
        )

    # -- Request construction ---------------------------------------------------

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"api_key": self._settings.gemini_api_key}
        http_options = self._http_options()
        if http_options is not None:
            kwargs["http_options"] = http_options
        return kwargs

    def _http_options(self) -> Any:
        """Best-effort request timeout from the existing configuration.

        ``google-genai`` expresses the HTTP timeout in milliseconds. If the
        installed SDK does not accept the option, we fall back to its
        default rather than failing the request.
        """
        if genai_types is None:
            return None
        try:
            return genai_types.HttpOptions(timeout=self._timeout_ms())
        except Exception:  # noqa: BLE001 - SDK version differences
            return None

    def _timeout_ms(self) -> int:
        try:
            seconds = int(self._settings.request_timeout)
        except (TypeError, ValueError):
            seconds = 30
        if seconds <= 0:
            seconds = 30
        return seconds * 1000

    # -- Response parsing -----------------------------------------------------

    @staticmethod
    def _extract_answer_text(response: Any) -> Optional[str]:
        """Pull the generated answer text out of a Gemini response object.

        Prefers the SDK's ``.text`` convenience accessor, then falls back to
        walking ``candidates -> content -> parts -> text``. Returns ``None``
        when no text was produced (which the caller treats as a failure,
        not as a successful empty answer).
        """
        try:
            text = response.text
        except Exception:  # noqa: BLE001 - .text can raise on non-text parts
            text = None
        if isinstance(text, str) and text.strip():
            return text

        collected: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    collected.append(part_text)
        if collected:
            return "".join(collected)

        return text if isinstance(text, str) else None

    @staticmethod
    def _extract_raw_citations(
        response: Any,
    ) -> Optional[list[dict[str, Optional[str]]]]:
        """Extract raw citation dicts from Gemini grounding metadata.

        Returns:
        - ``None`` when the response carries no grounding metadata at all
          (Gemini provided nothing -- "unavailable", never fabricated);
        - a list of ``{"url", "title", "domain"}`` dicts (possibly empty)
          when grounding metadata was present. Individual entries with no
          usable url/title are skipped here and by the normalization layer.
        """
        found_metadata = False
        raw: list[dict[str, Optional[str]]] = []

        for candidate in getattr(response, "candidates", None) or []:
            meta = getattr(candidate, "grounding_metadata", None)
            if meta is None:
                continue
            found_metadata = True
            for chunk in getattr(meta, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web is None:
                    continue
                uri = getattr(web, "uri", None)
                title = getattr(web, "title", None)
                if not uri and not title:
                    continue
                raw.append(
                    {
                        "url": uri if isinstance(uri, str) else None,
                        "title": title if isinstance(title, str) else None,
                        "domain": None,
                    }
                )

        if not found_metadata:
            return None
        return raw

    # -- Failure handling ---------------------------------------------------

    def _failure(self, prompt: PromptItem, message: str) -> AIResponse:
        return AIResponse(
            source_type=SourceType.LIVE_GEMINI,
            model_name=self.model_name,
            prompt_id=prompt.prompt_id,
            prompt=prompt.prompt,
            answer=None,
            citations=[],
            success=False,
            error_message=self._scrub(message) or "Gemini collection failed.",
        )

    def _describe_error(self, exc: BaseException) -> str:
        label = type(exc).__name__
        detail = self._scrub(str(exc))
        return f"{label}: {detail}" if detail else label

    def _scrub(self, text: str) -> str:
        """Defensive backstop: never let the API key leak through a message."""
        if not text:
            return ""
        key = self._settings.gemini_api_key
        if key and key in text:
            text = text.replace(key, "***REDACTED***")
        return text
