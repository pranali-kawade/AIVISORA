"""OpenRouter free-model collector (Phase 3C).

Sends one benchmark prompt to an OpenRouter **free model** via OpenRouter's
standard OpenAI-compatible chat-completions HTTP API, then funnels the raw
provider response through the shared Phase 3A normalization layer to
produce the existing ``schemas.models.AIResponse``. No new response schema
is introduced.

Flow:

    PromptItem
      -> OpenRouterCollector
      -> OpenRouter API (one request)
      -> raw response parsing
      -> utils.normalization
      -> existing AIResponse

Zero-budget / safety constraints honoured here:

- exactly one outbound request per ``collect()`` call -- no retry loop, no
  exponential backoff, no fallback provider, no second attempt;
- the collector is built for the configured free model
  (``OPENROUTER_MODEL``); it never overrides the model, never switches to a
  paid model, and never adds billing;
- if ``OPENROUTER_API_KEY`` is absent the collector never touches the
  network and returns a structured ``AIResponse(success=False, ...)``;
- the API key is read only from configuration, is never logged, never put
  into an exception message, and never stored on the returned object.
  Error strings are additionally scrubbed of the key as a defensive
  backstop.

The HTTP call uses only the Python standard library (``urllib`` + ``json``)
so no new dependency is required. The module is import-safe with or without
an API key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from collectors.base import BaseCollector
from config import Settings, load_settings
from schemas.models import AIResponse, PromptItem, SourceType
from utils.normalization import normalize_answer_text, normalize_citations

# Centralized endpoint: defined once here rather than scattered through the
# collector. Overridable per-instance via the constructor for testing.
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"

_MISSING_KEY_MESSAGE = (
    "OPENROUTER_API_KEY is not configured; the OpenRouter collector cannot "
    "make a live request. Set OPENROUTER_API_KEY in the environment or .env "
    "to enable it."
)

# NOTE: OpenRouter supports optional "HTTP-Referer" / "X-Title" attribution
# headers, but they require a real, verified project URL/name. This project
# does not have a confirmed public repository yet, so those headers are
# deliberately not sent -- a fictional URL must never be transmitted. They
# can be reinstated (ideally sourced from configuration) once a real URL
# exists.


def _perform_request(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> tuple[int, str]:
    """Make a single POST and return ``(status_code, body_text)``.

    A non-2xx HTTP response is returned as a normal ``(code, body)`` tuple
    (not raised) so the caller can translate it into a structured failure.
    Transport-level problems (DNS, refused connection, timeout, ...) are
    allowed to propagate and are handled by the caller.
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = getattr(response, "status", None) or getattr(response, "code", 200) or 200
            return int(status), body
    except urllib.error.HTTPError as http_error:
        try:
            body = http_error.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - error body is best-effort only
            body = ""
        return int(http_error.code), body


class OpenRouterCollector(BaseCollector):
    """Collects a single answer from an OpenRouter free model for one prompt.

    Always returns a validated :class:`AIResponse` -- ``success=True`` with a
    normalized answer on success, or ``success=False`` with an
    ``error_message`` on any failure (missing key, network failure, timeout,
    HTTP error, malformed JSON, missing/empty content). It never raises for
    those expected failure modes and never turns a failure into a
    successful empty response.
    """

    source_type: SourceType = SourceType.OPENROUTER_FREE

    def __init__(
        self,
        settings: Optional[Settings] = None,
        api_url: Optional[str] = None,
    ) -> None:
        # Snapshot configuration once per instance (not per call) so tests
        # can inject settings / mutate the environment and build a fresh
        # collector.
        self._settings: Settings = settings if settings is not None else load_settings()
        self._api_url: str = api_url or OPENROUTER_CHAT_COMPLETIONS_URL

    # -- Public API --------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The configured OpenRouter model id (never a secret)."""
        return self._settings.openrouter_model

    def collect(self, prompt: PromptItem) -> AIResponse:
        # 1. No API key -> no network, structured failure.
        if not self._settings.openrouter_configured:
            return self._failure(prompt, _MISSING_KEY_MESSAGE)

        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt.prompt}],
        }

        # 2. Exactly one request. No retry loop, no fallback.
        try:
            status, body = _perform_request(
                self._api_url, payload, headers, self._timeout_seconds()
            )
        except Exception as exc:  # noqa: BLE001 - network / timeout / etc.
            return self._failure(prompt, self._describe_error(exc))

        # 3. HTTP-level failure.
        if status < 200 or status >= 300:
            return self._failure(
                prompt,
                f"OpenRouter HTTP {status}: {self._scrub(self._shorten(body))}",
            )

        # 4. Parse JSON body.
        try:
            data = json.loads(body)
        except (ValueError, TypeError) as exc:
            return self._failure(
                prompt,
                "OpenRouter returned a malformed JSON body: " + self._describe_error(exc),
            )

        # 5. Provider-signalled error object (can arrive with HTTP 200).
        if isinstance(data, dict) and data.get("error"):
            return self._failure(
                prompt,
                "OpenRouter returned an error: "
                + self._scrub(self._shorten(json.dumps(data.get("error")))),
            )

        # 6. Extract answer + any genuine citation data.
        try:
            answer_raw = self._extract_answer_text(data)
            raw_citations = self._extract_raw_citations(data)
        except Exception as exc:  # noqa: BLE001
            return self._failure(
                prompt,
                "OpenRouter response could not be parsed: " + self._describe_error(exc),
            )

        # 7. Normalize (Phase 3A layer).
        answer = normalize_answer_text(answer_raw)
        if not answer:
            return self._failure(
                prompt,
                "OpenRouter returned an empty or content-free response.",
            )

        # `normalize_citations` returns None ("unavailable") when the
        # response carried no citation structure at all; AIResponse.citations
        # can only hold a list, so that collapses to an empty list here.
        citations = normalize_citations(raw_citations) or []

        # 8. Return the existing schema. Original prompt preserved verbatim.
        return AIResponse(
            source_type=SourceType.OPENROUTER_FREE,
            model_name=self.model_name,
            prompt_id=prompt.prompt_id,
            prompt=prompt.prompt,
            answer=answer,
            citations=citations,
            success=True,
        )

    # -- Request construction -------------------------------------------------

    def _timeout_seconds(self) -> int:
        try:
            seconds = int(self._settings.request_timeout)
        except (TypeError, ValueError):
            seconds = 30
        return seconds if seconds > 0 else 30

    # -- Response parsing -------------------------------------------------------

    @staticmethod
    def _extract_answer_text(data: Any) -> Optional[str]:
        """Pull the assistant message text out of an OpenRouter chat
        completion. Returns ``None`` when no usable content is present
        (which the caller treats as a failure, not a successful empty
        answer).
        """
        if not isinstance(data, dict):
            return None
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None

        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            # Some providers return content as a list of typed parts.
            if isinstance(content, list):
                parts = [
                    part.get("text")
                    for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                ]
                if parts:
                    return "".join(parts)

        # Legacy / text-completion shape.
        text = first.get("text")
        if isinstance(text, str):
            return text
        return None

    @staticmethod
    def _extract_raw_citations(data: Any) -> Optional[list[dict[str, Optional[str]]]]:
        """Extract raw citation dicts from an OpenRouter response, only when
        genuine citation/URL data is actually present.

        Returns:
        - ``None`` when the response contains no citation structure at all
          ("unavailable" -- never fabricated);
        - a list of ``{"url", "title", "domain"}`` dicts (possibly empty)
          when a citation structure was present. Entries with no usable
          url/title are skipped here and by the normalization layer.
        """
        if not isinstance(data, dict):
            return None

        found_structure = False
        raw: list[dict[str, Optional[str]]] = []

        # Shape 1: choices[0].message.annotations[] with url_citation entries.
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                annotations = message.get("annotations")
                if isinstance(annotations, list):
                    found_structure = True
                    for annotation in annotations:
                        if not isinstance(annotation, dict):
                            continue
                        cite = annotation.get("url_citation")
                        if not isinstance(cite, dict):
                            cite = annotation if annotation.get("type") == "url_citation" else None
                        if not isinstance(cite, dict):
                            continue
                        url = cite.get("url")
                        title = cite.get("title")
                        if not url and not title:
                            continue
                        raw.append(
                            {
                                "url": url if isinstance(url, str) else None,
                                "title": title if isinstance(title, str) else None,
                                "domain": None,
                            }
                        )

        # Shape 2: top-level "citations": list of URL strings or dicts.
        top_level = data.get("citations")
        if isinstance(top_level, list):
            found_structure = True
            for item in top_level:
                if isinstance(item, str) and item.strip():
                    raw.append({"url": item.strip(), "title": None, "domain": None})
                elif isinstance(item, dict):
                    url = item.get("url")
                    title = item.get("title")
                    if not url and not title:
                        continue
                    raw.append(
                        {
                            "url": url if isinstance(url, str) else None,
                            "title": title if isinstance(title, str) else None,
                            "domain": None,
                        }
                    )

        if not found_structure:
            return None
        return raw

    # -- Failure handling ---------------------------------------------------

    def _failure(self, prompt: PromptItem, message: str) -> AIResponse:
        return AIResponse(
            source_type=SourceType.OPENROUTER_FREE,
            model_name=self.model_name,
            prompt_id=prompt.prompt_id,
            prompt=prompt.prompt,
            answer=None,
            citations=[],
            success=False,
            error_message=self._scrub(message) or "OpenRouter collection failed.",
        )

    def _describe_error(self, exc: BaseException) -> str:
        label = type(exc).__name__
        detail = self._scrub(str(exc))
        return f"{label}: {detail}" if detail else label

    def _scrub(self, text: str) -> str:
        """Defensive backstop: never let the API key leak through a message."""
        if not text:
            return ""
        key = self._settings.openrouter_api_key
        if key and key in text:
            text = text.replace(key, "***REDACTED***")
        return text

    @staticmethod
    def _shorten(text: str, limit: int = 300) -> str:
        text = (text or "").strip()
        return text if len(text) <= limit else text[:limit] + "..."
