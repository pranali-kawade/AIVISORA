"""Tavily web-acquisition adapter (Phase 5A).

One job: acquire website page content through the Tavily Crawl API and
return it in a small project-owned result that Phase 5B can parse. No HTML
parsing, schema detection, scoring, recursion, retries, caching, or LLM
calls happen here.

Verified against docs.tavily.com:

* POST https://api.tavily.com/crawl
* Auth: ``Authorization: Bearer <TAVILY_API_KEY>`` header only (the key is
  never put in the URL, query string, or request body).
* Each item in the 200 response's ``results`` array exposes ``url`` and
  ``raw_content``. Those are the only fields Phase 5B needs; nothing else
  is kept, and nothing Tavily does not return (title, status, final URL)
  is invented.
* Failures come back as non-2xx with body ``{"detail": {"error": "..."}}``.

Uses only the standard library (``urllib`` + ``json``); no new dependency.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

from config import Settings, load_settings

TAVILY_CRAWL_URL = "https://api.tavily.com/crawl"

# Conservative defaults -- safe for the Tavily free allocation.
DEFAULT_MAX_DEPTH = 1
DEFAULT_MAX_BREADTH = 20
DEFAULT_LIMIT = 20
DEFAULT_TIMEOUT = 60.0

_TAVILY_TOKEN = re.compile(r"tvly-[A-Za-z0-9_\-]+")


@dataclass(frozen=True)
class AcquiredPage:
    """A single acquired page: its URL and raw extracted content."""

    url: str
    content: Optional[str]


@dataclass(frozen=True)
class WebAcquisitionError:
    """Structured acquisition failure. Never carries a secret.

    ``error_type`` is one of: ``missing_api_key``, ``invalid_url``,
    ``timeout``, ``network_error``, ``http_error`` (any non-2xx --
    ``status_code`` carries the specifics), ``malformed_response``.
    """

    error_type: str
    message: str
    status_code: Optional[int] = None


@dataclass(frozen=True)
class WebAcquisitionResult:
    """What the rest of AEO Radar consumes -- no raw Tavily payload leaks."""

    ok: bool
    start_url: str
    pages: tuple[AcquiredPage, ...]
    error: Optional[WebAcquisitionError]


def _perform_request(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> tuple[int, str]:
    """One POST -> ``(status_code, body_text)``. Non-2xx is returned, not
    raised; transport errors (timeout, connection) propagate to the caller.
    This is the seam the tests replace so no real request is ever made.
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(getattr(response, "status", 200) or 200), body
    except urllib.error.HTTPError as http_error:
        try:
            body = http_error.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - error body is best-effort only
            body = ""
        return int(http_error.code), body


class TavilyWebAcquirer:
    """Acquires website pages through the Tavily Crawl API.

    Always returns a :class:`WebAcquisitionResult`; expected failures come
    back as ``ok=False`` with a structured :class:`WebAcquisitionError`
    rather than raising. The API key is read only from configuration and is
    never placed in a returned object or an error message.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_breadth: int = DEFAULT_MAX_BREADTH,
        limit: int = DEFAULT_LIMIT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._settings = settings if settings is not None else load_settings()
        self._max_depth = _clamp(max_depth, 1, 5)
        self._max_breadth = _clamp(max_breadth, 1, 500)
        self._limit = _clamp(limit, 1, 1000)
        self._timeout = float(_clamp(timeout, 10, 150))

    def crawl(self, start_url: str) -> WebAcquisitionResult:
        key = self._settings.tavily_api_key
        if not key:
            return _fail(start_url, "missing_api_key", "TAVILY_API_KEY is not configured.")

        url = _valid_url(start_url)
        if url is None:
            return _fail(start_url, "invalid_url", f"not a valid http(s) URL: {start_url!r}")

        payload = {
            "url": url,
            "max_depth": self._max_depth,
            "max_breadth": self._max_breadth,
            "limit": self._limit,
            "allow_external": False,  # stay on the requested site
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        try:
            status, body = _perform_request(TAVILY_CRAWL_URL, payload, headers, self._timeout)
        except Exception as exc:  # noqa: BLE001 - translate every transport failure
            reason = getattr(exc, "reason", exc)
            kind = "timeout" if isinstance(reason, TimeoutError) or "timed out" in str(exc).lower() else "network_error"
            return _fail(url, kind, _scrub(f"{type(exc).__name__}: {exc}", key))

        if not 200 <= status < 300:
            return _fail(url, "http_error", f"Tavily HTTP {status}: {_scrub(_error_text(body), key)}", status)

        try:
            data = json.loads(body)
        except ValueError:
            return _fail(url, "malformed_response", "Tavily returned a non-JSON body.", status)

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return _fail(url, "malformed_response", "Tavily response has no 'results' array.", status)

        pages = tuple(
            AcquiredPage(
                url=item["url"] if isinstance(item.get("url"), str) else url,
                content=_content(item.get("raw_content")),
            )
            for item in results
            if isinstance(item, dict)
        )
        return WebAcquisitionResult(ok=True, start_url=url, pages=pages, error=None)


def _fail(start_url: str, error_type: str, message: str, status: Optional[int] = None) -> WebAcquisitionResult:
    return WebAcquisitionResult(
        ok=False,
        start_url=start_url,
        pages=(),
        error=WebAcquisitionError(error_type=error_type, message=message, status_code=status),
    )


def _scrub(text: str, key: Optional[str]) -> str:
    if key:
        text = text.replace(key, "***REDACTED***")
    return _TAVILY_TOKEN.sub("***REDACTED***", text)


def _clamp(value: Any, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return low


def _valid_url(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    parsed = urlparse(candidate)
    return candidate if parsed.scheme in ("http", "https") and parsed.netloc else None


def _error_text(body: str) -> str:
    text = (body or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("detail"), dict):
            return str(data["detail"].get("error") or text)[:200]
    except ValueError:
        pass
    return text[:200] or "no response body"


def _content(raw: Any) -> Optional[str]:
    return raw.strip() or None if isinstance(raw, str) else None
