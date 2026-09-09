"""Focused deterministic tests for the Tavily -> urllib acquisition fallback.

No real network: the Tavily HTTP seam (``crawler.tavily._perform_request``)
and ``urllib.request.urlopen`` are both replaced with in-memory fakes.

Run: python3 tests_website_acquisition_fallback_manual.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

os.environ.setdefault("TAVILY_API_KEY", "tvly-FAKE-KEY-FOR-TESTS-ONLY")

import crawler.tavily as tv
from agents import graph
from agents.graph import _default_acquire, _direct_fetch
from crawler.parser import PageEvidence

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, PASS if cond else FAIL, detail))


class _CaptureLog(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


_cap = _CaptureLog()
_glog = logging.getLogger("agents.graph")
_glog.addHandler(_cap)
_glog.setLevel(logging.INFO)

HTML = (
    "<html><head><title>Example Domain</title></head>"
    "<body><h1>Example Domain</h1><p>This domain is for illustrative examples.</p></body></html>"
)


def _tavily(status: int, body: str) -> None:
    tv._perform_request = lambda url, payload, headers, timeout: (status, body)


class _FakeHeaders:
    def get_content_charset(self):
        return "utf-8"


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = _FakeHeaders()

    def read(self, n=None):
        return self._body if n is None else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen(body: bytes | None = None, exc: Exception | None = None):
    def _fake(request, timeout=None):
        if exc is not None:
            raise exc
        return _FakeResp(body or b"")

    graph.urllib.request.urlopen = _fake


_TAVILY_OK_PAGE = json.dumps({"results": [
    {"url": "https://acme.example/", "raw_content": "# Acme\n\nAcme makes project management software."}
]})
_TAVILY_ZERO = json.dumps({"results": []})
_TAVILY_HTTP_ERR = json.dumps({"detail": {"error": "Unauthorized: invalid API key."}})


# 1. Tavily returns pages -> primary path preserved, fallback NOT used.
_tavily(200, _TAVILY_OK_PAGE)
_direct_called = {"n": 0}
_orig_direct = graph._direct_fetch
graph._direct_fetch = lambda url: (_direct_called.__setitem__("n", _direct_called["n"] + 1) or HTML)
_cap.messages.clear()
ev1 = _default_acquire("https://acme.example/")
graph._direct_fetch = _orig_direct
check("1. Tavily pages present -> PageEvidence from Tavily, fallback not invoked",
      isinstance(ev1, PageEvidence) and ev1.content_present and _direct_called["n"] == 0)

# 2. Tavily zero pages -> urllib fallback succeeds -> parsed PageEvidence.
_tavily(200, _TAVILY_ZERO)
_urlopen(body=HTML.encode())
_cap.messages.clear()
ev2 = _default_acquire("https://example.com")
check("2. Tavily zero pages -> direct GET recovers PageEvidence (content_present, title parsed)",
      isinstance(ev2, PageEvidence) and ev2.content_present and ev2.title == "Example Domain",
      repr(ev2)[:140])
check("2b. log records Tavily 'zero pages' then a direct-GET recovery",
      any("zero pages" in m for m in _cap.messages)
      and any("direct HTTP GET recovered evidence" in m for m in _cap.messages), str(_cap.messages))

# 3. Tavily hard failure (HTTP 401) -> urllib fallback succeeds.
_tavily(401, _TAVILY_HTTP_ERR)
_urlopen(body=HTML.encode())
_cap.messages.clear()
ev3 = _default_acquire("https://example.com")
check("3. Tavily HTTP 401 -> direct GET recovers PageEvidence",
      isinstance(ev3, PageEvidence) and ev3.content_present)
check("3b. log names the Tavily failure (http_error / 401) before falling back, no secret",
      any("http_error" in m and "401" in m for m in _cap.messages)
      and not any("tvly-" in m for m in _cap.messages), str(_cap.messages))

# 4. Tavily zero pages AND direct GET fails -> None, with a 'both failed' log.
_tavily(200, _TAVILY_ZERO)
_urlopen(exc=OSError("connection refused"))
_cap.messages.clear()
ev4 = _default_acquire("https://example.com")
check("4. Tavily unusable AND direct GET fails -> None", ev4 is None)
check("4b. final log states BOTH: Tavily unusable AND direct HTTP GET failed",
      any("Tavily unusable" in m and "direct HTTP GET" in m for m in _cap.messages)
      and any("direct HTTP GET failed" in m for m in _cap.messages), str(_cap.messages))

# 5. _direct_fetch unit: fake transport body -> decoded text.
_urlopen(body=b"<html><body><p>hi</p></body></html>")
check("5. _direct_fetch returns decoded body on a 200", _direct_fetch("https://x.example") ==
      "<html><body><p>hi</p></body></html>")

# 6. _direct_fetch unit: transport raises -> None + warning logged.
_urlopen(exc=OSError("dns failure"))
_cap.messages.clear()
check("6. _direct_fetch returns None and logs on transport error",
      _direct_fetch("https://x.example") is None
      and any("direct HTTP GET failed" in m for m in _cap.messages), str(_cap.messages))

# 7. _direct_fetch unit: empty body -> None (no fabricated evidence).
_urlopen(body=b"   ")
check("7. _direct_fetch treats a blank body as no evidence (None)",
      _direct_fetch("https://x.example") is None)

print("\n=== Acquisition fallback (Tavily -> urllib) ===")
p = f = 0
for n, s, d in results:
    p += s == PASS
    f += s == FAIL
    print(f"[{s}] {n}" + (f" -- {d}" if d and s == FAIL else ""))
print(f"\nTOTAL: {p} passed, {f} failed")
print("NETWORK CALLS = 0 (Tavily seam + urllib.request.urlopen both faked)")
sys.exit(0 if f == 0 else 1)
