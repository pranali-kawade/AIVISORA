"""Focused diagnostic for the DIAGNOSE=all-UNAVAILABLE symptom.

Traces the website-evidence data flow deterministically. No real network:
the Tavily HTTP seam (``crawler.tavily._perform_request``) is replaced, and
the end-to-end pipeline check injects a fake ``acquire`` callable.

Run: python3 tests_website_acquisition_diag_manual.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

# Let TavilyWebAcquirer() get past its missing-key guard without touching .env.
os.environ.setdefault("TAVILY_API_KEY", "tvly-FAKE-KEY-FOR-TESTS-ONLY")

import crawler.tavily as tv
from agents.graph import _default_acquire, run_pipeline
from crawler.parser import Headings, PageEvidence
from schemas.models import AIResponse, BrandMention, PromptItem, SourceType

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
_glog.setLevel(logging.WARNING)


def _seam(status: int, body: str) -> None:
    tv._perform_request = lambda url, payload, headers, timeout: (status, body)


# 2 / 3 / 4. Tavily ok + one real page -> PageEvidence(content_present=True)
_seam(200, json.dumps({"results": [{
    "url": "https://acme.example/",
    "raw_content": "# Acme\n\nAcme makes project management software. Pricing and features here.",
}]}))
_cap.messages.clear()
ev = _default_acquire("https://acme.example/")
check("2-4. Tavily ok + 1 page -> PageEvidence, content_present=True",
      isinstance(ev, PageEvidence) and ev.content_present is True, repr(ev)[:140])

# 7. Tavily HTTP error -> None, and the reason is now reported (not swallowed)
_seam(401, json.dumps({"detail": {"error": "Unauthorized: invalid API key."}}))
_cap.messages.clear()
none_ev = _default_acquire("https://acme.example/")
check("7. Tavily HTTP 401 -> _default_acquire returns None", none_ev is None)
check("7b. failure reason logged (http_error / HTTP 401), no secret leaked",
      any("http_error" in m and "401" in m for m in _cap.messages)
      and not any("tvly-" in m for m in _cap.messages), str(_cap.messages))

# 3. Tavily ok but zero pages -> None, logged distinctly
_seam(200, json.dumps({"results": []}))
_cap.messages.clear()
check("3. Tavily ok + 0 pages -> None, logged as 'no pages'",
      _default_acquire("https://acme.example/") is None
      and any("no pages" in m for m in _cap.messages), str(_cap.messages))

# 1 / 5 / 6. End-to-end: an injected acquire's evidence reaches diagnosis.
EVIDENCE = {
    "https://acme.example": PageEvidence(
        url="https://acme.example", title="Acme", meta_description=None,
        headings=Headings(h1=("Acme",)), sections=("Acme makes PM software.",),
        internal_links=(), external_links=(), canonical=None, robots=None,
        json_ld=(), schema_types=(), content_present=True),
    "https://brewco.example": PageEvidence(
        url="https://brewco.example", title="BrewCo", meta_description=None,
        headings=Headings(h1=("BrewCo",)), sections=("BrewCo FAQ.",),
        internal_links=(), external_links=(), canonical=None, robots=None,
        json_ld=(), schema_types=(), content_present=True),
}
seen_urls: list[str] = []


def fake_acquire(url: str):
    seen_urls.append(url)
    return EVIDENCE.get(url)


resp = AIResponse(
    source_type=SourceType.MOCK, prompt_id="p1", prompt="best PM tools?",
    answer="Acme and BrewCo are common picks.",
    brand_mentions=[BrandMention(brand_name="Acme", mentioned=True),
                    BrandMention(brand_name="BrewCo", mentioned=True)],
    success=True,
)
state = {
    "target_brand": "Acme", "category": "Project management software",
    "website": "https://acme.example", "competitors": ["BrewCo"],
    "competitor_sites": {"BrewCo": "https://brewco.example"},
    "prompts": [PromptItem(prompt_id="p1", prompt="best PM tools?")],
    "ai_responses": [resp], "prompt_count": 1,
}
final = run_pipeline(state, providers=[], acquire=fake_acquire)

check("1. target website URL reaches acquire_website_evidence()",
      "https://acme.example" in seen_urls)
check("5. website_data present in final pipeline state as PageEvidence",
      isinstance(final.get("website_data"), PageEvidence))
check("6. competitor_data populated for the supplied competitor site",
      isinstance(final.get("competitor_data", {}).get("BrewCo"), PageEvidence))
_findings = getattr(final.get("gap_results"), "findings", ())
check("5b. with evidence present, findings are NOT all UNAVAILABLE",
      bool(_findings) and any(f.status.value != "UNAVAILABLE" for f in _findings),
      str([f"{f.category.value}:{f.status.value}" for f in _findings]))

print("\n=== Website-acquisition diagnostic ===")
p = f = 0
for n, s, d in results:
    p += s == PASS
    f += s == FAIL
    print(f"[{s}] {n}" + (f" -- {d}" if d and s == FAIL else ""))
print(f"\nTOTAL: {p} passed, {f} failed")
print("NETWORK CALLS = 0 (Tavily HTTP seam replaced; end-to-end uses injected acquire)")
sys.exit(0 if f == 0 else 1)
