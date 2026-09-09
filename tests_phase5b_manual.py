"""Manual Phase 5B verification script: Website Evidence Parser.

Not a pytest suite (same "minimal dependencies" rule as Phases 1-5A). It
exercises `crawler/parser.py` and `crawler/schema_detector.py` against
deterministic in-memory `AcquiredPage` fixtures, then re-runs every earlier
manual phase as a regression check.

ZERO network requests: the parser is pure text processing (stdlib only).
No LLM, no API, no randomness, no timestamps.

Run with: python3 tests_phase5b_manual.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from crawler.tavily import AcquiredPage, WebAcquisitionResult
from crawler.parser import (
    Headings,
    JsonLdBlock,
    PageEvidence,
    parse_page,
    parse_result,
)
from crawler.schema_detector import SchemaEvidence, detect_schema

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))


def page(content, url="https://acme.example/coffee"):
    return AcquiredPage(url=url, content=content)


# ---------------------------------------------------------------------------
# Fixtures (deterministic, in-memory)
# ---------------------------------------------------------------------------
HTML_PAGE = """<html><head>
<title>Acme Camping Coffee Makers</title>
<meta name="description" content="Everything about Acme camping coffee makers.">
<meta name="robots" content="index,follow">
<link rel="canonical" href="https://acme.example/coffee">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"Acme Pour-Over",
 "brand":{"@type":"Brand","name":"Acme"},
 "offers":{"@type":"Offer","price":"29.99","priceCurrency":"USD"},
 "aggregateRating":{"@type":"AggregateRating","ratingValue":"4.6"}}
</script>
<script type="application/ld+json">{ not : valid json ,,, }</script>
</head><body>
<h1>Acme Camping Coffee Makers</h1>
<p>Acme makes durable pour-over gear.</p>
<h2>Features</h2>
<p>Lightweight and no batteries.</p>
<h2>Reviews</h2>
<h3>What buyers say</h3>
<p>Great for backpacking.</p>
<a href="/about">About</a>
<a href="/about">About again</a>
<a href="https://acme.example/contact">Contact</a>
<a href="https://competitor.example/review">Competitor review</a>
<a href="mailto:hi@acme.example">Email</a>
<a href="#top">Back to top</a>
</body></html>"""

MARKDOWN_PAGE = """# Acme Camping Coffee Makers

Acme makes durable pour-over gear for backpackers.

## Features

- Lightweight
- No batteries required

## Where to buy

Check [our shop](https://acme.example/shop) or read a [review](https://competitor.example/r).
Also see <https://acme.example/guide> and https://acme.example/faq
"""

FAQ_PAGE = """<html><body>
<h1>FAQ</h1>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"FAQPage","mainEntity":[
  {"@type":"Question","name":"Is it durable?","acceptedAnswer":{"@type":"Answer","text":"Yes."}},
  {"@type":"Question","name":"Batteries?","acceptedAnswer":{"@type":"Answer","text":"No."}}
]}
</script>
</body></html>"""

GRAPH_PAGE = """<html><body>
<script type="application/ld+json">
{"@context":"https://schema.org","@graph":[
  {"@type":"Organization","name":"Acme","url":"https://acme.example"},
  {"@type":["WebPage","AboutPage"],"name":"About"},
  {"@type":"BreadcrumbList","itemListElement":[{"@type":"ListItem","position":1,"name":"Home"}]}
]}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# 1-5. Headings, title, meta, ordering, sections (HTML)
# ---------------------------------------------------------------------------
ev = parse_page(page(HTML_PAGE))
check("1/4. H1/H2/H3 extracted in document order",
      ev.headings.h1 == ("Acme Camping Coffee Makers",)
      and ev.headings.h2 == ("Features", "Reviews")
      and ev.headings.h3 == ("What buyers say",),
      f"{ev.headings}")
check("2. title extracted", ev.title == "Acme Camping Coffee Makers", repr(ev.title))
check("3. meta description extracted", ev.meta_description == "Everything about Acme camping coffee makers.", repr(ev.meta_description))
check("5. body sections extracted as an ordered list of blocks, not one flat string",
      isinstance(ev.sections, tuple) and len(ev.sections) >= 3
      and ev.sections[0] == "Acme makes durable pour-over gear."
      and ev.sections[1] == "Lightweight and no batteries."
      and "Great for backpacking." in ev.sections,
      str(ev.sections))
check("5. headings are not duplicated into sections", "Features" not in ev.sections and "What buyers say" not in ev.sections)

# ---------------------------------------------------------------------------
# 6-7. internal vs external links, dedup, first-seen order
# ---------------------------------------------------------------------------
check("6. relative + same-domain links classified internal (resolved absolute)",
      ev.internal_links == ("https://acme.example/about", "https://acme.example/contact"),
      str(ev.internal_links))
check("6. other-domain link classified external",
      ev.external_links == ("https://competitor.example/review",), str(ev.external_links))
check("7. duplicate link kept once, first-seen order preserved",
      ev.internal_links.count("https://acme.example/about") == 1)
check("7. mailto:/tel:/fragment links are dropped, not classified",
      all("mailto:" not in x and not x.endswith("#top") for x in ev.internal_links + ev.external_links))

# ---------------------------------------------------------------------------
# 8-9. canonical + robots
# ---------------------------------------------------------------------------
check("8. canonical URL extracted when present", ev.canonical == "https://acme.example/coffee", repr(ev.canonical))
check("9. robots directive extracted when present", ev.robots == "index,follow", repr(ev.robots))
check("8/9. canonical/robots are None when absent from content",
      parse_page(page("<html><body><h1>Bare</h1></body></html>")).canonical is None
      and parse_page(page("<html><body><h1>Bare</h1></body></html>")).robots is None)

# ---------------------------------------------------------------------------
# 10-11. JSON-LD extraction; malformed JSON-LD does not crash
# ---------------------------------------------------------------------------
check("10. valid JSON-LD block parsed and preserved as structured data",
      len(ev.json_ld) == 2 and ev.json_ld[0].valid is True
      and isinstance(ev.json_ld[0].data, dict) and ev.json_ld[0].data.get("@type") == "Product")
check("11. malformed JSON-LD recorded as invalid, parser still returns evidence",
      ev.json_ld[1].valid is False and ev.json_ld[1].data is None and ev.json_ld[1].error
      and isinstance(ev, PageEvidence))
check("11. a page whose ONLY JSON-LD is malformed still parses",
      parse_page(page('<html><body><h1>x</h1><script type="application/ld+json">{oops</script></body></html>')).headings.h1 == ("x",))

# ---------------------------------------------------------------------------
# 12-13. schema type detection + nested detection
# ---------------------------------------------------------------------------
check("12. schema_types on PageEvidence lists detected @type values (nested included)",
      ev.schema_types == ("Product", "Brand", "Offer", "AggregateRating"), str(ev.schema_types))
se = detect_schema(ev.json_ld)
check("12. detect_schema returns SchemaEvidence with the same types",
      isinstance(se, SchemaEvidence) and se.types == ev.schema_types and se.has_faq is False)
check("12. interesting properties recorded per entity (no scoring)",
      any(e.type == "Product" and "brand" in e.properties and "offers" in e.properties and "aggregateRating" in e.properties
          for e in se.entities))

faq = parse_page(page(FAQ_PAGE))
faq_se = detect_schema(faq.json_ld)
check("13. nested FAQPage/Question/Answer detected", set(("FAQPage", "Question", "Answer")).issubset(set(faq.schema_types)), str(faq.schema_types))
check("13. has_faq flag set for FAQ structured data", faq_se.has_faq is True)

graph = parse_page(page(GRAPH_PAGE))
check("13. @graph + list @type + nested itemListElement all detected",
      {"Organization", "WebPage", "AboutPage", "BreadcrumbList", "ListItem"}.issubset(set(graph.schema_types)),
      str(graph.schema_types))

# ---------------------------------------------------------------------------
# Markdown path (title/meta/canonical/robots unavailable, headings+links work)
# ---------------------------------------------------------------------------
md = parse_page(page(MARKDOWN_PAGE, url="https://acme.example/"))
check("Markdown: H1/H2 extracted from '#'/'##'", md.headings.h1 == ("Acme Camping Coffee Makers",) and md.headings.h2 == ("Features", "Where to buy"), str(md.headings))
check("Markdown: title/meta/canonical/robots reported unavailable (not invented)",
      md.title is None and md.meta_description is None and md.canonical is None and md.robots is None)
check("Markdown: markdown/autolink/bare links extracted, classified, deduped",
      md.internal_links == ("https://acme.example/shop", "https://acme.example/guide", "https://acme.example/faq")
      and md.external_links == ("https://competitor.example/r",),
      f"{md.internal_links} | {md.external_links}")
check("Markdown: sections captured as separate blocks", "Acme makes durable pour-over gear for backpackers." in md.sections and len(md.sections) >= 2)

# ---------------------------------------------------------------------------
# 14. missing / empty content
# ---------------------------------------------------------------------------
for label, c in (("None", None), ("empty", ""), ("whitespace", "   \n\t ")):
    e = parse_page(page(c))
    check(f"14. {label} content -> safe empty evidence, no crash",
          isinstance(e, PageEvidence) and e.title is None and e.headings == Headings()
          and e.sections == () and e.internal_links == () and e.external_links == ()
          and e.json_ld == () and e.schema_types == () and e.content_present is False
          and e.url == "https://acme.example/coffee")

# ---------------------------------------------------------------------------
# 15. malformed HTML resilience -> partial evidence
# ---------------------------------------------------------------------------
broken = parse_page(page("<html><body><h1>Broken <p>no close <a href=/x>link</body>", url="https://acme.example/p"))
check("15. malformed HTML does not raise; returns a PageEvidence", isinstance(broken, PageEvidence))
check("15. malformed HTML still yields partial evidence (heading + link)",
      broken.headings.h1 and broken.headings.h1[0].startswith("Broken")
      and "https://acme.example/x" in broken.internal_links, f"{broken.headings.h1} {broken.internal_links}")

# ---------------------------------------------------------------------------
# 16. project-owned output models (no Tavily / no Pydantic response schema)
# ---------------------------------------------------------------------------
check("16. output is the project-owned PageEvidence dataclass from crawler.parser",
      type(ev).__module__ == "crawler.parser" and type(ev).__name__ == "PageEvidence")
check("16. nested models are project-owned too (Headings, JsonLdBlock, SchemaEvidence)",
      isinstance(ev.headings, Headings) and all(isinstance(b, JsonLdBlock) for b in ev.json_ld)
      and isinstance(se, SchemaEvidence))
check("16. PageEvidence fields match the Phase 5B contract",
      set(PageEvidence.__dataclass_fields__) == {
          "url", "title", "meta_description", "headings", "sections",
          "internal_links", "external_links", "canonical", "robots",
          "json_ld", "schema_types", "content_present",
      })

# ---------------------------------------------------------------------------
# 17. deterministic repeated parsing
# ---------------------------------------------------------------------------
check("17. same input -> identical output across repeated parses",
      parse_page(page(HTML_PAGE)) == parse_page(page(HTML_PAGE))
      and parse_page(page(MARKDOWN_PAGE)) == parse_page(page(MARKDOWN_PAGE))
      and detect_schema(parse_page(page(GRAPH_PAGE)).json_ld) == detect_schema(parse_page(page(GRAPH_PAGE)).json_ld))

# ---------------------------------------------------------------------------
# Phase 5A integration: parse_result over a WebAcquisitionResult
# ---------------------------------------------------------------------------
res = WebAcquisitionResult(
    ok=True, start_url="https://acme.example",
    pages=(AcquiredPage("https://acme.example/coffee", HTML_PAGE), AcquiredPage("https://acme.example/", MARKDOWN_PAGE)),
    error=None,
)
parsed = parse_result(res)
check("5A integration: parse_result yields one PageEvidence per acquired page",
      len(parsed) == 2 and all(isinstance(p, PageEvidence) for p in parsed))
check("5A integration: a failed WebAcquisitionResult yields no evidence",
      parse_result(WebAcquisitionResult(ok=False, start_url="x", pages=(), error=None)) == ())

# ---------------------------------------------------------------------------
# 18. zero network requests (structural: inspect actual import statements)
# ---------------------------------------------------------------------------
import re as _re

_FORBIDDEN_IMPORTS = {
    "urllib.request", "http", "http.client", "socket", "ssl", "requests",
    "httpx", "aiohttp", "openai", "anthropic", "google", "langchain",
    "bs4", "beautifulsoup4", "scrapy", "selenium", "playwright",
}
for mod in ("crawler/parser.py", "crawler/schema_detector.py"):
    src = Path(mod).read_text(encoding="utf-8")
    imported = set(_re.findall(r"(?m)^\s*(?:from|import)\s+([\w.]+)", src))
    roots = {name.split(".")[0] for name in imported} | imported
    bad = roots & _FORBIDDEN_IMPORTS
    check(f"18. {mod} imports nothing network / LLM / crawler-framework / browser related",
          not bad, f"imports={sorted(imported)} forbidden_hit={sorted(bad)}")
    check(f"18. {mod} never calls a URL opener", "urlopen" not in src and "URLopener" not in src)

# ---------------------------------------------------------------------------
# 19 + 20. regression -- earlier phases unchanged and still pass
# ---------------------------------------------------------------------------
for label, script in (
    ("Phase 5A", "tests_phase5a_manual.py"),
    ("Phase 4", "tests_phase4_manual.py"),
    ("Phase 3D", "tests_phase3d_manual.py"),
    ("Phase 3C", "tests_phase3c_manual.py"),
    ("Phase 3B", "tests_phase3b_manual.py"),
    ("Phase 3A", "tests_phase3a_manual.py"),
    ("Phase 2", "tests_phase2_manual.py"),
    ("Phase 1", "tests_phase1_manual.py"),
):
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=360)
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()
        check(f"Regression: {label} ({script}) exits 0", proc.returncode == 0, last)
    except Exception as exc:  # noqa: BLE001
        check(f"Regression: {label} ({script}) exits 0", False, repr(exc))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n=== Phase 5B Manual Verification Results ===")
passed = failed = 0
for name, status, detail in results:
    passed += status == PASS
    failed += status == FAIL
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

print(f"\nTOTAL: {passed} passed, {failed} failed")
print("NETWORK REQUESTS = 0 (deterministic in-memory parsing only)")
print("OVERALL:", "ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
sys.exit(0 if failed == 0 else 1)
