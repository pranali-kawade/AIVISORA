"""Website evidence parser (Phase 5B).

Deterministic, provider-independent parsing of the ``AcquiredPage`` objects
produced by ``crawler.tavily``. Input is the already-acquired page URL and
text (Tavily returns markdown by default; HTML is also handled). Output is
a project-owned :class:`PageEvidence`.

This phase is parsing/extraction only -- no gap analysis, visibility
scoring, optimization, or LLM reasoning. It makes ZERO network requests: if
a value (title, meta description, canonical, robots) is not present in the
supplied content it is returned as unavailable, never fetched separately.

Standard library only (``html.parser``, ``re``, ``json``, ``urllib.parse``);
no BeautifulSoup / Scrapy / browser automation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

from crawler.schema_detector import detect_schema
from utils.normalization import derive_domain_from_url, normalize_whitespace

_clean = normalize_whitespace


# ---------------------------------------------------------------------------
# Project-owned output models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Headings:
    h1: tuple[str, ...] = ()
    h2: tuple[str, ...] = ()
    h3: tuple[str, ...] = ()


@dataclass(frozen=True)
class JsonLdBlock:
    valid: bool
    data: Any
    raw: str
    error: Optional[str] = None


@dataclass(frozen=True)
class PageEvidence:
    url: str
    title: Optional[str]
    meta_description: Optional[str]
    headings: Headings
    sections: tuple[str, ...]
    internal_links: tuple[str, ...]
    external_links: tuple[str, ...]
    canonical: Optional[str]
    robots: Optional[str]
    json_ld: tuple[JsonLdBlock, ...]
    schema_types: tuple[str, ...]
    content_present: bool


# ---------------------------------------------------------------------------
# JSON-LD (works for HTML or any text that kept its <script> blocks)
# ---------------------------------------------------------------------------
_JSONLD_RE = re.compile(
    r'<script[^>]*type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


def _extract_json_ld(content: str) -> tuple[JsonLdBlock, ...]:
    blocks: list[JsonLdBlock] = []
    for match in _JSONLD_RE.finditer(content):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            blocks.append(JsonLdBlock(valid=True, data=json.loads(raw), raw=raw))
        except (ValueError, TypeError) as exc:
            blocks.append(JsonLdBlock(valid=False, data=None, raw=raw, error=str(exc)))
    return tuple(blocks)


# ---------------------------------------------------------------------------
# HTML path
# ---------------------------------------------------------------------------
_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "main", "nav",
    "li", "ul", "ol", "table", "tr", "blockquote", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
}


class _HTMLEvidenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.meta_description: Optional[str] = None
        self.robots: Optional[str] = None
        self.canonical: Optional[str] = None
        self.h1: list[str] = []
        self.h2: list[str] = []
        self.h3: list[str] = []
        self.links: list[str] = []
        self.sections: list[str] = []
        self._capture: Optional[str] = None
        self._buf: list[str] = []
        self._block: list[str] = []
        self._skip = 0  # inside <script>/<style>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "title":
            self._capture, self._buf = "title", []
        elif tag in ("h1", "h2", "h3"):
            self._flush_block()
            self._capture, self._buf = tag, []
        elif tag == "meta":
            name = (a.get("name") or a.get("property") or "").lower()
            content = a.get("content", "").strip()
            if content and name == "description" and self.meta_description is None:
                self.meta_description = content
            elif content and name == "robots" and self.robots is None:
                self.robots = content
        elif tag == "link":
            if "canonical" in a.get("rel", "").lower().split() and self.canonical is None:
                self.canonical = a.get("href", "").strip() or None
        elif tag == "a":
            href = a.get("href", "").strip()
            if href:
                self.links.append(href)
        if tag in _BLOCK_TAGS:
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if self._capture and tag == self._capture:
            self._commit_capture()
        if tag in _BLOCK_TAGS:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._capture:
            self._buf.append(data)
        elif data.strip():
            self._block.append(data)

    def close(self) -> None:  # flush anything left open by malformed markup
        super().close()
        if self._capture:
            self._commit_capture()
        self._flush_block()

    def _commit_capture(self) -> None:
        text = _clean(" ".join(self._buf))
        if self._capture == "title":
            if text:
                self.title_parts.append(text)
        elif text:
            getattr(self, self._capture).append(text)
        self._capture, self._buf = None, []

    def _flush_block(self) -> None:
        if self._block:
            text = _clean(" ".join(self._block))
            if text:
                self.sections.append(text)
            self._block = []


# ---------------------------------------------------------------------------
# Markdown / plain-text path
# ---------------------------------------------------------------------------
_MD_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_MD_LINK = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_MD_AUTOLINK = re.compile(r"<((?:https?://)[^>\s]+)>")
_MD_BARE_URL = re.compile(r"(?<![\(<\"'\]])\bhttps?://[^\s)\]<>\"']+")
_MD_LEADING = re.compile(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s?)")


def _parse_markdown(content: str) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    h1: list[str] = []
    h2: list[str] = []
    h3: list[str] = []
    for line in content.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            (h1, h2, h3)[len(m.group(1)) - 1].append(_clean(m.group(2)))

    hrefs = (
        _MD_LINK.findall(content)
        + _MD_AUTOLINK.findall(content)
        + _MD_BARE_URL.findall(content)
    )

    sections: list[str] = []
    for block in re.split(r"\n\s*\n", content):
        block = block.strip()
        if not block or (re.match(r"^#{1,6}\s+\S", block) and "\n" not in block):
            continue
        text = _clean(_MD_LEADING.sub("", block))
        if text:
            sections.append(text)

    return h1, h2, h3, hrefs, sections


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------
def _classify_links(hrefs: list[str], page_url: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    page_domain = derive_domain_from_url(page_url) if page_url else None
    seen: set[str] = set()
    internal: list[str] = []
    external: list[str] = []
    for href in hrefs:
        href = (href or "").strip()
        if not href or href.startswith("#") or href.lower().startswith(
            ("mailto:", "tel:", "javascript:", "data:")
        ):
            continue
        resolved = urljoin(page_url or "", href).split("#", 1)[0]
        parsed = urlparse(resolved)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        domain = derive_domain_from_url(resolved)
        same = page_domain is None or (
            bool(domain)
            and (
                domain == page_domain
                or domain.endswith("." + page_domain)
                or page_domain.endswith("." + domain)
            )
        )
        (internal if same else external).append(resolved)
    return tuple(internal), tuple(external)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_HTML_HINT = re.compile(
    r"</?(?:html|head|body|div|p|span|h[1-6]|a|ul|ol|li|table|script|style|"
    r"meta|link|section|article|header|footer|nav|title|img|br)\b",
    re.IGNORECASE,
)


def parse_page(page: Any) -> PageEvidence:
    """Parse one ``AcquiredPage`` into :class:`PageEvidence`. Never raises for
    malformed input -- it returns whatever partial evidence it could derive.
    """
    url = getattr(page, "url", "") or ""
    content = getattr(page, "content", None) or ""
    content_present = bool(content.strip())

    json_ld = _extract_json_ld(content)
    title = meta = robots = canonical = None
    h1: tuple[str, ...] = ()
    h2: tuple[str, ...] = ()
    h3: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    hrefs: list[str] = []

    if content_present and _HTML_HINT.search(content):
        parser = _HTMLEvidenceParser()
        try:
            parser.feed(content)
            parser.close()
        except Exception:  # noqa: BLE001 - malformed HTML -> keep partial evidence
            pass
        title = _clean(" ".join(parser.title_parts)) or None
        meta, robots, canonical = parser.meta_description, parser.robots, parser.canonical
        h1, h2, h3 = tuple(parser.h1), tuple(parser.h2), tuple(parser.h3)
        sections = tuple(parser.sections)
        hrefs = parser.links
    elif content_present:
        md_h1, md_h2, md_h3, hrefs, md_sections = _parse_markdown(content)
        h1, h2, h3 = tuple(md_h1), tuple(md_h2), tuple(md_h3)
        sections = tuple(md_sections)

    internal_links, external_links = _classify_links(hrefs, url)

    return PageEvidence(
        url=url,
        title=title,
        meta_description=meta,
        headings=Headings(h1=h1, h2=h2, h3=h3),
        sections=sections,
        internal_links=internal_links,
        external_links=external_links,
        canonical=canonical,
        robots=robots,
        json_ld=json_ld,
        schema_types=detect_schema(json_ld).types,
        content_present=content_present,
    )


def parse_result(result: Any) -> tuple[PageEvidence, ...]:
    """Parse every page of a successful ``WebAcquisitionResult`` (Phase 5A)."""
    if not getattr(result, "ok", False):
        return ()
    return tuple(parse_page(page) for page in getattr(result, "pages", ()))
