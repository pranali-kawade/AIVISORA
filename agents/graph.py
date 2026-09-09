"""The compiled LangGraph orchestration graphs.

Phase 10 ``graph`` (unchanged):

    START -> analyze_responses -> visibility -> diagnose -> optimize -> END

Phase 11 ``full_graph`` -- the end-to-end pipeline over the same node
functions plus prompt generation, provider collection and website
acquisition:

    START -> generate_prompts -> collect_ai_responses -> analyze_responses
          -> visibility -> acquire_website_evidence -> diagnose -> optimize -> END

Both are plain sequential chains: no routing, retries, loops, parallel
branches, persistence, checkpoints, or streaming. ``run_pipeline`` injects
the external dependencies (provider collectors, website acquisition) so
tests run fully offline; omitting them uses the real implementations.
"""

from __future__ import annotations

import logging
import urllib.request

from langgraph.graph import END, START, StateGraph

from agents import nodes
from agents.state import PipelineState

_UNSET = object()
_log = logging.getLogger(__name__)

# Fallback direct-fetch limits (standard library only; no retries/caching).
_DIRECT_FETCH_TIMEOUT = 15
_DIRECT_FETCH_MAX_BYTES = 2_000_000


def build_graph():
    """Phase 10 graph: analyze_responses -> visibility -> diagnose -> optimize."""
    graph = StateGraph(PipelineState)
    graph.add_node("analyze_responses", nodes.analyze_responses)
    graph.add_node("visibility", nodes.visibility)
    graph.add_node("diagnose", nodes.diagnose)
    graph.add_node("optimize", nodes.optimize)
    graph.add_edge(START, "analyze_responses")
    graph.add_edge("analyze_responses", "visibility")
    graph.add_edge("visibility", "diagnose")
    graph.add_edge("diagnose", "optimize")
    graph.add_edge("optimize", END)
    return graph.compile()


def build_full_graph():
    """Phase 11 end-to-end graph."""
    graph = StateGraph(PipelineState)
    graph.add_node("generate_prompts", nodes.generate_prompts_node)
    graph.add_node("collect_ai_responses", nodes.collect_ai_responses)
    graph.add_node("analyze_responses", nodes.analyze_responses)
    graph.add_node("visibility", nodes.visibility)
    graph.add_node("acquire_website_evidence", nodes.acquire_website_evidence)
    graph.add_node("diagnose", nodes.diagnose)
    graph.add_node("optimize", nodes.optimize)
    graph.add_edge(START, "generate_prompts")
    graph.add_edge("generate_prompts", "collect_ai_responses")
    graph.add_edge("collect_ai_responses", "analyze_responses")
    graph.add_edge("analyze_responses", "visibility")
    graph.add_edge("visibility", "acquire_website_evidence")
    graph.add_edge("acquire_website_evidence", "diagnose")
    graph.add_edge("diagnose", "optimize")
    graph.add_edge("optimize", END)
    return graph.compile()


#: Phase 10 compiled graph (kept for direct ``graph.invoke`` with pre-supplied data).
graph = build_graph()
#: Phase 11 compiled end-to-end graph.
full_graph = build_full_graph()


def _default_providers() -> list:
    """The real provider collectors -- constructed lazily so importing this
    module never pulls in a provider SDK, and tests that inject fakes never
    touch them.
    """
    from collectors.gemini import GeminiCollector
    from collectors.openrouter import OpenRouterCollector

    return [GeminiCollector(), OpenRouterCollector()]


def _direct_fetch(url: str) -> str | None:
    """Fallback acquisition: a single standard-library HTTP GET of ``url``.
    Returns the response body text, or ``None`` on any failure. No retries,
    no caching, no recursion -- one request only.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "AIVISORA/1.0 (+website-evidence)"})
    try:
        with urllib.request.urlopen(request, timeout=_DIRECT_FETCH_TIMEOUT) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(_DIRECT_FETCH_MAX_BYTES)
    except Exception as exc:  # noqa: BLE001 - any transport/decode failure -> no evidence
        _log.warning("direct HTTP GET failed for %s: %s: %s", url, type(exc).__name__, exc)
        return None
    return raw.decode(charset, errors="replace").strip() or None


def _default_acquire(url: str):
    """Website acquisition for the pipeline.

    PRIMARY: Tavily Crawl -> AcquiredPage -> parse_page -> PageEvidence.
    FALLBACK: when Tavily returns no usable pages (failure OR zero pages), a
    single ``urllib`` HTTP GET of the supplied URL, fed through the SAME
    parser. If both routes come up empty the result is ``None`` -- evidence
    is never fabricated. Imported lazily. Failure reasons are logged
    (secret-free) so an all-UNAVAILABLE diagnosis is explainable.
    """
    from crawler.parser import parse_page
    from crawler.tavily import AcquiredPage, TavilyWebAcquirer

    result = TavilyWebAcquirer().crawl(url)
    if result.ok and result.pages:
        return parse_page(result.pages[0])

    if not result.ok:
        err = result.error
        tavily_reason = (
            f"{getattr(err, 'error_type', 'unknown')}"
            f"{f' (HTTP {err.status_code})' if getattr(err, 'status_code', None) else ''}"
            f" -- {getattr(err, 'message', 'no detail')}"
        )
    else:
        tavily_reason = "ok but zero pages"
    _log.warning("Tavily acquisition unusable for %s: %s; trying direct HTTP GET", url, tavily_reason)

    body = _direct_fetch(url)
    if body:
        evidence = parse_page(AcquiredPage(url=url, content=body))
        if evidence.content_present:
            _log.info("direct HTTP GET recovered evidence for %s (%d chars)", url, len(body))
            return evidence
        _log.warning("direct HTTP GET for %s returned content the parser found empty", url)

    _log.warning(
        "website acquisition failed for %s: Tavily unusable (%s) AND direct HTTP GET produced no usable evidence",
        url,
        tavily_reason,
    )
    return None


def _ingest_aio(observations) -> list:
    from collectors.google_aio_observed import GoogleAIOObservedAdapter

    adapter = GoogleAIOObservedAdapter()
    batch = adapter.ingest_records(observations) if isinstance(observations, list) else adapter.load_file(observations)
    return list(batch.responses)


def run_pipeline(state: PipelineState, *, providers=_UNSET, acquire=_UNSET, aio_observations=None) -> dict:
    """Run the Phase 11 end-to-end graph.

    ``providers`` -- list of collector objects (each exposing ``.collect(prompt)``);
    ``acquire`` -- callable ``url -> PageEvidence | None``. Pass either
    explicitly (including ``[]`` / a fake) to override; omit to use the real
    implementations. ``aio_observations`` -- observation records (list) or a
    JSON file path; ingested via the unchanged Phase 4 adapter and seeded as
    observed ``ai_responses``. Missing observations are simply absent.
    """
    seeded: PipelineState = dict(state)  # shallow copy; never mutate the caller's dict
    if aio_observations is not None:
        seeded["ai_responses"] = list(seeded.get("ai_responses", [])) + _ingest_aio(aio_observations)

    config = {
        "configurable": {
            "providers": _default_providers() if providers is _UNSET else providers,
            "acquire": _default_acquire if acquire is _UNSET else acquire,
        }
    }
    return full_graph.invoke(seeded, config=config)
