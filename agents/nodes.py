"""LangGraph nodes.

Orchestration ONLY. Each node calls one existing Phase 2 / 6-9 / optimizer
function and returns its slice of the state. No business logic is
implemented here.

Phase 11 adds ``generate_prompts_node``, ``collect_ai_responses`` and
``acquire_website_evidence``. Their external dependencies (provider
collector objects, a website-acquisition callable) are read from the
LangGraph ``config["configurable"]`` mapping -- injected in tests, defaulted
to the real implementations by ``agents.graph.run_pipeline``. When a
dependency is absent a node simply passes the corresponding state through
unchanged, so the graph is also runnable Phase 10 style with pre-supplied
data.
"""

from __future__ import annotations

from agents.state import PipelineState
from analysis.attribution import analyze_attribution
from analysis.gaps import analyze_gaps
from analysis.response_analyzer import analyze_response
from analysis.visibility import compute_visibility
from optimizer.action_planner import plan_actions
from prompts.generator import generate_prompts


def _configurable(config, key, default=None):
    return ((config or {}).get("configurable") or {}).get(key, default)


def collect(state: PipelineState) -> dict:
    """Phase 10: pass through the already-provided AI responses. Real
    provider collection is intentionally not wired here yet.
    """
    return {"ai_responses": list(state.get("ai_responses", []))}


def generate_prompts_node(state: PipelineState, config=None) -> dict:
    if state.get("prompts"):
        return {}
    category = state.get("category")
    if not category:
        return {"prompts": []}
    return {
        "prompts": generate_prompts(
            brand=state["target_brand"],
            category=category,
            competitors=state.get("competitors") or None,
            prompt_count=state.get("prompt_count") or 8,
        )
    }


def collect_ai_responses(state: PipelineState, config=None) -> dict:
    """Run each injected provider collector over the generated prompts and
    append the resulting ``AIResponse`` objects (keeping any already in
    state, e.g. ingested Google AIO observed data). With no provider
    injected, the existing ``ai_responses`` pass through unchanged.
    """
    responses = list(state.get("ai_responses", []))
    providers = _configurable(config, "providers")
    if providers:
        for provider in providers:
            for prompt in state.get("prompts", []):
                responses.append(provider.collect(prompt))
    return {"ai_responses": responses}


def acquire_website_evidence(state: PipelineState, config=None) -> dict:
    """Turn the target and competitor website URLs into ``PageEvidence`` via
    the injected ``acquire`` callable (URL -> ``PageEvidence | None``). With
    no callable injected, any existing evidence in state passes through.
    An acquisition failure yields ``None`` -- never fabricated evidence.
    """
    acquire = _configurable(config, "acquire")
    if acquire is None:
        return {}
    update: dict = {}
    if state.get("website"):
        update["website_data"] = acquire(state["website"])
    competitor_sites = state.get("competitor_sites") or {}
    if competitor_sites:
        update["competitor_data"] = {brand: acquire(url) for brand, url in competitor_sites.items()}
    return update


def analyze_responses(state: PipelineState) -> dict:
    target = state["target_brand"]
    competitors = state.get("competitors", [])
    return {
        "analyzed_responses": [
            analyze_response(response, target_brand=target, competitors=competitors)
            for response in state.get("ai_responses", [])
        ]
    }


def visibility(state: PipelineState) -> dict:
    return {
        "visibility_results": compute_visibility(
            state.get("analyzed_responses", []),
            target_brand=state["target_brand"],
            competitors=state.get("competitors", []),
        )
    }


def diagnose(state: PipelineState) -> dict:
    responses = state.get("analyzed_responses", [])
    target_evidence = state.get("website_data")
    competitor_evidence = state.get("competitor_data", {})
    attribution = analyze_attribution(
        responses,
        target_brand=state["target_brand"],
        competitors=state.get("competitors", []),
        target_evidence=target_evidence,
        competitor_evidence=competitor_evidence,
    )
    gaps = analyze_gaps(
        responses,
        attribution=attribution,
        target_evidence=target_evidence,
        competitor_evidence=competitor_evidence,
    )
    return {"attribution_results": attribution, "gap_results": gaps}


def optimize(state: PipelineState) -> dict:
    return {"action_plan": plan_actions(state["gap_results"])}
