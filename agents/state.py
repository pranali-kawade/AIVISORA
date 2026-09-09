"""Phase 10 -- minimal LangGraph shared state for the orchestration pipeline.

A plain ``TypedDict``: LangGraph merges each node's partial dict return into
it. It carries only existing project models -- no duplicate domain types.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from analysis.attribution import AttributionSummary
from analysis.gaps import GapReport
from analysis.visibility import VisibilityMetrics
from crawler.parser import PageEvidence
from optimizer.action_planner import ActionPlan
from schemas.models import AIResponse, PromptItem


class PipelineState(TypedDict, total=False):
    # --- inputs (supplied by the caller) ---
    target_brand: str
    website: Optional[str]
    competitors: list[str]
    category: Optional[str]                     # for prompt generation
    prompt_count: int                           # for prompt generation
    competitor_sites: dict[str, str]            # competitor brand -> website URL
    prompts: list[PromptItem]
    ai_responses: list[AIResponse]
    website_data: Optional[PageEvidence]
    competitor_data: dict[str, PageEvidence]
    # --- outputs (filled in by the nodes) ---
    analyzed_responses: list[AIResponse]
    visibility_results: Optional[VisibilityMetrics]
    attribution_results: Optional[AttributionSummary]
    gap_results: Optional[GapReport]
    action_plan: Optional[ActionPlan]
