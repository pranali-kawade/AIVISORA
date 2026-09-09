"""Foundational Pydantic schemas for AEO Radar.

These models define the data contracts shared across the whole application
(collectors, analysis, optimizer, UI). They are intentionally minimal in
Phase 1: only the structures that are stable enough to define now are
implemented here. Analysis-stage schemas (VisibilityMetrics, Gap,
ActionItem, AttributionResult, CompetitorAnalysis, OptimizationScore) are
deliberately deferred to later phases, once the analysis design is settled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class SourceType(str, Enum):
    """Identifies where an AI response or observation actually came from.

    Every collected response must be tagged with exactly one of these
    values so that LIVE data, OPEN/FREE MODEL data, and OBSERVED/SAMPLE
    data are never silently mixed together. No value here may claim to be
    ChatGPT, Claude, Perplexity, or a live Google AI Overview API, because
    the project never has access to any of those as paid/live services.
    """

    LIVE_GEMINI = "live_gemini"
    OPENROUTER_FREE = "openrouter_free"
    GOOGLE_AIO_OBSERVED = "google_aio_observed"
    MOCK = "mock"


class PromptItem(BaseModel):
    """A single prompt to be sent to an AI collector (or replayed against
    observed data). Kept intentionally small and extensible.
    """

    prompt_id: str = Field(..., min_length=1, description="Stable unique identifier for this prompt")
    prompt: str = Field(..., min_length=1, description="The prompt text itself")
    intent: Optional[str] = Field(
        default=None,
        description="Category/intent label, e.g. 'comparison', 'recommendation', 'how_to'",
    )


class Citation(BaseModel):
    """A normalized citation extracted from an AI response.

    Not every provider returns citations, so every field here is optional.
    Absence of citation data must never be treated as a citation with
    empty values -- callers should check whether a Citation exists at all.
    """

    url: Optional[HttpUrl] = Field(default=None, description="Citation URL, if provided by the source")
    title: Optional[str] = Field(default=None, description="Citation title, if provided by the source")
    domain: Optional[str] = Field(default=None, description="Source domain, if known or derivable")


class BrandMention(BaseModel):
    """An observed mention of a brand within an AI response.

    This only records the observation itself (what was mentioned, where,
    and how) -- it does not calculate any visibility score, ranking, or
    aggregate metric. Those calculations belong to the analysis layer in
    a later phase.
    """

    brand_name: str = Field(..., min_length=1, description="Name of the brand as it was mentioned")
    mentioned: bool = Field(..., description="Whether the brand was actually mentioned in the response")
    position: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based position/rank of the mention within the response, if determinable",
    )
    recommended: Optional[bool] = Field(
        default=None,
        description="Whether the response appears to recommend this brand, if determinable",
    )
    context_snippet: Optional[str] = Field(
        default=None,
        description="Short snippet of surrounding text where the mention occurred",
    )


class AIResponse(BaseModel):
    """Normalized representation of a single AI collector response.

    Must support both successful and failed provider responses without
    causing the application to crash: `success=False` responses are valid
    and expected (rate limits, timeouts, malformed provider output, etc.)
    """

    source_type: SourceType = Field(..., description="Which data source/mode produced this response")
    model_name: Optional[str] = Field(default=None, description="Underlying model name/id, if applicable")
    prompt_id: str = Field(..., min_length=1, description="Identifier of the PromptItem this responds to")
    prompt: str = Field(..., min_length=1, description="The prompt text that was sent")
    answer: Optional[str] = Field(default=None, description="Raw answer text returned by the provider")
    citations: list[Citation] = Field(default_factory=list, description="Citations extracted from the response")
    brand_mentions: list[BrandMention] = Field(
        default_factory=list,
        description="Brand mentions detected in this response, if any extraction has been performed",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when this response was collected",
    )
    success: bool = Field(..., description="Whether the collection attempt succeeded")
    error_message: Optional[str] = Field(
        default=None,
        description="Human-readable error description when success=False",
    )

    def model_post_init(self, __context: object) -> None:
        """Enforce that failed responses always carry an explanation, and
        that successful responses are not silently empty.
        """
        if not self.success and not self.error_message:
            raise ValueError("error_message is required when success=False")
