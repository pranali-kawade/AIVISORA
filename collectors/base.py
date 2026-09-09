"""Provider-agnostic collector interface.

This module defines the contract that every future AI response collector
(Gemini, OpenRouter, Google AIO observed-data replay, mock, etc.) must
follow. It intentionally contains no network calls, no HTTP client, and
no provider-specific logic -- those belong in dedicated collector modules
implemented in a later phase.

Keeping this interface stable means the analysis layer can depend on
`AIResponse` without caring which concrete collector produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.models import AIResponse, PromptItem, SourceType


class BaseCollector(ABC):
    """Abstract base class for all AI response collectors.

    Concrete implementations are responsible for:
    - applying a network timeout to any request they make;
    - catching and translating connection/provider errors into a
      non-crashing `AIResponse(success=False, error_message=...)`;
    - never fabricating a successful response when the provider failed.
    """

    #: Concrete subclasses must declare which SourceType they produce.
    source_type: SourceType

    @abstractmethod
    def collect(self, prompt: PromptItem) -> AIResponse:
        """Collect a single AI response for the given prompt.

        Must always return a validated `AIResponse`, even on failure.
        Must never raise an unhandled exception for expected failure
        modes such as timeouts, rate limits, or malformed provider
        output -- those should be captured in the returned response's
        `success` / `error_message` fields instead.
        """
        raise NotImplementedError
