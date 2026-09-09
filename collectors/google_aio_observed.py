"""Google AI Overview -- observed-data adapter (Phase 4).

This module is the project's third AI environment, integrated **strictly as
observed data**. It is deliberately NOT a live collector and NOT an API
client:

- there is no Google AI Overview API and this module never calls one;
- it performs no network request, no scraping, and no browser automation;
- it never fabricates or generates AI Overview content.

What it does: load AI Overview results that a human has *manually observed*
and recorded, validate each record strictly, normalize the observed text
and citations with the existing Phase 3A utilities, tag each record with
``SourceType.GOOGLE_AIO_OBSERVED``, and yield the existing common
``schemas.models.AIResponse`` so downstream analysis can treat observed
AIO data uniformly alongside the live collectors -- while the
``GOOGLE_AIO_OBSERVED`` identity keeps it explicitly, permanently distinct
from a live provider query.

Flow:

    manually recorded observation record
      -> AIOObservation validation
      -> utils.normalization (text + citations)
      -> AIResponse (success or structured failure)  + ingest metadata

The distinction "citation information was unavailable" is preserved as a
first-class state (``citations_status == "unavailable"``) and is never
silently turned into "zero citations", matching the semantic rule
established in Phase 3A.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from schemas.models import AIResponse, SourceType
from utils.normalization import normalize_answer_text, normalize_citations

DEFAULT_OBSERVATIONS_PATH = "data/observations.json"

# Message used for the "no AI Overview was shown" case. It is phrased so it
# can never be read as a claim that Google AIO was queried programmatically.
ANSWER_NOT_OBSERVED_MESSAGE = (
    "No Google AI Overview was observed for this query "
    "(manually recorded observation; not an API result)."
)

# Citation availability states -- preserves the Phase 3A distinction that
# AIResponse.citations (a plain list) cannot express on its own.
CITATIONS_WITH = "with_citations"
CITATIONS_WITHOUT = "without_citations"
CITATIONS_UNAVAILABLE = "unavailable"

# Answer observation states.
ANSWER_OBSERVED = "observed"
ANSWER_NOT_OBSERVED = "not_observed"


# ---------------------------------------------------------------------------
# Structured validation error
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AIOValidationError:
    """A single, clear, structured reason a record was rejected."""

    observation_id: Optional[str]
    field: Optional[str]
    message: str
    error_type: str  # missing | empty | type | timestamp | citation | source | structure | value | json | io


# ---------------------------------------------------------------------------
# Input models -- the smallest dedicated shape for a manually recorded record
# ---------------------------------------------------------------------------
class AIORawCitation(BaseModel):
    """A source as written down by the person who observed the AI Overview.

    All fields optional individually, but an entry must carry at least one
    of them -- a completely blank citation is a malformed record, not a
    citation with empty values (Phase 3A rule).
    """

    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None

    @model_validator(mode="after")
    def _needs_at_least_one_field(self) -> "AIORawCitation":
        if not any(
            (value or "").strip() for value in (self.url, self.title, self.domain)
        ):
            raise ValueError(
                "citation entry must contain at least one of a non-empty url, title, or domain"
            )
        return self


class AIOObservation(BaseModel):
    """A single manually recorded Google AI Overview observation.

    ``citations`` is tri-state on purpose:
      * ``None``  -> citation information was not captured / unavailable
      * ``[]``    -> the AI Overview was checked and showed no sources
      * ``[...]`` -> sources were observed
    """

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(..., min_length=1)
    prompt_id: Optional[str] = Field(default=None, min_length=1)
    query: str = Field(..., min_length=1)
    brand: str = Field(..., min_length=1)
    category: Optional[str] = Field(default=None, min_length=1)
    answer_observed: bool
    answer_text: Optional[str] = None
    citations: Optional[list[AIORawCitation]] = None
    observed_at: datetime
    source: str = Field(default=SourceType.GOOGLE_AIO_OBSERVED.value)
    notes: Optional[str] = None

    @field_validator("source")
    @classmethod
    def _source_must_be_observed(cls, value: str) -> str:
        expected = SourceType.GOOGLE_AIO_OBSERVED.value
        if value != expected:
            raise ValueError(
                f"source must be '{expected}' for an observed Google AI Overview record; "
                "observed data is never tagged as a live provider"
            )
        return value

    @model_validator(mode="after")
    def _answer_consistency(self) -> "AIOObservation":
        has_text = bool(self.answer_text and self.answer_text.strip())
        if self.answer_observed and not has_text:
            raise ValueError(
                "answer_text is required and must be non-empty when answer_observed is true"
            )
        if not self.answer_observed and has_text:
            raise ValueError(
                "answer_text must be omitted when answer_observed is false"
            )
        if not self.answer_observed and self.citations:
            raise ValueError(
                "citations cannot be present when answer_observed is false"
            )
        return self


# ---------------------------------------------------------------------------
# Ingest results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AIOIngestResult:
    """Outcome of ingesting one observation record.

    On success ``response`` is the common ``AIResponse``; ``answer_status``
    and ``citations_status`` carry the observed-data semantics that do not
    fit inside ``AIResponse`` itself. On failure ``errors`` explains why.
    """

    observation_id: Optional[str]
    ok: bool
    response: Optional[AIResponse]
    errors: tuple[AIOValidationError, ...]
    answer_status: Optional[str]
    citations_status: Optional[str]


@dataclass(frozen=True)
class AIOIngestBatch:
    """The result of ingesting a whole observations document."""

    results: tuple[AIOIngestResult, ...]

    @property
    def ok(self) -> tuple[AIOIngestResult, ...]:
        return tuple(result for result in self.results if result.ok)

    @property
    def failures(self) -> tuple[AIOIngestResult, ...]:
        return tuple(result for result in self.results if not result.ok)

    @property
    def responses(self) -> list[AIResponse]:
        """The validated records as the common downstream representation."""
        return [r.response for r in self.results if r.ok and r.response is not None]


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------
_TYPE_BUCKET = {
    "missing": "missing",
    "string_too_short": "empty",
    "datetime_parsing": "timestamp",
    "datetime_from_date_parsing": "timestamp",
    "datetime_type": "timestamp",
    "extra_forbidden": "structure",
    "bool_parsing": "type",
    "bool_type": "type",
    "string_type": "type",
    "list_type": "type",
    "int_parsing": "type",
    "int_type": "type",
    "dict_type": "type",
    "model_type": "type",
    "model_attributes_type": "type",
}


def _translate_errors(
    exc: ValidationError, observation_id: Optional[str]
) -> tuple[AIOValidationError, ...]:
    translated: list[AIOValidationError] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field = ".".join(str(part) for part in loc) if loc else None
        top = str(loc[0]) if loc else ""
        bucket = _TYPE_BUCKET.get(err.get("type", ""), "value")
        if top == "citations":
            bucket = "citation"
        elif top == "source":
            bucket = "source"
        elif top == "observed_at":
            bucket = "timestamp"
        translated.append(
            AIOValidationError(
                observation_id=observation_id,
                field=field,
                message=err.get("msg", "invalid value"),
                error_type=bucket,
            )
        )
    return tuple(translated)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------
class GoogleAIOObservedAdapter:
    """Loads and validates manually recorded Google AI Overview observations.

    Not a live collector, not an API client. No method here makes a network
    request or touches a browser. It reads a local JSON document, validates
    every record, and returns the common ``AIResponse`` representation plus
    structured failures.
    """

    #: The data environment every record produced here belongs to.
    source_type: SourceType = SourceType.GOOGLE_AIO_OBSERVED

    # -- Loading -----------------------------------------------------------

    def load_file(self, path: str | Path = DEFAULT_OBSERVATIONS_PATH) -> AIOIngestBatch:
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _single_failure(None, f"observations file not found: {target}", "io")
        except OSError as exc:
            return _single_failure(None, f"observations file could not be read: {exc}", "io")
        return self.load_json_text(text)

    def load_json_text(self, text: str) -> AIOIngestBatch:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            return _single_failure(None, f"observations data is not valid JSON: {exc}", "json")
        return self.ingest_document(data)

    def ingest_document(self, data: object) -> AIOIngestBatch:
        if isinstance(data, dict):
            if "observations" not in data:
                return _single_failure(
                    None, "observations document must contain an 'observations' array", "structure"
                )
            records = data["observations"]
        elif isinstance(data, list):
            records = data
        else:
            return _single_failure(
                None, "observations document must be a JSON object or array", "structure"
            )
        if not isinstance(records, list):
            return _single_failure(None, "'observations' must be a JSON array", "structure")
        return self.ingest_records(records)

    def ingest_records(self, raw_records: list[object]) -> AIOIngestBatch:
        return AIOIngestBatch(results=tuple(self._ingest_one(raw) for raw in raw_records))

    # -- Per-record ingestion --------------------------------------------

    def _ingest_one(self, raw: object) -> AIOIngestResult:
        if not isinstance(raw, dict):
            return AIOIngestResult(
                observation_id=None,
                ok=False,
                response=None,
                errors=(
                    AIOValidationError(None, None, "observation record must be a JSON object", "structure"),
                ),
                answer_status=None,
                citations_status=None,
            )

        raw_id = raw.get("observation_id")
        obs_id = raw_id if isinstance(raw_id, str) and raw_id else None

        try:
            observation = AIOObservation.model_validate(raw)
        except ValidationError as exc:
            return AIOIngestResult(
                observation_id=obs_id,
                ok=False,
                response=None,
                errors=_translate_errors(exc, obs_id),
                answer_status=None,
                citations_status=None,
            )

        response, answer_status, citations_status = _to_response(observation)
        return AIOIngestResult(
            observation_id=observation.observation_id,
            ok=True,
            response=response,
            errors=(),
            answer_status=answer_status,
            citations_status=citations_status,
        )


# ---------------------------------------------------------------------------
# Conversion: validated observation -> common AIResponse
# ---------------------------------------------------------------------------
def _to_response(obs: AIOObservation) -> tuple[AIResponse, str, str]:
    prompt_id = obs.prompt_id or obs.observation_id

    if not obs.answer_observed:
        response = AIResponse(
            source_type=SourceType.GOOGLE_AIO_OBSERVED,
            model_name=None,
            prompt_id=prompt_id,
            prompt=obs.query,
            answer=None,
            citations=[],
            timestamp=obs.observed_at,
            success=False,
            error_message=ANSWER_NOT_OBSERVED_MESSAGE,
        )
        return response, ANSWER_NOT_OBSERVED, CITATIONS_UNAVAILABLE

    answer = normalize_answer_text(obs.answer_text)

    raw_citations = (
        None if obs.citations is None else [citation.model_dump() for citation in obs.citations]
    )
    normalized = normalize_citations(raw_citations)
    if normalized is None:
        citations_status = CITATIONS_UNAVAILABLE
        citations: list = []
    elif len(normalized) == 0:
        citations_status = CITATIONS_WITHOUT
        citations = []
    else:
        citations_status = CITATIONS_WITH
        citations = normalized

    response = AIResponse(
        source_type=SourceType.GOOGLE_AIO_OBSERVED,
        model_name=None,
        prompt_id=prompt_id,
        prompt=obs.query,
        answer=answer,
        citations=citations,
        timestamp=obs.observed_at,
        success=True,
        error_message=None,
    )
    return response, ANSWER_OBSERVED, citations_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _single_failure(obs_id: Optional[str], message: str, bucket: str) -> AIOIngestBatch:
    return AIOIngestBatch(
        results=(
            AIOIngestResult(
                observation_id=obs_id,
                ok=False,
                response=None,
                errors=(AIOValidationError(obs_id, None, message, bucket),),
                answer_status=None,
                citations_status=None,
            ),
        )
    )


def load_observations(path: str | Path = DEFAULT_OBSERVATIONS_PATH) -> AIOIngestBatch:
    """Convenience: load + validate the manually recorded observations file."""
    return GoogleAIOObservedAdapter().load_file(path)
