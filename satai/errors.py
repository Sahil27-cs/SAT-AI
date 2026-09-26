"""Typed error hierarchy for SAT-AI.

Errors carry enough structure for the API layer to turn them into RFC 7807
problem-detail responses without a chain of ``isinstance`` checks, and for the
agent layer to tell a user *why* something is unavailable rather than silently
degrading.

The distinction that matters most here is between
:class:`DataUnavailableError` and :class:`DataQualityError`. The first means
"we could not get the data"; the second means "we got it and it is not good
enough to answer with". Both must surface. Neither may ever be resolved by
substituting a plausible number (requirement 39).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentError",
    "ConfigurationError",
    "DataQualityError",
    "DataUnavailableError",
    "GroundingViolationError",
    "ModelUnavailableError",
    "ProviderError",
    "SatAIError",
    "ValidationError",
]


class SatAIError(Exception):
    """Base class for every SAT-AI error.

    Parameters
    ----------
    message:
        Human-readable explanation, safe to show a user.
    detail:
        Structured context for logs and problem-detail responses. Must never
        contain credentials.
    """

    http_status: int = 500
    error_type: str = "internal_error"

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail

    def to_problem(self) -> dict[str, Any]:
        """Render as an RFC 7807 problem-detail body."""
        return {
            "type": f"https://sat-ai.example/errors/{self.error_type}",
            "title": self.error_type.replace("_", " ").title(),
            "status": self.http_status,
            "detail": self.message,
            **self.detail,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, {self.detail!r})"


class ConfigurationError(SatAIError):
    """A required setting or credential is missing or malformed."""

    http_status = 500
    error_type = "configuration_error"


class ProviderError(SatAIError):
    """An upstream data provider failed (network, auth, quota, outage)."""

    http_status = 502
    error_type = "provider_error"

    def __init__(self, message: str, *, provider: str, **detail: Any) -> None:
        super().__init__(message, provider=provider, **detail)
        self.provider = provider


class DataUnavailableError(SatAIError):
    """No data exists for the requested area, time, or modality.

    Legitimate and common: no Sentinel-1 pass over the AOI in the window, full
    cloud cover on every optical scene, a region outside the study area. The
    correct response is to say so.
    """

    http_status = 404
    error_type = "data_unavailable"


class DataQualityError(SatAIError):
    """Data was retrieved but fails quality gates.

    e.g. cloud fraction above threshold, incompatible orbit geometry between a
    pre/post SAR pair, or a DEM void over the AOI.
    """

    http_status = 422
    error_type = "data_quality"


class ModelUnavailableError(SatAIError):
    """A model artifact is missing, corrupt, or the wrong version."""

    http_status = 503
    error_type = "model_unavailable"


class ValidationError(SatAIError):
    """Input failed validation (bad bbox, unknown hazard, impossible date range)."""

    http_status = 400
    error_type = "validation_error"


class AgentError(SatAIError):
    """The agent plane failed: LLM unreachable, tool loop exhausted, bad routing.

    The ML plane must keep working when this is raised. The API falls back to
    returning structured tool output with no natural-language layer.
    """

    http_status = 503
    error_type = "agent_error"


class GroundingViolationError(AgentError):
    """A generated response contained a value not traceable to any envelope.

    This is the load-bearing safety error of the whole system. It is raised by
    the grounding validator (Phase 11), logged, counted, and reported as an
    evaluation metric -- the measured hallucination rate under constraint.
    """

    http_status = 500
    error_type = "grounding_violation"

    def __init__(
        self, message: str, *, ungrounded: list[float] | None = None, **detail: Any
    ) -> None:
        super().__init__(message, ungrounded=ungrounded or [], **detail)
        self.ungrounded = ungrounded or []
