"""SAT-AI: multi-hazard prediction and early-warning research prototype.

Shared core library imported by both ``backend/`` (the FastAPI serving plane)
and ``ml/`` (the training and batch-inference plane), so that configuration,
logging, error types, and -- above all -- the provenance contract are defined
exactly once and cannot drift apart between the two.

SCOPE NOTE, restated here because it governs the whole codebase:

  SAT-AI performs hazard *risk assessment*, *detection*, *monitoring*, and
  *post-event damage assessment*. It does not forecast earthquakes, cyclone
  tracks, or wildfire ignition. Its risk levels are research outputs and are
  not official warnings.
"""

from __future__ import annotations

from satai.config import Settings, get_settings
from satai.errors import (
    DataQualityError,
    DataUnavailableError,
    GroundingViolationError,
    ModelUnavailableError,
    ProviderError,
    SatAIError,
)
from satai.logging import configure_logging, get_logger
from satai.provenance import (
    DataSourceRef,
    HazardType,
    ProvenanceEnvelope,
    SourceKind,
    SourceRef,
    SpatialRef,
    TemporalValidity,
    envelope,
)

__version__ = "0.1.0"

__all__ = [
    "DataQualityError",
    "DataSourceRef",
    "DataUnavailableError",
    "GroundingViolationError",
    "HazardType",
    "ModelUnavailableError",
    "ProvenanceEnvelope",
    "ProviderError",
    "SatAIError",
    "Settings",
    "SourceKind",
    "SourceRef",
    "SpatialRef",
    "TemporalValidity",
    "__version__",
    "configure_logging",
    "envelope",
    "get_logger",
    "get_settings",
]
