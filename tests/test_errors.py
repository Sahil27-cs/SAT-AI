"""Tests for the error hierarchy."""

from __future__ import annotations

import pytest

from satai.errors import (
    AgentError,
    DataQualityError,
    DataUnavailableError,
    GroundingViolationError,
    ProviderError,
    SatAIError,
)


def test_problem_detail_shape() -> None:
    err = DataUnavailableError("No Sentinel-1 pass over the AOI in the window", aoi="bihar_ganga")
    problem = err.to_problem()
    assert problem["status"] == 404
    assert problem["title"] == "Data Unavailable"
    assert problem["aoi"] == "bihar_ganga"
    assert "Sentinel-1" in problem["detail"]


def test_unavailable_and_quality_are_distinct_conditions() -> None:
    """'We could not get the data' and 'the data is not good enough' are
    different answers and must reach the user as different answers."""
    assert DataUnavailableError.http_status == 404
    assert DataQualityError.http_status == 422
    assert not issubclass(DataQualityError, DataUnavailableError)


def test_provider_error_records_which_provider_failed() -> None:
    err = ProviderError("quota exhausted", provider="cdse", quota="processing_units")
    assert err.provider == "cdse"
    assert err.to_problem()["provider"] == "cdse"


def test_grounding_violation_is_an_agent_error_and_lists_offending_values() -> None:
    err = GroundingViolationError("response stated ungrounded values", ungrounded=[0.82, 1400.0])
    assert isinstance(err, AgentError)
    assert err.ungrounded == [0.82, 1400.0]
    assert err.to_problem()["ungrounded"] == [0.82, 1400.0]


def test_agent_failure_does_not_imply_pipeline_failure() -> None:
    """Requirement 39: the ML system must survive the LLM being unavailable.

    503 signals a degraded optional layer, not a broken pipeline.
    """
    assert AgentError.http_status == 503


@pytest.mark.parametrize("cls", [DataUnavailableError, DataQualityError, ProviderError, AgentError])
def test_everything_descends_from_the_base_error(cls: type[SatAIError]) -> None:
    assert issubclass(cls, SatAIError)
