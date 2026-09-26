"""Tests that enforce the project's scientific commitments as executable rules.

These are not ordinary unit tests. They encode decisions that are easy to erode
silently over months of development -- averaging incommensurable hazard scores,
quietly dropping the vulnerability proxy caveat, adding a hazard coupling with
no physical mechanism. Making them fail the build is cheaper than catching them
in a viva.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from satai.config import get_settings

CONFIGS = get_settings().configs_dir


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def risk() -> dict[str, Any]:
    return _load("risk.yaml")


@pytest.fixture(scope="module")
def couplings() -> dict[str, Any]:
    return _load("couplings.yaml")


class TestRiskEngine:
    def test_risk_is_multiplicative(self, risk: dict[str, Any]) -> None:
        """Zero exposure must imply zero risk.

        An additive or averaged formulation assigns substantial risk to an
        uninhabited floodplain, which is wrong, and is the defect this project
        explicitly sets out not to reproduce.
        """
        assert risk["formulation"]["method"] == "multiplicative"

    def test_exponents_default_to_one(self, risk: dict[str, Any]) -> None:
        """Defaults are neutral; any departure must be argued, not tuned in."""
        exponents = risk["formulation"]["exponents"]
        assert exponents == {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}

    def test_missing_components_are_never_substituted(self, risk: dict[str, Any]) -> None:
        assert risk["formulation"]["missing_component_policy"] == "flag_and_exclude"

    def test_naive_hazard_averaging_is_forbidden(self, risk: dict[str, Any]) -> None:
        assert risk["multi_hazard"]["forbid_naive_average"] is True
        assert risk["multi_hazard"]["primary_output"] == "risk_vector"

    def test_vulnerability_is_declared_a_proxy_and_carries_its_caveat(
        self, risk: dict[str, Any]
    ) -> None:
        vuln = risk["vulnerability"]
        assert vuln["is_proxy"] is True
        assert len(vuln["caveat"]) > 100
        assert "socioeconomic" in vuln["caveat"].lower()

    @pytest.mark.parametrize("block", ["exposure", "vulnerability"])
    def test_component_weights_sum_to_one(self, risk: dict[str, Any], block: str) -> None:
        total = sum(c["weight"] for c in risk[block]["components"].values())
        assert total == pytest.approx(1.0), f"{block} weights sum to {total}"

    def test_earthquake_has_no_hazard_model(self, risk: dict[str, Any]) -> None:
        """SAT-AI performs post-event earthquake damage assessment only."""
        eq = risk["hazard"]["earthquake"]
        assert eq["source"] is None
        assert "prediction" in eq["not_a_claim"].lower()

    @pytest.mark.parametrize("hazard", ["flood", "wildfire", "cyclone", "earthquake"])
    def test_every_hazard_states_what_it_does_not_claim(
        self, risk: dict[str, Any], hazard: str
    ) -> None:
        entry = risk["hazard"][hazard]
        assert entry["claim"]
        assert entry["not_a_claim"]

    def test_risk_levels_carry_the_not_an_official_warning_disclaimer(
        self, risk: dict[str, Any]
    ) -> None:
        disclaimer = risk["levels"]["disclaimer"].upper()
        assert "NOT" in disclaimer
        assert "OFFICIAL" in disclaimer
        for authority in ("IMD", "NDMA"):
            assert authority in risk["levels"]["disclaimer"]

    def test_risk_bands_are_monotonic_and_complete(self, risk: dict[str, Any]) -> None:
        bands = risk["levels"]["bands"]
        assert [b["name"] for b in bands] == ["GREEN", "YELLOW", "ORANGE", "RED"]
        percentiles = [b["max_percentile"] for b in bands]
        assert percentiles == sorted(percentiles)
        assert percentiles[-1] == 100

    def test_sensitivity_analysis_is_mandatory(self, risk: dict[str, Any]) -> None:
        """Arbitrary exponents become a reported result only if they are varied."""
        sens = risk["sensitivity_analysis"]
        assert sens["enabled"] is True
        for param in ("alpha", "beta", "gamma"):
            assert 1.0 in sens["parameters"][param]
            assert len(sens["parameters"][param]) >= 3


class TestCouplings:
    def test_every_active_coupling_states_a_physical_mechanism(
        self, couplings: dict[str, Any]
    ) -> None:
        for coupling in couplings["couplings"]:
            mechanism = coupling.get("mechanism", "")
            assert len(mechanism) > 150, (
                f"{coupling['id']}: a coupling without a writable physical "
                "mechanism does not belong in this file"
            )

    def test_couplings_declare_direction_and_lag(self, couplings: dict[str, Any]) -> None:
        for coupling in couplings["couplings"]:
            assert coupling["from_hazard"] != coupling["to_hazard"]
            assert coupling["lag"]

    def test_coupling_strengths_are_bounded(self, couplings: dict[str, Any]) -> None:
        for coupling in couplings["couplings"]:
            assert 0.0 < coupling["strength"] <= 1.0

    def test_no_earthquake_coupling_exists(self, couplings: dict[str, Any]) -> None:
        for coupling in couplings["couplings"]:
            assert "earthquake" not in (coupling["from_hazard"], coupling["to_hazard"])

    def test_rejected_couplings_are_recorded_with_reasons(self, couplings: dict[str, Any]) -> None:
        """Rejections are kept so the reasoning survives and is not re-litigated."""
        rejected = couplings["rejected"]
        assert len(rejected) >= 3
        assert {r["id"] for r in rejected} >= {"generic_cooccurrence"}
        for entry in rejected:
            assert len(entry["reason"]) > 80

    def test_burn_to_flood_carries_the_indian_stubble_burning_caveat(
        self, couplings: dict[str, Any]
    ) -> None:
        entry = next(c for c in couplings["couplings"] if c["id"] == "burn_to_flood")
        assert "caveat" in entry
        assert "residue" in entry["caveat"].lower() or "stubble" in entry["caveat"].lower()
