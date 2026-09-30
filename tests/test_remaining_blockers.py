"""The pieces that closed the remaining blockers, tested without a network.

Each test pins a mistake that was either made during this work or is easy to
make next time:

* the ratio-only arm computes VV - VH, and ancillary arms refuse to be built
  without the chip they belong to;
* Sentinel-2 L2A reflectance applies the -1000 offset from baseline 04.00, which
  NBR does not cancel;
* a map overlay is reprojected to lon/lat and leaves unlisted classes clear;
* the Otsu baseline is applied tile by tile and a unimodal tile yields no water;
* the C1 tool mapping marks a benchmark tool with no deployed equivalent as not
  applicable instead of passing it;
* asset signing goes through ``/sign``, because the per-collection token did not
  authenticate against Copernicus DEM's container and every read came back 403.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="the dataset module imports torch")

from ml.flood import dataset  # noqa: E402
from ml.flood.dataset import BandSelection, assemble_stack, stack_with_ratio  # noqa: E402

# --- band selections ------------------------------------------------------


def _sar(vv: float = -10.0, vh: float = -17.0) -> np.ndarray:
    return np.stack([np.full((4, 4), vv), np.full((4, 4), vh)])


def test_ratio_only_is_vv_minus_vh() -> None:
    stack = stack_with_ratio(_sar(), BandSelection.RATIO)
    assert stack.shape == (1, 4, 4)
    assert np.allclose(stack, 7.0)


def test_ratio_only_cancels_a_common_offset() -> None:
    """The reason the arm exists: an offset on both polarisations disappears."""
    shifted = stack_with_ratio(_sar(-10.0 + 2.5, -17.0 + 2.5), BandSelection.RATIO)
    assert np.allclose(shifted, stack_with_ratio(_sar(), BandSelection.RATIO))


@pytest.mark.parametrize(
    "selection", [BandSelection.SAR_DEM, BandSelection.SAR_RAIN, BandSelection.FULL]
)
def test_ancillary_arms_refuse_to_build_without_a_chip(selection: BandSelection) -> None:
    """Their extra bands live on disk per chip; guessing them is not an option."""
    with pytest.raises(ValueError, match="ancillary"):
        stack_with_ratio(_sar(), selection)


def test_band_order_puts_ancillary_after_sar() -> None:
    assert BandSelection.FULL.bands == (
        "vv_db",
        "vh_db",
        "vv_vh_ratio",
        "elevation_m",
        "slope_deg",
        "rain_72h_mm",
    )
    assert BandSelection.SAR_DEM.tag == "sar_dem"
    assert BandSelection.RATIO.tag == "ratio_only"


def test_assemble_stack_reads_co_registered_rasters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    monkeypatch.setattr(dataset, "ANCILLARY_DIR", tmp_path)
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "crs": "EPSG:4326",
        "transform": from_origin(86.0, 27.0, 0.0001, 0.0001),
        "dtype": "float32",
    }
    (tmp_path / "dem").mkdir()
    with rasterio.open(tmp_path / "dem" / "Chip_1.tif", "w", count=2, **profile) as dst:
        dst.write(np.full((4, 4), 120.0, dtype=np.float32), 1)
        dst.write(np.full((4, 4), 3.5, dtype=np.float32), 2)

    stack = assemble_stack(_sar(), "Chip_1", BandSelection.SAR_DEM)
    assert stack.shape == (5, 4, 4)
    assert np.allclose(stack[2], 7.0)  # ratio still computed
    assert np.allclose(stack[3], 120.0)  # elevation
    assert np.allclose(stack[4], 3.5)  # slope


def test_a_missing_ancillary_raster_names_the_command_that_builds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("rasterio")
    monkeypatch.setattr(dataset, "ANCILLARY_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"build_ancillary\.py --dem"):
        assemble_stack(_sar(), "Nowhere_1", BandSelection.SAR_DEM)


# --- Sentinel-2 reflectance -------------------------------------------------


def test_l2a_offset_is_applied_from_baseline_four() -> None:
    from ml.hazards.run_burn_severity import reflectance

    dn = np.array([[0, 1000, 3000]], dtype=np.uint16)
    modern = reflectance(dn, "05.10")
    assert np.isnan(modern[0, 0])  # zero is no-data, not black
    assert modern[0, 1] == pytest.approx(0.0)
    assert modern[0, 2] == pytest.approx(0.2)


def test_no_offset_before_baseline_four() -> None:
    from ml.hazards.run_burn_severity import reflectance

    old = reflectance(np.array([[3000]], dtype=np.uint16), "03.01")
    assert old[0, 0] == pytest.approx(0.3)


def test_an_unapplied_offset_would_shrink_every_nbr() -> None:
    """Why the offset matters for a ratio: it does not cancel."""
    from satai.hazards.wildfire import nbr

    nir, swir = np.array([4000.0]), np.array([2000.0])
    right = nbr((nir - 1000) / 1e4, (swir - 1000) / 1e4)
    wrong = nbr(nir / 1e4, swir / 1e4)
    assert right[0] == pytest.approx(0.5)
    assert wrong[0] == pytest.approx(1 / 3)


# --- map overlays -----------------------------------------------------------


def test_categorical_overlay_is_lon_lat_and_leaves_other_classes_clear(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    pytest.importorskip("PIL")
    from ml.hazards.overlay import categorical_overlay
    from PIL import Image
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    codes = np.zeros((40, 40), dtype=np.int16)
    codes[:20] = 3
    profile = {
        # A UTM grid, so the reprojection has something to do.
        "crs": CRS.from_epsg(32645),
        "transform": from_origin(500_000.0, 2_950_000.0, 20.0, 20.0),
        "height": 40,
        "width": 40,
    }
    overlay = categorical_overlay(
        codes, profile, {3: (200, 50, 20, 255)}, tmp_path / "o.png", max_dim=64
    )

    west, north = overlay.coordinates[0]
    east, south = overlay.coordinates[2]
    assert 86 < west < east < 88 and 26 < south < north < 27
    image = np.asarray(Image.open(overlay.path))
    assert image[..., 3].max() == 255  # the listed class is drawn
    assert image[-1, :, 3].max() == 0  # class 0 is not listed, so it is clear


# --- Otsu over a scene ------------------------------------------------------


def test_otsu_by_tile_finds_water_only_where_the_histogram_is_bimodal() -> None:
    from ml.flood.otsu_scene import TILE, otsu_by_tile

    from satai.ml.baseline import OtsuHandBaseline

    rng = np.random.default_rng(0)
    scene = rng.normal(-8.0, 1.0, (TILE, 2 * TILE))  # dry land everywhere
    scene[:200, :200] = rng.normal(-22.0, 1.0, (200, 200))  # water in tile one only

    water, stats = otsu_by_tile(scene, OtsuHandBaseline())
    assert water[:200, :200].mean() > 0.95
    assert water[:, TILE:].sum() == 0  # the unimodal tile is not bisected
    assert stats["tiles_called_unimodal"] == 1
    assert stats["tiles_thresholded"] == 1


# --- C1 tool mapping --------------------------------------------------------


def test_a_benchmark_tool_without_a_deployed_equivalent_is_not_applicable() -> None:
    from ml.experiments.run_c1_benchmark import mapped

    from satai.agents.benchmark import QUESTIONS

    weather = next(q for q in QUESTIONS if "get_weather_data" in q.expected_tools)
    _translated, applicable = mapped(weather)
    assert applicable is False


def test_mapped_names_are_the_deployed_tools() -> None:
    from ml.experiments.run_c1_benchmark import TOOL_MAP, mapped

    from satai.agents.benchmark import QUESTIONS

    deployed = {
        "get_study_area",
        "get_hazard_result",
        "get_model_info",
        "get_experiment",
        "show_on_map",
    }
    assert {v for v in TOOL_MAP.values() if v} <= deployed
    for question in QUESTIONS:
        translated, _ = mapped(question)
        assert set(translated.expected_tools) <= deployed


# --- asset signing ----------------------------------------------------------


def test_assets_are_signed_through_the_sign_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from satai.providers import planetary

    seen: list[str] = []

    def fake_request(url: str, **_: Any) -> bytes:
        seen.append(url)
        return json.dumps({"href": "https://example.blob/x.tif?sig=1"}).encode()

    monkeypatch.setattr(planetary, "_request", fake_request)
    signed = planetary.sign_href("https://elevationeuwest.blob/x.tif", "cop-dem-glo-30")

    assert signed.endswith("sig=1")
    assert seen and seen[0].startswith(planetary.SIGN_URL)
    assert "href=https" in seen[0]
