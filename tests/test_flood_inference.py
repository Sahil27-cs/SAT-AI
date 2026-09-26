"""The inference and explanation paths, checked where they can be checked.

`ml/flood/predict.py` and `ml/flood/explain.py` produce the artifacts a reader
sees: a flood-extent polygon with an area in km2, and a per-band attribution
that says what the model leaned on. Both are easy to get quietly wrong --

* a pixel area computed as if a degree of longitude were a degree of latitude
  overstates flooded area in northern India by about 13 %, and nothing in the
  output looks wrong;
* an integrated-gradients loop that forgets to divide by its step count, or
  integrates from the wrong endpoint, still returns plausible-looking numbers
  with the wrong magnitude.

So the two are pinned against quantities that can be derived independently:
pixel area against the spheroid constants by hand, and attribution against the
completeness property integrated gradients is defined by. The model here is a
1x1 convolution with known weights, deliberately -- with a real checkpoint the
expected answer would itself have to come from the code under test.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="ADR-004: torch is installed separately from the dev extras"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.flood.explain import integrated_gradients, occlusion  # noqa: E402
from ml.flood.predict import _pixel_area_km2, _polygonise, _write_geotiff  # noqa: E402

rasterio = pytest.importorskip("rasterio", reason="ADR-004: the GDAL chain is conda-only")
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

# One Sen1Floods11 chip: 512x512, EPSG:4326, ~10 m pixels.
PIXEL_DEG = 8.983152841195215e-05


def _profile(*, centre_lat: float, epsg: int = 4326, height: int = 512) -> dict[str, Any]:
    """A raster profile whose grid centre sits at `centre_lat`."""
    top = centre_lat + PIXEL_DEG * height / 2.0
    return {
        "transform": Affine(PIXEL_DEG, 0.0, 86.0, 0.0, -PIXEL_DEG, top),
        "crs": CRS.from_epsg(epsg),
        "height": height,
        "width": height,
        "count": 1,
        "dtype": "float32",
        "nodata": float("nan"),
    }


# --- pixel area --------------------------------------------------------------


def test_pixel_area_matches_the_spheroid_constants_by_hand() -> None:
    area, method = _pixel_area_km2(_profile(centre_lat=0.0))
    assert method == "cosine_latitude"
    expected = (PIXEL_DEG * 111.320 * math.cos(0.0)) * (PIXEL_DEG * 110.574)
    assert area == pytest.approx(expected, rel=1e-9)


def test_area_shrinks_with_latitude() -> None:
    """A degree of longitude is shorter in Bihar than on the equator.

    Without this correction the flooded-area figure is an overestimate that
    grows with latitude, which is exactly the direction that matters: the study
    areas are all in the northern subtropics.
    """
    equator, _ = _pixel_area_km2(_profile(centre_lat=0.0))
    bihar, _ = _pixel_area_km2(_profile(centre_lat=26.0))
    assert bihar < equator
    assert bihar / equator == pytest.approx(math.cos(math.radians(26.0)), rel=1e-3)


def test_a_projected_grid_is_left_alone() -> None:
    """In UTM the pixel is already in metres; a cosine term would be a bug."""
    profile = _profile(centre_lat=26.0, epsg=32645)
    profile["transform"] = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 2_900_000.0)
    area, method = _pixel_area_km2(profile)
    assert method == "projected_crs"
    assert area == pytest.approx(100.0 / 1_000_000.0)


# --- polygonisation ----------------------------------------------------------


def test_polygonise_emits_one_feature_per_blob_and_nothing_for_dry_ground() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:5, 2:5] = True
    mask[10:13, 10:13] = True
    probability = np.where(mask, 0.9, 0.1).astype(np.float32)

    features = _polygonise(mask, _profile(centre_lat=26.0, height=16), probability)

    assert len(features) == 2
    for feature in features:
        assert feature["properties"]["class"] == "flood_extent"
        # 0.9 everywhere inside the mask, so the scene mean is 0.9 exactly.
        assert feature["properties"]["mean_probability_scene"] == pytest.approx(0.9, abs=1e-4)


def test_polygonise_writes_lon_lat_even_from_a_projected_source() -> None:
    """GeoJSON coordinates are WGS84 by specification.

    A UTM easting of 500000 passed through unchanged would put Bihar in the
    Gulf of Guinea.
    """
    profile = _profile(centre_lat=26.0, epsg=32645, height=16)
    profile["transform"] = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 2_900_000.0)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True

    features = _polygonise(mask, profile, np.full((16, 16), 0.8, dtype=np.float32))

    assert len(features) == 1
    ring = features[0]["geometry"]["coordinates"][0]
    for lon, lat in ring:
        assert -180.0 <= lon <= 180.0
        assert -90.0 <= lat <= 90.0


def test_an_empty_mask_yields_no_features() -> None:
    features = _polygonise(
        np.zeros((8, 8), dtype=bool),
        _profile(centre_lat=26.0, height=8),
        np.zeros((8, 8), dtype=np.float32),
    )
    assert features == []


# --- raster writing ----------------------------------------------------------


def test_a_uint8_mask_is_written_without_a_nan_nodata(tmp_path: Path) -> None:
    """The source profile carries a NaN nodata, which uint8 rejects outright.

    Carried over blindly the write aborts; and 0 in this raster means "not
    flooded", not "not observed", so the band has no nodata value to declare.
    """
    path = tmp_path / "mask.tif"
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1

    _write_geotiff(path, mask, _profile(centre_lat=26.0, height=16), dtype="uint8")

    with rasterio.open(path) as src:
        assert src.nodata is None
        assert src.dtypes[0] == "uint8"
        assert src.profile["tiled"] is True
        assert src.overviews(1)  # cloud-optimised: overviews present
        assert int(src.read(1).sum()) == 16


def test_a_float_probability_keeps_nan_as_nodata(tmp_path: Path) -> None:
    path = tmp_path / "probability.tif"
    array = np.full((16, 16), 0.42, dtype=np.float32)

    _write_geotiff(path, array, _profile(centre_lat=26.0, height=16), dtype="float32")

    with rasterio.open(path) as src:
        assert math.isnan(src.nodata)
        assert src.read(1)[0, 0] == pytest.approx(0.42)


# --- explainability ---------------------------------------------------------


class _LinearBands(torch.nn.Module):
    """A pixelwise weighted sum of bands: known attribution by construction.

    A 1x1 convolution keeps every pixel independent, which is what makes the
    completeness check below exact rather than approximate: the gradient at a
    pixel outside the annotated region is zero, so restricting the sum to
    annotated pixels does not discard any of the integral.
    """

    def __init__(self, weights: tuple[float, ...]) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(len(weights), 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor(weights).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _inputs(
    weights: tuple[float, ...], *, size: int = 8, all_valid: bool = True
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = _LinearBands(weights).eval()
    tensor = torch.randn(1, len(weights), size, size)
    valid = torch.ones(1, 1, size, size)
    if not all_valid:
        valid[:, :, size // 2 :, :] = 0.0
    return model, tensor, valid


def _mean_probability(model: torch.nn.Module, x: torch.Tensor, valid: torch.Tensor) -> float:
    with torch.no_grad():
        return float((torch.sigmoid(model(x)) * valid).sum() / valid.sum())


def test_integrated_gradients_satisfies_completeness() -> None:
    """Sum of attributions equals f(input) minus f(baseline).

    This is the property integrated gradients is *defined* by, and it is what
    catches the two mistakes the loop invites: forgetting the division by the
    step count, and accumulating at the wrong point of each interval.
    """
    model, tensor, valid = _inputs((1.5, -0.75, 0.25))
    attribution = integrated_gradients(model, tensor, valid, steps=256)

    delta = _mean_probability(model, tensor, valid) - _mean_probability(
        model, torch.zeros_like(tensor), valid
    )
    assert float(attribution.sum()) == pytest.approx(delta, abs=2e-4)


def test_integrated_gradients_ignores_unannotated_pixels() -> None:
    """Sen1Floods11 labels about a third of every chip as -1.

    An attribution that summed over those pixels would be reporting on ground
    truth that does not exist.
    """
    model, tensor, valid = _inputs((1.0, -1.0), all_valid=False)
    attribution = integrated_gradients(model, tensor, valid, steps=64)

    delta = _mean_probability(model, tensor, valid) - _mean_probability(
        model, torch.zeros_like(tensor), valid
    )
    assert float(attribution.sum()) == pytest.approx(delta, abs=2e-3)


def test_occlusion_reports_the_unmodified_probability_as_its_reference() -> None:
    model, tensor, valid = _inputs((2.0, 0.0))
    deltas, reference = occlusion(model, tensor, valid)

    assert reference == pytest.approx(_mean_probability(model, tensor, valid), abs=1e-6)
    assert deltas.shape == (2,)


def test_occluding_a_band_the_model_does_not_use_changes_nothing() -> None:
    """A zero-weight band is the control.

    If blanking it moves the prediction, the occlusion is touching something it
    should not.
    """
    model, tensor, valid = _inputs((2.0, 0.0))
    deltas, _ = occlusion(model, tensor, valid)
    assert deltas[1] == pytest.approx(0.0, abs=1e-6)
    assert abs(deltas[0]) > 1e-3


def test_the_two_methods_agree_on_a_model_where_the_answer_is_known() -> None:
    """On a linear model both methods should find band 0 the most influential.

    Disagreement here would mean one of the implementations is wrong. On the
    real checkpoint the two *do* disagree about the ratio band, which is a fact
    about that model -- and only interpretable because this case shows the
    methods agree when there is nothing to disagree about.
    """
    model, tensor, valid = _inputs((3.0, 0.1))
    ig = integrated_gradients(model, tensor, valid, steps=64)
    occ, _ = occlusion(model, tensor, valid)
    assert int(np.argmax(np.abs(ig))) == int(np.argmax(np.abs(occ))) == 0
