"""Turn a result raster into a small PNG overlay a web map can drape.

The analysis rasters are the record and stay on disk: a flood probability
raster for one study area is ~500 MB, far beyond anything a serverless function
or a browser should carry. What the website needs is a picture of the result in
the right place, and MapLibre's ``image`` source does exactly that given a PNG
and its four corners in longitude and latitude.

Two properties matter, and both are enforced here rather than trusted:

* **It is reprojected to EPSG:4326 first.** MapLibre places an image source by
  its corners in lon/lat and stretches it linearly between them. A UTM raster
  handed over as-is would be subtly rotated and sheared, and every pixel would
  sit slightly in the wrong place, which on a flood map means the wrong street.
* **Downsampling is stated, not hidden.** The overlay records the resolution it
  was drawn at, so nobody mistakes a 100 m picture for the 10 m analysis.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["Overlay", "categorical_overlay", "continuous_overlay"]

#: Longest side of an overlay, in pixels. Large enough to show structure at
#: study-area zoom, small enough to ship as a static file in the frontend bundle.
MAX_DIM = 1600


@dataclass(frozen=True)
class Overlay:
    """A PNG plus where it goes on the map and how it was made."""

    path: Path
    #: [[west, north], [east, north], [east, south], [west, south]], the order
    #: MapLibre's image source expects.
    coordinates: list[list[float]]
    width: int
    height: int
    resolution_deg: float
    approx_resolution_m: float

    def describe(self) -> dict[str, Any]:
        return {
            "coordinates": self.coordinates,
            "width_px": self.width,
            "height_px": self.height,
            "resolution_deg": round(self.resolution_deg, 7),
            "approx_resolution_m": round(self.approx_resolution_m, 1),
            "note": (
                "A downsampled picture of the result for display. The analysis "
                "itself ran at full resolution and is kept on disk."
            ),
        }

    def data_uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.path.read_bytes()).decode("ascii")


def _to_lonlat(
    array: npt.NDArray[Any],
    profile: dict[str, Any],
    *,
    categorical: bool,
    max_dim: int,
) -> tuple[npt.NDArray[Any], list[list[float]], float]:
    """Reproject one band onto a regular lon/lat grid no larger than ``max_dim``."""
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, transform_bounds

    height, width = array.shape
    left = profile["transform"].c
    top = profile["transform"].f
    right = left + profile["transform"].a * width
    bottom = top + profile["transform"].e * height
    west, south, east, north = transform_bounds(
        profile["crs"], "EPSG:4326", left, bottom, right, top, densify_pts=21
    )

    span = max(east - west, north - south)
    resolution = span / max_dim
    out_w = max(1, round((east - west) / resolution))
    out_h = max(1, round((north - south) / resolution))

    source = np.asarray(array, dtype=np.float32)
    destination = np.full((out_h, out_w), np.nan, dtype=np.float32)
    reproject(
        source=source,
        destination=destination,
        src_transform=profile["transform"],
        src_crs=profile["crs"],
        dst_transform=from_bounds(west, south, east, north, out_w, out_h),
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        # Classes must never be averaged into a class that does not exist; a
        # continuous field is better represented by the mean of what it covers.
        resampling=Resampling.mode if categorical else Resampling.average,
    )
    corners = [[west, north], [east, north], [east, south], [west, south]]
    return destination, corners, resolution


def _save(rgba: npt.NDArray[np.uint8], path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    path.write_bytes(buffer.getvalue())


def categorical_overlay(
    codes: npt.NDArray[Any],
    profile: dict[str, Any],
    palette: dict[int, tuple[int, int, int, int]],
    path: Path,
    *,
    max_dim: int = MAX_DIM,
) -> Overlay:
    """Draw integer classes with a fixed colour each; unlisted codes are clear."""
    lonlat, corners, resolution = _to_lonlat(
        codes.astype(np.float32), profile, categorical=True, max_dim=max_dim
    )
    rgba = np.zeros((*lonlat.shape, 4), dtype=np.uint8)
    for code, colour in palette.items():
        rgba[lonlat == code] = colour
    _save(rgba, path)
    return _overlay(path, corners, rgba.shape, resolution)


def continuous_overlay(
    values: npt.NDArray[Any],
    profile: dict[str, Any],
    path: Path,
    *,
    vmin: float,
    vmax: float,
    colour: tuple[int, int, int],
    floor: float | None = None,
    max_dim: int = MAX_DIM,
) -> Overlay:
    """Draw a [vmin, vmax] field as one hue whose opacity follows the value.

    One hue with varying opacity rather than a rainbow scale: the underlying
    basemap stays readable, and a stronger colour always means a larger value,
    which a multi-hue ramp does not guarantee. Values at or below ``floor`` are
    left fully transparent.
    """
    lonlat, corners, resolution = _to_lonlat(values, profile, categorical=False, max_dim=max_dim)
    scaled = np.clip((lonlat - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    alpha = np.where(np.isfinite(lonlat), scaled * 230, 0).astype(np.uint8)
    if floor is not None:
        alpha[~np.isfinite(lonlat) | (lonlat <= floor)] = 0
    rgba = np.zeros((*lonlat.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = colour
    rgba[..., 3] = alpha
    _save(rgba, path)
    return _overlay(path, corners, rgba.shape, resolution)


def _overlay(
    path: Path, corners: list[list[float]], shape: tuple[int, ...], resolution: float
) -> Overlay:
    mid_lat = (corners[0][1] + corners[2][1]) / 2.0
    metres = resolution * 111_320.0 * float(np.cos(np.radians(mid_lat)))
    return Overlay(
        path=path,
        coordinates=[[round(x, 6), round(y, 6)] for x, y in corners],
        width=int(shape[1]),
        height=int(shape[0]),
        resolution_deg=resolution,
        approx_resolution_m=metres,
    )
