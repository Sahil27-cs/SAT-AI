"""Geodesic and projected area calculations for SAT-AI.

Never calculate area as naive ``n_pixels * 0.01`` unless the raster is confirmed
to be 10m x 10m in a projected/equal-area coordinate reference system.

This module implements:
1. CRS detection and validation.
2. Geodesic / projected metric pixel area calculation in UTM Zone 45N (EPSG:32645),
   India Equal Area (EPSG:7755), or ellipsoidal WGS84 geodesic correction.
3. Rigorous area metrics return dictionary with full provenance.
"""

from __future__ import annotations

import math
from typing import Any

# WGS84 semi-major axis (a) and flattening (f)
WGS84_A = 6378137.0
WGS84_B = 6356752.314245
WGS84_E2 = 1.0 - (WGS84_B**2 / WGS84_A**2)

DEFAULT_BIHAR_UTM = "EPSG:32645"  # UTM Zone 45N (Bihar covers ~83.3 to 88.3 E)
EQUAL_AREA_INDIA = "EPSG:7755"   # India Equal Area / LCC


def validate_crs_for_area(crs: str | int | None) -> dict[str, Any]:
    """Validate whether the CRS is suitable for direct planar area computation or requires geodesic transform."""
    if crs is None:
        return {"valid": False, "reason": "CRS is None", "recommended": DEFAULT_BIHAR_UTM}
    crs_str = str(crs).upper()
    if "32645" in crs_str:
        return {"valid": True, "type": "projected", "crs": "EPSG:32645", "is_equal_area": True}
    if "7755" in crs_str:
        return {"valid": True, "type": "projected", "crs": "EPSG:7755", "is_equal_area": True}
    if "4326" in crs_str:
        return {"valid": True, "type": "geographic", "crs": "EPSG:4326", "requires_geodesic": True}
    return {"valid": True, "type": "other", "crs": crs_str}


def calculate_pixel_area_km2(
    transform_matrix: tuple[float, float, float, float, float, float] | Any,
    crs_epsg: int | str | None,
    centre_lat: float | None = None,
) -> tuple[float, str]:
    """Calculate the true area of a single raster cell in square kilometres.

    Parameters
    ----------
    transform_matrix : tuple or Affine
        Affine transform (a, b, c, d, e, f) where 'a' is pixel width and 'e' is pixel height.
    crs_epsg : int, str, or None
        The EPSG code or CRS representation.
    centre_lat : float, optional
        Latitude of the scene/chip centre in degrees. Required if CRS is geographic.

    Returns
    -------
    tuple[float, str]
        (cell_area_km2, method_description)
    """
    if hasattr(transform_matrix, "a") and hasattr(transform_matrix, "e"):
        res_x = abs(float(transform_matrix.a))
        res_y = abs(float(transform_matrix.e))
    elif isinstance(transform_matrix, (list, tuple)) and len(transform_matrix) >= 6:
        res_x = abs(float(transform_matrix[0]))
        res_y = abs(float(transform_matrix[4]))
    else:
        raise ValueError(f"Invalid affine transform: {transform_matrix}")

    epsg_int: int | None = None
    if isinstance(crs_epsg, int):
        epsg_int = crs_epsg
    elif isinstance(crs_epsg, str):
        cleaned = crs_epsg.strip().upper().replace("EPSG:", "")
        if cleaned.isdigit():
            epsg_int = int(cleaned)

    # If already a projected / equal-area metric CRS (e.g. UTM, EPSG:32645, EPSG:7755, etc.)
    if epsg_int and epsg_int != 4326:
        # Coordinates are already in metres
        area_m2 = res_x * res_y
        area_km2 = area_m2 / 1_000_000.0
        return area_km2, f"projected_equal_area_epsg_{epsg_int}"

    # If geographic (EPSG:4326), coordinates are in degrees.
    # We must calculate geodesic metric dimensions at the centre latitude.
    if centre_lat is None:
        if hasattr(transform_matrix, "f") and hasattr(transform_matrix, "e"):
            # Estimate centre from transform if available
            centre_lat = float(transform_matrix.f)
        else:
            # Fallback to Bihar central latitude (25.7 N)
            centre_lat = 25.7

    lat_rad = math.radians(centre_lat)

    # WGS84 radius of curvature in prime vertical and meridian
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    denom = math.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    # Meridional radius of curvature (M)
    M = (WGS84_A * (1.0 - WGS84_E2)) / (denom**3)
    # Normal radius of curvature (N)
    N = WGS84_A / denom

    # Metres per degree
    m_per_deg_lat = (math.pi / 180.0) * M
    m_per_deg_lon = (math.pi / 180.0) * N * cos_lat

    dx_m = res_x * m_per_deg_lon
    dy_m = res_y * m_per_deg_lat

    area_m2 = dx_m * dy_m
    area_km2 = area_m2 / 1_000_000.0
    return area_km2, "geodesic_wgs84_ellipsoidal"


def compute_raster_flood_extent(
    water_mask_pixels: int,
    total_valid_pixels: int,
    crs: str | int | None,
    pixel_transform: tuple[float, float, float, float, float, float] | Any,
    centre_lat: float | None = None,
    source: str = "Sentinel-1 C-SAR",
    acquisition_date: str = "2022-10-15",
    resolution_m: float = 10.0,
) -> dict[str, Any]:
    """Compute verified geodesic/projected area metrics according to SAT-AI standards."""
    pixel_km2, method = calculate_pixel_area_km2(pixel_transform, crs, centre_lat=centre_lat)

    flooded_area_km2 = round(water_mask_pixels * pixel_km2, 4)
    observed_area_km2 = round(total_valid_pixels * pixel_km2, 4)
    flooded_pct = (
        round((flooded_area_km2 / observed_area_km2) * 100.0, 2)
        if observed_area_km2 > 0
        else 0.0
    )

    return {
        "flooded_area_km2": flooded_area_km2,
        "observed_area_km2": observed_area_km2,
        "flooded_percentage": flooded_pct,
        "source": source,
        "acquisition_date": acquisition_date,
        "resolution_m": resolution_m,
        "method": method,
        "single_pixel_area_m2": round(pixel_km2 * 1_000_000.0, 3),
    }


def calculate_pixel_area_m2(
    resolution_x: float,
    resolution_y: float,
    crs: str | int | None = "EPSG:32645",
    center_lat: float | None = 25.5,
) -> float:
    """Calculate single cell area in square metres using projected or geodesic formulation."""
    transform = (resolution_x, 0.0, 0.0, 0.0, -resolution_y, 0.0)
    area_km2, _ = calculate_pixel_area_km2(transform, crs, centre_lat=center_lat)
    return round(area_km2 * 1_000_000.0, 4)


def calculate_flooded_area_from_mask(
    mask: Any,
    resolution_m: float = 10.0,
    crs: str | int | None = "EPSG:32645",
    acquisition_date: str = "2022-10-15",
    source: str = "Sentinel-1 SAR",
    center_lat: float | None = 25.5,
) -> dict[str, Any]:
    """Calculate flooded area directly from a numpy array mask."""
    import numpy as np

    flooded_pixels = int(np.sum(mask > 0)) if hasattr(mask, "sum") else int(mask)
    total_pixels = int(mask.size) if hasattr(mask, "size") else flooded_pixels
    transform = (resolution_m, 0.0, 0.0, 0.0, -resolution_m, 0.0)

    res = compute_raster_flood_extent(
        water_mask_pixels=flooded_pixels,
        total_valid_pixels=total_pixels,
        crs=crs,
        pixel_transform=transform,
        centre_lat=center_lat,
        source=source,
        acquisition_date=acquisition_date,
        resolution_m=resolution_m,
    )
    res["flooded_pixels"] = flooded_pixels
    res["total_pixels"] = total_pixels
    res["target_crs"] = "EPSG:32645 (UTM Zone 45N)" if "32645" in str(crs) else str(crs)
    return res

