"""
tests/test_bihar_impact.py
==========================
Comprehensive tests for Bihar Flood Impact Assessment System:
1. Geodesic and projected metric calculations in UTM Zone 45N (EPSG:32645).
2. GIS Impact Engine (38 Bihar districts, exposure layers, SAT-AI Impact Index).
3. Crop damage model inference and BFCD-22 benchmarks.
4. All 10 backend tools executed through execute_flood_tool dispatcher.
5. Natural language question interpretation, zero-hallucination policy, and Hindi/Hinglish queries.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "backend" / "api"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from satai.geo.area import (
    calculate_pixel_area_m2,
    calculate_flooded_area_from_mask,
    validate_crs_for_area,
)
from satai.geo.bihar_impact import (
    load_bihar_districts,
    get_event_inundation,
    calculate_district_impact,
    get_bihar_crop_damage_stats,
    BIHAR_FLOOD_EVENTS,
    DEFAULT_WEIGHTS,
)
from backend.api.agent_tools import execute_flood_tool, FLOOD_TOOL_DECLARATIONS
from backend.api.satai_agents import _natural_fallback_answer


# ============================================================================
# 1. Geodesic & Projected Metric CRS Calculations (UTM Zone 45N)
# ============================================================================

def test_pixel_area_projected_utm45n():
    """Verify UTM Zone 45N pixel area calculation for 10m Sentinel raster."""
    area_m2 = calculate_pixel_area_m2(resolution_x=10.0, resolution_y=10.0, crs="EPSG:32645")
    assert area_m2 == 100.0


def test_pixel_area_geographic_wgs84():
    """Verify WGS84 geodesic pixel area at Bihar latitude (~25.5° N)."""
    # 0.0001 deg is approx 11.1m lat, 10.0m lon
    area_m2 = calculate_pixel_area_m2(
        resolution_x=0.0001,
        resolution_y=0.0001,
        crs="EPSG:4326",
        center_lat=25.5,
    )
    # Cosine of 25.5 deg is ~0.9026
    # 11.132km/deg * 11.132km/deg * 0.9026 * 1e-8 * 1e6 ~ 111.9 m^2
    assert 100.0 < area_m2 < 125.0


def test_calculate_flooded_area_from_mask():
    """Verify projected area calculation from binary raster mask."""
    # 1000 x 1000 pixels at 10m resolution = 10,000 m * 10,000 m = 100 km^2
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[200:400, 200:400] = 1  # 200 x 200 = 40,000 flooded pixels = 4 km^2

    result = calculate_flooded_area_from_mask(
        mask=mask,
        resolution_m=10.0,
        crs="EPSG:32645",
        acquisition_date="2022-10-15",
        source="Sentinel-1 SAR",
    )

    assert result["flooded_pixels"] == 40000
    assert result["total_pixels"] == 1000000
    assert result["flooded_area_km2"] == 4.0
    assert result["observed_area_km2"] == 100.0
    assert result["flooded_percentage"] == 4.0
    assert result["target_crs"] == "EPSG:32645 (UTM Zone 45N)"
    assert "epsg_32645" in result["method"].lower() or "utm" in result["method"].lower()


# ============================================================================
# 2. Bihar GIS Impact Engine & Districts
# ============================================================================

def test_load_bihar_districts_count():
    """Verify all 38 districts of Bihar are loaded with required attributes."""
    districts = load_bihar_districts()
    assert len(districts) == 38

    # Spot check high-hazard districts
    muzaffarpur = next((d for d in districts if d["name"] == "Muzaffarpur"), None)
    assert muzaffarpur is not None
    assert muzaffarpur["district_area_km2"] > 3000
    assert muzaffarpur["cropland_area_km2"] > 1500
    assert muzaffarpur["historical_hazard_class"] == "Very High"


def test_calculate_district_impact_ranking():
    """Verify district aggregation and impact index calculation."""
    impact = calculate_district_impact("2022-10-15", sort_by="flooded_area_km2")
    assert impact["status"] == "success"
    assert impact["total_districts_analyzed"] == 38
    assert impact["total_flooded_area_km2"] > 3000

    # Top district for 2022-10-15 event is Muzaffarpur or Darbhanga
    top = impact["rankings"][0]
    assert "district_name" in top
    assert top["flooded_area_km2"] > 0
    assert top["sat_ai_impact_index"] > 0
    assert "provenance" in impact
    assert impact["provenance"]["status_label"] == "RESEARCH ANALYSIS — NOT AN OFFICIAL WARNING"


def test_bfcd22_crop_damage_stats():
    """Verify BFCD-22 crop damage statistics for Muzaffarpur benchmark."""
    crop_stats = get_bihar_crop_damage_stats("Muzaffarpur")
    assert crop_stats["status"] == "success"
    assert crop_stats["district"] == "Muzaffarpur"
    assert "classes" in crop_stats
    assert crop_stats["classes"]["0_no_damage"]["name"] == "No Damage"
    assert crop_stats["classes"]["1_partial_damage"]["name"] == "Partial Damage"
    assert crop_stats["classes"]["2_full_damage"]["name"] == "Full Damage"
    assert crop_stats["validation_metrics"]["macro_f1"] == 0.7420


# ============================================================================
# 3. Backend Tool Declarations & Dispatcher
# ============================================================================

def test_all_10_tools_declared():
    """Verify that all 10 Bihar impact tools are declared in tool schema."""
    declared_names = {t["name"] for t in FLOOD_TOOL_DECLARATIONS}
    expected_tools = [
        "get_bihar_flood_status",
        "get_bihar_district_impact",
        "get_flooded_area",
        "get_crop_damage_summary",
        "get_building_exposure",
        "get_road_exposure",
        "get_historical_flood_hazard",
        "get_flood_event_dates",
        "get_impact_index",
        "get_impact_provenance",
    ]
    for tool_name in expected_tools:
        assert tool_name in declared_names, f"Tool {tool_name} not found in FLOOD_TOOL_DECLARATIONS"


def test_execute_bihar_flood_status():
    """Test get_bihar_flood_status tool execution."""
    res = execute_flood_tool("get_bihar_flood_status", {"event_date": "2022-10-15"})
    assert res.get("available") is True
    assert res["region"] == "Bihar"
    assert res["flooded_area_km2"] > 3000
    assert res["flooded_percentage"] > 3.0


def test_execute_bihar_district_impact():
    """Test get_bihar_district_impact tool execution."""
    res = execute_flood_tool("get_bihar_district_impact", {"district": "Muzaffarpur"})
    assert res.get("available") is True
    d = res.get("district", res)
    assert d["district_name"] == "Muzaffarpur"
    assert d["flooded_area_km2"] == 412.6
    assert d["cropland_affected_km2"] == 284.5


def test_execute_get_flooded_area():
    """Test get_flooded_area tool execution."""
    res = execute_flood_tool("get_flooded_area", {"region": "Bihar", "event_date": "2022-10-15"})
    assert res.get("available") is True
    assert res["flooded_area_km2"] > 3000
    assert "utm" in res["method"].lower() or "projected" in res["method"].lower()


def test_execute_get_crop_damage_summary():
    """Test get_crop_damage_summary tool execution."""
    res = execute_flood_tool("get_crop_damage_summary", {"district": "Muzaffarpur"})
    assert res.get("available") is True
    assert "damage_distribution" in res
    assert res["damage_distribution"]["partial_damage_km2"] == 38.6
    assert res["damage_distribution"]["full_damage_km2"] == 25.6


def test_execute_get_building_and_road_exposure():
    """Test get_building_exposure and get_road_exposure tools."""
    b_res = execute_flood_tool("get_building_exposure", {"district": "Muzaffarpur"})
    assert b_res.get("available") is True
    assert b_res["buildings_exposed_count"] == 42100

    r_res = execute_flood_tool("get_road_exposure", {"district": "Muzaffarpur"})
    assert r_res.get("available") is True
    assert r_res["roads_exposed_km"] == 194.2


def test_execute_get_historical_hazard():
    """Test get_historical_flood_hazard tool execution."""
    res = execute_flood_tool("get_historical_flood_hazard", {"district": "Muzaffarpur"})
    assert res.get("available") is True
    assert res["historical_hazard_class"] == "Very High"
    assert "1998-2019" in res["data_source"]


def test_execute_get_flood_event_dates():
    """Test get_flood_event_dates tool."""
    res = execute_flood_tool("get_flood_event_dates", {})
    assert res.get("available") is True
    assert len(res["available_events"]) >= 2


def test_execute_get_impact_index():
    """Test get_impact_index tool execution."""
    res = execute_flood_tool("get_impact_index", {"district": "Muzaffarpur"})
    assert res.get("available") is True
    assert res["sat_ai_impact_index"] > 0
    assert res["weights"]["w1_flood_fraction"] == 0.40


def test_execute_get_impact_provenance():
    """Test get_impact_provenance tool execution."""
    res = execute_flood_tool("get_impact_provenance", {"topic": "crop_damage"})
    assert res.get("available") is True
    assert "BFCD-22" in str(res["ground_truth_status"])


# ============================================================================
# 4. Zero-Hallucination & Question Handling
# ============================================================================

def test_unvalidated_future_date_no_hallucination():
    """Ensure system does not fabricate flood numbers for invalid dates."""
    res = execute_flood_tool("get_bihar_flood_status", {"event_date": "2099-01-01"})
    assert res.get("available") is False
    assert "No validated SAT-AI scene is available" in res["reason"]


def test_natural_fallback_hindi_queries():
    """Test natural language Hinglish/Hindi question answering without raw JSON."""
    tool_res = [execute_flood_tool("get_bihar_district_impact", {"event_date": "2022-10-15"})]

    ans1 = _natural_fallback_answer("bihar mein flood kaha zyada hai?", tool_res)
    assert "Muzaffarpur" in ans1 or "Darbhanga" in ans1
    assert "km²" in ans1
    assert "{" not in ans1  # No raw JSON

    ans2 = _natural_fallback_answer("which district is most affected?", tool_res)
    assert "Muzaffarpur" in ans2
    assert "{" not in ans2

    crop_res = [execute_flood_tool("get_crop_damage_summary", {"district": "Muzaffarpur"})]
    ans3 = _natural_fallback_answer("kitni kheti damage hui?", crop_res)
    assert "cropland" in ans3.lower() or "damage" in ans3.lower()
    assert "{" not in ans3

    empty_res = [execute_flood_tool("get_bihar_flood_status", {"event_date": "2099-01-01"})]
    ans4 = _natural_fallback_answer("abhi bihar mein flood kitna hai?", empty_res)
    assert "real-time" in ans4.lower() or "archived" in ans4.lower() or "validated" in ans4.lower()
    assert "{" not in ans4


# ============================================================================
# 5. Crop Damage ML Model Architecture & Weights
# ============================================================================

def test_crop_damage_unet_structure():
    """Test CropDamageUNet initialization and forward pass with dummy tensor."""
    import torch
    from ml.crop_damage.model import CropDamageUNet

    model = CropDamageUNet(in_channels=7, num_classes=3, base_features=32)
    dummy_input = torch.randn(2, 7, 64, 64)
    output = model(dummy_input)

    assert output.shape == (2, 3, 64, 64)


def test_crop_damage_trained_checkpoint_exists():
    """Verify that the trained BFCD-22 weights exist and can be loaded."""
    import torch
    from ml.crop_damage.model import CropDamageUNet

    weights_path = _ROOT / "models" / "crop_damage" / "crop_damage_unet_bfcd22.pt"
    assert weights_path.exists(), "Trained crop damage checkpoint missing"

    ckpt = torch.load(weights_path, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "val_loss" in ckpt

    model = CropDamageUNet(in_channels=7, num_classes=3, base_features=32)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()


# ============================================================================
# 6. Vegetation Change Analysis & Place Profile Tools
# ============================================================================

def test_vegetation_analysis_sentinel2():
    """Verify dual-temporal Sentinel-2 vegetation analysis (NDVI pre vs post)."""
    res = execute_flood_tool("get_vegetation_analysis", {"event_date": "2022-10-15", "district": "Muzaffarpur"})
    assert res["available"] is True
    assert res["event_date"] == "2022-10-15"
    assert res["district"] == "Muzaffarpur"

    metrics = res["vegetation_metrics"]
    assert metrics["pre_event_ndvi_mean"] == 0.64
    assert metrics["post_event_ndvi_mean"] == 0.46
    assert metrics["delta_ndvi"] == -0.18
    assert metrics["relative_change_percentage"] == -28.1

    # Verify mandatory scientific disclaimer
    assert "scientific_disclaimer" in res
    assert "NDVI change does not prove permanent crop destruction" in res["scientific_disclaimer"]

    # Test unavailable date
    unavail = execute_flood_tool("get_vegetation_analysis", {"event_date": "2099-01-01"})
    assert unavail["available"] is False
    assert "DATA UNAVAILABLE" in unavail["status"]


def test_get_place_profile():
    """Verify place profile generation for Bihar state and specific districts."""
    res = execute_flood_tool("get_place_profile", {"place_name": "Bihar"})
    assert res["available"] is True
    assert res["place_name"] == "Bihar"
    assert res["total_area_km2"] == 94163.0

    env = res["environmental_profile"]
    assert env["vegetation"]["forest_cover_km2"] == 7380.79
    assert env["agriculture"]["net_cropped_area_km2"] == 52780.0
    assert "Ganga" in env["water"]["major_rivers"]
    assert "SRTM" in env["terrain"]["source"]

    # Test district profile
    dist_res = execute_flood_tool("get_place_profile", {"place_name": "Muzaffarpur"})
    assert dist_res["available"] is True
    assert dist_res["place_name"] == "Muzaffarpur"
    assert dist_res["total_area_km2"] == 3172.0
    assert dist_res["historical_hazard_class"] == "Very High"

    # Test unknown place
    unknown = execute_flood_tool("get_place_profile", {"place_name": "Atlantis"})
    assert unknown["available"] is False
    assert "DATA UNAVAILABLE" in unknown["status"]


def test_bihar_event_pipeline_execution():
    """Verify the 10-step Bihar Flood Event Pipeline produces valid artifacts."""
    from satai.geo.bihar_impact import run_bihar_event_pipeline
    import json

    res = run_bihar_event_pipeline("2022-10-15")
    assert res["status"] == "success"
    assert res["total_flooded_km2"] == 3482.4
    assert res["crs_validation"]["is_equal_area"] is True

    # Check generated files
    dist_stats_file = Path(res["district_stats_file"])
    crop_exp_file = Path(res["cropland_exposure_file"])
    veg_file = Path(res["vegetation_change_file"])

    assert dist_stats_file.exists()
    assert crop_exp_file.exists()
    assert veg_file.exists()

    data_dist = json.loads(dist_stats_file.read_text(encoding="utf-8"))
    assert len(data_dist) == 38  # all 38 districts ranked


def test_bihar_data_foundation_schema():
    """Verify that the 9 foundation categories are loaded with complete metadata."""
    from satai.geo.bihar_foundation import BIHAR_DATA_FOUNDATION

    cats = BIHAR_DATA_FOUNDATION["categories"]
    required_cats = [
        "administrative_boundaries",
        "land_cover",
        "vegetation",
        "agriculture",
        "water_bodies",
        "terrain",
        "historical_flood_hazard",
        "flood_events",
        "satellite_observations",
    ]
    for cat in required_cats:
        assert cat in cats, f"Missing category {cat}"
        entry = cats[cat]
        assert "source" in entry
        assert "url" in entry
        assert "license" in entry
        assert "resolution" in entry
        assert "crs" in entry
        assert "limitations" in entry
        assert len(entry["limitations"]) > 0


# ============================================================================
# 8. Acceptance Test Queries Validation
# ============================================================================

def test_all_8_acceptance_queries():
    """Verify that all 8 final acceptance queries produce grounded, natural responses."""
    queries = [
        ("What is the situation in Bihar?", ["3482.4", "muzaffarpur", "cropland", "sentinel-1"]),
        ("Which areas are affected by floods?", ["muzaffarpur", "darbhanga", "flooded area", "km²"]),
        ("How much area is flooded?", ["3482.4 km²", "3.7%"]),
        ("What is the vegetation situation?", ["0.64", "0.46", "-0.18", "-28.1%", "permanent crop destruction"]),
        ("How much cropland is affected?", ["2,498.2 km²", "64.2 km²", "muzaffarpur"]),
        ("When does Bihar usually experience floods?", ["july to september", "monsoon", "73.06%"]),
        ("What happened during the selected flood event?", ["3482.4 km²", "16", "sentinel-1"]),
        ("How reliable is this analysis?", ["sentinel-1", "utm zone 45n", "worldcover", "limitations"]),
    ]

    for q, expected_substrings in queries:
        ans = _natural_fallback_answer(q, [])
        assert "{" not in ans, f"Raw JSON found in response to: {q}"
        assert "adr" not in ans.lower(), f"ADR found in response to: {q}"
        assert "get_" not in ans, f"Tool name found in response to: {q}"
        for sub in expected_substrings:
            assert sub.lower() in ans.lower(), f"Expected '{sub}' in answer for: '{q}'\nAnswer was: {ans}"



