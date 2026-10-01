"""
tests/test_recent_satellite_intelligence.py
===========================================
Phase 11: Current / Recent Satellite Event Intelligence Test Suite

Validates:
1. Sentinel scene discovery & AOI intersection
2. Acquisition date filtering & cataloging
3. Preprocessing metadata & SNAP RTC steps
4. Projected metric CRS handling (UTM Zone 45N / EPSG:32645)
5. Metric area calculations
6. Distribution gate (PASS on 2024-07-28, FAIL on 2024-09-27)
7. Recent inference with explicit MODEL-INFERRED FLOOD EXTENT labeling
8. Missing ground truth representation (honest nulls)
9. 38 Bihar district aggregation with null building/road exposures
10. Recent vegetation comparison (Sentinel-2 NDVI pre/post)
11. No-data behavior ("No validated recent satellite scene is currently available")
12. Provenance tracking integrity
13. Chatbot grounding via tools
14. Current-vs-archived safety wording disclaiming live telemetry
15. Explicit test cases 1 to 4.
"""

import sys
from pathlib import Path
import pytest

# Ensure repo root and backend are in path
_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "backend" / "api"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from satai.geo.sentinel_discovery import (
    discover_sentinel1_scenes,
    load_satellite_catalog,
    BIHAR_AOI_BBOX,
    BIHAR_AOI_GEOJSON,
)
from satai.geo.recent_scene_engine import (
    check_distribution_gate,
    run_recent_flood_inference,
    compute_recent_district_impact,
    analyze_recent_vegetation_change,
    build_current_event_object,
)
from backend.api.flood_tools import execute_flood_tool, FLOOD_TOOL_DECLARATIONS
from backend.api.satai_agents import _natural_fallback_answer


# ============================================================================
# 1. Sentinel Scene Discovery & AOI Intersection
# ============================================================================

def test_scene_discovery_aoi():
    """Verify scene discovery detects scenes intersecting Bihar AOI."""
    scenes = discover_sentinel1_scenes(
        aoi_bbox=BIHAR_AOI_BBOX,
        start_date="2024-01-01",
        end_date="2024-10-01",
        force_refresh=False,
    )
    assert len(scenes) >= 2
    for sc in scenes:
        assert "footprint" in sc
        assert sc["mode"] == "IW"
        assert sc["product_type"] == "GRD"
        assert "VV" in sc["polarization"] and "VH" in sc["polarization"]


def test_scene_discovery_date_filtering():
    """Verify scene discovery filters strictly by requested acquisition date range."""
    # Filter tightly to July 2024
    july_scenes = discover_sentinel1_scenes(
        aoi_bbox=BIHAR_AOI_BBOX,
        start_date="2024-07-01",
        end_date="2024-07-31",
    )
    assert len(july_scenes) >= 1
    for sc in july_scenes:
        assert sc["acquisition_datetime"].startswith("2024-07")

    # Filter out-of-range should return empty
    empty_scenes = discover_sentinel1_scenes(
        aoi_bbox=BIHAR_AOI_BBOX,
        start_date="2023-01-01",
        end_date="2023-01-02",
    )
    assert len(empty_scenes) == 0


def test_catalog_caching():
    """Verify satellite catalog cache file exists and contains valid JSON."""
    catalog = load_satellite_catalog()
    assert "provider" in catalog
    assert "scenes" in catalog
    assert len(catalog["scenes"]) >= 2


# ============================================================================
# 2. Preprocessing Metadata & Projected CRS Handling
# ============================================================================

def test_preprocessing_metadata():
    """Verify preprocessing metadata records calibration, thermal noise, and terrain correction."""
    res = execute_flood_tool("get_recent_event_provenance", {"scene_id": "20240728"})
    assert res["available"] is True
    assert "RTC" in res["processing_level"] or "Range-Doppler Terrain Corrected" in res["processing_level"]
    assert "Sentinel-1A" in res["platform"]
    assert res["instrument"] == "C-SAR (5.405 GHz)"


def test_crs_terminology_not_equal_area():
    """Verify EPSG:32645 is described as a projected metric CRS and NOT an equal-area CRS."""
    event = build_current_event_object("20240728")
    crs_desc = event["provenance"]["projection"]
    assert "projected" in crs_desc.lower()
    assert "equal-area" not in crs_desc.lower()


# ============================================================================
# 3. Distribution Gate (PASS vs FAIL)
# ============================================================================

def test_distribution_gate_pass_scene():
    """Verify 2024-07-28 scene passes distribution gate."""
    gate = check_distribution_gate("20240728")
    assert gate["status"] == "PASS"
    assert gate["vv_wasserstein_db"] < 2.5
    assert gate["valid_pixel_fraction"] >= 0.90


def test_distribution_gate_fail_scene():
    """Verify 2024-09-27 scene fails distribution gate and inference is withheld."""
    gate = check_distribution_gate("20240927")
    assert gate["status"] == "FAIL"
    assert gate["vv_wasserstein_db"] >= 3.5

    # Verify run_recent_flood_inference withholds inference on FAIL
    inf = run_recent_flood_inference("20240927")
    assert inf["inference_status"] == "WITHHELD"
    assert "significant distribution mismatch" in inf["message"]
    assert inf["available"] is False


# ============================================================================
# 4. Recent Flood Inference & Area Calculation
# ============================================================================

def test_recent_flood_inference_labeling():
    """Verify compatible scene produces model inference explicitly labeled MODEL-INFERRED FLOOD EXTENT."""
    inf = run_recent_flood_inference("20240728")
    assert inf["available"] is True
    assert inf["inference_label"] == "MODEL-INFERRED FLOOD EXTENT"
    assert inf["resolution_m"] == 10.0
    assert inf["flooded_area_km2"] == 2841.5
    assert inf["flooded_percentage"] == 3.02


def test_missing_ground_truth_honesty():
    """Verify ground truth is explicitly declared unavailable for recent inference."""
    inf = run_recent_flood_inference("20240728")
    gt = inf["ground_truth"]
    assert gt["available"] is False
    assert gt["source"] is None


# ============================================================================
# 5. District Aggregation (All 38 Districts & Null Handling)
# ============================================================================

def test_district_aggregation_38_districts():
    """Verify all 38 Bihar districts are accounted for in recent impact calculation."""
    dist_impact = compute_recent_district_impact("20240728")
    assert dist_impact["available"] is True
    assert dist_impact["total_districts_evaluated"] == 38
    districts = dist_impact["district_ranking"]
    assert len(districts) == 38

    # Top impacted district should have non-zero flood area
    top_d = districts[0]
    assert top_d["flooded_area_km2"] > 0
    assert top_d["flood_percentage"] > 0

    # Buildings and roads must be null (unmeasured for recent scene, never estimated)
    for d in districts:
        assert d["buildings_exposed"] is None
        assert d["roads_exposed_km"] is None


# ============================================================================
# 6. Recent Vegetation Analysis (Sentinel-2 NDVI Change)
# ============================================================================

def test_recent_vegetation_analysis():
    """Verify dual-temporal Sentinel-2 NDVI change calculation and scientific notice."""
    veg = analyze_recent_vegetation_change("20240728")
    assert veg["status"] == "SUCCESS"
    assert veg["ndvi_pre"] == 0.62
    assert veg["ndvi_post"] == 0.49
    assert veg["delta_ndvi"] == -0.13
    assert veg["relative_change_percentage"] == -21.0
    # Scientific wording check
    assert "Observed vegetation-index change" in veg["scientific_notice"]
    assert "does not prove permanent crop destruction" in veg["scientific_notice"]


# ============================================================================
# 7. Unified Phase 11I Event Schema
# ============================================================================

def test_current_event_object_schema():
    """Verify unified event object matches Phase 11I schema."""
    event = build_current_event_object("20240728")
    required_keys = [
        "event_id",
        "region",
        "hazard",
        "event_type",
        "acquisition_datetime",
        "satellite",
        "sensor",
        "resolution_m",
        "analysis_status",
        "flood_extent",
        "vegetation",
        "cropland",
        "distribution_gate",
        "ground_truth",
        "provenance",
        "limitations",
    ]
    for k in required_keys:
        assert k in event
    assert event["event_type"] == "recent_satellite_observation"
    assert len(event["limitations"]) >= 3


# ============================================================================
# 8. Backend Tools Dispatcher (Phase 11J)
# ============================================================================

def test_all_8_recent_tools_registered():
    """Verify all 8 Phase 11 backend tools are present in FLOOD_TOOL_DECLARATIONS."""
    tool_names = [t["name"] for t in FLOOD_TOOL_DECLARATIONS]
    expected_tools = [
        "get_recent_satellite_scenes",
        "get_recent_satellite_scene",
        "get_recent_flood_inference",
        "get_recent_flood_area",
        "get_recent_district_impact",
        "get_recent_vegetation_change",
        "get_recent_event_summary",
        "get_recent_event_provenance",
    ]
    for et in expected_tools:
        assert et in tool_names, f"Missing tool: {et}"


def test_execute_recent_tools_dispatcher():
    """Verify execution of Phase 11 tools returns structured dictionaries."""
    # 1. get_recent_satellite_scenes
    res1 = execute_flood_tool("get_recent_satellite_scenes", {"region": "Bihar"})
    assert "scenes" in res1
    assert res1["scenes_count"] >= 2

    # 2. get_recent_satellite_scene
    res2 = execute_flood_tool("get_recent_satellite_scene", {"scene_id": "20240728"})
    assert res2["available"] is True
    assert "20240728" in res2["scene"]["scene_id"]

    # 3. get_recent_flood_inference
    res3 = execute_flood_tool("get_recent_flood_inference", {"scene_id": "20240728"})
    assert res3["inference_label"] == "MODEL-INFERRED FLOOD EXTENT"

    # 4. get_recent_flood_area
    res4 = execute_flood_tool("get_recent_flood_area", {"scene_id": "20240728"})
    assert res4["flooded_area_km2"] == 2841.5

    # 5. get_recent_district_impact
    res5 = execute_flood_tool("get_recent_district_impact", {"scene_id": "20240728"})
    assert res5["total_districts_evaluated"] == 38

    # 6. get_recent_vegetation_change
    res6 = execute_flood_tool("get_recent_vegetation_change", {"scene_id": "20240728"})
    assert "delta_ndvi" in res6

    # 7. get_recent_event_summary
    res7 = execute_flood_tool("get_recent_event_summary", {"scene_id": "20240728"})
    assert res7["unified_event"]["region"] == "Bihar"

    # 8. get_recent_event_provenance
    res8 = execute_flood_tool("get_recent_event_provenance", {"scene_id": "20240728"})
    assert "source" in res8


# ============================================================================
# 9. Explicit Cases 1 to 4 & Chatbot Behavior
# ============================================================================

def test_case_1_no_recent_scene_available():
    """CASE 1: No recent scene available -> System reports no validated scene."""
    ans = _natural_fallback_answer("Is there a recent satellite image for Rajasthan?")
    assert "No validated recent satellite scene is currently available" in ans


def test_case_2_distribution_mismatch_withheld():
    """CASE 2: Recent scene available but distribution mismatch -> Inference withheld or flagged."""
    res = execute_flood_tool("get_recent_flood_inference", {"scene_id": "20240927"})
    assert res["distribution_gate"]["status"] == "FAIL"
    assert "significant distribution mismatch" in res["message"]
    assert res["inference_status"] == "WITHHELD"


def test_case_3_recent_scene_compatible_model_inferred():
    """CASE 3: Recent scene available and compatible -> Model inference with explicit MODEL-INFERRED label."""
    res = execute_flood_tool("get_recent_flood_inference", {"scene_id": "20240728"})
    assert res["distribution_gate"]["status"] == "PASS"
    assert res["inference_label"] == "MODEL-INFERRED FLOOD EXTENT"
    assert res["flooded_area_km2"] == 2841.5


def test_case_4_user_asks_today_or_right_now():
    """CASE 4: User asks 'today' or 'right now' -> Gives date and disclaims continuous real-time monitoring."""
    queries = [
        "What's happening in Bihar right now?",
        "Is this a live satellite observation?",
        "What is happening today in Bihar?",
    ]
    for q in queries:
        ans = _natural_fallback_answer(q)
        assert "SAT-AI does not provide continuous real-time flood monitoring" in ans
        assert "not live telemetry" in ans.lower() or "not be interpreted as the current situation today" in ans


def test_chatbot_recent_districts_query():
    """Verify chatbot answers which districts are affected in the latest scene without exposing raw JSON."""
    ans = _natural_fallback_answer("Which districts are affected in the latest scene?")
    assert "Muzaffarpur" in ans or "Darbhanga" in ans
    assert "2024-07-28" in ans
    assert "{" not in ans
    assert "}" not in ans
    assert "def " not in ans
