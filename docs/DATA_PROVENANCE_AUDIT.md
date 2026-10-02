# SAT-AI Data Provenance & Numerical Audit

**Audit Date:** 2026-10-01  
**Auditor:** SAT-AI Systems & Geospatial Engineering  
**Scope:** All numerical values displayed in Hero, Place Profiles, Environmental Cards, Disaster Timelines, Satellite Observations, Change Detection, Cropland Exposure, and Chatbot responses.  
**Standard:** Strict Zero-Hallucination & Scientific Honesty Policy.

---

## 1. Provenance Classification Schema

| Status | Definition | Action in SAT-AI |
| :--- | :--- | :--- |
| **`VERIFIED`** | Direct measurement or calculation from an authentic, authoritative dataset (e.g. Census 2011, ISFR 2021, SRTM DEM). | Display with explicit source citation. |
| **`MODEL_INFERENCE`** | Deterministic output of a trained, validated machine learning model (e.g. SAT-AI U-Net on Sentinel-1 SAR or BFCD-22 Crop Damage U-Net). | Display with model label, sensor, acquisition date, and resolution in Evidence drawer. |
| **`REFERENCE`** | Peer-reviewed or official government archive statistics (e.g. NRSC 22-year flood hazard atlas 1998–2019). | Display clearly labeled as Historical Reference Data. |
| **`PLACEHOLDER`** | Synthetic, illustrative, or uncalculated number entered during UI scaffolding. | **MUST BE REMOVED IMMEDIATELY.** Replaced with `"Data not currently available"` or `"Analysis available after selecting a satellite event"`. |
| **`UNVERIFIED`** | Number originating from an uncalibrated third-party source or lacking provenance metadata. | Withheld until verified against primary raster/vector datasets. |

---

## 2. Comprehensive Numerical Audit Table

| UI Component | Field | Current / Scaffolding Value | Verified Source / Dataset | Calculation / Derivation Method | Acquisition / Record Date | Audit Status | Resolution / Action Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hero Card** | Observed Inundation | `3,842 km²` | Sentinel-1 SAR IW GRD | Typo of 3,482.4 km² event; placeholder in static card | 2022-10-15 | **`PLACEHOLDER`** | **REMOVED from static hero.** Replaced with: *"Analysis available after selecting a satellite event"*. |
| **Hero Card** | Vegetation Change | `-18.4% ΔNDVI` | None (arbitrary demo value) | Synthetic guess | N/A | **`PLACEHOLDER`** | **REMOVED from static hero.** Replaced with: *"Data not currently available (requires dual-temporal optical scene)"*. |
| **Hero Card** | Affected Cropland | `2,140 km²` | None (arbitrary demo value) | Synthetic guess | N/A | **`PLACEHOLDER`** | **REMOVED from static hero.** Replaced with: *"Analysis available after selecting a satellite event"*. |
| **Hero Card** | Analysis Date | `27 September 2026` | Future placeholder date | Hardcoded string | N/A | **`PLACEHOLDER`** | Replaced with verified event date: `2022-10-15` (Sentinel-1 C-SAR). |
| **Place Profile (Bihar)** | Total Area | `94,163 km²` | Census of India 2011 / Survey of India | Administrative boundary GIS polygon sum | 2011 | **`VERIFIED`** | Retained with authoritative source badge. |
| **Place Profile (Bihar)** | Coordinates | `25.37°N, 85.13°E` | Survey of India / WGS84 Centroid | Geographic centroid calculation | 2026 | **`VERIFIED`** | Retained. |
| **Environment: Vegetation** | Forest Cover Area | `7,381 km²` | Forest Survey of India (FSI) | State of Forest Report (ISFR 2021) wall-to-wall mapping | 2021 | **`VERIFIED`** | Retained with FSI citation. |
| **Environment: Vegetation** | Forest Cover % | `7.84%` | ISFR 2021 / FSI | `7380.79 / 94163.0 * 100` | 2021 | **`VERIFIED`** | Retained. |
| **Environment: Vegetation** | NDVI Range | `0.22 to 0.71` | Sentinel-2 L2A BOA Reflectance | Min/Max NDVI across dry season (April) to post-monsoon (October) | 2021–2022 | **`VERIFIED`** | Retained with sensor citation. |
| **Environment: Agriculture** | Net Cropped Area | `52,780 km²` | Directorate of Economics & Statistics (DES), Govt. of Bihar | Annual Agricultural Survey Records (2020-21) | 2021 | **`VERIFIED`** | Retained with DES citation. |
| **Environment: Agriculture** | Cropped Area % | `56.05%` | DES Bihar / MoA&FW | `52780.0 / 94163.0 * 100` | 2021 | **`VERIFIED`** | Retained. |
| **Environment: Water** | Surface Water Area | `3,520 km²` | NRSC / ISRO Bhuvan 1:50k LULC | Permanent river channel and perennial water body GIS vector area | 2015–2016 | **`REFERENCE`** | Retained with NRSC citation. |
| **Environment: Terrain** | Elevation Range | `35m to 880m` | USGS / NASA SRTM 30m DEM & ASTER GDEM | Digital Elevation Model zonal min/max (Someshwar Hills to Farakka confluence) | 2000–2020 | **`VERIFIED`** | Retained with SRTM citation. |
| **Environment: Built** | Arterial Roads | `14,820 km` | Ministry of Road Transport and Highways (MoRTH) | NH + SH official linear network inventory (2022) | 2022 | **`VERIFIED`** | Retained. |
| **Disaster Profile** | Flood Rating | `Very High (73.6% area)` | NRSC 22-Year Flood Hazard Atlas (1998–2019) | Frequency of inundation across 22 monsoon flood seasons | 1998–2019 | **`REFERENCE`** | Retained with NRSC 22-year atlas citation. |
| **Disaster Profile** | Earthquake Rating | `High (Zone IV & V)` | Bureau of Indian Standards (IS 1893:2002) | National Seismic Hazard Zonation Map | 2002 | **`REFERENCE`** | Retained with BIS citation. |
| **Disaster Profile** | Wildfire Rating | `Low` | NASA FIRMS / FSI Fire Alerts | Thermal anomalyMODIS/VIIRS counts 2018–2023 | 2023 | **`REFERENCE`** | Retained. |
| **Timeline (1998 Event)** | Inundated Area | `5,240 km²` | NRSC Flood Hazard Report (1998) | IRS-1C WiFS + Radarsat manual thresholding | 1998-08-14 | **`REFERENCE`** | Retained as historical observation. |
| **Timeline (1998 Event)** | ΔNDVI | `-32.5%` | None | Synthetic demo placeholder | 1998 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (1998 Event)** | Buildings Exposed | `480,000` | None | Synthetic demo placeholder | 1998 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (1998 Event)** | Roads Exposed | `2,650 km` | None | Synthetic demo placeholder | 1998 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (2004 Event)** | Inundated Area | `6,180 km²` | NRSC Remote Sensing Assessment | IRS-P6 AWiFS multi-temporal flood mapping | 2004-07-22 | **`REFERENCE`** | Retained as historical observation. |
| **Timeline (2004 Event)** | ΔNDVI | `-38.2%` | None | Synthetic demo placeholder | 2004 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (2004 Event)** | Buildings Exposed | `580,000` | None | Synthetic demo placeholder | 2004 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (2004 Event)** | Roads Exposed | `3,120 km` | None | Synthetic demo placeholder | 2004 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (2017 Event)** | Inundated Area | `4,920 km²` | Copernicus Sentinel-1 / NRSC | Thresholded SAR backscatter in EPSG:32645 | 2017-08-18 | **`REFERENCE`** | Retained. |
| **Timeline (2017 Event)** | ΔNDVI | `-29.0%` | None | Synthetic demo placeholder | 2017 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (2017 Event)** | Buildings Exposed | `460,000` | None | Synthetic demo placeholder | 2017 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (2017 Event)** | Roads Exposed | `2,450 km` | None | Synthetic demo placeholder | 2017 | **`PLACEHOLDER`** | **REMOVED.** Set to `null` -> `"Data not currently available"`. |
| **Timeline (2021 Event)** | Inundated Area | `4,812.0 km²` | Sentinel-1 C-SAR IW GRD | U-Net SAR segmentation projected to UTM 45N | 2021-08-28 | **`MODEL_INFERENCE`** | Retained with model provenance. |
| **Timeline (2021 Event)** | Cropland Affected | `3,486.0 km²` | Sentinel-1 + DES Cropland Layer | Binary intersection of detected water with cropland mask | 2021-08-28 | **`MODEL_INFERENCE`** | Retained. |
| **Timeline (2021 Event)** | Building Footprints | `445,200` | OpenStreetMap / Microsoft Footprints | Structural intersection count | 2021 | **`MODEL_INFERENCE`** | Retained with footprint caveat. |
| **Timeline (2021 Event)** | Roads Exposed | `2,410.5 km` | OSM Highway / MoRTH Vector | Linear intersection length in EPSG:32645 | 2021 | **`MODEL_INFERENCE`** | Retained with linear overlay caveat. |
| **Timeline (2022 Event)** | Inundated Area | `3,482.4 km²` | Sentinel-1 C-SAR IW GRD | U-Net SAR water segmentation (EPSG:32645) | 2022-10-15 | **`MODEL_INFERENCE`** | Retained with model provenance. |
| **Timeline (2022 Event)** | Cropland Affected | `2,498.2 km²` | BFCD-22 Ground Truth / DES Mask | Dual-temporal SAR + Optical damage model inference | 2022-10-15 | **`MODEL_INFERENCE`** | Retained. |
| **Timeline (2022 Event)** | ΔNDVI | `-28.1%` | Sentinel-2 MSI L2A | `(NDVI_post - NDVI_pre) / NDVI_pre * 100` on cloud-free pixels | 2022-10-15 | **`MODEL_INFERENCE`** | Retained with spectral disclaimer. |
| **District Breakdown** | 38 District Areas | Various | Census 2011 / DES Bihar | Official district gazetteer areas (e.g. Muzaffarpur 3,172 km², Patna 3,202 km²) | 2011 | **`VERIFIED`** | Retained. |
| **District Breakdown** | Muzaffarpur Flood | `412.6 km² (13.01%)` | Sentinel-1 SAR 2022-10-15 | District polygon clip & metric pixel area count | 2022-10-15 | **`MODEL_INFERENCE`** | Retained. |
| **District Breakdown** | Darbhanga Flood | `384.2 km² (16.86%)` | Sentinel-1 SAR 2022-10-15 | District polygon clip & metric pixel area count | 2022-10-15 | **`MODEL_INFERENCE`** | Retained. |
| **District Breakdown** | Samastipur Flood | `341.8 km² (11.77%)` | Sentinel-1 SAR 2022-10-15 | District polygon clip & metric pixel area count | 2022-10-15 | **`MODEL_INFERENCE`** | Retained. |

---

## 3. Mandatory Remediation Summary

1. **Hero Satellite Insight Card (`frontend/app/page.tsx`)**:
   - `3,842 km²` removed. Replaced with dynamic reference to the active event (`3,482.4 km²`) or `"Select an analyzed event"`.
   - `-18.4% ΔNDVI` removed. Replaced with `"-28.1% (Oct 2022 Event)"` or `"Analysis available after selecting a satellite event"`.
   - `2,140 km²` cropland removed. Replaced with `"2,498.2 km² (Oct 2022 Event)"`.
   - Analysis date updated from placeholder `"27 September 2026"` to `"15 October 2022"`.

2. **Historical Events 1998, 2004, 2010, 2017 (`frontend/lib/places-data.ts`)**:
   - Fabricated `vegetationChangePct`, `buildingsExposed`, and `roadsExposedKm` are set to `null`.
   - The UI displays: `"Data not currently available"` for these historical events, acknowledging sensor and data registry limitations honestly.

3. **Zero-Hallucination Integrity in Chatbot & GIS Tools (`satai/geo/bihar_impact.py`)**:
   - Uncomputed future dates or regions return an explicit `status: "UNAVAILABLE"` with explanatory caveats.
   - LLM safety validator strictly forbids fabricating numerical damages or official authority claims.
