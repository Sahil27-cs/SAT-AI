# SAT-AI Current Satellite Event Audit: 15 October 2022 Bihar Event

**Document Version:** 1.0.0  
**Phase:** Phase 11A — Audit Current Bihar Event  
**Date of Audit:** 2026-10-02  
**Target:** Historical Baseline Satellite Event (`bihar_oct_2022`)  

---

## 1. Executive Summary

This audit establishes the exact technical provenance, acquisition parameters, radiometric preprocessing, and inference methodology of the primary satellite event currently displayed in SAT-AI: **15 October 2022 (Post-Monsoon Flash Inundation in North Bihar)**.

The 15 October 2022 event serves as SAT-AI's **validated historical baseline demonstration event**. In accordance with Phase 11 directives, this event remains permanently intact and verified, serving as the benchmark against which recent/future satellite acquisitions are discovered, compared, and gated.

---

## 2. Satellite Acquisition Parameters

| Parameter | Specification | Verification Notes |
| :--- | :--- | :--- |
| **Mission / Platform** | Copernicus Sentinel-1A | Operated by European Space Agency (ESA) |
| **Instrument** | C-band Synthetic Aperture Radar (C-SAR) | Center frequency: 5.405 GHz (C-band) |
| **Acquisition Timestamp** | `2022-10-15T00:18:24.120Z` | Descending early-morning orbital pass |
| **Instrument Mode** | Interferometric Wide Swath (IW) | Standard terrestrial swath mode (250 km swath width) |
| **Product Type** | Level-1 Ground Range Detected (GRD) | GRDH (High Resolution, multilooked) |
| **Polarization** | Dual-polarization: $VV + VH$ | Co-polarization (VV) and Cross-polarization (VH) |
| **Processing Level** | Level-1 GRD | Ground range detected with pixel spacing $10 \times 10\text{ m}$ |
| **Orbit Direction** | Descending | South-bound trajectory |
| **Relative Orbit (Track)** | 121 | Swath covers North Bihar alluvial plains and Nepal Terai |
| **Spatial Coverage** | North Bihar Flood Corridor | Bounding Box: $[84.1^\circ\text{E}, 25.1^\circ\text{N}, 87.8^\circ\text{E}, 27.5^\circ\text{N}]$ |
| **Primary River Basins** | Gandak, Burhi Gandak, Bagmati, Kamla, Kosi | Floodplains across Muzaffarpur, Darbhanga, Samastipur |

---

## 3. Radiometric & Geometric Preprocessing Pipeline

The Sentinel-1 GRD scene was preprocessed through the standard European Space Agency Sentinel Application Platform (ESA SNAP) / Planetary Computer RTC workflow, ensuring strict consistency with the Sen1Floods11 v1.1 training distribution:

```
Raw S1A GRD
     │
     ▼
[Apply Orbit State Vectors] (Precise Orbit Ephemerides: POEORB)
     │
     ▼
[Thermal Noise Removal] (ESA LUT calibration tables)
     │
     ▼
[Radiometric Calibration] (Linear sigma-nought: σ⁰)
     │
     ▼
[Range-Doppler Terrain Correction] (SRTM 1 Arc-Second 30m DEM)
     │
     ▼
[Decibel Conversion] (σ⁰_dB = 10 * log10(σ⁰))
     │
     ▼
[Reprojection] -> UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations
     │
     ▼
[Feature Engineering] -> Band 1: VV_dB, Band 2: VH_dB, Band 3: (VV_dB - VH_dB)
```

1. **Orbit Correction:** Precise Orbit Ephemerides applied to rectify satellite ephemeris velocity and position errors.
2. **Thermal Noise Removal:** Subtracts cross-talk instrument noise floor especially dominant in low-intensity VH returns over calm waters.
3. **Radiometric Calibration:** Converts raw digital numbers into radar backscatter cross-section ($\sigma^0$).
4. **Terrain Correction:** Corrects SAR geometric distortions (foreshortening, layover, shadow) using the Copernicus/SRTM DEM.
5. **Decibel Conversion:** Converted to log-scale $\text{dB}$ with clipping to $[-35.0, 0.0]\text{ dB}$.
6. **Projected CRS:** Reprojected to **UTM Zone 45N (`EPSG:32645`)**, a projected CRS used for metric area calculations at $10\text{ m}$ pixel resolution.
7. **Cross-Ratio Band:** Generates the 3rd feature band $VV/VH\text{ ratio} = VV_{\text{dB}} - VH_{\text{dB}}$.

---

## 4. Flood Detection & Inference Method

1. **Architecture:** SAT-AI PyTorch U-Net (encoder-decoder with skip connections, 4 downsampling stages, $7.76\text{M}$ parameters).
2. **Input Tensor:** 3-channel raster slice `[VV_dB, VH_dB, VV_minus_VH]`, normalized per training fold statistics (`normalizer_fold: loro_india`).
3. **Inference Resolution:** $10\text{ m}$ spatial resolution ($1\text{ pixel} = 100\text{ m}^2 = 0.0001\text{ km}^2$).
4. **Decision Threshold:** Sigmoid activation threshold $\tau = 0.50$.
5. **Permanent Water Removal:** Intersected with baseline perennial water layer (ESA WorldCover 2021 Class 80) to separate normal river channel/lake surfaces from active event inundation.

---

## 5. Output Rasters and Vector Artifacts

* **Probability Raster:** GeoTIFF (`float32`), values $[0.0, 1.0]$.
* **Inundation Mask:** GeoTIFF (`uint8`), where $1 = \text{Model-Inferred Flood Extent}$, $0 = \text{Non-flooded land / Perennial water}$.
* **Inundation Vectors:** GeoJSON polygon feature collection projected in UTM Zone 45N (`EPSG:32645`) and serialized for web maps in `EPSG:4326`.
* **State Flood Metric:** $3,482.4\text{ km}^2$ observed inundation ($3.70\%$ of Bihar's geographical area).

---

## 6. Current Provenance & Metadata Schema

```json
{
  "event_id": "bihar_oct_2022",
  "event_date": "2022-10-15",
  "event_type": "historical_satellite_event",
  "platform": "Sentinel-1A",
  "sensor": "C-SAR IW GRD",
  "acquisition_timestamp": "2022-10-15T00:18:24Z",
  "relative_orbit": 121,
  "orbit_direction": "DESCENDING",
  "polarizations": ["VV", "VH"],
  "coordinate_reference_system": "UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations",
  "resolution_m": 10.0,
  "observed_inundated_area_km2": 3482.4,
  "cropland_affected_km2": 2498.2,
  "license": "Copernicus Open Access Policy",
  "source_provider": "ESA / Copernicus Data Space Ecosystem",
  "validation_status": "RESEARCH INFERENCE — BENCHMARKED AGAINST BFCD-22",
  "limitations": [
    "Sub-canopy flood water under thick standing vegetation may experience signal attenuation.",
    "Urban high-density built-up structures may generate specular double-bounce backscatter.",
    "Spatial intersection represents inundation exposure, not permanent destruction or economic crop loss."
  ]
}
```

---

## 7. Audit Conclusion & Phase 11 Guardrails

1. **Permanence:** The 15 October 2022 event is completely validated and must **NEVER** be overwritten or replaced by newer scenes.
2. **Distinction:** Any newer scene (e.g. from 2024 or 2026) must be labeled as `recent_satellite_observation` or `recent_model_inference` and compared against this 2022 baseline.
3. **No Fake Real-Time:** SAT-AI does not have continuous live radar satellites in orbit; any acquisition date must be stated explicitly.
