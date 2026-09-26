# Data sources

Every dataset SAT-AI uses, what it measures, how old it is when it arrives, and
what it may and may not be used to claim.

The latency column is the one that matters most. A system that mixes a 3-hour
fire detection with a 5-day-old reanalysis field and a 9-day-old SAR scene, and
presents the result as "current conditions", is wrong in a way no accuracy
metric detects.

---

## Satellite imagery

### Sentinel-1 (C-band SAR)

| | |
|---|---|
| Provider | ESA Copernicus |
| Access | CDSE STAC (no credentials needed for search), Google Earth Engine |
| Collection | `COPERNICUS/S1_GRD` |
| Resolution | 10 m (IW GRDH) |
| Revisit | 6–12 days over India |
| Latency | Hours to days for the scene; **6–12 days** for the next look |
| Licence | Copernicus open licence — free, attribution required |
| Source kind | `OBSERVATION` (raw); derived extent is `MODEL` |

**Why it is the backbone.** C-band penetrates cloud. Floods in the Indian
monsoon happen under cloud, so optical flood mapping is substantially a study
of when the sky cleared.

**What it cannot do.** It cannot see flooding beneath dense canopy. In dense
urban fabric, double-bounce between building walls and standing water *raises*
backscatter where the method expects a drop, inverting the signature. Smooth
dry surfaces — tarmac, sand, dry riverbeds — mimic water's low return.

### Sentinel-2 (optical, multispectral)

| | |
|---|---|
| Provider | ESA Copernicus |
| Access | CDSE, Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`) |
| Resolution | 10–60 m by band |
| Revisit | ~5 days, **cloud permitting** |
| Latency | Hours to days, when a usable scene exists at all |
| Licence | Copernicus open licence |
| Source kind | `OBSERVATION`; indices are `DERIVED` |

Used for NDVI, NDWI and NBR/dNBR. During the monsoon the effective revisit is
far worse than 5 days, which is precisely why flood mapping does not depend on
it.

### Copernicus DEM (GLO-30)

| | |
|---|---|
| Provider | ESA |
| Resolution | 30 m |
| Latency | Static |
| Licence | Free for non-commercial and most commercial use |
| Source kind | `OBSERVATION` (a measured surface); slope/HAND are `DERIVED` |

Supplies elevation, slope, drainage and HAND (height above nearest drainage).
HAND is what separates water from tarmac in the flood baseline.

Known limitation: voids and artefacts over water bodies and steep terrain, and
a surface model rather than a terrain model — canopy and buildings are included
in the height.

---

## Meteorology

### GPM IMERG Early

| | |
|---|---|
| Provider | NASA/JAXA |
| Variable | Precipitation |
| Resolution | 0.1° (~11 km) |
| Latency | **~4 hours** |
| Licence | Open |
| Source kind | `OBSERVATION` (satellite-retrieved, gauge-adjusted in later runs) |

The near-real-time rainfall input to the flood risk modulator. The Early run is
uncalibrated relative to Final; the difference is a real error term and is not
hidden.

### ERA5 / ERA5-Land

| | |
|---|---|
| Provider | ECMWF / Copernicus C3S |
| Variables | Temperature, humidity, wind, soil moisture |
| Resolution | 0.25° (ERA5), 0.1° (ERA5-Land) |
| Latency | **~5 days** — retrospective |
| Licence | Copernicus open licence |
| Source kind | `REANALYSIS` |

ERA5 is a model-assimilated reconstruction, not a measurement, and it is five
days behind. An ERA5-driven fire-danger index is a statement about last week,
and the `REANALYSIS` source kind exists so the interface cannot present it as
anything else.

---

## Hazard observations

### NASA FIRMS (VIIRS / MODIS active fire)

| | |
|---|---|
| Provider | NASA |
| Resolution | 375 m (VIIRS), 1 km (MODIS) |
| Latency | **~3 hours** |
| Licence | Open; API key required for high-volume access |
| Source kind | `OBSERVATION` — **never a prediction** |

A FIRMS record is a thermal anomaly detected at satellite overpass time. It is
not a forecast, and the grounding validator treats describing it as one as a
violation.

What it is not: a complete census. Fires under cloud, fires between overpasses,
and fires below the detection threshold are absent. Agricultural residue
burning is indistinguishable from forest fire without land-cover stratification.

### IBTrACS

| | |
|---|---|
| Provider | NOAA NCEI |
| Content | Historical cyclone best-track archive |
| Latency | Historical; near-real-time provisional tracks separately |
| Licence | Open |
| Source kind | `CATALOGUE` |

Used for historical track and landfall statistics and for exposure along an
observed or supplied track. **Not** used for forecasting: track prediction
requires assimilated observations and numerical weather prediction, which IMD
operates.

---

## Exposure and vulnerability

### WorldPop

| | |
|---|---|
| Provider | University of Southampton |
| Content | Gridded population |
| Resolution | 100 m |
| Licence | CC BY 4.0 |
| Source kind | `MODEL` — it is a modelled redistribution, not a census |

### GHSL (Global Human Settlement Layer)

| | |
|---|---|
| Provider | European Commission JRC |
| Content | Built-up surface and density |
| Resolution | 10–100 m |
| Licence | Open |
| Source kind | `MODEL` |

### OpenStreetMap

| | |
|---|---|
| Content | Roads, buildings, water bodies |
| Licence | ODbL — **attribution and share-alike required** |
| Source kind | `CATALOGUE` |

Completeness varies enormously by region. Rural Indian road networks are less
complete than urban ones, which biases any accessibility-based vulnerability
proxy in a direction that disadvantages exactly the populations most exposed.
This is stated in the vulnerability caveat attached to every risk result.

---

## Labels and ground truth

### Sen1Floods11

| | |
|---|---|
| Provider | Cloud to Street / Google |
| Content | **446 hand-labelled** Sentinel-1 flood chips across 11 events |
| Indian chips | **68** |
| Resolution | 10 m, 512×512 |
| Licence | CC BY 4.0 |
| Source kind | `OBSERVATION` (hand-labelled from imagery) |

The binding constraint on this entire project. Not the model architecture, not
the compute — the labels.

Two properties that shape the methodology:

1. **The official splits are not region-disjoint.** Measured: every region
   except Bolivia appears in train, validation and test at roughly 58/21/21.
   See ADR-009.
2. **The labels are drawn from imagery, not surveyed on the ground.** Every
   evaluation in this project is therefore satellite against satellite.

### xBD

| | |
|---|---|
| Provider | DIU / CMU |
| Content | Pre/post building damage, ordinal 4-class |
| Licence | CC BY-NC-SA 4.0 — **non-commercial** |
| Source kind | `OBSERVATION` |

Overwhelmingly non-Indian building stock. The domain gap to Indian construction
is untested and, without Indian damage labels, untestable — so damage transfer
is reported qualitatively and labelled as such.

---

## Latency summary

| Stream | Class | Typical age on arrival |
|---|---|---|
| FIRMS active fire | near-real-time | ~3 h |
| GPM IMERG Early | near-real-time | ~4 h |
| ERA5 | retrospective | ~5 days |
| Sentinel-1 flood extent | revisit-limited batch | 6–12 days |
| Sentinel-2 optical | revisit-limited, cloud-gated | 5 days to never |
| API response | precomputed lookup | < 500 ms target |

Every value SAT-AI returns carries its observation age. Mixing these streams
without doing so would be the system's most consequential error, and it would
not show up in any accuracy metric.

---

## Attribution

Required wherever SAT-AI outputs are reproduced:

- Contains modified Copernicus Sentinel data (2024–2026)
- Copernicus DEM © ESA
- GPM IMERG © NASA/JAXA
- ERA5 © ECMWF / Copernicus Climate Change Service
- FIRMS © NASA
- IBTrACS © NOAA NCEI
- WorldPop © University of Southampton (CC BY 4.0)
- GHSL © European Commission JRC
- © OpenStreetMap contributors (ODbL)
- Sen1Floods11 © Cloud to Street (CC BY 4.0)
- xBD © DIU/CMU (CC BY-NC-SA 4.0)

The xBD non-commercial clause is the binding one for any downstream reuse of
the damage module.
