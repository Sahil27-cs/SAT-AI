# SAT-AI

**An AI-driven multi-hazard risk assessment and early-warning research prototype
using satellite remote sensing and intelligent conversational agents.**

[![CI](https://github.com/Sahil27-cs/SAT-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Sahil27-cs/SAT-AI/actions/workflows/ci.yml)

**Live:** [sat-ai-sahil.vercel.app](https://sat-ai-sahil.vercel.app) ·
[API](https://sat-ai-api-sahil.vercel.app/docs) ·
[health](https://sat-ai-api-sahil.vercel.app/health)

The deployed dashboard shows **no hazard results**, because none have been
computed. Every hazard panel says so and states what would produce them. That
is the interface working as designed: for a system whose contribution is about
grounding, a placeholder number would undo the argument.

---

## What this is, and what it is not

SAT-AI combines Sentinel-1 SAR, Sentinel-2 optical, terrain, rainfall and
weather data to assess hazard risk, detect affected areas, assess post-event
damage, and explain its outputs in natural language through tool-using agents.

**It is a student research prototype.** Its risk levels are research outputs,
not official warnings. Official warnings for India come from IMD, NDMA and State
Disaster Management Authorities.

### Scope, stated precisely

| Hazard | What SAT-AI does | What it explicitly does not do |
|---|---|---|
| **Flood** | susceptibility mapping, extent detection from SAR, affected-area statistics, rainfall-modulated risk | forecast flood timing or depth |
| **Wildfire** | fire-danger estimation, ingestion of active-fire *observations*, burned-area and severity mapping | predict where a fire will ignite |
| **Cyclone** | extreme wind and rainfall risk, historical track analysis, exposure assessment along a track | forecast cyclone tracks or intensity |
| **Earthquake** | post-event building damage assessment from pre/post imagery | **predict earthquakes, in any form** |

The distinction between *prediction*, *detection*, *monitoring*, *risk
assessment* and *post-event assessment* is enforced in the code, not only in the
prose: `satai.provenance.SourceKind` separates an instrument observation from a
model output from a configurable index, and the API and UI surface that
distinction to the user.

### It is not real-time

Latency is measured per stream and reported honestly:

| Stream | Class | Approximate |
|---|---|---|
| FIRMS active fire | near-real-time | ~3 h |
| GPM IMERG Early rainfall | near-real-time | ~4 h |
| ERA5 weather | retrospective | ~5 days |
| Sentinel-1 flood extent | revisit-limited batch | 6–12 days |
| API response | precomputed lookup | < 500 ms target |

System-level: **near-real-time for weather-driven risk indices; revisit-limited
batch for satellite-derived extent and damage.**

---

## Architecture

Four planes, separated by latency class (ADR-001). The separation is what makes
the API fast enough to deploy and the results reproducible enough to report.

```
DATA PLANE     GEE + CDSE + FIRMS + GPM + ERA5   →  COG / Parquet / manifests
  (hours-days, offline)
       ↓
ML PLANE       training (Colab/Kaggle GPU) + batch inference (CPU, ONNX)
  (minutes-hours, batch)                       →  hazard rasters + XAI JSON
       ↓
SERVING PLANE  FastAPI + PostGIS + object storage
  (sub-second, online)                         →  reads precomputed artifacts
       ↓
AGENT PLANE    keyword router → 3 agents → typed tools → grounding validator
  (seconds, online)                            →  Google Gemini
       ↓
FRONTEND       Next.js + MapLibre GL on Vercel
```

**The provenance contract.** Every value crossing a plane boundary is wrapped in
a `ProvenanceEnvelope` carrying its source model and version, the satellite
scenes behind it, its native resolution, its acquisition time, its confidence,
and its caveats. The language model may rephrase an envelope; it may not author
one. A validator re-extracts every number from generated text and rejects
anything untraceable — and the violation rate is reported as an evaluation
metric rather than assumed to be zero. See ADR-003.

**One validator, two copies, checked against each other.** The serving plane
runs a dependency-light mirror of the validator so the serverless bundle stays
inside its size budget, and that mirror had drifted. It checked only numeric
traceability, while the instrument scored in experiment 9a also checked
fabricated authority, observation/prediction confusion and dropped caveats — so
the rate reported for C1 was measured by a stricter validator than the one
standing in front of users. All four checks now exist in both, and
[`tests/test_grounding_parity.py`](tests/test_grounding_parity.py) runs a shared
corpus through each implementation and fails if their verdicts diverge.

---

## Status

**Built, trained and deployed.** 661 tests green, `mypy --strict` clean over
`satai/`, `backend/` **and** `ml/`, `ruff check` and `ruff format` clean — all
four run in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)),
alongside the frontend type-check and build and a `gitleaks` scan of the full
history.

| | |
|---|---|
| ✅ Phase 0 | Gap analysis against the 7-paper corpus; contributions revised to C1–C4 |
| ✅ Phase 1 | Repository, environment, config layer, provenance contract, CI, ADRs |
| ✅ Phase 2 | Provider layer (CDSE / Earth Engine / FIRMS), manifests, AOI measurement |
| ✅ Phase 3a | Preprocessing **Track A** — LORO splits, band math, leakage-safe normalisation |
| ✅ Phase 4 | Flood baselines: Otsu with a unimodality guard, per-pixel ensemble — **scored** |
| ✅ Phase 5 | Segmentation metrics, Track A/B distribution gate |
| ✅ Phase 6 | **Deep flood model trained and evaluated** — `ml/flood/`, U-Net, region-disjoint |
| ✅ Phase 7 | Inference artifacts (COG + GeoJSON + provenance), per-band explainability, ONNX export verified against the checkpoint |
| ✅ Phase 8 | Risk engine + sensitivity analysis — **C4 executed**; the flood model is wired into it |
| ✅ Phase 9 | FastAPI serving plane, deployed and verified |
| ✅ Phase 10 | Agent plane on **Google Gemini** with native function calling and map control |
| ✅ Phase 11 | Next.js + MapLibre frontend, deployed and verified |
| ◐ | **C2 partially executed** (2 of 5 arms); **C1 and C3 blocked** — see below |

### The flood model, measured

Trained on Sen1Floods11 (446 hand-labelled chips, 11 flood events) under
leave-one-region-out. India was held out entirely and influenced neither
training nor checkpoint selection.

| Held-out India, 68 chips | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| Otsu baseline | 0.3754 | 0.5459 | 0.7293 | 0.4362 |
| **U-Net (VV, VH, ratio)** | **0.5230** | **0.6868** | 0.7506 | 0.6331 |
| delta | **+0.1476** | | | |

Otsu with a unimodality guard is a strong operational method, and a U-Net that
could not beat it would have demonstrated nothing. Two numbers belong beside
the headline rather than behind it:

- **Validation IoU was 0.8679** — on Mekong, the fold's *validation* region.
  The 0.345 gap is exactly what quoting a validation score would have
  overstated.
- **Per-chip median IoU is 0.223** against a pooled 0.523. The pooled figure is
  dominated by chips with large water bodies: the model does well where there
  is a lot of water and poorly where there is little.

The baseline's separability guard moved from 0.75 to **0.66 on measurement**,
selected leave-one-region-out. All eleven folds chose it independently, so the
region-disjoint estimate and the all-data optimum agree to four decimals.

**C2, so far: adding the VV−VH ratio band is worth −0.0019 IoU.** A negative
result, reported as one. Explainability makes it stranger rather than clearer:
integrated gradients assign the ratio band a **53.4 % attribution share**, the
largest of the three, and removing it costs nothing. The two methods do not even
agree on its sign — occlusion says positive, gradients say negative
([`ml/experiments/flood_xai/`](ml/experiments/flood_xai)). High attribution is
not necessity, which is the whole reason C2 removes a band and measures rather
than reading an attribution chart.

**What the model can serve.** `ml/flood/predict.py` writes a cloud-optimised
GeoTIFF, a WGS84 GeoJSON extent and a provenance envelope per chip — on the
eight held-out India chips, 35.9 km² flooded of 151.3 km² observed.
`ml/flood/export.py` exports to ONNX for CPU serving and then *checks* the
export: the largest per-pixel probability disagreement with the checkpoint is
**7.7e-07**, and no pixel crosses the 0.5 boundary
([`models/flood/flood_unet_loro_india_sar_ratio.parity.json`](models/flood)).
An export nobody compared against the checkpoint would serve numbers that are
not the ones reported above, so the registry records an export only together
with its parity result.

### What is not built, and what is merely blocked

Two different things, kept apart because conflating them is how a project
starts describing itself as more finished than it is.

**Not written at all** — `ml/wildfire/`, `ml/cyclone/` and `ml/damage/` do not
exist as pipelines. Their *computations* do: `satai/hazards/` implements burn
severity (dNBR against published USGS breaks), a multiplicative fire-danger
index, rainfall anomaly against a climatology, modified-Rankine cyclone track
exposure, and SAR change detection — all deterministic, all tested. What is
missing is the acquisition that would feed them real inputs.

**Built and tested, waiting on a resource:**

| Blocker | Blocks |
|---|---|
| `GEMINI_API_KEY` not set | **C1.** The Gemini tool-calling loop, the 28-question benchmark, the 4-check validator and the `chat_turns` audit log all exist and the validator is parity-tested against the scored instrument. Only the run is missing. |
| `SUPABASE_SERVICE_KEY` not set | The C1 audit log. `/health` reports `audit_log: disabled` until it is set. |
| No labelled urban Indian flood imagery | **C3.** The model exists and the protocol is implemented; Sen1Floods11's India chips are not urban, so the *target* is what is missing, not code. |
| No co-registered rainfall or DEM rasters | **C2's** three remaining arms. Needs a GPM IMERG pull per chip acquisition window and a Copernicus DEM GLO-30 pull per footprint. |
| No exposure raster (population or assets) | The flood **risk** map. `R = H^a * E^b * V^g` needs E, and [`ml/flood/to_risk.py`](ml/flood/to_risk.py) refuses to substitute a constant — a flat exposure field would make the risk map a rescaling of the hazard map while presenting itself as a multi-factor result. The coupling itself is implemented and tested. |
| Earth Engine / CDSE credentials | Track B acquisition |

Full register, with the code that exists stated per experiment:
[`docs/evaluation.md`](docs/evaluation.md).

---

## Quick start

```bash
conda env create -f environment.yml
conda activate satai
pip install -e ".[dev]"

python scripts/check_env.py      # hardware, packages, credentials, AOIs
pytest -q                        # should be all green
pre-commit install               # gitleaks, ruff, mypy on every commit
```

A local database, with the schema, read views and row-level security applied in
order:

```bash
docker compose up -d db
```

Note that the `api` service does **not** talk to that database: the serving
plane speaks PostgREST over HTTP and carries no Postgres driver, deliberately,
because a pooled connection per serverless invocation is a liability. The local
`db` is there for `psql`, Adminer and the batch plane.

Windows specifics, including the GDAL failure mode and the GPU decision:
[`docs/setup-windows.md`](docs/setup-windows.md).

---

## Layout

```
satai/          shared core — config, logging, errors, PROVENANCE CONTRACT, geo
backend/        FastAPI serving plane            (Phase 9)
  api/          the two modules Vercel deploys
  app/db/migrations/   numbered SQL: schema, read views, grants + RLS
ml/             training, evaluation, experiments (Phase 4+)
  flood/        U-Net, losses, dataset, train, evaluate, predict, explain,
                export (ONNX), to_risk — the trained pipeline
  registry/     models.json, DERIVED from artifacts on disk — never hand-written
  experiments/  one runnable script per numbered experiment, with its reports
frontend/       Next.js + MapLibre               (Phase 11)
configs/        aoi.yaml, risk.yaml, couplings.yaml — documented parameters
data/           gitignored except manifests/ and samples/
docs/adr/       architecture decision records (11)
docs/           methodology, evaluation, deployment, data-sources,
                research-contribution, limitations (living), research-gap
tests/          661 tests across 34 modules
scripts/
.github/        CI: lint, mypy, pytest, frontend build, gitleaks
```

`satai/` is a shared installable package imported by both `backend/` and `ml/`,
so the provenance contract is defined once and cannot drift between them.

---

## Data sources

All open, all with official access points. Credentials via `.env`; none are
committed.

| Source | Provides | Access |
|---|---|---|
| Sentinel-1 GRD | SAR backscatter, 10 m, cloud-independent | CDSE / GEE |
| Sentinel-2 L2A | multispectral, 10–20 m | CDSE / GEE |
| Copernicus DEM GLO-30 | elevation → slope, HAND, TWI | CDSE / AWS / GEE |
| GPM IMERG | 30-min precipitation, ~4 h latency | NASA GES DISC |
| ERA5 / ERA5-Land | reanalysis weather | Copernicus CDS |
| NASA FIRMS | active fire detections | FIRMS API |
| IBTrACS v4 | cyclone best tracks | NOAA NCEI |
| WorldPop / GHSL | population, built-up → exposure | WorldPop / JRC |
| Sen1Floods11 | 446 hand-labelled flood masks over 11 events (India: 68) | public |
| Copernicus EMS | rapid-mapping delineations | Copernicus EMS |
| xBD | pre/post building damage labels | xView2 |

---

## Evaluation discipline

Leakage prevention comes before metrics; a metric from a leaked split is worse
than no metric, because it looks credible.

- Flood segmentation: **leave-one-region-out splits built by this project**
  (ADR-009), reported per region and per land-cover class. Sen1Floods11's
  *official* splits are **not** region-disjoint — every region except Bolivia
  appears in train, validation and test at ~58/21/21, so chips from one flood
  event straddle the boundary. SAT-AI reports both protocols and the gap between
  them, which quantifies the inflation the standard protocol carries.
- Susceptibility: **spatial block cross-validation**, never random k-fold.
  Random folds on spatially autocorrelated data inflate AUC substantially.
- Fire danger: **temporal split** plus spatial blocking.
- Damage: xBD official splits; India as explicit **zero-shot domain transfer**.
- Agents: a versioned 28-question benchmark measuring grounding-violation rate,
  tool-invocation accuracy, refusal correctness and latency. About a third of
  the set must be refused — a benchmark of answerable questions measures
  fluency, not whether the system knows where its competence ends.
  **The validator producing those numbers was itself scored first**
  (experiment 9a: 100 % detection over 15 labelled cases, after two real
  defects were found and fixed).

**No advanced model is reported as better than its baseline unless it beats that
baseline on a region-disjoint test set, with the delta and a confidence
interval.** A negative result reported honestly is a contribution. An inflated
positive is not.

Classical baselines are built *first* — Otsu thresholding with HAND masking is a
strong method, and a deep model that cannot beat it has not demonstrated
anything.

---

## Research contributions

Revised after the Phase 0 gap analysis against the seven-paper corpus
([`docs/research-gap.md`](docs/research-gap.md)). The word *novel* appears
nowhere until the supplementary structured search in that document's §6 is done.

**What the corpus establishes, and SAT-AI therefore does not claim:**
SAR-anchored fusion of radar, precipitation and terrain is the *dominant* flood
architecture (Kemarau et al. 2026, from 176 studies); encoder–decoder
segmentation of Sentinel-1 is standard (Jones et al. 2023; Akhyar et al. 2024);
deriving susceptibility labels from AI-generated flood extents is published
practice (Jones et al. 2023); and LLM-driven agents over remote-sensing tools
already form a recognised class (Shang et al. 2026). SAT-AI adopts all of this
deliberately and without novelty claim.

**SAT-AI locates its contribution in measurement, not architecture:**

- **C1 — Measured grounding.** Reported grounding-violation rate,
  tool-invocation accuracy and refusal correctness for an agent layer over EO
  model outputs. Shang et al. (2026) state that reasoning-consistency and
  tool-invocation accuracy *"must be integrated into benchmarks"* for
  remote-sensing agents; none of the corpus papers that put a language model
  over EO outputs measures whether the generated text is faithful to them.
- **C2 — Quantified degradation under modality loss**, with modality-attribution
  XAI as the instrument. An extension of the established fusion practice.
- **C3 — Quantified rural→urban domain-transfer gap** for SAR flood segmentation
  in an Indian setting, under region-disjoint evaluation. Kemarau et al. (2026)
  name urban SAR saturation as *"a critical frontier … requiring dedicated
  methodological innovation"*.
- **C4 — Reproducible, sensitivity-analysed multi-hazard risk implementation.**
  An engineering and reproducibility contribution, not a novelty claim.
  **Executed.** The exponents barely change *which* places rank riskiest
  (Spearman rho never below 0.957) but move up to 18.7 % of cells between
  colour bands. So SAT-AI may report a ranking with confidence and may not
  report a cell's band with the same confidence — a distinction that only
  exists because the analysis was run.

**Comparability caveat:** no paper in the primary corpus reports a
region-disjoint or spatially-blocked evaluation protocol, and two report
*accuracy* on heavily imbalanced segmentation tasks. Their headline numbers are
therefore not directly comparable to SAT-AI's, and the results chapter says so.

---

## Limitations

Maintained continuously in [`docs/limitations.md`](docs/limitations.md), not
written at the end. Includes cloud cover, revisit gaps, resolution mismatch,
urban SAR failure modes, circularity in susceptibility labels, proxy
vulnerability, and the absence of any ground-survey validation.

---

## Documentation

| Document | What it covers |
|---|---|
| [`docs/methodology.md`](docs/methodology.md) | How observations become risk statements, and what each step may claim |
| [`docs/evaluation.md`](docs/evaluation.md) | The nine-experiment register, protocol, and the two executed results |
| [`docs/research-contribution.md`](docs/research-contribution.md) | C1–C4, what is new and what is assembly |
| [`docs/deployment.md`](docs/deployment.md) | Live URLs, database access model, redeploy instructions |
| [`docs/data-sources.md`](docs/data-sources.md) | Every dataset, its latency, licence and limits |
| [`docs/limitations.md`](docs/limitations.md) | Where each step fails (living document) |
| [`docs/research-gap.md`](docs/research-gap.md) | The Phase 0 corpus analysis |
| [`docs/adr/`](docs/adr/) | Ten architecture decision records |

---

## Security

- No credential is ever committed. `.env` is gitignored; `.env.example` documents
  every variable, including the Supabase ones the serving plane reads.
- `satai/config.py` is the only module reading the environment **in the library
  and the ML plane**; secrets there are `SecretStr` and do not appear in `repr`
  or logs. The serving plane (`backend/api/`) is a deliberate exception and
  reads `os.environ` directly: importing `pydantic-settings` would pull the
  config layer into a serverless bundle that has a size budget. Stated rather
  than glossed, because the rule reads as absolute and is not.
- `gitleaks` runs in pre-commit ([`.pre-commit-config.yaml`](.pre-commit-config.yaml))
  and in CI over the full history, since a hook can be bypassed with
  `--no-verify` and a pipeline cannot.
- The only write credential is `SUPABASE_SERVICE_KEY`. It has no fallback
  anywhere, is never read by the browser bundle, and its sole use is appending
  to the C1 audit log.
- Nothing secret goes behind a `NEXT_PUBLIC_` prefix — that prefix ships to the
  browser.
- Row-level security and the read-only view surface are defined in
  [`backend/app/db/migrations/`](backend/app/db/migrations/), so the access
  model is reproducible from a clone rather than living only in the hosted
  project.

---

## License

MIT — see [`LICENSE`](LICENSE). Note that the **datasets** carry their own
licences (Copernicus, NASA, OSM/ODbL, xBD) and those terms govern their use.

## Author

Chirag Pednekar — B.Tech Electronics & Telecommunications, Vidyalankar Institute
of Technology, Mumbai.
