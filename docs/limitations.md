# SAT-AI — Limitations

**A living document.** Started in Phase 1, updated in every phase. Limitations
discovered during development are added here immediately, while the reason is
still understood, rather than reconstructed at write-up time.

Entries are dated and tagged with the phase that found them. An entry is removed
only when the limitation is genuinely resolved, and the resolution is noted.

---

## 0. Scope boundaries (permanent, by design)

These are not deficiencies to be fixed later. They are the boundary of what the
system claims.

- **SAT-AI does not predict earthquakes.** The earthquake module, if built, is
  post-event damage assessment from pre/post imagery. No component estimates
  earthquake probability, location, magnitude or timing.
- **SAT-AI does not forecast cyclone tracks or intensity.** Operational cyclone
  forecasting requires assimilated observations and numerical weather prediction
  run by national meteorological agencies. SAT-AI performs historical track
  analysis, extreme wind/rainfall risk indices, and exposure assessment along an
  observed or supplied track.
- **SAT-AI does not predict wildfire ignition.** It estimates fire *danger* from
  fire-weather and fuel proxies, and it *ingests* active fire detections, which
  are observations rather than predictions.
- **SAT-AI does not forecast flood timing or depth.** It estimates flood
  susceptibility, detects flood extent from acquired imagery, and modulates risk
  by observed antecedent rainfall.
- **SAT-AI risk levels are not official warnings.** GREEN/YELLOW/ORANGE/RED are
  research outputs from a student prototype. Official warnings for India come
  from IMD, NDMA and State Disaster Management Authorities.
- **SAT-AI is not real-time.** See §3.

---

## 1. Data limitations

*(Phase 1 — anticipated; to be confirmed or corrected with measurements in Phase 2)*

- **Cloud cover defeats optical sensing exactly when it is needed.** Floods in
  India occur during the monsoon, when Sentinel-2 usability collapses. This is
  the asymmetry that motivates SAR-primary design and multimodal fusion.
- **Sentinel-1 revisit is 6–12 days.** Many flood peaks recede between
  acquisitions and are never observed. Sentinel-1D began user data distribution
  in 2026 and orbital reconfiguration is under way, so cadence should improve
  during the project — Phase 2 must *measure* the actual interval over the AOI
  rather than quote a nominal figure.
- **Resolution mismatch spans three orders of magnitude**: 10 m (S1/S2), 30 m
  (DEM), ~9 km (ERA5-Land), ~11 km (IMERG). Coarse fields resampled to a 10 m
  grid carry no sub-kilometre information, and rendering them at 10 m implies a
  precision that does not exist. `SpatialRef.resolution_m` records native
  resolution for exactly this reason.
- **ERA5 has roughly a five-day lag.** It is retrospective and cannot support
  any near-real-time claim.
- **Free-tier quotas cap throughput.** GEE noncommercial: 150 EECU-hours/month
  (Community). CDSE: 10,000 SH processing units/month. Both bind at project
  scale and must be monitored.
- **Ground truth is scarce and geographically biased.** Sen1Floods11 covers 11
  global events. xBD's disaster domains do not include Indian building stock.
  There is no ground-survey validation anywhere in this project — every
  evaluation is satellite against satellite, or satellite against another
  analyst's satellite interpretation.

## 2. Methodological limitations

- **SAR flood detection fails predictably in cities.** Double-bounce between
  building walls and standing water raises backscatter where the method expects
  a drop. Expect materially worse performance in the Mumbai AOI; this is
  reported, not tuned away (ADR-007).
- **SAR flood detection fails under vegetation.** Canopy attenuates the water
  signal; flooding beneath forest is largely invisible to C-band.
- **Smooth dry surfaces produce false positives.** Tarmac, sand and dry
  riverbeds mimic water's low backscatter. Terrain priors (HAND, slope) suppress
  but do not eliminate this.
- **Flood susceptibility labels are model-generated.** The inventory used to
  train the susceptibility model comes from SAT-AI's own extent model, so its
  errors propagate. This circularity is a legitimate published technique and a
  real weakness; both must be stated.
- **Active fire detection is not ignition detection.** FIRMS reports thermal
  anomalies at overpass times, missing fires between overpasses and under cloud,
  and conflating agricultural residue burning with forest fire — the dominant
  confound in the Indian context.
- **dNBR thresholds are transferred from the literature**, not calibrated
  against Indian field data.
- **Vulnerability is a proxy.** No socioeconomic determinants. See ADR-008.

## 3. Latency — what "real-time" would and would not mean here

Measured per stage rather than claimed globally:

| Stream | Latency class | Approximate |
|---|---|---|
| FIRMS active fire | near-real-time | ~3 h |
| GPM IMERG Early rainfall | near-real-time | ~4 h |
| ERA5 weather | retrospective | ~5 days |
| Sentinel-1 flood extent | revisit-limited batch | 6–12 days |
| API response | fast, serving precomputed results | < 500 ms target |

The honest system-level description is therefore: **near-real-time for
weather-driven risk indices; revisit-limited batch for satellite-derived extent
and damage.** A fast API response reflects a precomputed artifact, not fresh
observation, and the interface displays data age everywhere.

## 4. Model limitations

*(To be populated with measured results from Phases 4 onward.)*

- **Sen1Floods11 is small: 446 hand-labelled chips across 11 events, India 68.**
  Verified 2026-09-22. Roughly 400 chips per leave-one-region-out fold is few for
  training a segmentation network from scratch, which raises the priority of a
  pretrained encoder and of the foundation-model ablation.
- **Sen1Floods11's official splits are NOT region-disjoint.** Every region except
  Bolivia appears in train, validation and test at ~60/20/20, so chips from one
  flood event straddle the train/test boundary and spatial autocorrelation leaks
  across it. Published benchmark figures on this dataset — including foundation-
  model fine-tunes — therefore measure interpolation within an event, not
  generalisation to a new one. SAT-AI builds its own LORO splits (ADR-009) and
  reports both, so the inflation is quantified rather than inherited.
- Sen1Floods11's 11 events under-represent Indian monsoon flooding.
- xBD → India damage transfer is untested; any result is a qualitative
  demonstration with a stated domain gap, not a validated metric.
- Spatial autocorrelation means even spatially-blocked cross-validation is
  optimistic.
- No uncertainty quantification in v1. MC-dropout or deep ensembles are a
  stretch goal.
- Severe class imbalance: water typically occupies 2–15 % of pixels.

**Measured results go here as they are produced. Until an experiment has run,
the entry reads `[RESULT TO BE GENERATED]` — never a plausible placeholder
number.**

## 5. System limitations

- Single-region validation; transferability beyond the selected AOI is untested.
- Agent behaviour depends on a third-party API. Documented fallback: the ML and
  serving planes return structured output with no natural-language layer.
- No authentication or multi-tenancy — a research prototype, not a service.
- Free-tier hosting means cold starts and constrained memory.

## 6. What would be needed to make this operational

Recorded so that the gap between a research prototype and a warning system is
explicit rather than implied:

- Ground-truth validation against field survey or authoritative inundation maps
- Continuous monitoring, alerting and on-call operation
- Formal uncertainty quantification with calibrated intervals
- Institutional integration with IMD / NDMA / SDMA
- Independent expert review
- Legal and ethical review of false-negative consequences
- Service-level guarantees on data availability

---

## Change log

| Date | Phase | Change |
|---|---|---|
| 2026-09-22 | 1 | Document created with anticipated limitations and permanent scope boundaries |
| 2026-09-22 | 0 | Gap analysis added §2 entry: the susceptibility-label circularity is **published practice** (Jones et al. 2023), which legitimises the method without removing the weakness — both facts now stated |
| 2026-09-22 | 0 | Added comparability caveat: no paper in the primary corpus reports a region-disjoint protocol, and two report accuracy on heavily imbalanced tasks, so their numbers are not directly comparable to SAT-AI's |
| 2026-09-22 | 2 | Sentinel-1 revisit moved from "anticipated" to **measurable** — `measure_aoi_availability.py` reports the observed median gap per AOI. The nominal 6–12 day figure must not be quoted once a measurement exists |
| 2026-09-22 | 2 | New limitation: **relative-orbit heterogeneity.** A pre/post SAR pair from different relative orbits has different incidence geometry, so a cross-orbit change signal mixes flooding with viewing angle. Change detection must be restricted to within-orbit pairs, which reduces usable pairs below the raw scene count |
| 2026-09-22 | 2 | New limitation: **catalogue truncation.** A capped search returns a lower bound, not a count. Propagated as a manifest warning rather than silently accepted |
| 2026-09-22 | 2 | **CORRECTION.** Phases 0–1 repeatedly stated that Sen1Floods11 ships region-disjoint splits. Verification showed it does not. Evaluation protocol replaced with project-built leave-one-region-out splits (ADR-009); README, research-gap.md and ADR-007 corrected. The original claim came from recollection of the paper rather than inspection of the data — the same failure mode ADR-007 was written to prevent, caught one layer later than it should have been |
| 2026-09-22 | 2 | **Open question:** the `bihar_ganga` bbox in `configs/aoi.yaml` was invented before the Indian chip locations were known. `verify_sen1floods11.py --footprints` resolves it; **must be run before Phase 3b** |
| 2026-09-22 | 3 | New **major** risk: **training/inference preprocessing domain mismatch** (ADR-010). Sen1Floods11 ships chips its authors already calibrated and terrain-corrected; SAT-AI's operational path starts from raw GRD. If Track B's radiometric conventions differ from Track A's, the model produces confident, plausible, wrong flood masks — with no error raised anywhere. Not discussed in the Phase 0 corpus, because those papers train and test on the same dataset. A distribution-comparison test gates Phase 5, and the measured distance is reported as part of C3 |
| 2026-09-22 | 3 | New limitation: **validation region confounds the LORO estimate.** Each fold holds out a whole region for validation as well as test, so early stopping is tuned on a region the model also never trained on. This is the honest choice, but it means the validation signal is itself out-of-distribution and may stop training earlier than a within-domain criterion would |
| 2026-09-22 | 3 | Recorded: **dNBR severity thresholds are transferred from the literature**, not calibrated against Indian field data (`satai/preprocessing/indices.py`) |
