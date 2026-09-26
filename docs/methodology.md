# Methodology

How SAT-AI turns satellite observations into risk statements, and what each step
is allowed to claim. Written to be read alongside `docs/limitations.md`, which
states where each step fails.

---

## 1. The distinction everything rests on

Five things are routinely conflated in multi-hazard dashboards. SAT-AI keeps
them apart in code, in the database schema, in the API response and in the
language the agent is permitted to use.

| Class | What it is | Example in SAT-AI | Source kind |
|---|---|---|---|
| **Detection** | Something happened; it was measured | NASA FIRMS thermal anomaly | `OBSERVATION` |
| **Mapping** | Where it happened, inferred from imagery | SAR flood extent | `MODEL` |
| **Susceptibility** | Where it tends to happen, from terrain | Flood susceptibility surface | `MODEL` |
| **Risk index** | Susceptibility combined with exposure and vulnerability | `R = H^α · E^β · V^γ` | `INDEX` |
| **Forecast** | What will happen, and when | **Not produced by SAT-AI** | — |

The `SourceKind` enum in `satai/provenance.py` is that table, made executable.
Every value crossing a plane boundary carries one, and the interface renders it
as a badge beside the number. A reader who sees `MODEL OUTPUT` next to a flood
extent knows they are looking at an inference from backscatter, not a
measurement of water.

The fifth row has no implementation. That is the point of the row.

---

## 2. Architecture: four planes, separated by latency class

ADR-001 sets out the reasoning. The short version: the planes are separated
because they have irreconcilable latency budgets, not because layering is
tidy.

```
  Data plane        acquisition, preprocessing        minutes → hours
  ML plane          training, batch inference         minutes → hours (GPU)
  Serving plane     read precomputed artifacts        < 500 ms target
  Agent plane       tool calls + language             seconds
```

A request for "the flood risk in Bihar" cannot run a U-Net. Sentinel-1 revisit
is 6–12 days, so there is nothing to gain by trying: the answer would be the
same as the one computed on the last overpass. The serving plane therefore does
lookups and arithmetic only, and the batch planes do everything expensive.

This is also why the agent cannot invent a number even in principle. It has no
access to a model — only to tools that read the serving plane.

---

## 3. Data sources

Full table with licences and access routes in `docs/data-sources.md`. The
operative distinctions:

- **Sentinel-1 C-band SAR** is the backbone because it is cloud-independent.
  Floods happen under cloud; optical flood mapping in the Indian monsoon is
  largely a study of when the sky cleared.
- **Sentinel-2 optical** supplies vegetation and burn indices, when visible.
- **Copernicus DEM (30 m)** supplies slope, drainage and HAND.
- **GPM IMERG Early** supplies near-real-time rainfall (~4 h latency).
- **ERA5** supplies the meteorological fields, and is *retrospective* — roughly
  five days behind. An ERA5-driven index is never a current-conditions
  statement, and the envelope's `REANALYSIS` kind says so.
- **NASA FIRMS** supplies active-fire detections. These are observations at
  overpass time, never forecasts, and the validator treats describing them as
  predictions as a violation.
- **Sen1Floods11** supplies the only hand-labelled flood ground truth: 446
  chips across 11 events, 68 of them Indian.

---

## 4. SAR preprocessing

Terrain-corrected, calibrated to σ⁰, converted to decibels, speckle-filtered.
Two conventions are fixed in code because getting either wrong inverts the
result:

**Polarisation ratio.** In decibels, division becomes subtraction:

```
VV/VH ratio  =  VV_dB − VH_dB
```

Computing `VV_dB / VH_dB` produces a number that looks plausible, varies
smoothly, and means nothing. `satai/preprocessing/indices.py` implements the
subtraction and the test suite asserts it.

**Change sign.** The log-ratio is `post − pre`, so flooding — a drop in
backscatter as smooth water reflects radar away from the sensor — is
**negative**. Reversing it turns every flood map into its own complement.

**Normalisation.** Band statistics are fitted on the training partition only
and tagged with the fold identity that produced them
(`satai/preprocessing/normalize.py`). Fitting on the full dataset leaks test
distribution into training and inflates the reported score by an amount nobody
can afterwards estimate.

---

## 5. Two tracks, and the gate between them (ADR-010)

Sen1Floods11 ships pre-processed chips. SAT-AI's own pipeline processes raw
GRD granules. These are not the same distribution, and a model trained on one
and deployed on the other is running out of domain without anyone noticing.

- **Track A** — Sen1Floods11 chips. Comparable to published work, and the only
  route to hand-labelled ground truth.
- **Track B** — SAT-AI's GRD pipeline over the study areas. What the system
  will actually see in operation.

Between them sits a **distribution gate** (`satai/ml/distribution_gate.py`).
Per band it computes a KS statistic and a Wasserstein-1 distance, and it
fails the gate when the Wasserstein distance exceeds 3 dB — about a third of a
typical land/water separation, which is enough to move any threshold learned on
Track A.

The KS p-value is computed, reported, and deliberately **not** used as the
criterion. With 10⁵ pixels every real difference is significant; the p-value
restates the sample size, not the effect.

A failed gate does not stop the project. It changes what may be claimed: the
Track A number stays a Track A number and is not presented as performance over
India.

---

## 6. Evaluation splits (ADR-009)

Sen1Floods11's official splits are **not region-disjoint**. This was measured,
not assumed: every region except Bolivia appears in train, validation and test
at roughly 58/21/21. Chips from a single flood event — adjacent tiles, same
day, same sensor geometry — straddle the split boundary.

A model can therefore score well by recognising the event rather than by
recognising water, and that score will not survive contact with a new region.

SAT-AI reports **both**:

- the official split, for comparability with published numbers;
- a **leave-one-region-out** split (`satai/preprocessing/splits.py`), which is
  the honest estimate of transfer.

The gap between them is not an inconvenience. It is a measurement of how much
the standard protocol inflates, and it is reported as a result.

---

## 7. Models, in order

Baselines first, and not as a formality.

1. **Otsu + HAND** (`satai/ml/baseline.py`). Unsupervised thresholding of the
   VV histogram, with a height-above-nearest-drainage mask to remove the false
   positives Otsu cannot avoid — tarmac, sand and dry riverbeds are smooth and
   scatter radar away exactly as water does.

   Otsu assumes bimodality and will happily bisect a dry chip's single mode,
   reporting half of it as flooded. The guard is the separability η — the
   fraction of variance the split explains. Splitting one Gaussian explains
   2/π ≈ 0.637; two modes 10 dB apart explain over 0.9. The cut-off is 0.75.

2. **Per-pixel ensemble.** Random Forest / gradient boosting over the band
   stack, with **no spatial context**. This is the control that gives the deep
   model's margin a meaning: the difference between the two is the value of
   spatial context, isolated from architecture and optimisation.

3. **U-Net.** Encoder–decoder segmentation, the standard architecture for this
   task. It has to beat both baselines on leave-one-region-out to have
   demonstrated anything.

---

## 8. Metrics

Binary segmentation of a rare class. Three rules, each guarding a real failure:

- **Accuracy is not reported alone.** At a 5 % water fraction, predicting
  "all dry" scores 95 %. `SegmentationMetrics.accuracy_is_misleading` flags
  this automatically below a 20 % positive rate so the caveat travels with the
  number.
- **Ignored pixels are excluded from every count.** Sen1Floods11 marks no-data
  as −1. Scoring those as negatives makes the reported IoU a function of how
  much of the chip was masked — a property of the scene, not the model.
- **Dataset IoU pools confusion counts; it never averages per-chip IoU.**
  Averaging weights a 10-pixel chip like a 10 000-pixel one and is dominated by
  near-empty chips where IoU is unstable by construction. The test suite
  contains a worked case where the two differ by 0.25 IoU.

Thresholds are selected on the **validation** fold and applied unchanged at
test time.

---

## 9. Risk (ADR-008)

```
R_h  =  H_h^α  ·  E^β  ·  V^γ
```

Multiplicative, deliberately. An uninhabited floodplain has high hazard and
near-zero risk; an additive or averaged form assigns it substantial risk and
is the single most common defect in multi-hazard dashboards. The multiplicative
form enforces the boundary condition *zero exposure implies zero risk*,
following the Crichton risk-triangle decomposition.

Per-hazard scores are **never averaged into one number**. Flood risk and fire
danger come from different models with different base rates; their mean denotes
nothing. The engine returns a vector and a dominant hazard.

Bands (GREEN/YELLOW/ORANGE/RED) are **percentiles within the area of interest**,
not absolute thresholds. Every area gets a top decile, including a safe one. The
caveat is attached to every result rather than left to the reader.

Vulnerability is a **proxy** from building density, road access and elevation
above drainage. It omits income, age structure, disability and tenure, and so
systematically under-represents the vulnerability of marginalised populations.
That sentence is in the envelope caveats, not only in this document.

The exponents default to 1.0 because no evidence supports anything else — which
makes them an arbitrary choice, and the reason the sensitivity analysis in
§11 is not optional.

---

## 10. The agent plane

Three agents (risk analyst, emergency, recovery) over nine tools. Two design
decisions carry the weight:

**Routing is heuristic first.** `route_heuristic()` needs no network, no model
and no API key. Model-assisted routing is an optimisation on top. This is what
lets the system degrade to structured tool output when the language layer is
unreachable, rather than failing.

**Tools raise rather than default.** A tool either returns a provenance
envelope or raises `DataUnavailableError`. There is no path returning a
plausible zero, because the language layer cannot distinguish a default from a
measurement, and neither can the validator downstream — it checks that numbers
come from envelopes, not that the envelopes are true.

**Generation is validated, then regenerated once.** Every number in the
response is extracted deterministically and checked against the union of
`groundable_values()` from the turn's envelopes, at 2 % tolerance. A violation
triggers one regeneration with the offending values named. A second violation
falls back to structured tool output with the failure declared.

Fabricated authority — an invented helpline, an evacuation order, a claimed
official warning — is never regenerated. A retry might produce a
better-worded version of the same lie.

---

## 11. Sensitivity analysis

One-at-a-time over α, β, γ on a {0.5, 0.75, 1.0, 1.5, 2.0} grid, measuring
three things against the default configuration: Spearman rank correlation,
the fraction of cells that change band, and the Jaccard overlap of the
top-decile sets.

The finding, from the executed run: rank order is stable (min Spearman 0.957 at
ρ=0, 0.998 at ρ=0.9) while band membership is not (18.7 % down to 4.8 %
reassigned). So SAT-AI may state with confidence *which places rank riskiest*,
and may not state with the same confidence *what colour a given cell is*. Those
are different claims and only one of them survives the analysis.

The design's limit is stated with the result: one-at-a-time varies each exponent
with the others held fixed and does not explore interactions.

---

## 12. Reproducibility

- Every configuration that affects an output is hashed into
  `RiskConfig.config_hash()` and stored with the result.
- Normalisation statistics carry the fold identity that produced them.
- Every experiment writes a JSON artifact with its git SHA, timestamp and
  full parameter set.
- Unmeasured metrics render as `[RESULT TO BE GENERATED]` and serialise as
  `null` — never as `0.0`, which is itself a result.

---

## 13. What is deliberately absent

- **No earthquake prediction**, in any form. Earthquake scope is post-event
  damage assessment from pre/post imagery.
- **No cyclone track or intensity forecasting.** That needs assimilated
  observations and numerical weather prediction, which IMD operates.
- **No wildfire ignition prediction.** Fire danger is estimated; ignition is
  not predicted.
- **No flood timing or depth forecasting.**
- **No monetary damage estimate.** There is no asset-value data, and producing
  a figure without it would be fabrication.
- **No ground-survey validation anywhere in this project.** Every evaluation is
  satellite against satellite.
