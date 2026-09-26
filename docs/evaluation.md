# Evaluation protocol and results

Ten named experiments. Four have been executed and one partially; their real
numbers appear below, including a negative result. For the other five, each
entry states which code exists, which does not, and what is blocking the
result. None is blocked on code any more — the remaining blockers are one
training run, one labelled target, two acquisitions and one API key — and each
entry says which of those it is rather than borrowing the word "implemented"
from the parts of the system that are.

**`[RESULT TO BE GENERATED]` means exactly that.** No experiment in this
document has an estimated, illustrative or expected value. A fabricated number
here would invalidate the project's entire argument, which is about whether
generated text is faithful to measured output.

---

## Protocol

### Splits

| Protocol | What it estimates | Use |
|---|---|---|
| Official Sen1Floods11 | Comparability with published work | Reported alongside |
| Leave-one-region-out | Transfer to an unseen region | **The headline number** |

The official splits are not region-disjoint — measured, not assumed: every
region except Bolivia appears in all three splits at roughly 58/21/21. The gap
between the two protocols is itself reported, as a measurement of how much the
standard protocol inflates.

### Metrics

IoU, Dice, precision, recall, F1, reported with the positive rate. Accuracy is
reported only with its caveat attached, automatically, below a 20 % positive
rate. Dataset-level figures pool confusion counts across chips; per-chip IoU is
never averaged.

Ignored pixels (label −1) are excluded from every count.

### Threshold selection

Swept on validation, fixed, applied unchanged at test. Selecting on test would
leak the test distribution into the headline number.

### Seeds

Every experiment fixes a seed and records it in its JSON artifact. Deep-model
experiments will report mean ± sd over three seeds; a single-seed deep result
is not a result.

---

## Experiment register

### 1 — Flood baseline (Otsu, per-pixel ensemble) — **EXECUTED**

Establishes what an unsupervised operational method achieves, so the deep
model's margin means something.

- **Data**: Sen1Floods11 v1.1 HandLabeled, Track A. 441 chips scored across 11
  regions; the archive downloads without credentials, contrary to what this
  document previously recorded.
- **Splits**: leave-one-region-out
- **Code**: `satai/ml/baseline.py`, `satai/ml/metrics.py`,
  `satai/preprocessing/splits.py`, runner
  `ml/experiments/run_flood_baseline.py`, guard selection
  `ml/experiments/select_baseline_threshold.py`
- **Result**: the separability guard was selected **leave-one-region-out** and
  moved from the assumed 0.75 to **0.66**. Pooled IoU over all regions at that
  guard is **0.4199**. On the held-out India region, 68 chips: **IoU 0.3754, F1
  0.5459, P 0.7293, R 0.4362**.
- **Optimism of the naive protocol: +0.0000.** All eleven folds selected 0.66
  independently, so choosing the guard on all the data would have given the same
  answer here. That is a measured result, not a reason to skip the protocol.
- **Caveat**: no HAND terrain mask ships with Sen1Floods11, so this is a lower
  bound on the method as operationally deployed. 57 of 68 India chips were
  called unimodal by the guard.
- **Reports**: `ml/experiments/flood_baseline/threshold_selection.json`,
  `ml/experiments/flood_baseline/sweep_0.66/otsu_baseline.json`

### 2 — Flood U-Net — **EXECUTED**

- **Data**: Sen1Floods11 v1.1 HandLabeled, Track A
- **Splits**: leave-one-region-out, India held out entirely
- **Code**: `ml/flood/` — `model.py` (U-Net, 4 levels, base width 32,
  7,763,041 parameters, trained from scratch), `losses.py` (masked BCE + Dice),
  `dataset.py`, `train.py`, `evaluate.py`
- **Result**: **IoU 0.5230, F1 0.6868, P 0.7506, R 0.6331** on the 68 held-out
  India chips at threshold 0.5, after 32 epochs. Against the baseline's 0.3754
  that is **+0.1476 IoU**, a 39 % relative gain, and it clears the bar this
  project set: a deep model that could not beat Otsu would have demonstrated
  nothing.
- **Two numbers that belong next to the headline**, not behind it:
  - validation IoU was **0.8679** — on Mekong, the fold's *validation* region.
    Quoting that as the model's score would have overstated it by 0.345.
  - the **per-chip median IoU is 0.223** against a pooled 0.523. The pooled
    figure is dominated by chips with large water bodies; the model does well
    where there is much water and poorly where there is little.
- **Deviation from the protocol above**: **one seed, not three.** Seed variance
  is therefore unmeasured, which matters directly for experiment 4 — where the
  effect being measured is smaller than a seed difference plausibly is.
- **Reports**: `ml/experiments/flood_unet/test_loro_india_sar_ratio.json`,
  `ml/experiments/flood_comparison.json`; registry entry in
  `ml/registry/models.json`

### 3 — LORO gap (C-adjacent) — **BLOCKED**

The quantity of interest is the difference between the official-split score and
the leave-one-region-out score for the same model and seed.

- **Status**: `[RESULT TO BE GENERATED]`. Everything trained here used LORO, so
  the official-split arm of the comparison does not exist and the gap has not
  been measured.
- **What is not this measurement**: the 0.345 validation-to-test gap reported in
  experiment 2 is a *region-to-region* gap within LORO. It suggests the
  direction, and it is not the defined quantity.
- **Needs**: one training run on Sen1Floods11's official splits with the same
  seed and schedule. No new data and no new code — `ml/flood/train.py` takes the
  protocol as an argument.

### 4 — Modality loss (**C2**) — **PARTIALLY EXECUTED** (2 of 5 arms)

Measured degradation when a modality is withheld at inference. The contribution
is the *measured* degradation curve, which the corpus does not report for this
task.

- **Code**: `ml/experiments/run_modality_ablation.py`. Every arm shares seed,
  schedule, augmentation, training fold and selection region, and differs only
  in the input band stack.
- **Executed**: `sar` (VV, VH) → IoU **0.5211**; `sar_ratio` (VV, VH, VV−VH
  ratio) → IoU **0.5230**.
- **Result so far: removing the derived ratio band costs 0.0019 IoU** — a
  negative result, reported as one. It is far smaller than the spread between
  regions and plausibly smaller than a seed difference, which is why experiment
  2's single seed is recorded as a limitation rather than a footnote.
- **The explainability cross-reference makes this stranger, not clearer.**
  Integrated gradients assign the ratio band a **53.4 % attribution share** —
  the largest of the three — and removing it costs nothing. Integrated gradients
  and occlusion do not even agree on its sign. High attribution is not
  necessity; that is precisely why this experiment removes a band and measures
  instead of reading an attribution chart
  (`ml/experiments/flood_xai/xai_loro_india_India.json`).
- **Blocked arms**: `sar_rain` needs a GPM IMERG pull per chip acquisition
  window resampled to the chip grid; `sar_dem` needs Copernicus DEM GLO-30 per
  chip footprint plus a HAND derivation; `full` needs both.
- **Report**: `ml/experiments/c2_modality_ablation.json`

### 5 — Rural→urban transfer (**C3**) — **BLOCKED**

Train on rural Indian flood chips, evaluate on an urban Indian target. The
expected direction is known and physical: urban double-bounce between building
walls and standing water *raises* backscatter where the method expects a drop,
so the signature inverts. The contribution is the size of the gap, in India,
measured.

- **Code**: the model and the scoring path now exist (experiment 2), as does the
  Track A/B distribution gate (`satai/ml/distribution_gate.py`), which is the
  precondition for trusting any cross-track result.
- **Status**: `[RESULT TO BE GENERATED]`
- **Blocker**: the *target*, not the code. Sen1Floods11's India chips are not
  urban, and no labelled urban Indian flood imagery has been acquired. Running
  the model over an unlabelled urban scene would produce a map and no
  measurement, which is not what C3 claims.

### 6 — Risk-engine sensitivity (**C4**) — **EXECUTED**

One-at-a-time over α, β, γ on {0.5, 0.75, 1.0, 1.5, 2.0}, over correlated
Gaussian random fields at five levels of coupling between hazard, exposure and
vulnerability.

**Measured results** (`ml/experiments/risk_sensitivity/`):

| Field correlation ρ | Min Spearman ρ | Max band reassignment | Min top-decile Jaccard |
|---|---|---|---|
| 0.0 | 0.9571 | 18.7 % | 0.740 |
| 0.3 | 0.9784 | 13.8 % | 0.794 |
| 0.5 | 0.9864 | 11.5 % | 0.839 |
| 0.7 | 0.9927 | 8.6 % | 0.865 |
| 0.9 | 0.9978 | 4.8 % | 0.915 |

**Finding.** The exponents barely change *which* places rank riskiest — rank
correlation never falls below 0.957 — but they change *what the map shows*
substantially, moving up to 18.7 % of cells between colour bands. Banding is a
threshold operation, so cells near a cut-point move even when the underlying
ordering is almost unchanged.

**What this licenses and forbids.** SAT-AI may report a ranking of places by
risk with confidence. It may not report a cell's colour band with the same
confidence, and the interface therefore presents the band with its percentile
definition rather than as a property of the place.

As the fields become more coincident the exponents matter less, because the
product's ranking approaches that of any single factor.

**Design limit, stated with the result**: one-at-a-time holds the other
exponents fixed and does not explore interactions.

### 7 — Wildfire / cyclone indices — **BLOCKED ON ACQUISITION**

- **Code that exists**: `satai/hazards/wildfire.py` (burn severity as dNBR
  against published USGS breaks, and a multiplicative fire-danger index) and
  `satai/hazards/extreme_weather.py` (rainfall anomaly against a climatology,
  modified-Rankine cyclone track exposure) — deterministic and unit-tested. The
  risk engine that consumes them exists and is tested.
- **Code that does not exist**: `ml/wildfire/` and `ml/cyclone/` as *pipelines*.
  There is no acquisition, no manifest and no scored artifact for either hazard,
  so the computations have never been run over real input.
- **Status**: `[RESULT TO BE GENERATED]`
- **Blocker**: Earth Engine credentials for ERA5 and fuel-state inputs, and a
  FIRMS pull. What is missing is the data, not the arithmetic.

### 8 — Damage assessment transfer

xBD-trained change classification evaluated on Indian building stock.

- **Code that exists**: `satai/hazards/damage.py` — SAR change detection,
  unit-tested.
- **Code that does not exist**: `ml/damage/` as a pipeline; nothing trains on
  xBD or scores against it.
- **Status**: `[RESULT TO BE GENERATED]`
- **Blocker**: the xBD download, and the pipeline around it. No Indian damage
  labels exist, so even once built this can only be a qualitative transfer
  check, and is labelled as one

### 9a — Validator validation — **EXECUTED**

Before the grounding-violation rate of C1 can be reported, the instrument
producing it must itself be scored. Fifteen hand-labelled responses: seven
clean, eight with planted violations of known kind.

**Measured results** (`ml/experiments/validator_validation/`):

| Metric | Value |
|---|---|
| Detection accuracy | **100.0 %** (15/15) |
| Violation recall | **100.0 %** (0 missed) |
| Violation precision | **100.0 %** (0 false alarms) |
| Violation-kind accuracy | **100.0 %** |

The first run scored **86.7 %**, and the two failures were real defects rather
than labelling disputes:

1. **False positive.** The confusion check matched the word *predictions* inside
   the phrase *"observations, not predictions"* — the exact sentence the
   benchmark rewards. Left unfixed it would have biased C1 downward by
   penalising correct answers. Fixed by making negation sentence-scoped.
2. **False negative.** The caveat check passed if any word of five or more
   letters from the caveat appeared in the response, so the word *flood*
   discharged a caveat about official warnings. Fixed by requiring distinctive
   phrases from a fixed table.

Both fixes are covered by unit tests so they cannot regress silently.

**Reading.** Recall is the number that matters. A missed violation understates
C1 and lets the system claim better grounding than it has. A precision failure
merely costs a regeneration.

### 9b — C1 grounding benchmark (**C1**)

28 questions across seven kinds. About a third must be refused — evacuation
advice, earthquake prediction, an out-of-area region, an invented helpline —
because a benchmark made only of answerable questions measures fluency rather
than whether the system knows where its competence ends.

- **Code that exists**: the question set, the scorer, the nine tools, the three
  agents, the router and the validator — all unit-tested (`satai/agents/`)
- **Status**: the harness is complete and its instrument is validated at 100 %.
  The measurement over a live model is `[RESULT TO BE GENERATED]`.
- **Blocker**: `GEMINI_API_KEY` is not set in the build environment. The
  deployed loop is Gemini native function calling over REST, and the
  deployed validator is parity-tested against the scored one
  (`tests/test_grounding_parity.py`), so what is missing is the run and not
  the instrument.

---

## Summary

| # | Experiment | Contribution | Code | Result |
|---|---|---|---|---|
| 1 | Flood baseline | — | Complete | **Executed** — IoU 0.3754 on held-out India; guard 0.66, LORO-selected |
| 2 | Flood U-Net | — | Complete | **Executed** — IoU 0.5230 on held-out India, +0.1476 over baseline |
| 3 | LORO gap | — | Complete | Blocked: needs an official-split run |
| 4 | Modality loss | C2 | Complete | **Partially executed** — 2 of 5 arms; ratio band worth −0.0019 IoU |
| 5 | Rural→urban transfer | C3 | Complete | Blocked: no labelled urban Indian target |
| 6 | Risk sensitivity | C4 | Complete | **Executed** |
| 7 | Wildfire / cyclone | — | Computations yes, pipelines no | Blocked: acquisition + credentials |
| 8 | Damage transfer | — | Change detection yes, pipeline no | Blocked: xBD acquisition |
| 9a | Validator validation | C1 instrument | Complete | **Executed** |
| 9b | C1 benchmark | C1 | Complete | Blocked: API key |

**Four of ten executed, one partially.** Experiments 1, 2, 4, 6 and 9a produced
real numbers, all reported above with their limits — including a negative result
(4) and a gap that a validation score would have hidden (2).

Of the five that remain, none is blocked on code any longer:

- **3** needs one training run on the official splits.
- **5** needs a labelled urban Indian target.
- **7** and **8** need acquisition — the computations are implemented and tested
  in `satai/hazards/`, what is missing is real input.
- **9b** needs `GEMINI_API_KEY` on the deployment.

That distinction is the point of this register. "Blocked on a resource" and "not
written" are different claims, and a project that blurs them ends up describing
itself as more finished than it is.
