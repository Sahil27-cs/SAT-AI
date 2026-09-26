# Evaluation protocol and results

Nine named experiments. Two have been executed and their real numbers appear
below. For the other seven, each entry states which code exists, which does
not, and what is blocking the result. Four of them have no implementation at
all, and say so rather than borrowing the word "implemented" from the parts of
the system that are.

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

### 1 — Flood baseline (Otsu + HAND, per-pixel ensemble)

Establishes what an unsupervised operational method and a context-free
supervised model achieve, so the deep model's margin means something.

- **Data**: Sen1Floods11, Track A
- **Splits**: official + LORO
- **Code that exists**: `satai/ml/baseline.py` (both baselines, unit-tested),
  `satai/ml/metrics.py`, `satai/preprocessing/splits.py`
- **Code that does not exist**: the runner that loads chips, applies the folds
  and writes the scored artifact. `ml/flood/` is an empty directory.
- **Status**: `[RESULT TO BE GENERATED]`
- **Blocker**: the runner, and the Sen1Floods11 archive —
  `storage.googleapis.com` is blocked by the build environment's egress policy

### 2 — Flood U-Net

- **Data**: Sen1Floods11, Track A
- **Splits**: official + LORO, 3 seeds
- **Code that exists**: the evaluation layer it would be scored with
  (`satai/ml/metrics.py`) and the split logic (`satai/preprocessing/splits.py`)
- **Code that does not exist**: the model itself. There is no `unet.py` and no
  training loop anywhere in this repository.
- **Status**: `[RESULT TO BE GENERATED]`
- **Blocker**: the model, the dataset, and a GPU. The build environment has
  none of the three; the target machine (RTX 4050, 6 GB) supplies the third.

### 3 — LORO gap (C-adjacent)

The quantity of interest is the difference between the official-split score and
the leave-one-region-out score for the same model and seed.

- **Status**: `[RESULT TO BE GENERATED]`
- **Depends on**: experiments 1 and 2, neither of which has a runner yet.
  The split logic this experiment measures *is* implemented and tested.

### 4 — Modality loss (**C2**)

Measured degradation when a modality is withheld at inference: SAR-only,
no-DEM, no-optical, all available. The contribution is the *measured*
degradation curve, which the corpus does not report for this task.

- **Status**: not implemented; `[RESULT TO BE GENERATED]`
- **Blocker**: depends on experiment 2, which has no model

### 5 — Rural→urban transfer (**C3**)

Train on rural Indian flood chips, evaluate on Mumbai MMR. The expected
direction is known and physical: urban double-bounce between building walls and
standing water *raises* backscatter where the method expects a drop, so the
signature inverts. The contribution is the size of the gap, in India, measured.

- **Code that exists**: the Track A/B distribution gate
  (`satai/ml/distribution_gate.py`, unit-tested), which is the precondition for
  trusting any cross-track result
- **Status**: the transfer experiment itself is not implemented;
  `[RESULT TO BE GENERATED]`
- **Blocker**: depends on experiment 2; also needs Track B acquisition

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

### 7 — Wildfire / cyclone indices

- **Status**: not implemented. `ml/wildfire/` and `ml/cyclone/` are empty
  directories; the risk engine that would consume their output exists and is
  tested. `[RESULT TO BE GENERATED]`
- **Blocker**: the modules, plus Earth Engine credentials for ERA5 and
  fuel-state inputs

### 8 — Damage assessment transfer

xBD-trained change classification evaluated on Indian building stock.

- **Status**: not implemented. `ml/damage/` is an empty directory.
  `[RESULT TO BE GENERATED]`
- **Blocker**: the module, and the xBD download. No Indian damage labels exist,
  so even once built this can only be a qualitative transfer check, and is
  labelled as one

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
- **Blocker**: `ANTHROPIC_API_KEY` is not set in the build environment

---

## Summary

| # | Experiment | Contribution | Status |
|---|---|---|---|
| # | Experiment | Contribution | Code | Result |
|---|---|---|---|---|
| 1 | Flood baseline | — | Models yes, runner no | Blocked: runner + dataset |
| 2 | Flood U-Net | — | **No model written** | Blocked: model + dataset + GPU |
| 3 | LORO gap | — | Split logic yes | Blocked: 1, 2 |
| 4 | Modality loss | C2 | **Not written** | Blocked: 2 |
| 5 | Rural→urban transfer | C3 | Gate yes, experiment no | Blocked: 2 + Track B |
| 6 | Risk sensitivity | C4 | Complete | **Executed** |
| 7 | Wildfire / cyclone | — | **Not written** | Blocked: modules + credentials |
| 8 | Damage transfer | — | **Not written** | Blocked: module + xBD |
| 9a | Validator validation | C1 instrument | Complete | **Executed** |
| 9b | C1 benchmark | C1 | Complete | Blocked: API key |

Two of nine executed. Both are the two that need no GPU, no dataset download
and no credential, and both produced real numbers, reported above with their
limits.

Three of the remaining seven have complete code and wait only on a resource
(9b on a key; 1 and 5 on a runner plus data). **Four have no code at all** —
the U-Net, the modality-loss study, the wildfire/cyclone indices and the damage
module. Saying they are "implemented" would be the same category of claim this
project exists to argue against, so the table says what is there.
