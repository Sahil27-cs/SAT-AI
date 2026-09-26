# Research contribution

What in SAT-AI is new, what is assembly of published work, and how to tell the
difference.

---

## The finding that shaped the project

Phase 0 was a gap analysis against seven papers, conducted before any code was
written. Its conclusion was uncomfortable and is the reason the rest of the
project is organised as it is:

> **Every architectural component SAT-AI proposed to build is already
> established in the literature.**

Specifically:

- **SAR + precipitation + DEM fusion is the dominant flood architecture.**
  Kemarau et al. (2026) synthesise 176 studies and identify it as the
  consensus approach. Building it is not a contribution.
- **Encoder–decoder segmentation of Sentinel-1 is standard.** Jones et al.
  (2023), Akhyar et al. (2024). A U-Net on Sen1Floods11 is a reimplementation.
- **Deriving susceptibility labels from AI-produced flood extents is published
  practice.** Jones et al. (2023). SAT-AI's plan to do this was not novel; it
  was, unknowingly, already the method.
- **LLM agents over remote-sensing tools are a recognised class.** Shang et al.
  (2026) name six such systems. "Add a chatbot" is not a contribution either.

A project whose claim is "we built a multi-hazard platform with an AI assistant"
would be claiming credit for reimplementation.

## Where the gap actually is

The same corpus that closes the architectural gap opens a different one. Shang
et al. (2026, §VI.D) state that for remote-sensing AI agents

> reasoning-consistency and tool-invocation accuracy must be integrated into
> benchmarks to ensure logical reliability

and that opaque reasoning limits adoption *"in high-risk applications like
territorial planning and emergency management"*.

None of the corpus papers that put a language model over Earth-observation
outputs — Aziz et al. (2025), Swain et al. (2025), Kundu et al. (2025) —
**measures whether the generated text is faithful to those outputs.** The
systems are demonstrated, not evaluated on that axis.

So the contribution moved from *architecture* to *measurement*. Four claims
follow, C1 to C4, each of which is a number this project produces and the
corpus does not.

---

## C1 — Grounding-violation rate and tool-invocation accuracy

**Claim.** A measured grounding-violation rate and tool-invocation accuracy for
a tool-using agent over Earth-observation model outputs, on a versioned
benchmark whose composition is published.

**Method.** Every serving-plane value is wrapped in a provenance envelope. The
union of `groundable_values()` across a turn's envelopes is the set of numbers
the response may state. After generation, every number is extracted
deterministically and checked at 2 % tolerance. Violations are classified into
five kinds and counted separately, because their causes differ:

| Kind | Meaning |
|---|---|
| `ungrounded_value` | A number no tool returned |
| `observation_prediction_confusion` | A measurement called a prediction, or the reverse |
| `unsupported_claim` | A categorical assertion with no tool output behind it |
| `missing_caveat` | An envelope's critical caveat dropped from the response |
| `fabricated_authority` | An invented helpline, evacuation order or official warning |

**Why deterministic extraction rather than an LLM judge.** Using a model to
check a model makes the checker's reliability load-bearing and unmeasured.
Regex extraction is exact and cheap, and its failure modes are enumerable —
which is what made experiment 9a possible.

**The benchmark.** 28 questions, seven kinds, versioned at 1.0.0. About a third
must be refused. That proportion is a design decision: a set of answerable
questions measures fluency, whereas refusal items measure whether the system
knows where its competence ends — which is the part that matters when someone
asks whether to evacuate.

**Status.** The instrument is validated at 100 % detection accuracy over 15
labelled cases (experiment 9a, executed). The measurement over a live model is
`[RESULT TO BE GENERATED]`, blocked on an API key.

**Honest scoping.** C1 measures whether quantities are *traceable*, not whether
reasoning is *good*. A response can be perfectly grounded and poorly argued.
That is a different measurement and this project does not pretend to make it.

---

## C2 — Degradation under induced modality loss

**Claim.** A measured degradation curve for a multi-modal flood model as
modalities are withheld at inference: SAR-only, no-DEM, no-optical, all.

**Why it is a gap.** Fusion papers report the score of the full stack. What an
operator needs to know is what happens on the day the optical scene is
cloud-covered or the DEM has a void — which is most days. The corpus does not
report that curve for this task.

**Status. Partially executed — 2 of 5 arms.** The flood model exists and is
trained (`ml/flood/`), and the ablation runner
(`ml/experiments/run_modality_ablation.py`) holds seed, schedule, augmentation,
training fold and selection region fixed so that the arms differ only in the
input band stack.

**Measured so far.** SAR (VV, VH) scores IoU **0.5211** on held-out India; SAR
plus the derived VV−VH dB ratio scores **0.5230**. Removing the ratio band
therefore costs **0.0019 IoU** — smaller than the spread between regions, and
plausibly smaller than a seed difference. A negative result, reported as one.

**What makes it interesting rather than merely small.** Integrated gradients
assign that same ratio band a **53.4 % attribution share**, the largest of the
three, and integrated gradients and occlusion disagree on its sign. High
attribution is not necessity — which is the entire reason this contribution
removes a band and measures, instead of reading an attribution chart and
calling the result an explanation.

**Still blocked.** The three remaining arms need co-registered rainfall (GPM
IMERG per chip acquisition window) and terrain (Copernicus DEM GLO-30 per chip
footprint, plus a HAND derivation). Neither ships with Sen1Floods11.
`[RESULT TO BE GENERATED]` for `sar_rain`, `sar_dem` and `full`.

---

## C3 — Rural-to-urban SAR transfer gap in India

**Claim.** The size of the performance gap when a flood model trained on rural
Indian chips is evaluated on Mumbai's urban fabric.

**Why it is a gap.** The *direction* is known and physical: urban double-bounce
between building walls and standing water raises backscatter where the method
expects a drop, so the signature inverts. The *magnitude*, in India, on a
leave-one-region-out protocol, is not published.

**Status. Blocked on the target, no longer on the code.** The flood model is
trained and its scoring path is in place (`ml/flood/evaluate.py`), and the
precondition for trusting any cross-track number — the Track A/B distribution
gate — is built and tested (`satai/ml/distribution_gate.py`).

What is missing is labelled urban Indian flood imagery. Sen1Floods11's India
chips are not urban, and running the model over an unlabelled urban scene would
produce a map and no measurement. `[RESULT TO BE GENERATED]`

---

## C4 — Sensitivity-analysed multi-hazard risk engine — **executed**

**Claim.** A reproducible multi-hazard risk formulation whose free parameters
are accompanied by a published sensitivity analysis, so that an arbitrary
modelling choice becomes a reported quantity.

**Why it is a gap.** Risk dashboards publish weighted composites with weights
that are neither derived nor justified nor tested. The weights are usually
presented as if they were physics.

**Measured result.** One-at-a-time over α, β, γ on
{0.5, 0.75, 1.0, 1.5, 2.0}, across five levels of correlation between hazard,
exposure and vulnerability fields:

| Field correlation ρ | Min Spearman ρ | Max band reassignment | Min top-decile Jaccard |
|---|---|---|---|
| 0.0 | 0.9571 | 18.7 % | 0.740 |
| 0.9 | 0.9978 | 4.8 % | 0.915 |

**Finding.** The exponents barely change *which* places rank riskiest, but
change *what the map shows* substantially. Rank correlation never falls below
0.957; band membership moves for up to 18.7 % of cells. Banding is a threshold
operation, so cells near a cut-point move even when the ordering does not.

**What it licenses.** SAT-AI may report a ranking of places by risk with
confidence. It may not report a cell's colour band with the same confidence —
and so the interface presents the band with its percentile definition attached
rather than as a property of the place.

---

## Secondary finding: Sen1Floods11's splits are not region-disjoint

Not planned as a contribution; found while implementing the split logic, and
verified against the data rather than assumed.

Every region except Bolivia appears in train, validation and test at roughly
58/21/21. Chips from a single flood event — adjacent tiles, same day, same
sensor geometry — straddle the split boundary. A model can therefore score well
by recognising the event rather than the water.

SAT-AI reports both protocols. The gap between them is a measurement of how
much the standard protocol inflates, and it is reported as a result rather than
as a footnote.

This correction was applied to four documents that had previously asserted the
opposite; ADR-009 records the correction and its evidence.

---

## Where the project stands against these claims

| | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| Instrument built and tested | yes | partly (metrics) | partly (gate) | yes |
| Experiment runnable today | yes, given a key | no | no | yes |
| **Measured** | not yet | no | no | **yes** |

C4 is delivered. C1's instrument is delivered and validated, and the
measurement needs only an API key. C2 and C3 are designs with some of their
supporting machinery built and the model they depend on unwritten. That is the
honest position, and it is stated here rather than left for a reader to
discover.

---

## What is explicitly *not* claimed

- Not a novel architecture. Every component is from the literature, cited.
- Not state-of-the-art accuracy on Sen1Floods11.
- Not an operational system. It is a student research prototype.
- Not validated against ground survey. Every evaluation is satellite against
  satellite; no field data exists anywhere in this project.
- Not a warning system, and not a substitute for one.

---

## Reproducing the executed results

```bash
python ml/experiments/run_risk_sensitivity.py --mode structural   # C4
python ml/experiments/run_validator_validation.py                 # C1 instrument
```

Both are deterministic under fixed seeds, need no GPU, no credentials and no
network, and write JSON artifacts with their git SHA and full parameter set to
`ml/experiments/`.

That they run anywhere is not incidental. The two results this project can
stand behind are the two that anyone can reproduce in under a minute.

---

## References

1. Kemarau, R. A., et al. (2026). Synthesis of 176 flood-mapping studies.
2. Jones, C., et al. (2023). AI-derived flood extents as susceptibility labels.
3. Akhyar, A., et al. (2024). Deep segmentation of Sentinel-1 for flood mapping.
4. Shang, C., et al. (2026). LLM agents for remote sensing. *IEEE JSTARS*.
5. Aziz, M., et al. (2025). Conversational interfaces over EO products.
6. Swain, S., et al. (2025). LLM-assisted disaster analytics.
7. Kundu, S., et al. (2025). Language interfaces for geospatial risk.
8. Bonafilia, D., et al. (2020). Sen1Floods11. *CVPR Workshops*.
9. Gupta, R., et al. (2019). xBD / Creating xBD. *CVPR Workshops*.
10. Crichton, D. (1999). The risk triangle. In *Natural Disaster Management*.
