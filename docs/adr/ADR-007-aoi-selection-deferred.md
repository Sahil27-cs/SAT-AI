# ADR-007: AOI selection deferred to measured evidence in Phase 2

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 1

## Context

The natural instinct is to pick Mumbai — the developer lives there, it floods
every monsoon, and it is easy to motivate in a presentation.

That instinct is the wrong one, for a specific technical reason. The binding
constraint on this project is **labelled ground truth**, not hazard relevance.
A study area with no labels supports no supervised model; it supports only the
transfer of a model trained elsewhere. And Mumbai is, additionally, the *hardest*
SAR flood-mapping case there is: in dense urban fabric, double-bounce scattering
between building walls and standing water *raises* backscatter, inverting the
low-backscatter signature the method depends on.

Picking a study area for convenience and then reporting whatever metrics come
out is how projects end up with an unexplained IoU of 0.4 and no account of why.

## Decision

**No AOI is marked `selected` in Phase 1.** `configs/aoi.yaml` carries four
candidates, all `status: candidate`, each with a written rationale.

Phase 2 opens by *measuring*, per candidate, and printing a comparison table:

1. Labelled ground-truth coverage — dominant criterion
2. Sentinel-1 scene count and actual revisit interval over the window
3. Sentinel-2 usable-scene count after cloud masking
4. Hazard event recurrence from the historical record
5. Exposure (population, built-up area)
6. Tile count against the compute budget

Only then is a candidate promoted, and this ADR is updated with the table and the
decision. `tests/test_geo_aoi.py::test_no_aoi_is_selected_before_phase_2_measures_availability`
fails if an AOI is promoted before that happens.

## Verification result (2026-09-22, Phase 2)

`scripts/verify_sen1floods11.py` was run against the public bucket. Two findings,
one confirming and one correcting.

**CONFIRMED — the India event exists.** 68 hand-labelled Sentinel-1 chips, of
446 in the whole hand-labelled set (15 %). At 512 px / 10 m that is roughly
1,800 km² of expert-annotated water mask. This was the assumption the whole AOI
choice rested on, and it holds.

**CORRECTED — the splits are not region-disjoint.** This ADR previously implied
that Sen1Floods11's official splits would give a clean spatial-generalisation
protocol. They do not: every region except Bolivia appears in train, validation
*and* test at ~60/20/20. India is 40/14/14. These are chip-level stratified
random splits, so chips from the same flood event sit on both sides of the
train/test boundary. See **ADR-009**, which replaces the evaluation protocol with
leave-one-region-out splits built by this project.

**STILL OPEN — does the bbox match?** "India" in a filename confirms the
country, not the basin. `configs/aoi.yaml`'s `bihar_ganga` bbox
(85.0–86.5 E, 25.2–26.2 N) was chosen before the chip locations were known.
`verify_sen1floods11.py --footprints` reads each chip's georeferencing from its
TIFF header over HTTP range requests and prints the union bounding box. **Run it
before Phase 3**, and if the footprints fall outside the configured bbox, replace
the bbox with the measured one and note the change here. Building a preprocessing
pipeline around a footprint that does not contain the labels would waste the
phase.

`mumbai_mmr` is retained deliberately as a **transfer-evaluation target, not a
training region**. Quantifying the degradation of a rural-trained model in dense
urban fabric is a reportable result. Poor IoU there is the finding, not a defect
to be tuned away.

## Alternatives considered

**Commit to Mumbai now.** Rejected: no dense labels, hardest SAR case, and
selection by familiarity is exactly the methodological weakness an examiner
probes first.

**Use a global benchmark only, with no Indian AOI.** Cleanest science, weakest
contribution. Rejected: regional applicability is part of the project's stated
value, and India-specific evaluation is one of its listed contributions.

**Select all four.** Rejected: roughly 1,700 tiles across four AOIs exceeds the
compute budget, and the project needs one AOI done properly before it needs four
done shallowly.

## Consequences

**Buys:** a study area chosen on evidence, a methodology chapter with a
defensible selection section, and a domain-transfer experiment that falls out of
the design for free.

**Costs:** Phase 2 gains a measurement step of a day or two before any modelling
begins, and the preprocessing pipeline must stay AOI-agnostic — which it should
be anyway.
