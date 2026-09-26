# ADR-010: Two preprocessing tracks, and the domain-match risk between them

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 3

## Context

The Phase 1 plan described a single preprocessing pipeline: raw Sentinel-1 GRD
through orbit correction, calibration, speckle filtering and terrain correction,
into a harmonised feature stack. Phase 2's verification makes clear that this is
actually **two pipelines with different inputs and a hard compatibility
requirement between them**.

**Track A — training.** Sen1Floods11 does not ship raw GRD scenes. It ships
446 hand-labelled chips that its authors already calibrated, terrain-corrected
and exported. Training the flood model needs none of the SAR preprocessing chain:
it needs chip loading, normalisation, augmentation and splitting.

**Track B — inference.** Running the trained model over a live AOI *does* start
from raw GRD and needs the full chain.

The requirement that links them is easy to state and easy to get wrong:

> **Track B must produce data whose statistical characteristics match Track A's,
> or the model transfers to nothing.**

A model trained on Sen1Floods11's γ⁰ in dB, with their speckle filter, their
DEM and their nodata convention, and then run on a stack built with different
choices at any of those steps, is being asked to generalise across a domain
shift that nobody measured. It will produce confident, plausible, wrong flood
masks — the exact failure mode this project is built to avoid, arriving through
the back door of a preprocessing mismatch rather than a modelling error.

This risk is not discussed in the Phase 0 corpus. Papers that train on
Sen1Floods11 generally also evaluate on Sen1Floods11, so the question never
arises for them. It arises for SAT-AI because SAT-AI is supposed to run
operationally over an Indian AOI.

## Decision

**Separate the tracks explicitly, and make the compatibility requirement a
tested constraint rather than an assumption.**

1. **Track A is built first and is AOI-independent.** Chip loading, the
   leave-one-region-out splitter (ADR-009), leakage-safe normalisation and band
   math. None of it depends on which AOI is selected, so it proceeds now and
   unblocks Phase 4 immediately.
2. **Track B is built after the AOI is selected**, because a harmonisation
   pipeline needs a footprint, a CRS and a tile grid, and inventing those before
   the AOI decision would mean building the wrong thing.
3. **Track B's specification is derived from Track A, not chosen
   independently.** Before any Track B code is written, the actual Sen1Floods11
   chip characteristics are measured — band count, data type, units, value
   distribution, CRS, nodata convention, pixel size — and recorded. Track B then
   targets those numbers.
4. **A domain-match test gates Phase 5.** After Track B produces its first AOI
   stack, compare its per-band distribution against the Sen1Floods11 training
   distribution (percentiles, mean, variance, and a two-sample statistic). A
   material mismatch is a blocker, not a footnote, and the measured distance
   between the two distributions is reported in the results chapter as part of
   C3's domain-transfer story.
5. **Normalisation statistics travel with the model**, fitted on the training
   fold only and serialised alongside the weights. Track B applies the training
   fold's statistics; it never recomputes its own from AOI data, which would
   quietly erase the very shift the test in (4) exists to detect.

## Alternatives considered

**One pipeline, reprocess Sen1Floods11 from raw GRD ourselves.** The cleanest
answer in principle: both tracks would then share every processing choice by
construction. Rejected: Sen1Floods11's chips are distributed as processed
products, the source scene identifiers and exact processing parameters would
have to be reconstructed for all 446 chips, and any reconstruction error would
corrupt the hand labels' alignment. Cost is high, benefit is uncertain.

**Ignore the mismatch and hope it is small.** What the surrounding literature
effectively does, because the question does not arise when training and testing
on the same dataset. Rejected for the reason above — it converts a measurable
risk into an invisible one.

**Train on our own AOI data instead, avoiding the transfer entirely.** Would
require expert flood annotation the project cannot produce. This is precisely
the label scarcity that made Sen1Floods11 the choice in the first place.

**Domain adaptation (histogram matching, CORAL, adversarial alignment).** A
reasonable Phase 13+ extension, and it would make a good ablation. Rejected as a
v1 dependency: adaptation should be applied after the shift is *measured*, not
instead of measuring it.

## Consequences

**Buys:** Phase 3 starts now instead of waiting on the AOI decision; Phase 4 is
unblocked; and the largest silent risk in the project becomes a number that gets
reported. It also strengthens C3 — the rural→urban transfer gap now sits
alongside a measured preprocessing-domain distance, so a poor Mumbai result can
be attributed rather than just observed.

**Costs:** two pipelines to maintain and keep in agreement, and a gate before
Phase 5 that could fail and force rework of Track B. That is the gate doing its
job.

**Makes harder later:** changing a Track A preprocessing choice after Track B is
built means revisiting both and re-running the domain-match test. Acceptable —
that coupling is real, and pretending otherwise is what this ADR prevents.

## Follow-up

- `scripts/inspect_sen1floods11_chips.py` establishes the Track A
  characteristics from the files themselves.
- The domain-match test is written in Phase 3b and gates Phase 5.
- `docs/limitations.md` carries the risk until the test has actually run.
