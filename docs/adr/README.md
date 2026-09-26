# Architecture Decision Records

An ADR is a short, dated note recording one decision: what was decided, why, what
was rejected, and what it costs.

## Why this project keeps them

Two reasons, one practical and one academic.

**Practical.** Six months into a project, nobody remembers why the flood model is
a U-Net rather than a Siamese network, and the temptation is to re-litigate it
from scratch. An ADR ends that conversation in thirty seconds.

**Academic.** When an examiner asks "why did you choose X?", the answer is
ADR-00n. Written incrementally, these records become roughly sixty per cent of
the methodology chapter. Written retrospectively the night before submission,
they become a reconstruction — and they read like one.

## Rules

- One decision per record. Numbered sequentially, never renumbered.
- Written **when the decision is made**, not afterwards.
- Superseded records are never deleted. Mark them `Superseded by ADR-0NN` and
  keep them. The fact that a decision was reversed is itself a finding.
- Every record names what was rejected and why. A record with no rejected
  alternative is not a decision, it is a description.

## Template

```markdown
# ADR-0NN: <title>

- **Status:** Proposed | Accepted | Superseded by ADR-0NN
- **Date:** YYYY-MM-DD
- **Phase:** N

## Context
What forces the decision. Constraints, requirements, what is actually true.

## Decision
What we are doing. One paragraph.

## Alternatives considered
Each with the reason it was not chosen.

## Consequences
What this buys, what it costs, and what it makes harder later.
```

## Index

| ADR | Title | Status | Phase |
|-----|-------|--------|-------|
| [001](ADR-001-four-plane-architecture.md) | Four-plane architecture separated by latency class | Accepted | 1 |
| [002](ADR-002-earth-engine-primary-data-plane.md) | Earth Engine as primary data plane, CDSE as secondary | Accepted | 1 |
| [003](ADR-003-provenance-and-config-contract.md) | Provenance envelope and single-source configuration | Accepted | 1 |
| [004](ADR-004-conda-for-geospatial-stack.md) | conda-forge for the geospatial stack, pip for the rest | Accepted | 1 |
| [005](ADR-005-model-selection-opus-and-router.md) | Opus 5 for reasoning, a small model for routing | **Superseded by 011** | 1 |
| [006](ADR-006-rasters-in-object-storage.md) | Rasters as COGs in object storage, not in PostGIS | Accepted | 1 |
| [007](ADR-007-aoi-selection-deferred.md) | AOI selection deferred to measured evidence in Phase 2 | Accepted | 1 |
| [008](ADR-008-multiplicative-risk-formulation.md) | Multiplicative hazard–exposure–vulnerability risk | Accepted | 1 |
| [009](ADR-009-evaluation-splits.md) | Leave-one-region-out splits, not Sen1Floods11's official ones | Accepted | 2 |
| [010](ADR-010-two-track-preprocessing.md) | Two preprocessing tracks, and the domain-match risk between them | Accepted | 3 |
| [011](ADR-011-google-gemini-for-the-agent-plane.md) | Google Gemini with native function calling, REST not SDK | Accepted | 10 |
