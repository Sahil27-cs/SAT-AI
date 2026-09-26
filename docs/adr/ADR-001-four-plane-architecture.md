# ADR-001: Four-plane architecture separated by latency class

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 1

## Context

SAT-AI has to do four things whose time constants differ by six orders of
magnitude:

| Work | Time constant |
|---|---|
| Acquire and preprocess a Sentinel-1 scene | minutes to tens of minutes |
| Train or batch-infer a segmentation model | minutes to hours |
| Answer an HTTP request | under a second |
| Answer a conversational question | a few seconds |

A Sentinel-1 GRD scene is roughly a gigabyte. Downloading one, applying orbit
correction, calibration, speckle filtering and terrain correction, then running
inference, cannot happen inside a web request. Any design in which a user clicks
"analyse Mumbai" and a scene is fetched synchronously will time out in
development and will not deploy at all on free or low-cost infrastructure.

This is the single most common structural failure in projects of this kind. It
is usually discovered at deployment, when the architecture is too entangled to
fix cheaply.

## Decision

Four planes, separated by latency class, communicating only through durable
artifacts:

1. **Data plane** (hours–days, offline) — acquisition, harmonisation, feature
   stacks. Writes COGs, Parquet and GeoJSON to object storage plus a manifest.
2. **ML plane** (minutes–hours, batch) — training and scheduled inference.
   Reads the data plane's artifacts; writes hazard rasters, explanation JSON and
   metrics.
3. **Serving plane** (sub-second, online) — FastAPI, PostGIS, object storage.
   Reads precomputed artifacts. Runs the risk engine, which is arithmetic and
   therefore cheap. **Never invokes a heavy model in a request.**
4. **Agent plane** (seconds, online) — a router and three agents calling typed
   tools that hit the serving plane.

## Status note (2026-09-22): LangGraph was planned and not adopted

This ADR originally named LangGraph as the agent plane's router, and the
dependency was pinned in `environment.yml` for several phases without ever
being imported. Recorded here rather than quietly deleted, because a pinned
dependency and an architecture diagram are both claims about how the system
works, and this one was not true.

What was built instead: `satai/agents/router.py` is a weighted keyword scorer
with deterministic entity extraction. The reason is requirement 39 — the system
must route when no API key is reachable at all — which means the fallback path
has to be complete on its own. Once it is, a graph framework is orchestrating
three agents and nine tools over a code path that already works, in exchange
for a large dependency tree, a second prompt abstraction and an upgrade
treadmill.

Revisit if the agent plane grows multi-turn state, parallel tool fan-out, or
mid-conversation replanning. None of those are present, and none are on the
register in `docs/evaluation.md`.

## Alternatives considered

**Monolithic FastAPI doing everything on demand.** Simpler to start, and the
usual choice. Rejected: it cannot deploy, it makes results irreproducible
(nothing is versioned or repeatable), and it makes the latency of each stage
unmeasurable — which would make the honest reporting required by the project's
own scope rules impossible.

**Async job queue (Celery/RQ) with live submission.** A user submits an
analysis, polls for completion. Genuinely reasonable, and a plausible future
extension. Rejected for now: it adds a broker, worker deployment and job-state
UI, and the demand pattern here (a handful of study areas, refreshed on a
satellite revisit cycle) is served better by scheduled precomputation. Revisit
if arbitrary user-specified AOIs become a requirement.

**Everything in notebooks, dashboard reads exported files.** Rejected: no API,
no agents, not deployable, not a system.

## Consequences

**Buys:** a genuinely fast API that deploys on free-tier infrastructure;
reproducible results, because every artifact comes from a versioned batch run;
independently measurable latency per stage, which is what allows the honest
"near-real-time for weather-driven indices, revisit-limited batch for
satellite-derived extent" claim instead of a vague "real-time"; and failure
isolation — the LLM going down does not take the maps with it.

**Costs:** results are as fresh as the last batch run, not as fresh as the
request. The UI must therefore display data age everywhere, which is more work
and less impressive-looking — and more honest. Arbitrary user-chosen AOIs are
not supported without a job queue.

**Makes harder later:** true on-demand analysis of a new region. Accepted
deliberately; adding a job queue later is a contained change, whereas
unpicking a monolith is not.
