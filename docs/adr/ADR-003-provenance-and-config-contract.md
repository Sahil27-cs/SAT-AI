# ADR-003: Provenance envelope and single-source configuration

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 1

## Context

SAT-AI puts a language model in front of scientific outputs. That combination has
exactly one catastrophic failure mode: the model states a number that no part of
the system computed. A user asking "what is the flood risk in Mumbai?" and
receiving a confident, invented `0.82` is worse off than a user who receives
nothing, because an invented number is indistinguishable from a real one.

The usual mitigation is a system prompt instructing the model not to invent
figures. That is a request, not a guarantee, and it cannot be measured.

A second, quieter problem: configuration read ad hoc from `os.environ` across a
codebase leaks credentials into tracebacks and logs, and fails late — forty
minutes into a download loop rather than at startup.

## Decision

**Two contracts, both structural.**

**1. The provenance envelope.** Every quantitative value crossing a plane
boundary is wrapped in `satai.provenance.ProvenanceEnvelope`, carrying the value,
the source kind (observation / model / derived / index / reanalysis / catalogue),
the model identity and version, the specific input scenes, the spatial reference
including native resolution, the temporal validity including acquisition time,
the confidence where one exists, and machine-generated caveats.

The agent plane never receives a bare float. It may rephrase an envelope; it may
not author one. `groundable_values()` yields the numbers a response is permitted
to state, and a validator in the agent graph (Phase 11) re-extracts every number
from generated text and rejects the response if any is untraceable. Violations
are logged and counted, and that count is reported as an evaluation metric.

**2. Single-source configuration.** `satai/config.py` is the only module that
reads the environment. Credentials are `SecretStr`. Configuration is validated
once, at import.

## Alternatives considered

**Prompt instructions alone.** Zero engineering cost. Rejected: unenforceable
and unmeasurable. "We asked it not to" is not a method section.

**Retrieval-augmented generation over a document store.** Reduces invention but
does not eliminate it, and it does not fit: the ground truth here is numeric
model output, not text.

**Post-hoc fact-checking with a second LLM.** Adds cost and latency, and makes
the second model's reliability load-bearing. Rejected in favour of deterministic
numeric extraction, which is cheap and exact.

**Plain dataclasses / dicts for provenance.** Rejected: no validation, and the
immutability that prevents a caveat being silently dropped downstream comes free
with a frozen Pydantic model.

## Consequences

**Buys:** an enforceable — and *measurable* — grounding guarantee, which is one
of the project's stated research contributions rather than a safety afterthought.
A complete audit trail, so any figure in the final report traces to the scene IDs
behind it. Automatic UI honesty, because `TemporalValidity` distinguishes when a
scene was acquired from when it was processed, making "this is nine days old"
unavoidable rather than optional. And credentials that cannot leak through a
`repr`.

**Costs:** verbosity. Every tool must construct envelopes rather than return
floats, which is perhaps a 20–30 % increase in tool-layer code. Accepted
deliberately — it is the part of the system a knowledgeable examiner will probe
hardest.

**Enforced by:** `tests/test_provenance.py`, and by the API schema layer, which
takes envelopes rather than scalars.
