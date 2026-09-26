# ADR-005: Opus 5 for reasoning, a small model for routing

- **Status:** **Superseded by [ADR-011](ADR-011-google-gemini-for-the-agent-plane.md)** (2026-09-26)
- **Date:** 2026-09-22
- **Phase:** 1

> The agent plane now runs on Google Gemini with native function calling. This
> record is left as written: it is what was decided on 2026-09-22 and why, and
> the constraints it reasons about — no model identifier at a call site, a
> deterministic router that needs no network, a grounding validator over the
> generated text — all carried over. Read ADR-011 for what is deployed.

## Context

The agent plane does two quite different jobs.

**Routing** is a short classification: which of three agents should handle this
query, and what location, hazard and timeframe does it mention? Input is a
sentence; output is a small structured object.

**Reasoning** is the hard part: interpreting SHAP attributions and modality
contributions into an explanation a non-specialist can act on, while staying
strictly inside what the tool envelopes actually returned.

Using the largest model for both is the obvious default and wastes most of its
cost on the easy half. It also adds latency to the step the user feels first.

## Decision

- **Reasoning: `claude-opus-5`**, temperature 0.2. Low temperature because the
  task is faithful interpretation of structured data, not composition.
- **Routing: `claude-haiku-4-5-20251001`** with structured output.

Both are configured through `satai.config.LLMSettings` and overridable by
environment variable. No model identifier is hard-coded at a call site.

**Model identifiers were verified against the live Anthropic documentation on
2026-09-22, not assumed.** The current catalogue also lists `claude-sonnet-5` and
`claude-fable-5-1`. `tests/test_config.py::test_verified_model_identifiers` pins
the two in use; if it fails after a documentation change, re-verify rather than
editing the expectation to match the code.

## Alternatives considered

**Opus 5 for everything.** Simplest. Rejected: several times the cost and
noticeably more latency on the step that gates every response, for a task a much
smaller model does reliably.

**A small model for everything.** Rejected: explanation quality is a stated
research contribution. Interpreting a SHAP vector into a faithful causal
narrative without overstepping the data is exactly where reasoning capability
shows, and where a weaker model tends to embellish — the failure this project is
built to prevent.

**A local open-weights model (Llama, Mistral) via Ollama.** Attractive: no API
cost, no external dependency, and a good story about self-hosting. Rejected as
primary: the machine has no confirmed GPU, so inference would be slow, and
grounding discipline under tool-use is materially weaker at the sizes that would
fit. Worth revisiting in Phase 19 as an *ablation* — "how does grounding
violation rate change with model size?" would be a genuinely interesting result.

**A rules-based router (keyword matching).** Rejected: brittle on natural
phrasing, and the cost saving over Haiku is negligible.

## Consequences

**Buys:** materially lower cost per conversation; faster first response; and a
provider abstraction that makes swapping models a configuration change.

**Costs:** two models to evaluate rather than one. The router becomes a failure
mode of its own — a misrouted query reaches the wrong agent and the wrong tools.
Router accuracy is therefore measured explicitly in the Phase 11 benchmark
rather than assumed.

**Dependency:** the agent plane requires a reachable Anthropic API. Per the
project's fallback requirement, the ML and serving planes must remain fully
functional without it, returning structured tool output with no natural-language
layer. `AgentError` carries HTTP 503 precisely to signal a degraded optional
layer rather than a broken pipeline.
