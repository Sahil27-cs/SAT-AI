# ADR-011: Google Gemini for the agent plane

- **Status:** Accepted
- **Date:** 2026-09-26
- **Phase:** 10
- **Supersedes:** [ADR-005](ADR-005-model-selection-opus-and-router.md)

## Context

ADR-005 chose `claude-opus-5` for reasoning and `claude-haiku-4-5` for routing,
configured through `satai.config.LLMSettings` and overridable by environment
variable. That decision stands as a record of what was decided and why; the
project has since moved the language layer to Google Gemini.

The move is a **provider** change, not a change to what the agent plane is
allowed to do. Every constraint ADR-005 was written around still holds: no model
identifier at a call site, a deterministic router that works with no network, and
a grounding validator that inspects the generated text against the envelopes the
tools actually returned.

Two things about the serving plane made the port non-trivial:

1. **Bundle size.** The serving plane is a Vercel Python function with a size
   budget. A vendor SDK pulls a dependency tree; the REST API needs `httpx`,
   which the function already has for PostgREST.
2. **Tool calling is not optional here.** The whole architecture depends on the
   model being unable to state a number it did not receive from a tool. A
   provider-agnostic chat shim that degrades tool calling into prose would remove
   the property the project is built to demonstrate.

## Decision

**`gemini-2.5-flash` through the REST API, with native function calling**, via
`backend/api/gemini.py` and `backend/api/agent_tools.py`.

- **No vendor SDK.** One `httpx` call to
  `generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`.
- **No OpenAI-compatibility shim.** Gemini's own `functionDeclarations` /
  `functionCall` / `functionResponse` shape is used directly. The shim's
  translation layer is exactly where a tool result would quietly become a
  suggestion.
- **The key travels in the `x-goog-api-key` header, never in the query string.**
  A key in a URL is a key in every access log, proxy log and error report along
  the path.
- **The key is backend-only**, with no fallback value anywhere in the
  repository: never behind `NEXT_PUBLIC_`, never read by the frontend, never in
  a response body, never in a log record. `tests/test_key_containment.py` sets
  `GEMINI_API_KEY` to a sentinel and asserts the value appears in none of the
  routes, the chat path, the refusal path, the validation errors, the log
  records, or anything the frontend reads.
- **Tool rounds are bounded** at `GEMINI_MAX_TOOL_ROUNDS` (default 3). A
  confused turn must answer or stop; without a bound it loops until the
  serverless function times out, which reads to a user as the system being down.
- **Routing stays heuristic-first.** `satai.agents.router.route_heuristic` needs
  no network and no key, so the serving plane still degrades to structured tool
  output instead of failing — which is the case where someone is most likely to
  be asking something urgent.

## Consequences

**A single model instead of two.** ADR-005's split — a large model for
reasoning, a small one for routing — collapses, because the deterministic router
already handles the classification the small model was there for, and a second
model would add a provider dependency to the step that gates every response.

**Refusals do not depend on the provider.** They are evaluated before any tool
call *and* before any model call, so a system that cannot reach Gemini still
refuses to advise evacuation rather than failing open. This is tested in
`tests/test_backend_agent.py`.

**The deployed validator must not drift from the scored one.** The serving plane
carries a dependency-light copy of the grounding validator, and the C1 number is
measured with `satai.agents.grounding`. That divergence has already cost this
project once. `tests/test_grounding_parity.py` runs a shared corpus through both
and fails if the verdicts differ.

**C1 is still not measured.** The loop, the 28-question benchmark, the four-check
validator and the `chat_turns` audit log all exist; no `GEMINI_API_KEY` is set on
the deployment, so the run has not happened. The register says so rather than
reporting a number.

## Alternatives considered

**Keep Anthropic.** No technical objection — ADR-005's reasoning was sound and
the code was working. Rejected because the project owner requires Gemini.

**A vendor SDK.** Less code to maintain. Rejected on the serverless size budget:
the SDK's dependency tree is larger than the function's entire allowance, and
`httpx` was already present.

**A provider-agnostic abstraction over both.** Attractive in principle.
Rejected: the two providers' tool-calling shapes differ enough that the
abstraction would either leak or flatten, and flattening tool calls into prose
would destroy the grounding property. One provider, used directly, with the
model name in configuration is the smaller risk.
