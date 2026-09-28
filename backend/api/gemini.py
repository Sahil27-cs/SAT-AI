"""Google Gemini client: generation and native function calling.

Talks to the REST API over ``httpx`` rather than through ``google-genai``. The
serving plane is a Vercel function with a bundle-size budget and exactly three
runtime dependencies (fastapi, pydantic, httpx); adding an SDK to issue one
POST would be the largest dependency in the deployment for no behaviour the
endpoint uses. The wire format is stable and documented, and keeping it visible
here means the tool-calling loop is readable rather than hidden behind a
framework.

Native function calling, not the OpenAI-compatibility shim. The compatibility
endpoint drops ``functionResponse`` round-trips into a translation layer and
loses Gemini's own notion of a tool turn; since tool invocation accuracy is a
reported metric (C1), the layer that records it should be the one the model
actually speaks.

**The key never leaves this module's request headers.** It is read from the
environment once, never logged, never echoed into a response, and never sent to
the browser. ``describe_configuration`` exists so /health can report whether a
key is present without revealing anything about it.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

__all__ = [
    "DEFAULT_MODEL",
    "GeminiError",
    "GeminiNotConfigured",
    "call_args",
    "call_gemini",
    "call_name",
    "describe_configuration",
    "is_configured",
]

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

#: Current production default. Overridden by GEMINI_MODEL, which is why the
#: constant is a starting point rather than a hard-coded assumption -- model
#: identifiers are retired on Google's schedule, not this project's, and a
#: deployment whose account has a different model available should not need a
#: code change. `scripts/check_env.py` lists what the configured key can
#: actually reach.
DEFAULT_MODEL = "gemini-3.8-flash"

TIMEOUT_S = 45.0


class GeminiError(RuntimeError):
    """The API was reached and refused, or returned something unusable."""


class GeminiNotConfigured(GeminiError):
    """No API key. Distinct from a failure, because the caller degrades rather
    than errors -- requirement 39: a missing language layer must not break the
    data plane."""


#: Characters that cannot appear in an HTTP header value. A key pasted into a
#: dashboard field picks up a trailing newline often enough that httpx raising
#: `LocalProtocolError` -- which names neither the header nor the cause -- is a
#: predictable and very confusing deployment failure.
_ILLEGAL_IN_HEADER = ("\r", "\n", "\t", "\0")


def _api_key() -> str:
    """The configured key, with copy-paste whitespace removed.

    Stripped rather than used verbatim: a leading or trailing space or newline
    is invisible in a dashboard, makes the key unusable as a header value, and
    produces an error that points nowhere near the cause.
    """
    return os.environ.get("GEMINI_API_KEY", "").strip()


def key_problem() -> str | None:
    """Why the configured key cannot be used, or None if it looks usable.

    Deliberately describes the defect without quoting any part of the key. The
    point is to tell an operator what to fix, not to print a credential into a
    health response or a log line.
    """
    raw = os.environ.get("GEMINI_API_KEY")
    if raw is None or not raw.strip():
        return "GEMINI_API_KEY is not set"
    key = raw.strip()
    if any(ch in key for ch in _ILLEGAL_IN_HEADER):
        return (
            "GEMINI_API_KEY contains a line break or tab, which cannot be sent "
            "as an HTTP header. Re-paste the key as a single line."
        )
    if " " in key:
        return (
            "GEMINI_API_KEY contains a space. Google API keys do not; the value "
            "probably includes a label or was pasted with surrounding text."
        )
    if not key.isascii():
        return "GEMINI_API_KEY contains non-ASCII characters and cannot be sent as a header."
    return None


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def is_configured() -> bool:
    return bool(_api_key())


def describe_configuration() -> dict[str, Any]:
    """What /health may say about the language layer.

    Reports presence and the model identifier, never the key, never a prefix of
    it, never its length. A key fragment in a health response is a key in every
    uptime monitor that polls it.
    """
    return {
        "provider": "google-gemini",
        "configured": is_configured(),
        "model": model_name() if is_configured() else None,
        # Names the defect, never any part of the value. A key that is present
        # but unusable is otherwise indistinguishable from a working one until
        # the first chat turn fails.
        "key_problem": key_problem(),
    }


def _extract(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Pull text and function calls out of one Gemini response.

    A single candidate can carry both -- a sentence of preamble and a tool call
    in the same turn -- so both are returned rather than one being chosen.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        # Usually a safety block or an empty prompt. promptFeedback carries the
        # reason and is worth surfacing: "the model returned nothing" is not a
        # diagnosis.
        feedback = payload.get("promptFeedback", {})
        raise GeminiError(f"no candidate returned; promptFeedback={feedback}")

    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(p["text"] for p in parts if "text" in p)
    # The whole part, not just the inner functionCall. Gemini 3 attaches a
    # `thoughtSignature` as a *sibling* of `functionCall`, and refuses the next
    # turn with "Function call is missing a thought_signature" if the history it
    # receives back has lost it. Reconstructing the part from its pieces is what
    # loses it, so nothing is reconstructed.
    calls = [p for p in parts if "functionCall" in p]
    return text, calls


async def call_gemini(
    contents: list[dict[str, Any]],
    *,
    system_instruction: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """One generation turn.

    Returns ``(text, function_calls, usage)``. The caller drives the tool loop;
    this function does exactly one round trip so that each round is separately
    visible to the C1 audit log.
    """
    key = _api_key()
    if not key:
        raise GeminiNotConfigured("GEMINI_API_KEY is not set on this deployment")

    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if tools:
        body["tools"] = [{"functionDeclarations": tools}]
        # ANY would force a call every turn, including the turn where the model
        # should be writing its final answer from results it already has. AUTO
        # lets it stop calling, which is what makes the loop terminate.
        body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

    # Checked before the request rather than after the failure: httpx reports a
    # malformed header as LocalProtocolError, which names neither the header nor
    # what was wrong with it.
    problem = key_problem()
    if problem is not None:
        raise GeminiNotConfigured(problem)

    return await _post(model_name(), key, body, allow_fallback=True)


#: Google's own wording when an identifier has been retired. Matched rather than
#: guessed at: the 404 body names the replacement model, which is the only
#: authoritative statement of what to use instead.
_RETIRED = "no longer available"


#: Statuses that mean "try again", not "you asked for something impossible".
#: 503 is what a newly-released model returns under load.
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_BACKOFF_S = 1.5


async def _post(
    model: str,
    key: str,
    body: dict[str, Any],
    *,
    allow_fallback: bool,
    attempt: int = 0,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """One call to one model, with a single retry onto the current default.

    Model identifiers are retired on Google's schedule. A deployment whose
    ``GEMINI_MODEL`` names a retired one would otherwise be permanently
    degraded, with the reason visible only in a chat note. It falls back once,
    to the identifier this code was written against, and **records which model
    actually answered** -- silently substituting a model would make the
    provenance of every response a guess.
    """
    url = f"{API_BASE}/models/{model}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.post(
                url,
                # Header rather than ?key=, so the credential never lands in a
                # URL that a proxy, a log line or an error message might keep.
                headers={"x-goog-api-key": key, "content-type": "application/json"},
                json=body,
            )
    except httpx.HTTPError as exc:
        raise GeminiError(f"Gemini unreachable ({type(exc).__name__})") from exc

    if response.status_code >= 400:
        # The body can echo request content; only the status and the API's own
        # short message are propagated, and never the key.
        detail = ""
        try:
            detail = response.json().get("error", {}).get("message", "")[:200]
        except (ValueError, AttributeError):
            detail = ""

        if (
            allow_fallback
            and response.status_code == 404
            and _RETIRED in detail
            and model != DEFAULT_MODEL
        ):
            return await _post(DEFAULT_MODEL, key, body, allow_fallback=False)

        # Capacity, not a defect in the request. Observed repeatedly against a
        # freshly-released model. Retried a few times with backoff, because
        # degrading a question the system could have answered -- and telling the
        # user the language layer is unavailable -- is a worse outcome than
        # waiting two seconds.
        if response.status_code in _TRANSIENT_STATUS and attempt < _MAX_ATTEMPTS - 1:
            await asyncio.sleep(_BACKOFF_S * (2**attempt))
            return await _post(model, key, body, allow_fallback=allow_fallback, attempt=attempt + 1)

        raise GeminiError(f"Gemini returned {response.status_code}: {detail}")

    payload = response.json()
    text, calls = _extract(payload)
    usage = dict(payload.get("usageMetadata", {}))
    # The model that actually produced this text, which is not necessarily the
    # one that was configured.
    usage["model"] = model
    return text, calls, usage


def user_turn(text: str) -> dict[str, Any]:
    return {"role": "user", "parts": [{"text": text}]}


def model_turn(text: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Echo the model's own turn back into the conversation.

    Required by the protocol: a ``functionResponse`` is only valid if the
    matching ``functionCall`` is present in the history. Dropping it produces a
    400 that reads as though the tool result were malformed.

    The parts are replayed exactly as they arrived. Gemini 3 carries a
    ``thoughtSignature`` alongside each ``functionCall``, and rebuilding the part
    from its name and arguments silently discards it -- which the API rejects
    with a message about the tool result rather than about the history.
    """
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"text": text})
    # Verbatim. `calls` are the model's own parts as it sent them, including any
    # `thoughtSignature`, and the protocol requires that signature to come back.
    parts.extend(calls)
    return {"role": "model", "parts": parts or [{"text": ""}]}


def call_name(part: dict[str, Any]) -> str:
    """The tool name inside a function-call part."""
    call: dict[str, Any] = part.get("functionCall") or {}
    return str(call.get("name", ""))


def call_args(part: dict[str, Any]) -> dict[str, Any]:
    """The arguments inside a function-call part."""
    call: dict[str, Any] = part.get("functionCall") or {}
    arguments: dict[str, Any] = call.get("args") or {}
    return arguments


def tool_turn(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """A tool's output, in the shape Gemini expects back."""
    return {
        "role": "user",
        "parts": [{"functionResponse": {"name": name, "response": result}}],
    }
