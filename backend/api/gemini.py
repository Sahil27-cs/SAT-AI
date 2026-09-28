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

import os
from typing import Any

import httpx

__all__ = [
    "DEFAULT_MODEL",
    "GeminiError",
    "GeminiNotConfigured",
    "call_gemini",
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
DEFAULT_MODEL = "gemini-2.5-flash"

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
    calls = [p["functionCall"] for p in parts if "functionCall" in p]
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

    url = f"{API_BASE}/models/{model_name()}:generateContent"
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
        raise GeminiError(f"Gemini returned {response.status_code}: {detail}")

    payload = response.json()
    text, calls = _extract(payload)
    usage = payload.get("usageMetadata", {})
    return text, calls, usage


def user_turn(text: str) -> dict[str, Any]:
    return {"role": "user", "parts": [{"text": text}]}


def model_turn(text: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Echo the model's own turn back into the conversation.

    Required by the protocol: a ``functionResponse`` is only valid if the
    matching ``functionCall`` is present in the history. Dropping it produces a
    400 that reads as though the tool result were malformed.
    """
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"text": text})
    parts.extend({"functionCall": call} for call in calls)
    return {"role": "model", "parts": parts or [{"text": ""}]}


def tool_turn(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """A tool's output, in the shape Gemini expects back."""
    return {
        "role": "user",
        "parts": [{"functionResponse": {"name": name, "response": result}}],
    }
