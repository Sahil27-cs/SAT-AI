"""A key that is present but unusable must say so, without quoting itself.

This file exists because of a real deployment failure. `GEMINI_API_KEY` was set
in production, `/health` reported `llm: configured`, and every chat turn came
back degraded with:

    Gemini unreachable (LocalProtocolError)

`LocalProtocolError` is what httpx raises when the outgoing request cannot be
framed — here, a header value that is not a legal header value. It names neither
the header nor what was wrong with it, so the visible symptom pointed at the
network while the cause was a copy-paste artifact in a dashboard field.

Two behaviours are pinned:

* **Harmless whitespace is absorbed.** A key pasted with a trailing newline is
  the same key. Failing on it would be pedantry at the operator's expense.
* **A genuinely unusable key is diagnosed before the request**, with a message
  that says what to fix — and that never contains any part of the key, because
  this message travels into health responses and logs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = REPO_ROOT / "backend" / "api"

pytest.importorskip("httpx", reason="httpx is a serving-plane dependency")

#: Deliberately NOT shaped like a Google API key.
#:
#: The first draft of this file used an "AIza..."-prefixed string, and
#: `test_the_repository_contains_no_gemini_shaped_key` failed on it -- correctly.
#: A fixture that trips the repository's own credential scanner teaches everyone
#: who meets it to ignore that scanner. Nothing here depends on the prefix: the
#: code under test checks for whitespace and non-ASCII, not for a vendor format.
FAKE_KEY = "test-key-not-a-real-credential-0123456789"


def _load() -> Any:
    if str(_BACKEND) not in sys.path:
        sys.path.append(str(_BACKEND))
    spec = importlib.util.spec_from_file_location("gemini_config_test", _BACKEND / "gemini.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gemini = _load()


# --- what counts as configured ----------------------------------------------


def test_an_unset_key_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert gemini.is_configured() is False
    assert gemini.key_problem() == "GEMINI_API_KEY is not set"


def test_a_whitespace_only_key_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty dashboard field sometimes saves as a space rather than nothing."""
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert gemini.is_configured() is False
    assert gemini.key_problem() == "GEMINI_API_KEY is not set"


def test_a_clean_key_has_no_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    assert gemini.is_configured() is True
    assert gemini.key_problem() is None


# --- paste artifacts ---------------------------------------------------------


@pytest.mark.parametrize(
    "wrapped",
    [
        FAKE_KEY + "\n",
        "\n" + FAKE_KEY,
        " " + FAKE_KEY + " ",
        FAKE_KEY + "\r\n",
        "\t" + FAKE_KEY,
    ],
)
def test_surrounding_whitespace_is_absorbed(monkeypatch: pytest.MonkeyPatch, wrapped: str) -> None:
    """A trailing newline is invisible in a dashboard and changes nothing real.

    Before this, it made the key unusable as a header value and the resulting
    error pointed at the network.
    """
    monkeypatch.setenv("GEMINI_API_KEY", wrapped)
    assert gemini.key_problem() is None
    assert gemini.is_configured() is True


def test_an_embedded_line_break_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza\nrest-of-the-key")
    problem = gemini.key_problem()
    assert problem is not None
    assert "line break" in problem


def test_a_key_pasted_with_its_label_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact mistake that is easy to make when copying from instructions."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini api key-AQ.Ab8RN6notarealvalue")
    problem = gemini.key_problem()
    assert problem is not None
    assert "space" in problem


def test_a_non_ascii_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A smart quote or an en dash picked up from a word processor.

    The character is built with `chr` rather than typed: a literal en dash in
    the source is itself flagged as an ambiguous character, and a lint fix that
    silently turned it into a hyphen would leave this test passing while testing
    nothing.
    """
    en_dash = chr(0x2013)
    monkeypatch.setenv("GEMINI_API_KEY", f"testkey{en_dash}notarealkey")
    problem = gemini.key_problem()
    assert problem is not None
    assert "non-ASCII" in problem


# --- the diagnostic must not become a leak -----------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "AIza\nsecret-tail-value",
        "gemini api key-SECRETVALUE123",
        "testkey" + chr(0x2013) + "SECRETVALUE123",
    ],
)
def test_the_diagnostic_never_quotes_the_key(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """This message goes into /health and into logs.

    A diagnostic that helpfully showed "the key is 'AIza\\nsecret...'" would put
    the credential in every uptime monitor that polls the endpoint.
    """
    monkeypatch.setenv("GEMINI_API_KEY", value)
    problem = gemini.key_problem() or ""
    for fragment in ("SECRETVALUE123", "secret-tail-value", value.strip()):
        assert fragment not in problem


def test_the_configuration_summary_carries_the_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    """`configured: true` alone made a broken key look like a working one."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza\nbroken")
    summary = gemini.describe_configuration()

    assert summary["configured"] is True
    assert summary["key_problem"] is not None
    assert "broken" not in str(summary)


def test_a_healthy_summary_reports_no_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    summary = gemini.describe_configuration()

    assert summary["key_problem"] is None
    assert summary["model"] == "gemini-2.5-flash"
    assert FAKE_KEY not in str(summary)


# --- the call path -----------------------------------------------------------


def test_an_unusable_key_is_refused_before_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checked before httpx is handed the header.

    Otherwise the failure surfaces as a transport error, and an operator reads
    it as "Google is down" rather than "fix the value in the dashboard".

    Driven with `asyncio.run` rather than a plugin: one coroutine does not
    justify a test dependency the rest of the suite has no use for.
    """
    import asyncio

    monkeypatch.setenv("GEMINI_API_KEY", "AIza\nbroken")
    with pytest.raises(gemini.GeminiNotConfigured) as caught:
        asyncio.run(gemini.call_gemini([{"role": "user", "parts": [{"text": "hello"}]}]))
    assert "line break" in str(caught.value)


# --- a retired model identifier ----------------------------------------------


class _Response:
    """Enough of an httpx response for the code under test."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _Client:
    """Records which model each request was addressed to."""

    # Class-level on purpose: the code under test constructs its own client, so
    # the recorder cannot be passed in. Reset by `_run_with_stub` before each use.
    calls: ClassVar[list[str]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    #: Models the stub answers 429 for, as a spent daily quota does.
    exhausted: ClassVar[set[str]] = set()

    async def post(self, url: str, **kwargs: Any) -> _Response:
        model = url.rsplit("/models/", 1)[1].split(":")[0]
        _Client.calls.append(model)
        if model in _Client.exhausted:
            return _Response(429, {"error": {"message": "You exceeded your current quota"}})
        if model == "gemini-2.5-flash":
            return _Response(
                404,
                {
                    "error": {
                        "message": (
                            "This model models/gemini-2.5-flash is no longer available "
                            "to new users. Please update your code to use "
                            "models/gemini-3.8-flash"
                        )
                    }
                },
            )
        return _Response(
            200,
            {
                "candidates": [{"content": {"parts": [{"text": "answered"}]}}],
                "usageMetadata": {"totalTokenCount": 11},
            },
        )


def _run_with_stub(monkeypatch: pytest.MonkeyPatch, configured_model: str) -> tuple[str, dict]:
    import asyncio

    _Client.calls = []
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", configured_model)
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    monkeypatch.setattr(gemini.httpx, "AsyncClient", _Client)
    # Backoff between transient retries is real time; the stub needs none.
    monkeypatch.setattr(gemini, "_BACKOFF_S", 0.0)
    text, _calls, usage = asyncio.run(
        gemini.call_gemini([{"role": "user", "parts": [{"text": "hi"}]}])
    )
    return text, usage


def test_a_retired_model_falls_back_once_to_the_fallback_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment naming a retired model would otherwise stay degraded forever.

    Google retires identifiers on its own schedule. The 404 body names the
    replacement, so the failure is recoverable -- but only once, and only onto
    the configured fallback. This key has been told exactly this about
    gemini-2.5-flash, which is also the configured default.
    """
    text, usage = _run_with_stub(monkeypatch, "gemini-2.5-flash")

    assert text == "answered"
    assert _Client.calls == ["gemini-2.5-flash", gemini.FALLBACK_MODEL]
    assert usage["fallback_from"] == "gemini-2.5-flash"
    assert usage["fallback_reason"] == "retired"


def test_the_model_that_actually_answered_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently substituting a model would make every answer's provenance a guess.

    The C1 measurement is a claim about a named model. If the configured one is
    retired and another answers, the artifact has to say which.
    """
    _text, usage = _run_with_stub(monkeypatch, "gemini-2.5-flash")
    assert usage["model"] == gemini.FALLBACK_MODEL
    assert usage["model"] != "gemini-2.5-flash"


def test_a_working_model_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback must not fire on a model that answered."""
    _text, usage = _run_with_stub(monkeypatch, gemini.FALLBACK_MODEL)
    assert _Client.calls == [gemini.FALLBACK_MODEL]
    assert usage["model"] == gemini.FALLBACK_MODEL
    assert "fallback_from" not in usage


def test_a_spent_quota_falls_back_once_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quota is counted per model, so the other model can still answer.

    The primary is retried first (a per-minute limit clears), then the fallback
    is tried once, and the usage records which model answered and why.
    """
    monkeypatch.setattr(_Client, "exhausted", {"gemini-custom-flash"})
    _text, usage = _run_with_stub(monkeypatch, "gemini-custom-flash")

    assert _Client.calls == ["gemini-custom-flash"] * gemini._MAX_ATTEMPTS + [gemini.FALLBACK_MODEL]
    assert usage["model"] == gemini.FALLBACK_MODEL
    assert usage["fallback_reason"] == "quota"


def test_a_spent_quota_on_both_models_is_an_error_not_a_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_Client, "exhausted", {"gemini-custom-flash", gemini.FALLBACK_MODEL})
    with pytest.raises(gemini.GeminiError, match="429"):
        _run_with_stub(monkeypatch, "gemini-custom-flash")
    assert _Client.calls.count(gemini.FALLBACK_MODEL) == gemini._MAX_ATTEMPTS


# --- generation settings -----------------------------------------------------


def test_the_default_model_is_gemini_2_5_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert gemini.DEFAULT_MODEL == "gemini-2.5-flash"
    assert gemini.model_name() == "gemini-2.5-flash"


def test_generation_settings_default_to_the_documented_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "GEMINI_TEMPERATURE",
        "GEMINI_MAX_OUTPUT_TOKENS",
        "GEMINI_TIMEOUT_S",
        "GEMINI_MAX_TOOL_ROUNDS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert gemini.generation_settings() == {
        "temperature": 0.2,
        "max_output_tokens": 1200,
        "timeout_s": 45.0,
        "max_tool_rounds": 3,
    }


def test_generation_settings_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.5")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "800")
    monkeypatch.setenv("GEMINI_TIMEOUT_S", "30")
    monkeypatch.setenv("GEMINI_MAX_TOOL_ROUNDS", "2")
    assert gemini.generation_settings() == {
        "temperature": 0.5,
        "max_output_tokens": 800,
        "timeout_s": 30.0,
        "max_tool_rounds": 2,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GEMINI_MAX_TOOL_ROUNDS", "0"),
        ("GEMINI_MAX_TOOL_ROUNDS", "50"),
        ("GEMINI_TEMPERATURE", "hot"),
        ("GEMINI_MAX_OUTPUT_TOKENS", "-5"),
    ],
)
def test_an_out_of_range_setting_falls_back_to_its_default(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """A dashboard typo must not take the chat endpoint down, nor unbound the loop."""
    monkeypatch.setenv(name, value)
    settings = gemini.generation_settings()
    assert settings["max_tool_rounds"] == 3
    assert settings["temperature"] == 0.2
    assert settings["max_output_tokens"] == 1200


def test_the_configuration_summary_never_carries_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    summary = gemini.describe_configuration()
    assert summary["max_tool_rounds"] == 3
    assert summary["fallback_model"] == gemini.fallback_model()
    assert FAKE_KEY not in json.dumps(summary)


# --- the tool loop's thought signature ---------------------------------------


def test_a_function_call_part_keeps_its_thought_signature() -> None:
    """Gemini 3 refuses the next turn without it.

    The signature sits beside `functionCall` in the part, not inside it, so
    rebuilding the part from a name and arguments discards it -- and the API
    rejects the result with a message about the tool response, pointing away
    from the actual cause.
    """
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "looking that up"},
                        {
                            "functionCall": {"name": "get_flood", "args": {"region": "bihar"}},
                            "thoughtSignature": "SIG-abc123",
                        },
                    ]
                }
            }
        ]
    }
    text, calls = gemini._extract(payload)

    assert text == "looking that up"
    assert len(calls) == 1
    assert calls[0]["thoughtSignature"] == "SIG-abc123"


def test_the_replayed_model_turn_carries_the_signature_through() -> None:
    call_part = {
        "functionCall": {"name": "get_flood", "args": {"region": "bihar"}},
        "thoughtSignature": "SIG-abc123",
    }
    turn = gemini.model_turn("looking that up", [call_part])

    assert turn["role"] == "model"
    signatures = [p.get("thoughtSignature") for p in turn["parts"] if "functionCall" in p]
    assert signatures == ["SIG-abc123"]


def test_the_accessors_read_a_whole_part() -> None:
    """The loop stopped indexing the call directly once the part became the unit."""
    part = {
        "functionCall": {"name": "get_flood", "args": {"region": "bihar_ganga"}},
        "thoughtSignature": "SIG",
    }
    assert gemini.call_name(part) == "get_flood"
    assert gemini.call_args(part) == {"region": "bihar_ganga"}


def test_the_accessors_tolerate_a_call_with_no_arguments() -> None:
    """Gemini omits `args` entirely for a no-argument tool."""
    part = {"functionCall": {"name": "list_study_areas"}}
    assert gemini.call_name(part) == "list_study_areas"
    assert gemini.call_args(part) == {}


# --- what a 429 says about itself ----------------------------------------------


def _quota_body(quota_id: str, delay: str) -> dict[str, Any]:
    return {
        "error": {
            "message": "You exceeded your current quota",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": quota_id}],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
            ],
        }
    }


class _QuotaClient(_Client):
    """Answers 429 with a chosen body for the first `refusals` calls to `model`."""

    body: ClassVar[dict[str, Any]] = {}
    refusals: ClassVar[int] = 0
    target: ClassVar[str] = ""

    async def post(self, url: str, **kwargs: Any) -> _Response:
        model = url.rsplit("/models/", 1)[1].split(":")[0]
        _Client.calls.append(model)
        if model == _QuotaClient.target and _QuotaClient.refusals > 0:
            _QuotaClient.refusals -= 1
            return _Response(429, _QuotaClient.body)
        return _Response(200, {"candidates": [{"content": {"parts": [{"text": "answered"}]}}]})


def _run_quota(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any], refusals: int
) -> tuple[dict[str, Any], list[float]]:
    import asyncio

    waits: list[float] = []

    async def no_wait(seconds: float) -> None:
        waits.append(seconds)

    _Client.calls = []
    monkeypatch.setattr(_QuotaClient, "body", body)
    monkeypatch.setattr(_QuotaClient, "refusals", refusals)
    monkeypatch.setattr(_QuotaClient, "target", "gemini-custom-flash")
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    monkeypatch.setattr(gemini.httpx, "AsyncClient", _QuotaClient)
    monkeypatch.setattr(gemini.asyncio, "sleep", no_wait)
    _text, _calls, usage = asyncio.run(
        gemini.call_gemini([{"role": "user", "parts": [{"text": "hi"}]}])
    )
    return usage, waits


def test_a_daily_quota_goes_straight_to_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying a spent daily quota only burns the turn's 60 s."""
    body = _quota_body("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "40s")
    usage, waits = _run_quota(monkeypatch, body, refusals=99)
    assert _Client.calls == ["gemini-custom-flash", gemini.FALLBACK_MODEL]
    assert waits == []
    assert usage["fallback_reason"] == "daily quota"


def test_a_short_per_minute_wait_is_honoured_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """1.5 s and 3 s of backoff end long before a per-minute window does."""
    body = _quota_body("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "8s")
    usage, waits = _run_quota(monkeypatch, body, refusals=1)
    assert _Client.calls == ["gemini-custom-flash", "gemini-custom-flash"]
    assert waits == [8.5]
    assert usage["model"] == "gemini-custom-flash"


def test_a_long_per_minute_wait_is_not_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _quota_body("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "50s")
    usage, waits = _run_quota(monkeypatch, body, refusals=99)
    assert all(w < 50 for w in waits)
    assert usage["model"] == gemini.FALLBACK_MODEL


def test_the_quota_error_says_which_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _quota_body("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "40s")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-custom-flash")
    with pytest.raises(gemini.GeminiError, match="daily quota exhausted"):
        _run_quota_no_fallback_env(monkeypatch, body)


def _run_quota_no_fallback_env(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> None:
    import asyncio

    async def no_wait(seconds: float) -> None:
        return None

    _Client.calls = []
    monkeypatch.setattr(_QuotaClient, "body", body)
    monkeypatch.setattr(_QuotaClient, "refusals", 99)
    monkeypatch.setattr(_QuotaClient, "target", "gemini-custom-flash")
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-flash")
    monkeypatch.setattr(gemini.httpx, "AsyncClient", _QuotaClient)
    monkeypatch.setattr(gemini.asyncio, "sleep", no_wait)
    asyncio.run(gemini.call_gemini([{"role": "user", "parts": [{"text": "hi"}]}]))
