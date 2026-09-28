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
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = REPO_ROOT / "backend" / "api"

pytest.importorskip("httpx", reason="httpx is a serving-plane dependency")

#: Distinctive, and shaped like a Google API key without being one.
FAKE_KEY = "AIzaSyTESTONLYnotarealkey0123456789abcd"


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
    monkeypatch.setenv("GEMINI_API_KEY", f"AIzaSy{en_dash}notarealkey")
    problem = gemini.key_problem()
    assert problem is not None
    assert "non-ASCII" in problem


# --- the diagnostic must not become a leak -----------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "AIza\nsecret-tail-value",
        "gemini api key-SECRETVALUE123",
        "AIzaSy" + chr(0x2013) + "SECRETVALUE123",
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
