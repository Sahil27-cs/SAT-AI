"""The Gemini API key must not leave the backend. Enforced, not promised.

Every other guarantee in this project is about being honest with numbers. This
one is about not leaking a credential, and it is the only defect here that costs
real money the moment it ships: a key in a response body, in an error message,
in a log line or in the frontend bundle is a key anyone can read and spend.

The rule is checked the only way worth checking it -- by setting the key to a
value that appears nowhere else in the repository and then looking for that
value everywhere the system can emit text:

* every route's response body and headers, on the happy path and on the failure
  paths, because an exception handler that echoes its configuration is the
  classic way a secret escapes;
* the configuration summary the health endpoint serves, which has to report
  whether a key is present without reporting what it is;
* log records, which on a serverless platform go to a dashboard a wider group
  can read than the one holding the key;
* the frontend source tree, statically, because anything reachable from there
  is compiled into a bundle the browser downloads.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi is a serving-plane dependency"
)
TestClient = fastapi_testclient.TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = REPO_ROOT / "backend" / "api"

#: Distinctive on purpose. A grep for this string in a response is unambiguous:
#: nothing else in the repository or in any dependency produces it by accident.
SENTINEL = "gemini-key-sentinel-3f9c2a-do-not-log"

#: Routes with no required parameters. The chat endpoint is driven separately
#: because it takes a body and because it is the one that talks to Gemini.
GET_ROUTES = (
    "/",
    "/health",
    "/api/v1/regions",
    "/api/v1/regions/bihar_ganga",
    "/api/v1/flood?region=bihar_ganga",
    "/api/v1/wildfire?region=bihar_ganga",
    "/api/v1/cyclone?region=bihar_ganga",
    "/api/v1/damage?region=bihar_ganga",
    "/api/v1/risk?region=bihar_ganga",
    "/api/v1/explanation?region=bihar_ganga&hazard=flood",
    "/api/v1/satellite?region=bihar_ganga",
    "/api/v1/weather?region=bihar_ganga",
    "/api/v1/historical?region=bihar_ganga",
    "/api/v1/experiments",
    "/openapi.json",
    "/docs",
)


def _load_api(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the serving plane with the sentinel key in the environment.

    The module cache is cleared first: these modules read the key at import
    time, so a copy imported by an earlier test would be holding the empty
    string and this file would pass without testing anything.
    """
    monkeypatch.setenv("GEMINI_API_KEY", SENTINEL)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "")
    for name in ("index", "satai_agents", "gemini", "agent_tools"):
        sys.modules.pop(name, None)
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))

    spec = importlib.util.spec_from_file_location("index", _BACKEND / "index.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["index"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = _load_api(monkeypatch)

    async def no_database(table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        """No rows. The degraded path is where error text gets constructed, and
        error text is where a configuration dump would appear."""
        return []

    monkeypatch.setattr(module, "fetch_rows", no_database, raising=False)
    return module


@pytest.fixture
def client(api: Any) -> Any:
    # raise_server_exceptions=False so a 500 is returned as a response and can be
    # searched, instead of being re-raised before anything inspects its body.
    return TestClient(api.app, raise_server_exceptions=False)


def _assert_clean(text: str, where: str) -> None:
    assert SENTINEL not in text, f"the Gemini key appears in {where}"
    # A partial leak is still a leak: the first characters of a key are enough
    # to identify a project, and enough to confirm a guess.
    assert SENTINEL[:12] not in text, f"a prefix of the Gemini key appears in {where}"


# --- responses ---------------------------------------------------------------


@pytest.mark.parametrize("route", GET_ROUTES)
def test_no_route_returns_the_key(client: Any, route: str) -> None:
    response = client.get(route)
    _assert_clean(response.text, f"the body of GET {route}")
    _assert_clean(json.dumps(dict(response.headers)), f"the headers of GET {route}")


def test_the_chat_endpoint_does_not_return_the_key(client: Any) -> None:
    """The one endpoint that actually holds the key while it works.

    There is no network here, so this exercises the failure path -- which is the
    path that formats an exception into a response.
    """
    response = client.post("/api/v1/chat", json={"message": "What is the flood extent?"})
    _assert_clean(response.text, "the chat response")


def test_a_refused_question_does_not_return_the_key(client: Any) -> None:
    """Refusals are constructed before any model call, so they are built from
    the configuration the service is holding at the time."""
    response = client.post("/api/v1/chat", json={"message": "Should residents evacuate?"})
    _assert_clean(response.text, "a refusal response")


def test_a_malformed_request_does_not_return_the_key(client: Any) -> None:
    """Validation errors echo the input back. They must not echo anything else."""
    for payload in ({}, {"message": ""}, {"message": "x" * 10_000}, {"wrong": "field"}):
        response = client.post("/api/v1/chat", json=payload)
        _assert_clean(response.text, f"the error for {payload!r}")


def test_health_reports_that_a_key_exists_without_reporting_the_key(client: Any) -> None:
    """Presence is operationally necessary; the value never is."""
    body = client.get("/health").json()
    _assert_clean(json.dumps(body), "the health payload")
    # The whole point of the endpoint is that it still answers the question: a
    # key is present, and which model it is for. Neither is a secret, and
    # without them a deployment that is merely misconfigured is
    # indistinguishable from one that is broken.
    assert body["checks"]["llm"] == "configured"
    assert body["checks"]["llm_provider"] == "google-gemini"


def test_the_configuration_summary_is_presence_and_model_only(api: Any) -> None:
    from gemini import describe_configuration

    summary = describe_configuration()
    _assert_clean(json.dumps(summary), "describe_configuration()")
    assert summary["configured"] is True, "presence must still be reported"
    assert any("model" in key for key in summary), "the model name is safe and useful"


# --- logs --------------------------------------------------------------------


def test_nothing_logs_the_key(client: Any, caplog: pytest.LogCaptureFixture) -> None:
    """On a serverless platform the log stream has a wider audience than the key.

    Driven through the chat path because that is where the key is read, and a
    debug line printing the request it is about to send is the obvious way it
    would end up here.
    """
    with caplog.at_level(logging.DEBUG):
        client.post("/api/v1/chat", json={"message": "Summarise the flood extent."})
        client.get("/health")

    for record in caplog.records:
        _assert_clean(record.getMessage(), f"a log record from {record.name}")
        _assert_clean(str(getattr(record, "__dict__", {})), f"the extras of {record.name}")


# --- the frontend ------------------------------------------------------------


def _frontend_sources() -> list[Path]:
    root = REPO_ROOT / "frontend"
    if not root.is_dir():
        return []
    return [
        path
        for pattern in ("**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx", "**/*.json", "**/*.css")
        for path in root.glob(pattern)
        if "node_modules" not in path.parts and ".next" not in path.parts
    ]


def test_the_frontend_never_reads_the_key() -> None:
    """Anything the frontend reads is compiled into a bundle the browser gets.

    `process.env.GEMINI_API_KEY` in a client component is not merely a leak of
    the value -- Next.js would inline it into the JavaScript it ships.

    Reads, not mentions: the dashboard *names* the variable in the caveat that
    explains why the C1 measurement has not been run, which is exactly the kind
    of honesty this project is for. A blanket ban on the string would have made
    that caveat unwriteable.
    """
    import re

    reads = re.compile(r"(process\.env|import\.meta\.env|env)\s*(\.|\[\s*[\"'])\s*GEMINI_API_KEY")
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _frontend_sources()
        if reads.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, f"frontend sources reading the key: {offenders}"


def test_no_public_environment_variable_is_named_like_a_secret() -> None:
    """`NEXT_PUBLIC_` is a promise that the value is public.

    Naming a secret that way is how a key gets exposed by a rename rather than
    by a mistake at the point of use.
    """
    import re

    pattern = re.compile(r"NEXT_PUBLIC_[A-Z0-9_]*(GEMINI|ANTHROPIC|SERVICE_KEY|SECRET|TOKEN)")
    searched = [
        *_frontend_sources(),
        *(p for p in (REPO_ROOT / ".env.example", REPO_ROOT / "vercel.json") if p.is_file()),
    ]
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in searched
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, f"secret-shaped public variables in: {offenders}"


def test_the_repository_contains_no_gemini_shaped_key() -> None:
    """Google API keys start with `AIza`. Nothing tracked here should match.

    CI runs gitleaks over the full history as well; this is the cheap check that
    fails in a second on the machine where the mistake was made.
    """
    import re
    import subprocess

    listing = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607 - fixed args; git comes from PATH by design
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        pytest.skip("not a git checkout")

    pattern = re.compile(r"AIza[0-9A-Za-z_\-]{30,}")
    offenders = []
    for name in filter(None, listing.stdout.split("\0")):
        path = REPO_ROOT / name
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(name)
    assert not offenders, f"Google-API-key-shaped strings in: {offenders}"
