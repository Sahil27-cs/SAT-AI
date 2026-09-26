"""Route tests for the deployed serving plane.

`backend/api/index.py` serves every number the dashboard and the agent show,
and it had no tests at all. That is how two endpoints came to accept a look-back
window, validate it, and then return an unrelated fixed slice: nothing ever
asserted that the parameter reached the query.

The database is replaced with a recorder that captures the PostgREST parameters
each handler sends. Asserting on those parameters is the point — a test that
only checked the response body would have passed against the broken version,
because the broken version returned perfectly well-formed rows for the wrong
window.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi is a serving-plane dependency"
)
TestClient = fastapi_testclient.TestClient

_BACKEND = Path(__file__).resolve().parents[1] / "backend" / "api"


def _load_api() -> Any:
    """Load the serving plane under the name its own agent module expects.

    `satai_agents` does `from index import ...`, so the module has to be
    registered as `index` rather than under a test-local alias.
    """
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    spec = importlib.util.spec_from_file_location("index", _BACKEND / "index.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["index"] = module
    spec.loader.exec_module(module)
    return module


api = _load_api()


REGION_ROW = {
    "id": "bihar_ganga",
    "name": "Middle Ganga plain, Bihar",
    "country": "India",
    "bbox": [83.0, 24.5, 88.0, 27.0],
    "area_km2": 24500.0,
    "utm_epsg": "EPSG:32645",
    "tile_count": 412,
    "study_role": "training",
    "primary_hazards": ["flood"],
    "label_sources": ["sen1floods11"],
    "selection_rationale": "Recurrent Ganga flooding with hand-labelled coverage.",
    "caveats": ["Prototype output, not an official warning."],
}

HAZARD_ROW = {
    "region_id": "bihar_ganga",
    "hazard": "flood",
    "risk_index": 0.8734,
    "risk_band": "ORANGE",
    "confidence": 0.71,
    "flooded_area_km2": 913.4,
    "population_exposed": 1_240_000,
    "source_kind": "model",
    "model_version_id": "flood_unet@0.3.1",
    "scene_ids": ["S1A_IW_GRDH_1SDV_20240715T003112"],
    "observed_at": "2024-07-15T00:31:12+00:00",
    "computed_at": "2024-07-16T04:00:00+00:00",
    "caveats": ["Urban SAR saturation is a known failure mode."],
}


class Recorder:
    """Stands in for PostgREST and remembers what each handler asked for."""

    def __init__(self, tables: dict[str, list[dict[str, Any]]]) -> None:
        self.tables = tables
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def __call__(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        self.calls.append((table, dict(params)))
        return list(self.tables.get(table, []))

    def params_for(self, table: str) -> dict[str, str]:
        for name, params in self.calls:
            if name == table:
                return params
        raise AssertionError(f"no query was issued against {table!r}; saw {self.calls}")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A client whose database returns the fixture rows above.

    The Anthropic key is cleared and the agent mirror dropped from the module
    cache, so the chat tests exercise the documented degraded path instead of
    making a paid API call on the machine of whoever happens to have a key in
    their environment. A test suite that behaves differently depending on an
    unrelated credential is not a test suite.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    sys.modules.pop("satai_agents", None)

    recorder = Recorder(
        {
            "regions": [REGION_ROW],
            "latest_hazard_results": [HAZARD_ROW],
            "hazard_results": [
                {
                    "risk_index": 0.8734,
                    "risk_band": "ORANGE",
                    "computed_at": "2024-07-16T04:00:00+00:00",
                    "observed_at": "2024-07-15T00:31:12+00:00",
                }
            ],
            "observations": [
                {"kind": "weather", "observed_at": "2024-07-15T00:00:00+00:00", "values": {}}
            ],
            "experiments": [],
            "explanations": [],
        }
    )
    monkeypatch.setattr(api, "_query", recorder)
    api._CHAT_CALLS.clear()
    test_client = TestClient(api.app)
    test_client.recorder = recorder  # type: ignore[attr-defined]
    return test_client


# --- the window bug ---------------------------------------------------------


def test_weather_sends_the_requested_window_to_the_database(client: Any) -> None:
    """The defect: `days` was validated and then never used.

    The response still looked right, so only an assertion on the outgoing query
    catches it.
    """
    response = client.get("/api/v1/weather", params={"region": "bihar_ganga", "days": 3})
    assert response.status_code == 200

    params = client.recorder.params_for("observations")
    assert "observed_at" in params, "the day window never reached the query"
    assert params["observed_at"].startswith("gte.")

    cutoff = datetime.fromisoformat(params["observed_at"].removeprefix("gte."))
    expected = datetime.now(UTC) - timedelta(days=3)
    assert abs((cutoff - expected).total_seconds()) < 60


def test_weather_reports_the_cutoff_it_actually_applied(client: Any) -> None:
    body = client.get("/api/v1/weather", params={"region": "bihar_ganga", "days": 14}).json()
    applied = datetime.fromisoformat(body["window_applied_from"])
    assert abs((datetime.now(UTC) - applied).days - 14) <= 1
    assert body["window_days"] == 14


def test_historical_sends_the_requested_window_to_the_database(client: Any) -> None:
    response = client.get("/api/v1/historical", params={"region": "bihar_ganga", "months": 2})
    assert response.status_code == 200

    params = client.recorder.params_for("hazard_results")
    assert "computed_at" in params, "the month window never reached the query"

    cutoff = datetime.fromisoformat(params["computed_at"].removeprefix("gte."))
    expected = datetime.now(UTC) - timedelta(days=60)
    assert abs((cutoff - expected).total_seconds()) < 120


def test_different_windows_produce_different_queries(client: Any) -> None:
    """The sharpest form of the regression: one month and ten years were identical."""
    client.get("/api/v1/historical", params={"region": "bihar_ganga", "months": 1})
    short = client.recorder.params_for("hazard_results")["computed_at"]

    client.recorder.calls.clear()
    client.get("/api/v1/historical", params={"region": "bihar_ganga", "months": 120})
    long = client.recorder.params_for("hazard_results")["computed_at"]

    assert short != long


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/api/v1/weather", {"region": "bihar_ganga", "days": 0}),
        ("/api/v1/weather", {"region": "bihar_ganga", "days": 91}),
        ("/api/v1/historical", {"region": "bihar_ganga", "months": 0}),
        ("/api/v1/historical", {"region": "bihar_ganga", "months": 121}),
    ],
)
def test_out_of_range_windows_are_rejected(client: Any, path: str, params: dict) -> None:
    assert client.get(path, params=params).status_code == 422


# --- honest absence ---------------------------------------------------------


def test_an_unconfigured_region_is_a_404_with_a_reason(client: Any) -> None:
    client.recorder.tables["regions"] = []
    response = client.get("/api/v1/regions/paris")
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "region_not_configured"


def test_a_hazard_with_no_run_returns_explained_absence_not_a_zero(client: Any) -> None:
    client.recorder.tables["latest_hazard_results"] = []
    body = client.get("/api/v1/flood", params={"region": "bihar_ganga"}).json()

    assert body["available"] is False
    assert "risk_index" not in body, "an absent result must not carry a number at all"
    assert body["what_would_produce_it"]


def test_a_hazard_result_carries_its_provenance_and_age(client: Any) -> None:
    body = client.get("/api/v1/flood", params={"region": "bihar_ganga"}).json()

    assert body["risk_index"] == 0.8734
    assert body["provenance"]["scene_ids"] == ["S1A_IW_GRDH_1SDV_20240715T003112"]
    assert body["provenance"]["observation_age_hours"] > 0
    assert any("NOT an official warning" in c for c in body["provenance"]["caveats"])


def test_the_risk_vector_is_not_averaged(client: Any) -> None:
    body = client.get("/api/v1/risk", params={"region": "bihar_ganga"}).json()
    assert "risk_vector" in body
    assert "mean" not in body and "average" not in body
    assert "NOT averaged" in body["note"]


# --- failure modes that used to be 500s -------------------------------------


def test_a_naive_timestamp_does_not_crash_the_handler(client: Any) -> None:
    """Subtracting a naive datetime from an aware one raises TypeError.

    That used to surface as a 500 on an otherwise valid row.
    """
    client.recorder.tables["latest_hazard_results"] = [
        {**HAZARD_ROW, "observed_at": "2024-07-15T00:31:12"}
    ]
    response = client.get("/api/v1/flood", params={"region": "bihar_ganga"})
    assert response.status_code == 200


def test_an_unparseable_timestamp_yields_an_unknown_age_not_an_error(client: Any) -> None:
    client.recorder.tables["latest_hazard_results"] = [
        {**HAZARD_ROW, "observed_at": "not a timestamp"}
    ]
    body = client.get("/api/v1/flood", params={"region": "bihar_ganga"}).json()
    assert body["provenance"]["observation_age_hours"] is None


def test_a_schema_mismatch_is_a_502_that_names_the_cause(client: Any) -> None:
    """A dropped column upstream is not an internal error in this service."""
    client.recorder.tables["regions"] = [{k: v for k, v in REGION_ROW.items() if k != "area_km2"}]
    response = client.get("/api/v1/regions")
    assert response.status_code == 502
    assert response.json()["detail"]["error"] == "schema_mismatch"


def test_null_optional_columns_are_tolerated(client: Any) -> None:
    """flooded_area_km2 and population_exposed are nullable in the schema."""
    client.recorder.tables["latest_hazard_results"] = [
        {
            **HAZARD_ROW,
            "flooded_area_km2": None,
            "population_exposed": None,
            "confidence": None,
            "scene_ids": None,
            "caveats": None,
        }
    ]
    body = client.get("/api/v1/flood", params={"region": "bihar_ganga"}).json()
    assert body["flooded_area_km2"] is None
    assert body["provenance"]["scene_ids"] == []


# --- cost guard -------------------------------------------------------------


def test_the_chat_endpoint_is_rate_limited(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The only route that spends money per call must not be unbounded."""
    monkeypatch.setattr(api, "CHAT_RATE_LIMIT", 3)
    api._CHAT_CALLS.clear()

    payload = {"message": "What is the flood risk?", "region": "bihar_ganga"}
    codes = [client.post("/api/v1/chat", json=payload).status_code for _ in range(5)]

    assert codes.count(429) >= 1, f"no request was throttled: {codes}"
    throttled = client.post("/api/v1/chat", json=payload)
    assert throttled.status_code == 429
    assert "Retry-After" in throttled.headers


def test_an_oversized_message_is_rejected_before_the_model(client: Any) -> None:
    response = client.post("/api/v1/chat", json={"message": "x" * 2001})
    assert response.status_code == 422


# --- the C1 audit log -------------------------------------------------------


def test_the_audit_log_is_disabled_rather_than_broken_without_a_service_key(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anon key cannot reach chat_turns by design, so absence is the state."""
    monkeypatch.setattr(api, "SUPABASE_SERVICE_KEY", "")
    posted: list[str] = []

    class NoPost:
        async def __aenter__(self) -> NoPost:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def post(self, url: str, **kwargs: Any) -> None:
            posted.append(url)

    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kw: NoPost())
    api._CHAT_CALLS.clear()

    response = client.post("/api/v1/chat", json={"message": "flood risk?", "region": "bihar_ganga"})
    assert response.status_code == 200
    assert posted == [], "nothing should be written without a write credential"
    assert client.get("/health").json()["checks"]["audit_log"] == "disabled"


def test_a_chat_turn_is_recorded_against_the_prefixed_view(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Posting straight at `satai.chat_turns` 404s: PostgREST routes only to
    exposed schemas, and `satai` is deliberately not one. Migration 002 creates
    the insert-only view in `public` that this must target instead."""
    monkeypatch.setattr(api, "SUPABASE_SERVICE_KEY", "sb_secret_test")
    captured: dict[str, Any] = {}

    class Capture:
        async def __aenter__(self) -> Capture:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def post(self, url: str, **kwargs: Any) -> Any:
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return type("R", (), {"status_code": 201, "text": ""})()

    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kw: Capture())
    api._CHAT_CALLS.clear()

    client.post(
        "/api/v1/chat",
        json={"message": "flood risk?", "region": "bihar_ganga", "session_id": "s-1"},
    )

    assert captured["url"].endswith("/rest/v1/satai_chat_turns")
    assert "Content-Profile" not in captured["headers"]
    row = captured["json"]
    assert row["session_id"] == "s-1"
    assert row["query"] == "flood risk?"
    assert row["grounded"] is True
    assert isinstance(row["latency_ms"], int)
    assert "get_location_statistics" in row["tools_called"]


def test_an_audit_write_failure_never_fails_the_users_request(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer does not depend on the bookkeeping succeeding."""
    monkeypatch.setattr(api, "SUPABASE_SERVICE_KEY", "sb_secret_test")

    class Broken:
        async def __aenter__(self) -> Broken:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def post(self, url: str, **kwargs: Any) -> Any:
            raise api.httpx.ConnectError("audit sink unreachable")

    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **kw: Broken())
    api._CHAT_CALLS.clear()

    response = client.post("/api/v1/chat", json={"message": "flood risk?", "region": "bihar_ganga"})
    assert response.status_code == 200
    assert response.json()["answer"]


# --- meta -------------------------------------------------------------------


def test_health_reports_each_dependency(client: Any) -> None:
    body = client.get("/health").json()
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["llm"] in {"configured", "not_configured"}
    assert "instance_age_s" in body, "uptime on serverless is instance age; name it that"


def test_health_degrades_rather_than_failing_when_the_database_is_down(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        raise api.HTTPException(status_code=502, detail={"error": "database_error"})

    monkeypatch.setattr(api, "_query", broken)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert "unavailable" in body["checks"]["database"]


def test_cors_does_not_default_to_a_wildcard() -> None:
    """A deployment that forgets to configure CORS should be closed, not open."""
    assert api.ALLOWED_ORIGINS != ["*"]


def test_the_hazard_routes_publish_a_typed_schema(client: Any) -> None:
    """Without response_model the union return leaves /docs describing `{}`."""
    schema = client.get("/openapi.json").json()
    for route in ("/api/v1/flood", "/api/v1/wildfire", "/api/v1/cyclone", "/api/v1/damage"):
        content = schema["paths"][route]["get"]["responses"]["200"]["content"]
        assert content["application/json"]["schema"], f"{route} has an empty response schema"


def test_the_root_document_states_what_the_system_will_not_do(client: Any) -> None:
    body = client.get("/").json()
    assert "predict earthquakes" in body["scope"]["does_not"]
    assert "NOT an official warning" in body["disclaimer"]
