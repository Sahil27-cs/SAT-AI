"""SAT-AI serving plane — FastAPI.

Architecture note (ADR-001): this is the **serving plane**. It reads
precomputed artifacts and runs only cheap arithmetic. It never invokes a
segmentation model, never processes a raster and never downloads a satellite
scene. That separation is what lets it answer in under a second on ordinary
serverless infrastructure, and it is why heavy inference belongs in the batch
plane rather than here.

Every response carries provenance (ADR-003): what produced the value, from
which scenes, how old the observation is, and what caveats attach to it. An
endpoint that cannot answer honestly returns 404 with an explanation rather
than a default, a zero or an estimate.

**Window parameters are applied, not echoed.** ``days`` and ``months`` become
real ``gte`` filters on the query. An endpoint that accepted a window,
validated it, and then returned an unrelated fixed slice would be asserting a
filter it had not applied -- the exact failure mode the provenance contract
exists to prevent, committed by the layer that enforces it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

# This module's own directory, on sys.path.
#
# The serving-plane modules import each other by bare name (`from gemini import
# ...`), which is how Vercel's Python runtime has historically laid them out.
# It does not always hold: depending on how the handler is loaded, the project
# root is on sys.path and `api/` is not, and every sibling import then fails
# with ModuleNotFoundError at request time. That is exactly what happened on the
# first Gemini deployment -- /health reported "No module named 'gemini'" while
# the study-area catalogue loaded fine, because the catalogue is read by file
# path and the modules are read by import.
#
# Inserting the directory explicitly makes both layouts work and costs nothing.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import httpx  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402

log = logging.getLogger("satai.api")
logging.basicConfig(level=os.environ.get("SATAI_LOG_LEVEL", "INFO"))

# The catalogue endpoint and its *publishable* key. These are not secrets: a
# Supabase publishable key is designed for untrusted clients, already ships in
# the browser bundle this API serves, and is read-only under row-level security.
# They exist only as a fallback so a deployment without configured environment
# variables degrades to read-only catalogue access rather than to a blank page.
# Environment variables always win. The service-role key is never referenced
# anywhere in this repository, and no write credential has a fallback.
PUBLIC_SUPABASE_URL = "https://akqhuzgekjsvrizysfmp.supabase.co"
PUBLIC_SUPABASE_ANON_KEY = "sb_publishable_ZqxJZMUFFB1LQmcpV92b5w_66qDwpiZ"

SUPABASE_URL = (os.environ.get("SUPABASE_URL") or PUBLIC_SUPABASE_URL).rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or PUBLIC_SUPABASE_ANON_KEY
TABLE_PREFIX = os.environ.get("SUPABASE_TABLE_PREFIX", "satai_")
API_VERSION = "0.5.0"
STARTED_AT = datetime.now(UTC)

if not os.environ.get("SUPABASE_URL"):
    # Said out loud rather than assumed. A deployment silently reading someone
    # else's catalogue is a confusing thing to debug from the outside.
    log.warning(
        "SUPABASE_URL is not set; falling back to the public demo catalogue at %s",
        PUBLIC_SUPABASE_URL,
    )

# CORS. The default is the local dev origin, matching satai.config.APISettings --
# a wildcard default means a deployment that forgets to configure it is open to
# every origin on the web, which is the wrong way round for a default.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

#: Per-instance request budget for the chat endpoint. See ``_rate_limit``.
CHAT_RATE_LIMIT = int(os.environ.get("API_RATE_LIMIT_PER_MINUTE", "20"))

#: Hard ceiling on rows returned by any single query, regardless of the window
#: asked for. A window parameter bounds *time*; this bounds *payload*.
MAX_ROWS = 1000

app = FastAPI(
    title="SAT-AI API",
    version=API_VERSION,
    description=(
        "Serving plane for SAT-AI, a research prototype for multi-hazard risk "
        "assessment from satellite remote sensing. Every value returned carries "
        "provenance. Risk levels are prototype research outputs, NOT official "
        "warnings — official warnings for India come from IMD, NDMA and State "
        "Disaster Management Authorities."
    ),
    docs_url="/docs",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "SAT-AI prototype risk level. Research output from a student research "
    "prototype. This is NOT an official warning and does not replace IMD, NDMA "
    "or State Disaster Management Authority advisories."
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Where a value came from. Attached to every quantitative response."""

    source_kind: str = Field(
        description="observation | model | derived | index | reanalysis | catalogue"
    )
    source_id: str
    version: str = "0.0.0"
    scene_ids: list[str] = Field(default_factory=list)
    observed_at: datetime | None = None
    computed_at: datetime | None = None
    observation_age_hours: float | None = None
    caveats: list[str] = Field(default_factory=list)


class Region(BaseModel):
    id: str
    name: str
    country: str
    bbox: list[float]
    area_km2: float
    utm_epsg: str
    tile_count: int | None = None
    study_role: str
    primary_hazards: list[str]
    label_sources: list[str]
    selection_rationale: str | None = None
    caveats: list[str]


class HazardResult(BaseModel):
    region_id: str
    hazard: str
    risk_index: float
    risk_band: str
    confidence: float | None = None
    flooded_area_km2: float | None = None
    population_exposed: int | None = None
    provenance: Provenance


class Experiment(BaseModel):
    id: str
    name: str
    contribution: str | None = None
    dataset: str | None = None
    split_protocol: str | None = None
    model: str | None = None
    metrics: dict[str, Any]
    status: str
    notes: list[str]
    completed_at: datetime | None = None


class Empty(BaseModel):
    """A deliberate, explained absence of data."""

    available: Literal[False] = False
    reason: str
    what_would_produce_it: str
    region_id: str | None = None
    hazard: str | None = None


class WindowedResponse(BaseModel):
    """Shared shape for the endpoints that take a look-back window.

    ``window_applied`` is the cut-off actually sent to the database, not a
    restatement of the request. A client can compare it against ``window_days``
    and see that the filter was real.
    """

    available: bool
    region_id: str
    window_days: int
    window_applied_from: datetime
    truncated: bool = Field(
        description="True when the row cap was reached, so the window is not fully covered."
    )
    caveats: list[str] = Field(default_factory=list)
    reason: str | None = None


class WeatherResponse(WindowedResponse):
    observations: list[dict[str, Any]]


class HistoricalResponse(WindowedResponse):
    hazard: str
    months: int
    series: list[dict[str, Any]]
    note: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    region: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    agent: str
    route_confidence: float
    route_method: str
    tools_called: list[str]
    grounded: bool
    provenance: list[Provenance]
    degraded: bool = Field(
        description="True when no LLM was reachable and this is tool output only."
    )
    map_actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Map intents the assistant expressed by calling show_on_map: which "
            "region to fly to and which layers to activate. The frontend executes "
            "these, which is what connects the agent to the map."
        ),
    )
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Study-area catalogue
# ---------------------------------------------------------------------------

#: Generated from configs/aoi.yaml by scripts/export_study_areas.py, which
#: writes the same payload here and into the frontend. Read from disk rather
#: than fetched: study areas are configuration and change on a pull request,
#: not on a satellite revisit, so a network round trip would add a failure mode
#: to data that is fixed at deploy time.
_STUDY_AREAS_PATH = Path(__file__).resolve().parent / "study_areas.generated.json"
_STUDY_AREAS_CACHE: list[dict[str, Any]] | None = None


def study_areas() -> list[dict[str, Any]]:
    """The configured study areas. Cached for the life of the instance."""
    global _STUDY_AREAS_CACHE
    if _STUDY_AREAS_CACHE is None:
        try:
            payload = json.loads(_STUDY_AREAS_PATH.read_text(encoding="utf-8"))
            _STUDY_AREAS_CACHE = list(payload["studyAreas"])
        except (OSError, ValueError, KeyError) as exc:
            # An unreadable catalogue disables the agent's region tools rather
            # than taking down every endpoint; /health reports it.
            log.error("study-area catalogue unreadable: %s", exc)
            _STUDY_AREAS_CACHE = []
    return _STUDY_AREAS_CACHE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_timestamp(value: Any) -> datetime | None:
    """Parse a PostgREST timestamp into an aware UTC datetime.

    Returns ``None`` rather than raising. PostgREST normally emits an offset,
    but a column written by a client that dropped it yields a naive string, and
    subtracting that from an aware ``now()`` raises ``TypeError`` deep inside a
    response handler -- a 500 on an otherwise valid row. A timestamp we cannot
    interpret means an age we cannot report, which the schema already models as
    ``None``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            log.warning("unparseable timestamp from database: %r", value)
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def age_hours(observed: datetime | None) -> float | None:
    """Hours since an observation, or ``None`` if it has no usable timestamp."""
    if observed is None:
        return None
    return (datetime.now(UTC) - observed).total_seconds() / 3600.0


async def _query(table: str, params: dict[str, str]) -> list[dict[str, Any]]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "database_not_configured",
                "message": "SUPABASE_URL and SUPABASE_ANON_KEY are not set on this deployment.",
            },
        )
    # PostgREST serves only the schemas on its exposed list, and exposing the
    # whole `satai` schema would publish scenes and chat_turns alongside the
    # catalogue. Instead the database publishes an explicit read-only view
    # surface in `public` under this prefix. The migration that creates it is
    # backend/app/db/migrations/002_public_read_views.sql, so the mapping is
    # reproducible from a clone rather than living only in the hosted project.
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_PREFIX}{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params, headers=headers)
    except httpx.HTTPError as exc:
        # A network failure is a 502 with a named cause, not an opaque 500.
        raise HTTPException(
            status_code=502,
            detail={
                "error": "database_unreachable",
                "message": f"{type(exc).__name__} contacting the catalogue.",
            },
        ) from exc
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={"error": "database_error", "message": response.text[:300]},
        )
    return list(response.json())


def _model_or_502(model: type[BaseModel], row: dict[str, Any], table: str) -> Any:
    """Build a response model from a database row, or fail with a named cause.

    A column renamed or dropped in the database is an upstream contract break,
    not a bug in the request. Letting Pydantic's ValidationError escape turns it
    into an unexplained 500; this reports it as a 502 naming the table, which is
    the difference between a five-minute diagnosis and an hour of guessing.
    """
    try:
        return model(**row)
    except ValidationError as exc:
        log.error("row from %s does not match %s: %s", table, model.__name__, exc)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "schema_mismatch",
                "message": (
                    f"A row from {table!r} does not match the {model.__name__} "
                    f"contract this API serves. The database schema and this "
                    f"deployment are out of step."
                ),
            },
        ) from exc


#: Per-instance sliding window of recent chat request timestamps, keyed by
#: client. This is a *cost guard*, not a security control, and the distinction
#: matters: each serverless instance keeps its own deque, so the effective
#: global limit is this number multiplied by the number of warm instances, and
#: a cold start resets it. It is here to stop one script from draining an
#: Anthropic budget in a loop. Anything stronger belongs at the edge (Vercel
#: firewall / WAF), where the state is actually shared.
_CHAT_CALLS: dict[str, deque[float]] = {}


def _rate_limit(request: Request) -> None:
    """Reject a client that has exceeded ``CHAT_RATE_LIMIT`` in the last minute."""
    client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    calls = _CHAT_CALLS.setdefault(client, deque())
    while calls and now - calls[0] > 60.0:
        calls.popleft()

    if len(calls) >= CHAT_RATE_LIMIT:
        retry_after = int(60.0 - (now - calls[0])) + 1
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "message": (
                    f"More than {CHAT_RATE_LIMIT} chat requests in a minute from "
                    f"this client. The agent endpoint calls a metered language "
                    f"model, so it is rate limited."
                ),
                "retry_after_s": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )
    calls.append(now)

    # Unbounded growth across distinct clients would be a slow leak on a warm
    # instance. Drop windows that are entirely expired.
    if len(_CHAT_CALLS) > 2048:
        for key in [k for k, v in _CHAT_CALLS.items() if not v or now - v[-1] > 60.0]:
            _CHAT_CALLS.pop(key, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["meta"])
async def root() -> dict[str, Any]:
    return {
        "name": "SAT-AI API",
        "version": API_VERSION,
        "description": (
            "Multi-hazard risk assessment from satellite remote sensing (research prototype)."
        ),
        "disclaimer": DISCLAIMER,
        "docs": "/docs",
        "scope": {
            "does": [
                "flood susceptibility and extent detection",
                "fire danger estimation and active-fire observation ingestion",
                "extreme wind and rainfall risk indices",
                "post-event damage assessment",
            ],
            "does_not": [
                "predict earthquakes",
                "forecast cyclone tracks or intensity",
                "predict wildfire ignition",
                "forecast flood timing or depth",
                "issue official warnings",
            ],
        },
    }


@app.get("/health", tags=["meta"])
async def health() -> dict[str, Any]:
    """Dependency-level health, so degradation is observable rather than guessed."""
    checks: dict[str, str] = {"api": "ok"}
    try:
        await _query("regions", {"select": "id", "limit": "1"})
        checks["database"] = "ok"
    except HTTPException as exc:
        checks["database"] = f"unavailable ({exc.status_code})"
    # Imported lazily: /health must answer even if the agent module fails to
    # import, because "the agent is broken" is exactly what it exists to report.
    try:
        from gemini import describe_configuration

        llm = describe_configuration()
        checks["llm"] = "configured" if llm["configured"] else "not_configured"
        checks["llm_provider"] = llm["provider"]
        if llm["model"]:
            checks["llm_model"] = llm["model"]
    except ImportError as exc:  # pragma: no cover - deployment fault
        checks["llm"] = f"unavailable ({exc})"
    checks["study_areas"] = str(len(study_areas()))
    checks["audit_log"] = "enabled" if os.environ.get("SUPABASE_SERVICE_KEY") else "disabled"
    return {
        "status": "ok" if checks.get("database") == "ok" else "degraded",
        "version": API_VERSION,
        # Time since THIS instance started, not since the service was deployed.
        # On serverless every cold start resets it, so it measures instance age
        # and nothing more. Named accordingly so it is not read as uptime.
        "instance_age_s": (datetime.now(UTC) - STARTED_AT).total_seconds(),
        "checks": checks,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/v1/regions", response_model=list[Region], tags=["regions"])
async def list_regions() -> list[Region]:
    rows = await _query("regions", {"select": "*", "order": "id"})
    return [_model_or_502(Region, r, "regions") for r in rows]


@app.get("/api/v1/regions/{region_id}", response_model=Region, tags=["regions"])
async def get_region(region_id: str) -> Region:
    rows = await _query("regions", {"select": "*", "id": f"eq.{region_id}"})
    if not rows:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "region_not_configured",
                "message": (
                    f"{region_id!r} is not a SAT-AI study area. The system covers "
                    f"only its configured regions and does not estimate risk "
                    f"elsewhere."
                ),
            },
        )
    result: Region = _model_or_502(Region, rows[0], "regions")
    return result


async def _hazard(region_id: str, hazard: str) -> HazardResult | Empty:
    await get_region(region_id)
    rows = await _query(
        "latest_hazard_results",
        {"select": "*", "region_id": f"eq.{region_id}", "hazard": f"eq.{hazard}"},
    )
    if not rows:
        return Empty(
            reason=(
                f"No {hazard} analysis has been computed for {region_id}. SAT-AI "
                f"produces results on a batch schedule tied to satellite revisit, "
                f"and this region has no completed run."
            ),
            what_would_produce_it=(
                "Run the Track B acquisition and the flood inference pipeline: "
                "ml/preprocessing/build_aoi_stack.py then ml/flood/infer.py"
            ),
            region_id=region_id,
            hazard=hazard,
        )
    row = rows[0]
    observed = parse_timestamp(row.get("observed_at"))
    return HazardResult(
        region_id=row["region_id"],
        hazard=row["hazard"],
        risk_index=row["risk_index"],
        risk_band=row["risk_band"],
        confidence=row.get("confidence"),
        flooded_area_km2=row.get("flooded_area_km2"),
        population_exposed=row.get("population_exposed"),
        provenance=Provenance(
            source_kind=row.get("source_kind", "model"),
            source_id=row.get("model_version_id") or "unknown",
            version=row.get("model_version_id") or "0.0.0",
            scene_ids=row.get("scene_ids") or [],
            observed_at=observed,
            computed_at=parse_timestamp(row.get("computed_at")),
            observation_age_hours=age_hours(observed),
            caveats=[DISCLAIMER, *(row.get("caveats") or [])],
        ),
    )


# `response_model` is set explicitly on each hazard route. Without it the union
# return annotation leaves the OpenAPI schema for the four most-used endpoints
# untyped, so /docs describes them as `{}` and a generated client gets nothing.
@app.get("/api/v1/flood", response_model=HazardResult | Empty, tags=["hazards"])
async def flood(region: str = Query(...)) -> HazardResult | Empty:
    return await _hazard(region, "flood")


@app.get("/api/v1/wildfire", response_model=HazardResult | Empty, tags=["hazards"])
async def wildfire(region: str = Query(...)) -> HazardResult | Empty:
    return await _hazard(region, "wildfire")


@app.get("/api/v1/cyclone", response_model=HazardResult | Empty, tags=["hazards"])
async def cyclone(region: str = Query(...)) -> HazardResult | Empty:
    return await _hazard(region, "cyclone")


@app.get("/api/v1/damage", response_model=HazardResult | Empty, tags=["hazards"])
async def damage(region: str = Query(...)) -> HazardResult | Empty:
    return await _hazard(region, "damage")


@app.get("/api/v1/risk", tags=["hazards"])
async def risk(region: str = Query(...)) -> dict[str, Any]:
    """Multi-hazard risk vector. Deliberately NOT averaged across hazards.

    Per-hazard scores come from different models with different base rates and
    are not commensurable; their mean would denote nothing (ADR-008).
    """
    await get_region(region)
    rows = await _query("latest_hazard_results", {"select": "*", "region_id": f"eq.{region}"})
    vector = {r["hazard"]: r["risk_index"] for r in rows}
    return {
        "region_id": region,
        "risk_vector": vector,
        "dominant_hazard": max(vector, key=lambda k: vector[k]) if vector else None,
        "available": bool(vector),
        "formulation": "R = H^alpha * E^beta * V^gamma (multiplicative; ADR-008)",
        "note": (
            "Per-hazard scores are NOT averaged: they are outputs of different "
            "models with different base rates and are not commensurable."
        ),
        "disclaimer": DISCLAIMER,
        **(
            {}
            if vector
            else {
                "reason": f"No hazard analyses have been computed for {region}.",
                "what_would_produce_it": "Run the hazard inference pipelines.",
            }
        ),
    }


@app.get("/api/v1/explanation", tags=["explainability"])
async def explanation(region: str = Query(...), hazard: str = Query("flood")) -> dict[str, Any]:
    await get_region(region)
    rows = await _query(
        "explanations",
        {
            "select": "*",
            "region_id": f"eq.{region}",
            "hazard": f"eq.{hazard}",
            "order": "computed_at.desc",
            "limit": "1",
        },
    )
    if not rows:
        return {
            "available": False,
            "region_id": region,
            "hazard": hazard,
            "reason": (
                "No explanation artifact exists for this region and hazard. "
                "Explanations are computed in the batch plane alongside the "
                "prediction, not on request."
            ),
            "what_would_produce_it": "ml/flood/explain.py after a completed inference run",
        }
    row = rows[0]
    return {
        "available": True,
        "region_id": region,
        "hazard": hazard,
        "method": row["method"],
        "version": row["version"],
        "drivers": row["drivers"],
        "caveats": [
            "Attributions describe what the model used, not physical causation.",
            *(row.get("caveats") or []),
        ],
        "computed_at": row["computed_at"],
    }


@app.get("/api/v1/satellite", tags=["provenance"])
async def satellite(region: str = Query(...), hazard: str = Query("flood")) -> dict[str, Any]:
    result = await _hazard(region, hazard)
    if isinstance(result, Empty):
        return result.model_dump()
    return {
        "available": True,
        "region_id": region,
        "hazard": hazard,
        "scene_ids": result.provenance.scene_ids,
        "observed_at": result.provenance.observed_at,
        "observation_age_hours": result.provenance.observation_age_hours,
        "latency_class": "revisit-limited batch",
        "note": (
            "Sentinel-1 revisit is 6-12 days. Satellite-derived extent is "
            "batch analysis, not real-time monitoring."
        ),
    }


@app.get("/api/v1/weather", response_model=WeatherResponse, tags=["observations"])
async def weather(region: str = Query(...), days: int = Query(7, ge=1, le=90)) -> WeatherResponse:
    """Weather observations within the requested look-back window.

    The window is a real filter on ``observed_at``. ``truncated`` says whether
    the row cap was hit before the window was exhausted, so a caller can tell a
    quiet fortnight from a clipped result.
    """
    await get_region(region)
    since = datetime.now(UTC) - timedelta(days=days)
    rows = await _query(
        "observations",
        {
            "select": "*",
            "region_id": f"eq.{region}",
            "kind": "eq.weather",
            "observed_at": f"gte.{since.isoformat()}",
            "order": "observed_at.desc",
            "limit": str(MAX_ROWS),
        },
    )
    return WeatherResponse(
        available=bool(rows),
        region_id=region,
        window_days=days,
        window_applied_from=since,
        truncated=len(rows) >= MAX_ROWS,
        observations=rows,
        caveats=[
            "ERA5 reanalysis has roughly a five-day lag and is retrospective.",
            "GPM IMERG Early is near-real-time at approximately 4 hours latency.",
            *(
                [
                    f"Result capped at {MAX_ROWS} rows; the {days}-day window is "
                    f"not fully represented. Narrow the window for a complete series."
                ]
                if len(rows) >= MAX_ROWS
                else []
            ),
        ],
        reason=(
            None if rows else f"No weather records ingested for {region} in the last {days} days."
        ),
    )


@app.get("/api/v1/historical", response_model=HistoricalResponse, tags=["hazards"])
async def historical(
    region: str = Query(...),
    hazard: str = Query("flood"),
    months: int = Query(12, ge=1, le=120),
) -> HistoricalResponse:
    """Past SAT-AI analyses within the requested window.

    ``months`` is converted to a cut-off on ``computed_at`` and sent to the
    database. Previously this parameter was validated and then discarded, so a
    one-month request and a ten-year request returned the same rows.
    """
    await get_region(region)
    # Calendar months vary in length; 30 days is the documented approximation
    # and is stated in the response so nobody reverse-engineers it from the data.
    since = datetime.now(UTC) - timedelta(days=30 * months)
    rows = await _query(
        "hazard_results",
        {
            "select": "risk_index,risk_band,computed_at,observed_at",
            "region_id": f"eq.{region}",
            "hazard": f"eq.{hazard}",
            "computed_at": f"gte.{since.isoformat()}",
            "order": "computed_at.desc",
            "limit": str(MAX_ROWS),
        },
    )
    return HistoricalResponse(
        available=bool(rows),
        region_id=region,
        hazard=hazard,
        months=months,
        window_days=30 * months,
        window_applied_from=since,
        truncated=len(rows) >= MAX_ROWS,
        series=rows,
        note=(
            "Historical values are previous SAT-AI model outputs, not independent "
            "observations of what occurred."
        ),
        caveats=[
            "A month is treated as 30 days; window_applied_from is the exact cut-off used.",
            *(
                [f"Result capped at {MAX_ROWS} rows; the window is not fully represented."]
                if len(rows) >= MAX_ROWS
                else []
            ),
        ],
        reason=(
            None
            if rows
            else f"No completed {hazard} analyses for {region} in the last {months} months."
        ),
    )


@app.get("/api/v1/experiments", response_model=list[Experiment], tags=["research"])
async def experiments() -> list[Experiment]:
    """The experiment registry — executed results and honest pending states."""
    rows = await _query("experiments", {"select": "*", "order": "id"})
    return [
        _model_or_502(
            Experiment, {k: r[k] for k in Experiment.model_fields if k in r}, "experiments"
        )
        for r in rows
    ]


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["agents"])
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """Conversational interface.

    When no LLM is configured the endpoint degrades to structured tool output
    rather than failing (requirement 39), and says so via ``degraded``. It never
    substitutes generated prose for missing data.

    Rate limited: this route is the only one that spends money per call.
    """
    _rate_limit(http_request)

    # Imported lazily and by bare name: Vercel puts `api/` on sys.path directly.
    # mypy resolves it through the `satai_agents` override in pyproject.toml.
    from satai_agents import answer

    started = time.perf_counter()
    response: ChatResponse = await answer(request)
    await log_chat_turn(request, response, int((time.perf_counter() - started) * 1000))
    return response


# ---------------------------------------------------------------------------
# Agent audit log (C1)
# ---------------------------------------------------------------------------

#: Writing to `satai.chat_turns` needs the service-role key: the table has RLS
#: enabled with no policy, so the anon key cannot reach it by design. Without
#: that key the audit log is disabled and says so via /health, rather than
#: failing every chat request or silently pretending to record.
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


async def log_chat_turn(request: ChatRequest, response: ChatResponse, latency_ms: int) -> None:
    """Record one turn for the C1 grounding measurement.

    The schema calls this table "a research artifact, not an operational log",
    and it is: the grounding-violation rate reported for contribution C1 is a
    count over these rows. A turn that is validated, returned to the browser and
    then discarded contributes nothing to the measurement, which is what was
    happening before this existed.

    **This must never fail a request.** A user's answer does not depend on the
    audit write succeeding, so every failure here is logged and swallowed.
    """
    if not SUPABASE_SERVICE_KEY:
        return
    row = {
        "session_id": request.session_id or "anonymous",
        "query": request.message,
        "routed_agent": response.agent,
        "route_confidence": response.route_confidence,
        "route_method": response.route_method,
        "tools_called": response.tools_called,
        "response": response.answer,
        "grounded": response.grounded,
        # The mirror reports violations as free-text notes; the kinds it can
        # distinguish are recorded here and the rest stay in `notes`.
        "violation_kinds": [
            kind
            for kind, marker in (
                ("ungrounded_value", "Grounding violation"),
                ("fabricated_authority", "fabricated authority"),
                ("observation_prediction_confusion", "observation/prediction"),
                ("missing_caveat", "omits the"),
            )
            if any(marker.lower() in n.lower() for n in response.notes)
        ],
        "regenerated": any("Regenerated once" in n for n in response.notes),
        "latency_ms": latency_ms,
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Posted to the PREFIXED view in `public`, not to `satai.chat_turns`.
            # PostgREST routes only to exposed schemas, and `satai` is
            # deliberately not exposed -- for writes as much as reads -- so a
            # direct post would 404 at the router regardless of which key is
            # held. Migration 002 creates the insert-only view this targets.
            result = await client.post(
                f"{SUPABASE_URL}/rest/v1/{TABLE_PREFIX}chat_turns",
                json=row,
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
        if result.status_code >= 400:
            log.warning("chat_turns write rejected (%s): %s", result.status_code, result.text[:200])
    except httpx.HTTPError as exc:
        log.warning("chat_turns write failed: %s", type(exc).__name__)
