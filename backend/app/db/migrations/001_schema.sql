-- 001_schema.sql — SAT-AI core schema (PostgreSQL + PostGIS)
--
-- Canonical DDL. Applied in filename order, so this file runs first both in
-- the local Docker stack (mounted at /docker-entrypoint-initdb.d) and against a
-- managed Postgres.
--
-- Everything lives in a dedicated `satai` schema rather than in `public`. Two
-- reasons, and the second is the operative one:
--
--   1. This database instance is shared with another application, and a bare
--      `regions` table is a name collision waiting to happen.
--   2. PostgREST publishes only the schemas on its exposed list. Keeping the
--      tables out of `public` means the default exposure is *nothing*, and what
--      the anonymous key can read is stated explicitly in 002 rather than being
--      whatever happened to be created. A read-only surface that has to be
--      opted into is a different security posture from one that has to be
--      remembered to close.
--
-- Design rule from ADR-006: this database stores METADATA and VECTOR results.
-- Raster pixels live in object storage as Cloud-Optimized GeoTIFFs and are
-- referenced here by URI. A single AOI at 10 m would exhaust a free-tier
-- database on its own, and PostGIS raster performance at that volume is poor.
--
-- Design rule from ADR-003: every stored result carries its provenance --
-- which model at which version, which scenes, which git commit, which config
-- hash. A number whose origin is unknown cannot be defended in a viva and
-- cannot be reproduced.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS satai;

-- ---------------------------------------------------------------------------
-- Study areas
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satai.regions (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    country             TEXT NOT NULL DEFAULT 'India',
    geom                GEOMETRY(MultiPolygon, 4326),
    bbox                DOUBLE PRECISION[4] NOT NULL,
    area_km2            DOUBLE PRECISION NOT NULL,
    utm_epsg            TEXT NOT NULL,
    tile_count          INTEGER,
    population          BIGINT,
    -- 'training' | 'transfer_evaluation' | 'candidate'. Carried through to the
    -- API so the interface can say WHY a region is in the study, which is the
    -- difference between a dashboard and a research instrument.
    study_role          TEXT NOT NULL DEFAULT 'candidate',
    primary_hazards     TEXT[] NOT NULL DEFAULT '{}',
    label_sources       TEXT[] NOT NULL DEFAULT '{}',
    selection_rationale TEXT,
    caveats             TEXT[] NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS regions_geom_idx ON satai.regions USING GIST (geom);

-- ---------------------------------------------------------------------------
-- Model registry. Every prediction points at exactly one row here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satai.model_versions (
    id                  TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL,
    version             TEXT NOT NULL,
    hazard              TEXT NOT NULL,
    architecture        TEXT,
    -- The evaluation protocol is stored, not assumed. Sen1Floods11's official
    -- splits are NOT region-disjoint (ADR-009), so a metric is meaningless
    -- without knowing which protocol produced it.
    split_protocol      TEXT,
    test_regions        TEXT[],
    train_regions       TEXT[],
    modalities          TEXT[] NOT NULL DEFAULT '{}',
    metrics             JSONB NOT NULL DEFAULT '{}',
    normalisation_uri   TEXT,
    weights_uri         TEXT,
    git_sha             TEXT,
    config_hash         TEXT,
    trained_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_id, version)
);

-- ---------------------------------------------------------------------------
-- Satellite scenes used by any result
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satai.scenes (
    scene_id        TEXT PRIMARY KEY,
    dataset         TEXT NOT NULL,
    provider        TEXT NOT NULL,
    acquired_at     TIMESTAMPTZ,
    footprint       GEOMETRY(Polygon, 4326),
    cloud_cover     DOUBLE PRECISION,
    orbit_direction TEXT,
    relative_orbit  INTEGER,
    platform        TEXT,
    access_url      TEXT,
    properties      JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS scenes_footprint_idx ON satai.scenes USING GIST (footprint);
CREATE INDEX IF NOT EXISTS scenes_acquired_idx ON satai.scenes (acquired_at DESC);

-- ---------------------------------------------------------------------------
-- Hazard results. One row per (region, hazard, run).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satai.hazard_results (
    id                  BIGSERIAL PRIMARY KEY,
    region_id           TEXT NOT NULL REFERENCES satai.regions(id) ON DELETE CASCADE,
    hazard              TEXT NOT NULL,
    model_version_id    TEXT REFERENCES satai.model_versions(id),

    risk_index          DOUBLE PRECISION NOT NULL CHECK (risk_index BETWEEN 0 AND 1),
    risk_band           TEXT NOT NULL CHECK (risk_band IN ('GREEN','YELLOW','ORANGE','RED')),
    confidence          DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    -- Nullable on purpose: a susceptibility run produces a risk index with no
    -- delineated extent. The API and the agent tools treat a NULL here as an
    -- absent quantity rather than a zero, because "0 km2 flooded" is a
    -- measurement and NULL is the absence of one.
    flooded_area_km2    DOUBLE PRECISION,
    population_exposed  BIGINT,
    changed_structures  INTEGER,

    -- source_kind mirrors satai.provenance.SourceKind. The interface must be
    -- able to tell a measurement from a model output from a composite index,
    -- and that distinction has to survive the database.
    source_kind         TEXT NOT NULL DEFAULT 'model'
                        CHECK (source_kind IN
                               ('observation','model','derived','index','reanalysis','catalogue')),

    extent_geom         GEOMETRY(MultiPolygon, 4326),
    raster_uri          TEXT,          -- COG in object storage (ADR-006)
    explanation_uri     TEXT,

    observed_at         TIMESTAMPTZ,   -- when the satellite looked
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),  -- when we processed it
    scene_ids           TEXT[] NOT NULL DEFAULT '{}',
    caveats             TEXT[] NOT NULL DEFAULT '{}',
    manifest_id         TEXT,
    git_sha             TEXT
);
CREATE INDEX IF NOT EXISTS hazard_region_time_idx
    ON satai.hazard_results (region_id, hazard, computed_at DESC);
CREATE INDEX IF NOT EXISTS hazard_extent_idx ON satai.hazard_results USING GIST (extent_geom);

-- ---------------------------------------------------------------------------
-- Explanations (SHAP / modality attribution)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satai.explanations (
    id                BIGSERIAL PRIMARY KEY,
    hazard_result_id  BIGINT REFERENCES satai.hazard_results(id) ON DELETE CASCADE,
    region_id         TEXT NOT NULL REFERENCES satai.regions(id) ON DELETE CASCADE,
    hazard            TEXT NOT NULL,
    method            TEXT NOT NULL,     -- shap | modality_ablation | occlusion
    version           TEXT NOT NULL,
    drivers           JSONB NOT NULL,    -- [{feature, value, contribution, percentile}]
    caveats           TEXT[] NOT NULL DEFAULT '{}',
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS explanations_region_idx
    ON satai.explanations (region_id, hazard, computed_at DESC);

-- ---------------------------------------------------------------------------
-- Observations: FIRMS detections and weather. Deliberately a separate table
-- from hazard_results, because an observation is not a prediction and merging
-- them would make it possible to confuse the two in a query.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satai.observations (
    id            BIGSERIAL PRIMARY KEY,
    region_id     TEXT NOT NULL REFERENCES satai.regions(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,        -- active_fire | weather | rainfall
    source        TEXT NOT NULL,        -- firms_viirs | era5_land | gpm_imerg
    geom          GEOMETRY(Point, 4326),
    observed_at   TIMESTAMPTZ NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    values        JSONB NOT NULL DEFAULT '{}',
    confidence    TEXT,
    caveats       TEXT[] NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS observations_region_time_idx
    ON satai.observations (region_id, kind, observed_at DESC);

-- ---------------------------------------------------------------------------
-- Experiment registry (requirement 34). Reproducibility lives here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satai.experiments (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    contribution        TEXT,           -- C1 | C2 | C3 | C4
    dataset             TEXT,
    split_protocol      TEXT,
    train_regions       TEXT[],
    val_region          TEXT,
    test_region         TEXT,
    modalities          TEXT[],
    model               TEXT,
    hyperparameters     JSONB NOT NULL DEFAULT '{}',
    seed                INTEGER,
    metrics             JSONB NOT NULL DEFAULT '{}',
    preprocessing_version TEXT,
    model_version       TEXT,
    git_sha             TEXT,
    result_uri          TEXT,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','running','complete','failed')),
    notes               TEXT[],
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- Agent audit log. The grounding-violation rate (C1) is computed from this,
-- so it is a research artifact, not an operational log.
--
-- Written by the serving plane on every chat turn, using the service-role key.
-- It is deliberately NOT reachable with the anonymous key: see 003, where RLS
-- is enabled with no policy. Conversation transcripts are not public data.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS satai.chat_turns (
    id                 BIGSERIAL PRIMARY KEY,
    session_id         TEXT NOT NULL,
    query              TEXT NOT NULL,
    routed_agent       TEXT,
    route_confidence   DOUBLE PRECISION,
    route_method       TEXT,
    tools_called       TEXT[] NOT NULL DEFAULT '{}',
    response           TEXT,
    grounded           BOOLEAN,
    violation_kinds    TEXT[] NOT NULL DEFAULT '{}',
    n_claims_checked   INTEGER,
    regenerated        BOOLEAN NOT NULL DEFAULT false,
    latency_ms         INTEGER,
    model              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_turns_session_idx ON satai.chat_turns (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chat_turns_grounded_idx ON satai.chat_turns (grounded, created_at DESC);

-- ---------------------------------------------------------------------------
-- Convenience view: the latest result per region and hazard.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW satai.latest_hazard_results AS
SELECT DISTINCT ON (region_id, hazard) *
FROM satai.hazard_results
ORDER BY region_id, hazard, computed_at DESC;
