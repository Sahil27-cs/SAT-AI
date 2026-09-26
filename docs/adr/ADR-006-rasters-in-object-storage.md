# ADR-006: Rasters as COGs in object storage, not in PostGIS

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 1

## Context

SAT-AI produces raster outputs: flood probability surfaces, susceptibility maps,
burn severity, risk grids. A single AOI at 10 m generates hundreds of megabytes
to a few gigabytes per run, and runs accumulate as models are versioned.

PostGIS supports a `raster` type, and a managed Postgres free tier (Neon,
Supabase) typically allows a few hundred megabytes to a few gigabytes of storage
in total.

## Decision

Rasters are written as **Cloud-Optimized GeoTIFFs** to object storage
(Cloudflare R2, Supabase Storage, or the local filesystem in development). The
database stores metadata and a URI: model run, timestamps, footprint geometry,
band descriptions, statistics — never the pixels.

Vector outputs small enough to be queried spatially (flood extent polygons,
fire detections, damage points, admin aggregates) **do** live in PostGIS, where
GIST-indexed spatial queries are the whole point.

## Alternatives considered

**PostGIS raster.** Keeps everything in one place and allows raster–vector joins
in SQL. Rejected: it would exhaust a free-tier database with a single AOI, and
PostGIS raster performance at this volume is poor. It also foregoes COG's
defining advantage — HTTP range requests, which let the frontend fetch only the
tiles in view.

**Rasters on the API server's local disk.** Simplest. Rejected: ephemeral
filesystems on the target hosts (Hugging Face Spaces, Render, Fly.io) mean data
vanishes on redeploy, and it does not scale past one instance.

**Everything as pre-rendered PNG tiles.** Fast to serve. Rejected as the sole
format: PNG discards the underlying values, so the API could no longer report
"flood probability at this point is 0.87 (95 % CI 0.71–0.93)" — the provenance
contract requires the numbers, not a picture of them. PNG/PMTiles are generated
*in addition*, for map display.

## Consequences

**Buys:** a database that stays inside a free tier; partial reads via HTTP range
requests, so the map fetches only visible tiles; and storage that scales
independently of the database.

**Costs:** two storage systems to keep consistent. Mitigated by making the
`model_runs` row the single source of truth — a raster with no row is garbage to
be collected, and a row whose URI does not resolve is a failing health check.

**Operational note:** object storage has its own cost and egress model. For the
demo scale (a handful of AOIs, a few model versions) this sits inside free
allowances, but it must be monitored rather than assumed — the same discipline
applied to GEE quota in ADR-002.
