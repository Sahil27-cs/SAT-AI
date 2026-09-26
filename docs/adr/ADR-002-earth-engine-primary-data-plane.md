# ADR-002: Earth Engine as primary data plane, CDSE as secondary

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 1

## Context

The project runs on a student laptop of modest specification. A single
Sentinel-1 GRD scene is around 1 GB; a Sentinel-2 L2A tile is comparable.
Covering the Bihar candidate AOI (~16,600 km², ~635 tiles at 512 px / 10 m)
across a multi-year window for susceptibility training means terabytes of raw
data if scenes are downloaded whole.

Two free access routes exist, verified 2026-09-22:

- **Google Earth Engine**, noncommercial: 150 EECU-hours per month (Community),
  or 1,000 EECU-hours (Contributor — free, billing account required only for
  identity verification; graduate students and researchers qualify). Processing
  runs on Google's infrastructure; only reduced results are transferred.
- **Copernicus Data Space Ecosystem**, free tier: 10,000 Sentinel Hub processing
  units/month, 10,000 SH requests/month, 12 TB per rolling 30 days, openEO
  10,000 credits/month.

CDSE's allowance is generous, but the data still lands on the laptop, where disk
and RAM are the constraint rather than bandwidth.

## Decision

**Earth Engine is the primary data plane.** Filtering, masking, index
computation, temporal compositing and spatial reduction run server-side; only
analysis-ready arrays and reduced statistics are exported.

**CDSE is the secondary plane**, used for three specific purposes: full-
resolution raw scenes where GEE's preprocessed collections are insufficient
(notably SAR products needing custom processing); independent verification that
results are not artifacts of GEE's particular preprocessing; and the documented
fallback when GEE quota or availability fails.

Both sit behind the same `satai.providers` interface, so neither is load-bearing
in the architecture.

## Alternatives considered

**CDSE only.** Philosophically cleaner — fully open, European, no Google
dependency, and the natural choice for pure Copernicus work. Rejected as
primary: it moves the compute to the weakest machine in the system. The download
and disk cost would dominate the project's schedule.

**Download everything, process locally with rasterio/SNAP.** Maximum control and
the most instructive path. Rejected as the default: at this AOI size and time
depth it is simply not feasible on the available hardware. Retained for the
handful of scenes where custom SAR processing is genuinely needed.

**Microsoft Planetary Computer.** Good STAC catalogue and free Hub compute.
Rejected as primary because its hosted-compute availability has been less
predictable than GEE's noncommercial tier; kept as a documented third fallback.

**AWS Open Data S3 + local Dask.** Excellent for scaling out. Rejected: needs
cluster compute the project does not have.

## Consequences

**Buys:** preprocessing that would take days locally takes minutes; disk stays
inside a student budget; multi-year time series become tractable, which is what
makes the flood-inventory generation in Phase 6 possible at all.

**Costs:** a dependency on a Google service with its own terms and quota, and
GEE's preprocessing choices (for instance its S1 calibration path) become part
of the method and must be stated in the paper rather than glossed over. Export
granularity is limited; very large exports need tiling.

**Mitigation:** every provider is behind an interface, every fetch writes a
manifest, and the CDSE path is exercised at least once per module so the
fallback is known to work rather than assumed to.

**Risk to watch:** quota exhaustion mid-phase. Monitor EECU consumption from
Phase 2; if the Community tier binds, apply for Contributor.

## Sources

- Earth Engine noncommercial tiers — <https://developers.google.com/earth-engine/guides/noncommercial_tiers>
- CDSE quotas — <https://documentation.dataspace.copernicus.eu/Quotas.html>
