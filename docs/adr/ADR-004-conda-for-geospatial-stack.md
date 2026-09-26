# ADR-004: conda-forge for the geospatial stack, pip for the rest

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 1

## Context

Development happens on Windows. `rasterio`, `geopandas`, `fiona` and `pyproj`
are thin Python bindings over compiled GDAL, GEOS and PROJ libraries. On Windows
these regularly fail at import with DLL-load errors when the wheels' bundled
libraries disagree with each other or with a system GDAL — and the error message
points nowhere useful. It is one of the most common reasons a geospatial project
stalls before it starts.

The machine already has miniconda installed.

## Decision

The entire compiled geospatial stack comes from **conda-forge**, pinned in
`environment.yml`, with `nodefaults` set so the `defaults` channel is never
mixed in (mixing channels is itself a frequent cause of ABI mismatch).

`pip` is used only for packages with no reliable conda-forge build or that move
too quickly for one: PyTorch, `segmentation-models-pytorch`, `earthengine-api`,
`sentinelhub`, `openeo`, `anthropic`, `langgraph`.

`pyproject.toml` declares only pure-Python runtime dependencies, so
`pip install -e .` inside the conda environment never tries to resolve GDAL.

## Alternatives considered

**pip + wheels only.** One tool, one lockfile, and it usually works on Linux.
Rejected: on Windows it is the single most likely thing to consume days of
debugging, and CI passing on Linux would give false confidence.

**Docker for all development.** Reproducible and clean. Rejected as the primary
path: it adds friction to the notebook-and-debugger workflow this project needs
most, and GPU passthrough on Windows is an extra complication. Docker is still
used for the local PostGIS instance and for backend deployment, where its value
is unambiguous.

**WSL2 + Linux pip.** Genuinely good, and a reasonable fallback. Rejected as the
default because it adds a filesystem boundary between the editor and the code
and complicates GPU access. Documented in `docs/setup-windows.md` as plan B.

**pixi / uv.** Faster and increasingly capable. Rejected for now: less
documentation for the geospatial stack specifically, and this project should not
also be an experiment in packaging tooling.

## Consequences

**Buys:** a geospatial stack that imports on the first attempt; reproducibility
via a pinned `environment.yml`; and an escape hatch for fast-moving pip packages.

**Costs:** two package managers, which must not be allowed to fight. The rule is
strict and stated in `environment.yml`: conda first, pip only inside the conda
environment, never `pip install` something conda already provides.

**CI note:** GitHub Actions installs the *pip* subset only. The `satai` core
package must therefore import and pass its tests without GDAL — which is why
`satai/geo/aoi.py` is deliberately dependency-free. Phases that add heavy
dependencies get their own CI jobs.
