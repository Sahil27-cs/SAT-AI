# SAT-AI serving plane.
#
# This image contains the serving plane only: FastAPI, httpx, pydantic. It
# reads precomputed artifacts and does cheap arithmetic, so it needs no
# geospatial stack, no CUDA and no model weights, and starts in about a second.
#
# The ML plane is deliberately NOT containerised here. It needs a GPU and a
# conda geospatial stack (GDAL, rasterio, PROJ), and an image that carries a
# model it cannot run is worse than no image: it invites someone to deploy it
# and wonder why inference never completes. Use environment.yml for that plane.

FROM python:3.11-slim

# curl is for the healthcheck below; nothing here compiles, so no build tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so a code change does not invalidate the dependency layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt uvicorn==0.34.0

COPY api/ ./api/

# Run unprivileged. The serving plane writes nothing to disk.
RUN useradd --create-home --uid 10001 satai
USER satai

EXPOSE 8000

# The healthcheck hits /health rather than /, because /health reports the
# database dependency. A container that is up but cannot reach the catalogue is
# degraded, and saying so is the whole point of that endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
