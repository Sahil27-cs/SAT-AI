"""Centralised configuration for SAT-AI.

This module is the **single place in the library and the ML plane** that reads
environment variables. Every other module there imports ``get_settings()``.
That rule exists for three reasons:

1. Secrets cannot leak into logs or tracebacks from scattered ``os.environ``
   calls -- credentials here are wrapped in :class:`pydantic.SecretStr`, whose
   ``repr`` is ``**********``.
2. Configuration is validated once, at startup, with clear error messages,
   instead of failing deep inside a data-download loop 40 minutes in.
3. Tests can override configuration without monkey-patching the environment.

**The one exception, stated rather than glossed.** ``backend/api/`` reads
``os.environ`` directly. Importing this module there would pull
``pydantic-settings`` into a serverless bundle that has a size budget, for
configuration the serving plane resolves once at cold start and never
revalidates. The classes here still model those variables --
:class:`SupabaseSettings` in particular -- so the names and defaults have one
definition even though the deployed code does not import them. Anything that
drifts between the two is a bug in this file's favour: `.env.example` documents
both readers.

See ``docs/adr/ADR-003-provenance-and-config-contract.md``.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, resolved from this file's location: satai/config.py -> ../
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    """Log rendering mode."""

    CONSOLE = "console"
    JSON = "json"


class _Base(BaseSettings):
    """Shared settings behaviour: read .env, ignore unrelated env vars."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class LLMSettings(_Base):
    """Google Gemini configuration for the agent plane.

    Gemini rather than a hosted Anthropic model: the agent needs native
    function calling with a tool-result round trip, which is what makes the
    tool-invocation accuracy reported for C1 measurable on the same channel the
    model actually speaks.

    The model identifier is configurable and the constant below is a starting
    point, not an assumption. Model names are retired on the provider's
    schedule, and a deployment whose account exposes a different one should not
    need a code change -- ``scripts/check_env.py`` lists what the configured key
    can actually reach.

    The key is backend-only. It is never read by the frontend, never placed
    behind a NEXT_PUBLIC_ prefix, and never included in a health response.
    """

    provider: Literal["google"] = Field(default="google", alias="LLM_PROVIDER")
    api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, alias="GEMINI_TEMPERATURE")
    max_output_tokens: int = Field(default=1200, gt=0, le=65_536, alias="GEMINI_MAX_OUTPUT_TOKENS")
    timeout_s: float = Field(default=45.0, gt=0, alias="GEMINI_TIMEOUT_S")
    max_tool_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        alias="GEMINI_MAX_TOOL_ROUNDS",
        description=(
            "Tool-calling rounds before the model must answer. Bounds a confused "
            "turn that would otherwise loop until the function times out."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_configured(self) -> bool:
        """True when an API key is present.

        The ML pipeline must remain fully functional when this is False
        (requirement 39: LLM unavailable must degrade, not break).
        """
        return self.api_key is not None and bool(self.api_key.get_secret_value())


class EarthEngineSettings(_Base):
    """Google Earth Engine -- the primary data plane (ADR-002)."""

    project_id: str | None = Field(default=None, alias="GEE_PROJECT_ID")
    service_account_email: str | None = Field(default=None, alias="GEE_SERVICE_ACCOUNT_EMAIL")
    service_account_key_path: Path | None = Field(
        default=None, alias="GEE_SERVICE_ACCOUNT_KEY_PATH"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_configured(self) -> bool:
        return self.project_id is not None


class CDSESettings(_Base):
    """Copernicus Data Space Ecosystem -- secondary / verification plane.

    Free tier as documented on 2026-09-22: 10,000 Sentinel Hub processing units
    per month, 10,000 SH requests per month, 12 TB per rolling 30 days.
    """

    client_id: str | None = Field(default=None, alias="CDSE_CLIENT_ID")
    client_secret: SecretStr | None = Field(default=None, alias="CDSE_CLIENT_SECRET")
    username: str | None = Field(default=None, alias="CDSE_USERNAME")
    password: SecretStr | None = Field(default=None, alias="CDSE_PASSWORD")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_configured(self) -> bool:
        return self.client_id is not None and self.client_secret is not None


class NASASettings(_Base):
    """NASA Earthdata (GPM IMERG) and FIRMS (active fire) credentials."""

    earthdata_username: str | None = Field(default=None, alias="EARTHDATA_USERNAME")
    earthdata_password: SecretStr | None = Field(default=None, alias="EARTHDATA_PASSWORD")
    firms_map_key: SecretStr | None = Field(default=None, alias="FIRMS_MAP_KEY")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def firms_configured(self) -> bool:
        return self.firms_map_key is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def earthdata_configured(self) -> bool:
        return self.earthdata_username is not None and self.earthdata_password is not None


class CDSSettings(_Base):
    """Copernicus Climate Data Store (ERA5 / ERA5-Land).

    These two variables were documented in ``.env.example`` without a field to
    read them, so filling them in did nothing and nothing said so. A documented
    variable that no code reads is worse than an undocumented one: it looks
    configured.
    """

    api_url: str = Field(default="https://cds.climate.copernicus.eu/api", alias="CDS_API_URL")
    api_key: SecretStr | None = Field(default=None, alias="CDS_API_KEY")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_configured(self) -> bool:
        return self.api_key is not None


class SupabaseSettings(_Base):
    """PostgREST catalogue access, as used by the serving plane.

    The deployed API reads these from ``os.environ`` directly rather than
    through this class -- see the note in ``.env.example`` -- but they are
    modelled here so that offline tooling, the environment check and anything
    in the ML plane that needs the catalogue share one definition and one set of
    names.

    ``service_key`` is a write credential. It is the only secret in this file
    with no fallback anywhere in the project, and the only thing it is used for
    is appending to the C1 audit log.
    """

    url: str | None = Field(default=None, alias="SUPABASE_URL")
    anon_key: SecretStr | None = Field(default=None, alias="SUPABASE_ANON_KEY")
    service_key: SecretStr | None = Field(default=None, alias="SUPABASE_SERVICE_KEY")
    table_prefix: str = Field(default="satai_", alias="SUPABASE_TABLE_PREFIX")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_configured(self) -> bool:
        return self.url is not None and self.anon_key is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def audit_log_enabled(self) -> bool:
        """Whether chat turns can be recorded for the C1 measurement."""
        return self.service_key is not None


class DatabaseSettings(_Base):
    """PostgreSQL + PostGIS connection."""

    url: str = Field(
        default="postgresql+psycopg://satai:satai@localhost:5432/satai",
        alias="DATABASE_URL",
    )
    echo: bool = Field(default=False, alias="DATABASE_ECHO")
    pool_size: int = Field(default=5, ge=1, le=50, alias="DATABASE_POOL_SIZE")


class StorageSettings(_Base):
    """Object storage for rasters and model artifacts.

    Large rasters are stored as Cloud-Optimized GeoTIFFs in object storage and
    referenced from the database by URI. Raster bytes are never stored in
    PostgreSQL (ADR-006).
    """

    backend: Literal["local", "s3"] = Field(default="local", alias="STORAGE_BACKEND")
    local_root: Path = Field(default=REPO_ROOT / "data" / "processed", alias="STORAGE_LOCAL_ROOT")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_bucket: str | None = Field(default=None, alias="S3_BUCKET")
    s3_access_key_id: SecretStr | None = Field(default=None, alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: SecretStr | None = Field(default=None, alias="S3_SECRET_ACCESS_KEY")
    s3_region: str = Field(default="auto", alias="S3_REGION")


class APISettings(_Base):
    """FastAPI server configuration."""

    host: str = Field(default="0.0.0.0", alias="API_HOST")  # noqa: S104
    port: int = Field(default=8000, gt=0, lt=65536, alias="API_PORT")
    v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    cors_allow_origins: str = Field(default="http://localhost:3000", alias="CORS_ALLOW_ORIGINS")
    rate_limit_per_minute: int = Field(default=60, gt=0, alias="API_RATE_LIMIT_PER_MINUTE")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origins(self) -> list[str]:
        """CORS origins parsed from the comma-separated environment value."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


class Settings(_Base):
    """Root settings object. Obtain via :func:`get_settings`."""

    env: Environment = Field(default=Environment.DEVELOPMENT, alias="SATAI_ENV")
    log_level: str = Field(default="INFO", alias="SATAI_LOG_LEVEL")
    log_format: LogFormat = Field(default=LogFormat.CONSOLE, alias="SATAI_LOG_FORMAT")
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="SATAI_DATA_DIR")
    models_dir: Path = Field(default=REPO_ROOT / "models", alias="SATAI_MODELS_DIR")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    gee: EarthEngineSettings = Field(default_factory=EarthEngineSettings)
    cdse: CDSESettings = Field(default_factory=CDSESettings)
    cds: CDSSettings = Field(default_factory=CDSSettings)
    nasa: NASASettings = Field(default_factory=NASASettings)
    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: APISettings = Field(default_factory=APISettings)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return upper

    @computed_field  # type: ignore[prop-decorator]
    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def configs_dir(self) -> Path:
        return REPO_ROOT / "configs"

    def capability_report(self) -> dict[str, bool]:
        """Which external services are configured right now.

        Drives graceful degradation (requirement 39) and the ``/health``
        endpoint. A missing credential must produce a clearly-labelled reduced
        capability, never a silent fabrication of data.
        """
        return {
            "llm": self.llm.is_configured,
            "earth_engine": self.gee.is_configured,
            "cdse": self.cdse.is_configured,
            "cds": self.cds.is_configured,
            "firms": self.nasa.firms_configured,
            "earthdata": self.nasa.earthdata_configured,
            "supabase": self.supabase.is_configured,
            # Reported separately from `supabase`: read access and the ability
            # to record a chat turn for C1 are different capabilities, and a
            # deployment can legitimately have the first without the second.
            "agent_audit_log": self.supabase.audit_log_enabled,
        }

    def ensure_directories(self) -> None:
        """Create the local data directories if they do not exist."""
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.manifests_dir,
            self.samples_dir,
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings.

    Cached so that ``.env`` is parsed once per process. In tests, call
    ``get_settings.cache_clear()`` after changing the environment.
    """
    return Settings()
