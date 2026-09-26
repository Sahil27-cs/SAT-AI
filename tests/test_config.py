"""Tests for the configuration layer."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from satai.config import (
    APISettings,
    CDSSettings,
    Environment,
    LLMSettings,
    Settings,
    SupabaseSettings,
    get_settings,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_defaults_are_usable_without_any_env() -> None:
    """Phase 1 must run with no .env and no credentials at all."""
    settings = Settings()
    assert settings.env is Environment.DEVELOPMENT
    assert settings.log_level == "INFO"
    assert settings.llm.model == "claude-opus-5"


def test_verified_model_identifiers() -> None:
    """Model IDs were checked against the live Anthropic docs, not guessed.

    If this test fails after a docs change, re-verify rather than editing the
    expectation to match whatever the code says.
    """
    llm = LLMSettings()
    assert llm.model == "claude-opus-5"
    assert llm.router_model == "claude-haiku-4-5-20251001"


def test_api_key_is_a_secret_and_does_not_leak_in_repr() -> None:
    llm = LLMSettings(ANTHROPIC_API_KEY="sk-ant-super-secret-value")  # type: ignore[call-arg]
    assert "super-secret-value" not in repr(llm)
    assert "super-secret-value" not in str(llm)
    assert isinstance(llm.api_key, SecretStr)
    assert llm.api_key.get_secret_value() == "sk-ant-super-secret-value"


def test_llm_is_not_configured_without_a_key() -> None:
    """Drives graceful degradation: no key means no agent layer, not a crash."""
    assert LLMSettings(ANTHROPIC_API_KEY=None).is_configured is False  # type: ignore[call-arg]
    assert LLMSettings(ANTHROPIC_API_KEY="").is_configured is False  # type: ignore[call-arg]
    assert LLMSettings(ANTHROPIC_API_KEY="sk-ant-x").is_configured is True  # type: ignore[call-arg]


def test_temperature_bounds() -> None:
    with pytest.raises(ValidationError):
        LLMSettings(ANTHROPIC_TEMPERATURE=2.5)  # type: ignore[call-arg]


def test_log_level_is_validated_and_normalised() -> None:
    assert Settings(SATAI_LOG_LEVEL="debug").log_level == "DEBUG"  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Settings(SATAI_LOG_LEVEL="VERBOSE")  # type: ignore[call-arg]


def test_cors_origins_are_parsed_from_csv() -> None:
    api = APISettings(  # type: ignore[call-arg]
        CORS_ALLOW_ORIGINS="http://localhost:3000, https://sat-ai.vercel.app ,"
    )
    assert api.cors_origins == ["http://localhost:3000", "https://sat-ai.vercel.app"]


def test_capability_report_lists_every_external_service() -> None:
    report = Settings().capability_report()
    assert set(report) == {
        "llm",
        "earth_engine",
        "cdse",
        "cds",
        "firms",
        "earthdata",
        "supabase",
        "agent_audit_log",
    }
    assert all(isinstance(v, bool) for v in report.values())


def test_documented_variables_are_actually_read() -> None:
    """CDS_API_KEY was in .env.example with no field to read it.

    Filling it in did nothing, and nothing said so — which is worse than
    leaving it undocumented, because it looks configured.
    """
    cds = CDSSettings(CDS_API_KEY="uid:key")  # type: ignore[call-arg]
    assert cds.is_configured is True
    assert cds.api_key is not None
    assert cds.api_key.get_secret_value() == "uid:key"
    assert CDSSettings(CDS_API_KEY=None).is_configured is False  # type: ignore[call-arg]


def test_the_service_key_is_a_secret_and_gates_the_audit_log() -> None:
    """Read access and the ability to record a chat turn are separate things."""
    read_only = SupabaseSettings(  # type: ignore[call-arg]
        SUPABASE_URL="https://example.supabase.co", SUPABASE_ANON_KEY="sb_publishable_x"
    )
    assert read_only.is_configured is True
    assert read_only.audit_log_enabled is False, "C1 logging must not run on the anon key"

    writable = SupabaseSettings(  # type: ignore[call-arg]
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_ANON_KEY="sb_publishable_x",
        SUPABASE_SERVICE_KEY="sb_secret_do_not_log",
    )
    assert writable.audit_log_enabled is True
    assert "do_not_log" not in repr(writable)
    assert "do_not_log" not in str(writable)


def test_derived_paths_sit_under_the_data_dir() -> None:
    settings = Settings()
    assert settings.raw_dir.parent == settings.data_dir
    assert settings.processed_dir.parent == settings.data_dir
    assert settings.manifests_dir.name == "manifests"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
