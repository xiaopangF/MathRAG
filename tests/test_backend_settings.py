import pytest

from backend.core.settings import BackendSettings, SettingsError


def test_backend_settings_load_operational_values(monkeypatch):
    monkeypatch.setenv("MATHRAG_MAX_UPLOAD_MB", "12")
    monkeypatch.setenv("MATHRAG_SQLITE_TIMEOUT_SECONDS", "4.5")
    monkeypatch.setenv("MATHRAG_MAX_JSON_BODY_MB", "2")
    monkeypatch.setenv("MATHRAG_JOB_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("MATHRAG_RAG_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("MATHRAG_LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("MATHRAG_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("MATHRAG_LOG_JSON", "false")
    monkeypatch.setenv(
        "MATHRAG_CORS_ORIGINS",
        "https://app.example.com, https://admin.example.com",
    )

    settings = BackendSettings.from_env()

    assert settings.max_upload_bytes == 12 * 1024 * 1024
    assert settings.sqlite_busy_timeout_ms == 4500
    assert settings.max_json_body_bytes == 2 * 1024 * 1024
    assert settings.job_max_attempts == 5
    assert settings.rag_max_concurrency == 3
    assert settings.llm_timeout_seconds == 45
    assert settings.llm_max_retries == 4
    assert settings.log_json is False
    assert settings.cors_origins == (
        "https://app.example.com",
        "https://admin.example.com",
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MATHRAG_MAX_UPLOAD_MB", "0"),
        ("MATHRAG_MAX_JSON_BODY_MB", "0"),
        ("MATHRAG_SQLITE_TIMEOUT_SECONDS", "invalid"),
        ("MATHRAG_JOB_MAX_ATTEMPTS", "100"),
        ("MATHRAG_LOG_JSON", "sometimes"),
        ("MATHRAG_LLM_TIMEOUT_SECONDS", "0"),
        ("MATHRAG_LLM_MAX_RETRIES", "11"),
    ],
)
def test_backend_settings_reject_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(SettingsError):
        BackendSettings.from_env()
