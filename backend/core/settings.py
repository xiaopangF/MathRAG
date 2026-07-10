import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from backend.core.paths import PROJECT_ROOT


class SettingsError(ValueError):
    """Raised when an operational environment variable is invalid."""


def _read_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be between {minimum} and {maximum}")
    return value


def _read_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be between {minimum} and {maximum}")
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be a boolean")


def _read_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise SettingsError(f"{name} must contain at least one value")
    return values


@dataclass(frozen=True, slots=True)
class BackendSettings:
    app_name: str
    app_version: str
    environment: str
    log_level: str
    log_json: bool
    cors_origins: tuple[str, ...]
    cors_origin_regex: str
    max_upload_bytes: int
    max_json_body_bytes: int
    sqlite_timeout_seconds: float
    job_max_attempts: int
    rag_max_concurrency: int
    rag_acquire_timeout_seconds: float
    llm_timeout_seconds: float
    llm_max_retries: int
    request_id_header: str = "X-Request-ID"

    @property
    def sqlite_busy_timeout_ms(self) -> int:
        return int(self.sqlite_timeout_seconds * 1000)

    @classmethod
    def from_env(cls) -> "BackendSettings":
        log_level = os.getenv("MATHRAG_LOG_LEVEL", "INFO").strip().upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise SettingsError("MATHRAG_LOG_LEVEL is invalid")

        max_upload_mb = _read_int(
            "MATHRAG_MAX_UPLOAD_MB",
            50,
            minimum=1,
            maximum=1024,
        )
        return cls(
            app_name="MathRAG API",
            app_version="0.2.0",
            environment=os.getenv("MATHRAG_ENVIRONMENT", "development").strip()
            or "development",
            log_level=log_level,
            log_json=_read_bool("MATHRAG_LOG_JSON", True),
            cors_origins=_read_csv(
                "MATHRAG_CORS_ORIGINS",
                (
                    "http://127.0.0.1:5173",
                    "http://localhost:5173",
                ),
            ),
            cors_origin_regex=os.getenv(
                "MATHRAG_CORS_ORIGIN_REGEX",
                r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
            ).strip(),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            max_json_body_bytes=_read_int(
                "MATHRAG_MAX_JSON_BODY_MB",
                1,
                minimum=1,
                maximum=20,
            )
            * 1024
            * 1024,
            sqlite_timeout_seconds=_read_float(
                "MATHRAG_SQLITE_TIMEOUT_SECONDS",
                10.0,
                minimum=0.1,
                maximum=120.0,
            ),
            job_max_attempts=_read_int(
                "MATHRAG_JOB_MAX_ATTEMPTS",
                3,
                minimum=1,
                maximum=20,
            ),
            rag_max_concurrency=_read_int(
                "MATHRAG_RAG_MAX_CONCURRENCY",
                2,
                minimum=1,
                maximum=64,
            ),
            rag_acquire_timeout_seconds=_read_float(
                "MATHRAG_RAG_ACQUIRE_TIMEOUT_SECONDS",
                2.0,
                minimum=0.0,
                maximum=60.0,
            ),
            llm_timeout_seconds=_read_float(
                "MATHRAG_LLM_TIMEOUT_SECONDS",
                30.0,
                minimum=1.0,
                maximum=300.0,
            ),
            llm_max_retries=_read_int(
                "MATHRAG_LLM_MAX_RETRIES",
                2,
                minimum=0,
                maximum=10,
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> BackendSettings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return BackendSettings.from_env()
