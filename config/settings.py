from __future__ import annotations

import json
import os
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized runtime configuration with production validation."""

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_api_keys: str = Field(default="", alias="GEMINI_API_KEYS")
    gemini_model: str = Field(default="", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(default="", alias="GEMINI_EMBEDDING_MODEL")
    # Backward-compatible aliases still used in existing .env files.
    openai_model: str = Field(default="", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="", alias="OPENAI_EMBEDDING_MODEL")

    mongodb_uri: str = Field(default="mongodb://localhost:27017/revops_ai", alias="MONGODB_URI")
    mongo_server_selection_timeout_ms: int = Field(
        default=1200,
        ge=100,
        le=10000,
        alias="MONGO_SERVER_SELECTION_TIMEOUT_MS",
    )
    mongo_connect_timeout_ms: int = Field(
        default=1200,
        ge=100,
        le=10000,
        alias="MONGO_CONNECT_TIMEOUT_MS",
    )
    mongo_socket_timeout_ms: int = Field(
        default=3000,
        ge=500,
        le=30000,
        alias="MONGO_SOCKET_TIMEOUT_MS",
    )

    mail_username: Optional[str] = Field(default=None, alias="MAIL_USERNAME")
    mail_password: Optional[str] = Field(default=None, alias="MAIL_PASSWORD")
    mail_from: Optional[str] = Field(default=None, alias="MAIL_FROM")
    mail_server: str = Field(default="smtp.gmail.com", alias="MAIL_SERVER")
    mail_port: int = Field(default=587, ge=1, le=65535, alias="MAIL_PORT")
    mail_tls: bool = Field(default=True, alias="MAIL_TLS")
    mail_ssl: bool = Field(default=False, alias="MAIL_SSL")
    sendgrid_api_key: str = Field(default="mock_key", alias="SENDGRID_API_KEY")

    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        alias="CORS_ORIGINS",
    )
    cors_origin_regex: str = Field(default="", alias="CORS_ORIGIN_REGEX")
    max_retries: int = Field(default=2, ge=0, le=5, alias="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, alias="RETRY_DELAY")
    agent_inter_call_delay_seconds: float = Field(
        default=1.5,
        ge=0.0,
        le=30.0,
        alias="AGENT_INTER_CALL_DELAY_SECONDS",
    )
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, alias="CONFIDENCE_THRESHOLD")
    faiss_index_path: str = Field(default="./memory/faiss_index", alias="FAISS_INDEX_PATH")

    auth_enabled: bool = Field(default=True, alias="AUTH_ENABLED")
    auth_username: str = Field(default="admin", alias="AUTH_USERNAME")
    auth_password: str = Field(default="ChangeMe-Admin#123", alias="AUTH_PASSWORD")
    auth_secret_key: str = Field(default="change-me", alias="AUTH_SECRET_KEY")
    auth_algorithm: str = Field(default="HS256", alias="AUTH_ALGORITHM")
    auth_access_token_expire_minutes: int = Field(
        default=60,
        ge=15,
        le=240,
        alias="AUTH_ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    auth_refresh_token_expire_days: int = Field(
        default=30,
        ge=7,
        le=90,
        alias="AUTH_REFRESH_TOKEN_EXPIRE_DAYS",
    )
    auth_cookie_name: str = Field(default="revops_access_token", alias="AUTH_COOKIE_NAME")
    auth_refresh_cookie_name: str = Field(
        default="revops_refresh_token",
        alias="AUTH_REFRESH_COOKIE_NAME",
    )
    auth_cookie_samesite: str = Field(default="lax", alias="AUTH_COOKIE_SAMESITE")
    auth_refresh_cookie_path: str = Field(default="/", alias="AUTH_REFRESH_COOKIE_PATH")
    auth_cookie_secure: Optional[bool] = Field(default=None, alias="AUTH_COOKIE_SECURE")
    auth_cookie_domain: Optional[str] = Field(default=None, alias="AUTH_COOKIE_DOMAIN")
    auth_password_min_length: int = Field(default=12, ge=10, le=128, alias="AUTH_PASSWORD_MIN_LENGTH")
    auth_password_require_upper: bool = Field(default=True, alias="AUTH_PASSWORD_REQUIRE_UPPER")
    auth_password_require_lower: bool = Field(default=True, alias="AUTH_PASSWORD_REQUIRE_LOWER")
    auth_password_require_digit: bool = Field(default=True, alias="AUTH_PASSWORD_REQUIRE_DIGIT")
    auth_password_require_special: bool = Field(default=True, alias="AUTH_PASSWORD_REQUIRE_SPECIAL")
    auth_issuer: str = Field(default="revops-ai", alias="AUTH_ISSUER")
    auth_audience: str = Field(default="revops-ai-api", alias="AUTH_AUDIENCE")

    global_rate_limit_requests_per_minute: int = Field(
        default=120,
        ge=10,
        le=5000,
        alias="GLOBAL_RATE_LIMIT_REQUESTS_PER_MINUTE",
    )
    login_rate_limit_attempts: int = Field(
        default=5,
        ge=1,
        le=25,
        alias="LOGIN_RATE_LIMIT_ATTEMPTS",
    )
    login_rate_limit_window_seconds: int = Field(
        default=900,
        ge=60,
        le=86400,
        alias="LOGIN_RATE_LIMIT_WINDOW_SECONDS",
    )

    default_page_size: int = Field(default=25, ge=1, le=200, alias="DEFAULT_PAGE_SIZE")
    max_page_size: int = Field(default=100, ge=10, le=500, alias="MAX_PAGE_SIZE")

    memory_ttl_seconds: int = Field(default=604800, ge=3600, alias="MEMORY_TTL_SECONDS")
    memory_max_documents_per_user: int = Field(
        default=200,
        ge=10,
        le=5000,
        alias="MEMORY_MAX_DOCUMENTS_PER_USER",
    )
    memory_max_total_documents: int = Field(
        default=5000,
        ge=100,
        le=50000,
        alias="MEMORY_MAX_TOTAL_DOCUMENTS",
    )

    @field_validator("environment")
    @classmethod
    def _normalize_environment(cls, value: str) -> str:
        return (value or "development").strip().lower()

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return (value or "INFO").strip().upper()

    @field_validator("cors_origin_regex")
    @classmethod
    def _normalize_cors_origin_regex(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("auth_cookie_samesite")
    @classmethod
    def _normalize_auth_cookie_samesite(cls, value: str) -> str:
        normalized = (value or "lax").strip().lower()
        if normalized not in {"lax", "strict", "none"}:
            raise ValueError("AUTH_COOKIE_SAMESITE must be one of: lax, strict, none")
        return normalized

    @field_validator("auth_refresh_cookie_path")
    @classmethod
    def _normalize_auth_refresh_cookie_path(cls, value: str) -> str:
        path = (value or "/").strip()
        if not path.startswith("/"):
            path = "/" + path
        return path.rstrip("/") or "/"



    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_test(self) -> bool:
        return self.environment == "test"

    @property
    def database_name(self) -> str:
        parsed = urlparse(self.mongodb_uri)
        path = (parsed.path or "").lstrip("/")
        return path or "revops_ai"

    @property
    def is_running_on_vercel(self) -> bool:
        return bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV") or os.getenv("NOW_REGION"))

    @property
    def is_localhost_mongo_uri(self) -> bool:
        parsed = urlparse(self.mongodb_uri)
        host = (parsed.hostname or "").strip().lower()
        return host in {"", "localhost", "127.0.0.1", "::1"}

    @property
    def cors_origins_list(self) -> list[str]:
        raw = (self.cors_origins or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def resolved_cors_origin_regex(self) -> Optional[str]:
        regex = (self.cors_origin_regex or "").strip()
        return regex or None

    @property
    def gemini_api_key_list(self) -> List[str]:
        keys: List[str] = []
        raw = (self.gemini_api_keys or "").strip()
        if raw:
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        keys.extend(str(item).strip() for item in parsed if str(item).strip())
                except json.JSONDecodeError:
                    pass
            if not keys:
                keys.extend(item.strip() for item in raw.split(",") if item.strip())

        single_key = (self.gemini_api_key or "").strip()
        if single_key:
            keys.insert(0, single_key)

        deduped: List[str] = []
        seen = set()
        for key in keys:
            if key and key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key_list)

    @property
    def resolved_gemini_model(self) -> str:
        return (
            self.gemini_model.strip()
            or self.openai_model.strip()
            or "gemini-2.5-flash"
        )

    @property
    def resolved_gemini_embedding_model(self) -> str:
        return (
            self.gemini_embedding_model.strip()
            or self.openai_embedding_model.strip()
            or "gemini-embedding-001"
        )

    @property
    def is_mock_email(self) -> bool:
        if not (self.mail_username and self.mail_password and self.mail_from):
            return True
        placeholders = ("your_", "example.com", "changeme", "dummy", "placeholder")
        joined = " ".join([self.mail_username, self.mail_password, self.mail_from]).lower()
        return any(marker in joined for marker in placeholders)

    @property
    def resolved_auth_cookie_secure(self) -> bool:
        if self.auth_cookie_secure is not None:
            return bool(self.auth_cookie_secure)
        return self.is_production

    @property
    def auth_access_token_expire_seconds(self) -> int:
        return self.auth_access_token_expire_minutes * 60

    @property
    def auth_refresh_token_expire_seconds(self) -> int:
        return self.auth_refresh_token_expire_days * 24 * 60 * 60

    def validate_runtime(self) -> None:
        problems: list[str] = []
        if not self.mongodb_uri.strip():
            problems.append("MONGODB_URI must be configured.")
        if self.is_running_on_vercel and self.is_localhost_mongo_uri:
            problems.append(
                "MONGODB_URI points to localhost in a Vercel runtime. Set MONGODB_URI to your cloud MongoDB URI (for example, MongoDB Atlas)."
            )
        if self.auth_cookie_samesite == "none" and not self.resolved_auth_cookie_secure:
            problems.append("AUTH_COOKIE_SECURE must be true when AUTH_COOKIE_SAMESITE is set to none.")
        if self.auth_enabled and self.is_production:
            secret = self.auth_secret_key.strip()
            if secret in {"", "change-me", "changeme", "change-me-please-use-a-long-random-string"}:
                problems.append("AUTH_SECRET_KEY must be replaced with a strong random secret.")
            if len(secret) < 32:
                problems.append("AUTH_SECRET_KEY must be at least 32 characters long.")
            if not self.auth_issuer.strip():
                problems.append("AUTH_ISSUER must be configured.")
            if not self.auth_audience.strip():
                problems.append("AUTH_AUDIENCE must be configured.")
        if self.is_production:
            if not self.gemini_api_key_list:
                problems.append("At least one Gemini API key (GEMINI_API_KEY or GEMINI_API_KEYS) is required in production.")
            if not self.cors_origins_list and not self.resolved_cors_origin_regex:
                problems.append("CORS_ORIGINS or CORS_ORIGIN_REGEX must be configured in production.")
            if not self.resolved_auth_cookie_secure:
                problems.append("AUTH_COOKIE_SECURE must resolve to true in production.")
        if problems:
            raise RuntimeError("Invalid configuration:\n- " + "\n- ".join(problems))


settings = Settings()
