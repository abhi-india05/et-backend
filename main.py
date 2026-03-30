"""RevOps AI — FastAPI application entry point.

All route logic lives in ``backend/api/routes/``.  This module is responsible
for application wiring: lifespan, middleware, exception handlers, and router
registration.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.router import api_router
from backend.agents.failure_recovery import get_recovery_engine
from backend.auth.passwords import PasswordValidationError, hash_password
from backend.config.settings import settings
from backend.db.mongo import close_clients, get_database
from backend.deps import get_refresh_token_repo, get_session_repo, get_user_repo
from backend.memory.vector_store import get_vector_store
from backend.repositories.users import UserRepository
from backend.services.auth_service import AuthServiceError, InvalidRefreshTokenError, RefreshTokenReuseError
from backend.services.observability import get_metrics_registry
from backend.services.rate_limit import get_rate_limiter
from backend.tools.email_tool import get_email_stats
from backend.utils.errors import APIError, error_response
from backend.utils.helpers import generate_request_id, now_iso
from backend.utils.logger import bind_context, clear_context, configure_logging, get_logger

configure_logging(log_level=settings.log_level, environment=settings.environment)
logger = get_logger("main")

RATE_LIMIT_EXEMPT_PATHS = {"/health", "/openapi.json", "/docs", "/redoc"}


# ---------------------------------------------------------------------------
# Client IP helper
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

async def _seed_admin_user(user_repo: UserRepository) -> None:
    if not settings.auth_enabled:
        return
    admin_password_hash = hash_password(settings.auth_password)
    await user_repo.ensure_admin_user(username=settings.auth_username, password_hash=admin_password_hash)


async def _initialize_runtime() -> None:
    settings.validate_runtime()
    store = get_vector_store()
    store.load()
    logger.info("vector_store_ready", **store.stats())

    if settings.is_test:
        return

    try:
        await get_database().command("ping")
        user_repo = get_user_repo()
        session_repo = get_session_repo()
        refresh_repo = get_refresh_token_repo()
        await user_repo.ensure_indexes()
        await session_repo.ensure_indexes()
        await refresh_repo.ensure_indexes()
        await _seed_admin_user(user_repo)
        logger.info("runtime_persistence_ready")
    except Exception as exc:
        if settings.is_production:
            raise
        logger.warning("runtime_persistence_unavailable", error=str(exc))


async def _database_health() -> Dict[str, Any]:
    if settings.is_test:
        return {"ready": True, "mode": "test"}
    try:
        await get_database().command("ping")
        return {"ready": True, "mode": "mongo"}
    except Exception as exc:
        return {"ready": False, "error": str(exc)}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "revops_ai_startup",
        environment=settings.environment,
        has_gemini_key=settings.has_gemini_key,
        mock_email=settings.is_mock_email,
    )
    await _initialize_runtime()
    yield
    logger.info("revops_ai_shutdown")
    get_vector_store().save()
    close_clients()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RevOps AI - Autonomous Sales and Revenue Intelligence",
    description="Production-grade RevOps AI backend with hardened auth, persistence, and agent workflows.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=settings.resolved_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or generate_request_id()
    client_ip = _client_ip(request)
    request.state.request_id = request_id
    bind_context(request_id=request_id, method=request.method, path=request.url.path, client_ip=client_ip)

    rate_limit_result = None
    status_code = 500
    started_at = time.perf_counter()

    if request.url.path not in RATE_LIMIT_EXEMPT_PATHS and request.method.upper() != "OPTIONS":
        rate_limit_result = get_rate_limiter().check(
            f"global:{client_ip}",
            limit=settings.global_rate_limit_requests_per_minute,
            window_seconds=60,
        )
        if not rate_limit_result.allowed:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            get_metrics_registry().record_request(
                method=request.method,
                path=request.url.path,
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                duration_ms=duration_ms,
            )
            response = error_response(
                request_id=request_id,
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                code="rate_limit_exceeded",
                message="Too many requests. Please try again later.",
                details={"retry_after_seconds": rate_limit_result.retry_after_seconds},
                headers={
                    "Retry-After": str(rate_limit_result.retry_after_seconds),
                    "X-RateLimit-Remaining": "0",
                },
            )
            clear_context()
            return response

    response: Optional[Response] = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        if rate_limit_result is not None:
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_result.remaining)
        logger.info(
            "http_request_complete",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
        )
        return response
    finally:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        get_metrics_registry().record_request(
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
        )
        clear_context()


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return error_response(
        request_id=getattr(request.state, "request_id", generate_request_id()),
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(
        request_id=getattr(request.state, "request_id", generate_request_id()),
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="validation_error",
        message="Request validation failed",
        details=exc.errors(),
    )


@app.exception_handler(PasswordValidationError)
async def password_validation_handler(request: Request, exc: PasswordValidationError) -> JSONResponse:
    return error_response(
        request_id=getattr(request.state, "request_id", generate_request_id()),
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="weak_password",
        message=str(exc),
    )


@app.exception_handler(AuthServiceError)
async def auth_service_error_handler(request: Request, exc: AuthServiceError) -> JSONResponse:
    status_code = status.HTTP_401_UNAUTHORIZED
    if isinstance(exc, RefreshTokenReuseError):
        status_code = status.HTTP_401_UNAUTHORIZED
    elif isinstance(exc, InvalidRefreshTokenError):
        status_code = status.HTTP_401_UNAUTHORIZED
    return error_response(
        request_id=getattr(request.state, "request_id", generate_request_id()),
        status_code=status_code,
        code=exc.code,
        message=str(exc),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", "http_error")
        message = detail.get("message", "Request failed")
        details = detail.get("details")
    else:
        code = "http_error"
        message = str(detail)
        details = None
    return error_response(
        request_id=getattr(request.state, "request_id", generate_request_id()),
        status_code=exc.status_code,
        code=code,
        message=message,
        details=details,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", generate_request_id())
    logger.error(
        "unhandled_exception",
        request_id=request_id,
        path=str(request.url),
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    message = "An internal error occurred"
    details = str(exc) if not settings.is_production else None
    return error_response(
        request_id=request_id,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        message=message,
        details=details,
    )


# ---------------------------------------------------------------------------
# Health check (un-prefixed so load balancers can hit /health directly)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    database = await _database_health()
    email_stats: Dict[str, Any]
    try:
        # Health is unscoped; use a synthetic namespace so stats lookup never crashes the endpoint.
        email_stats = get_email_stats(user_id="system_health")
    except Exception as exc:
        logger.warning("health_email_stats_failed", error=str(exc))
        email_stats = {
            "total": 0,
            "sent": 0,
            "failed": 0,
            "success_rate": 0.0,
            "error": str(exc),
        }

    return {
        "status": "healthy" if database.get("ready") else "degraded",
        "version": app.version,
        "environment": settings.environment,
        "timestamp": now_iso(),
        "gemini_configured": settings.has_gemini_key,
        "email_mode": "live" if not settings.is_mock_email else "mock",
        "vector_store": get_vector_store().stats(),
        "email_stats": email_stats,
        "database": database,
    }


# ---------------------------------------------------------------------------
# Mount versioned API
# ---------------------------------------------------------------------------

app.include_router(api_router)

# Backward-compatible non-prefixed mounts so existing tests and frontends
# continue working while migrations happen.
from backend.api.routes.auth import router as _auth_compat  # noqa: E402
from backend.api.routes.workflows import router as _workflows_compat  # noqa: E402
from backend.api.routes.admin import router as _admin_compat  # noqa: E402
from backend.api.routes.outreach import router as _outreach_compat  # noqa: E402

app.include_router(_auth_compat)
app.include_router(_workflows_compat)
app.include_router(_admin_compat)
app.include_router(_outreach_compat, prefix="/outreach")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=not settings.is_production)
