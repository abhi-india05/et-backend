"""Auth routes — login, register, refresh, logout, me."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request, Response, status

from backend.auth.deps import AuthUser, get_current_user
from backend.auth.jwt import create_access_token
from backend.auth.passwords import hash_password, verify_password
from backend.config.settings import settings
from backend.deps import get_refresh_token_repo, get_user_repo
from backend.models.schemas import (
    AuthLoginRequest,
    AuthMeResponse,
    AuthRefreshRequest,
    AuthRegisterRequest,
    AuthTokenResponse,
)
from backend.repositories.refresh_tokens import RefreshTokenRepository
from backend.repositories.users import UserRepository
from backend.services.auth_service import (
    issue_token_pair,
    revoke_refresh_token,
    rotate_refresh_token,
)
from backend.services.rate_limit import get_rate_limiter
from backend.utils.errors import APIError
from backend.utils.helpers import now_iso
from backend.utils.logger import bind_context, record_audit

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _set_access_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        httponly=True,
        secure=settings.resolved_auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        domain=settings.auth_cookie_domain,
        max_age=settings.auth_access_token_expire_seconds,
        path="/",
    )


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_refresh_cookie_name,
        value=token,
        httponly=True,
        secure=settings.resolved_auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        domain=settings.auth_cookie_domain,
        max_age=settings.auth_refresh_token_expire_seconds,
        path=settings.auth_refresh_cookie_path,
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(
        key=settings.auth_cookie_name,
        domain=settings.auth_cookie_domain,
        path="/",
    )
    response.delete_cookie(
        key=settings.auth_refresh_cookie_name,
        domain=settings.auth_cookie_domain,
        path=settings.auth_refresh_cookie_path,
    )


@router.post("/login", response_model=AuthTokenResponse)
async def auth_login(
    req: AuthLoginRequest,
    request: Request,
    response: Response,
    user_repo: UserRepository = Depends(get_user_repo),
    refresh_repo: RefreshTokenRepository = Depends(get_refresh_token_repo),
) -> AuthTokenResponse:
    if not settings.auth_enabled:
        access = create_access_token(subject="anonymous", username="anonymous", role="user")
        _set_access_cookie(response, access["token"])
        return AuthTokenResponse(
            access_token=access["token"],
            expires_in=settings.auth_access_token_expire_seconds,
            refresh_expires_in=0,
            role="user",
        )

    login_limit = get_rate_limiter().check(
        f"login:{_client_ip(request)}:{req.username.lower()}",
        limit=settings.login_rate_limit_attempts,
        window_seconds=settings.login_rate_limit_window_seconds,
    )
    if not login_limit.allowed:
        raise APIError(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            code="login_rate_limited",
            message="Too many login attempts. Please try again later.",
            details={"retry_after_seconds": login_limit.retry_after_seconds},
        )

    user = await user_repo.get_by_username(req.username)
    if not user or not verify_password(req.password, user.password_hash):
        raise APIError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="invalid_credentials",
            message="Invalid username or password",
        )

    updated_user = await user_repo.update_last_login(user.user_id) or user
    issued = await issue_token_pair(user=updated_user, refresh_repo=refresh_repo)
    _set_access_cookie(response, issued.access_token)
    _set_refresh_cookie(response, issued.refresh_token)
    bind_context(
        user_id=updated_user.user_id,
        username=updated_user.username,
        role=updated_user.role,
        session_id=issued.session_id,
    )
    record_audit(
        session_id=issued.session_id,
        agent_name="auth",
        action="login",
        input_summary=f"User login for {updated_user.username}",
        output_summary="Access and refresh tokens issued",
        status="success",
        confidence=1.0,
    )
    return AuthTokenResponse(
        access_token=issued.access_token,
        expires_in=issued.access_expires_in,
        refresh_expires_in=issued.refresh_expires_in,
        role=issued.role,
    )


@router.post("/register", response_model=AuthTokenResponse)
async def auth_register(
    req: AuthRegisterRequest,
    request: Request,
    response: Response,
    user_repo: UserRepository = Depends(get_user_repo),
    refresh_repo: RefreshTokenRepository = Depends(get_refresh_token_repo),
) -> AuthTokenResponse:
    if not settings.auth_enabled:
        raise APIError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="auth_disabled",
            message="Registration is disabled when authentication is disabled.",
        )

    # Rate limit registration
    register_limit = get_rate_limiter().check(
        f"register:{_client_ip(request)}",
        limit=settings.login_rate_limit_attempts,
        window_seconds=settings.login_rate_limit_window_seconds,
    )
    if not register_limit.allowed:
        raise APIError(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            code="register_rate_limited",
            message="Too many registration attempts. Please try again later.",
            details={"retry_after_seconds": register_limit.retry_after_seconds},
        )

    if req.username.lower() == settings.auth_username.lower():
        raise APIError(
            status_code=status.HTTP_409_CONFLICT,
            code="reserved_username",
            message="Username is reserved",
        )

    try:
        user = await user_repo.create_user(
            username=req.username,
            password_hash=hash_password(req.password),
            role="user",
        )
    except Exception as exc:
        if "duplicate" in str(exc).lower() or "exists" in str(exc).lower():
            raise APIError(
                status_code=status.HTTP_409_CONFLICT,
                code="username_exists",
                message="Username already exists",
            ) from exc
        raise

    issued = await issue_token_pair(user=user, refresh_repo=refresh_repo)
    _set_access_cookie(response, issued.access_token)
    _set_refresh_cookie(response, issued.refresh_token)
    bind_context(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
        session_id=issued.session_id,
    )
    record_audit(
        session_id=issued.session_id,
        agent_name="auth",
        action="register",
        input_summary=f"User registration for {user.username}",
        output_summary="User created and tokens issued",
        status="success",
        confidence=1.0,
    )
    return AuthTokenResponse(
        access_token=issued.access_token,
        expires_in=issued.access_expires_in,
        refresh_expires_in=issued.refresh_expires_in,
        role=issued.role,
    )


@router.post("/refresh", response_model=AuthTokenResponse)
async def auth_refresh(
    request: Request,
    response: Response,
    payload: Optional[AuthRefreshRequest] = None,
    user_repo: UserRepository = Depends(get_user_repo),
    refresh_repo: RefreshTokenRepository = Depends(get_refresh_token_repo),
) -> AuthTokenResponse:
    # Rate limit refresh
    refresh_limit = get_rate_limiter().check(
        f"refresh:{_client_ip(request)}",
        limit=30,
        window_seconds=60,
    )
    if not refresh_limit.allowed:
        raise APIError(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            code="refresh_rate_limited",
            message="Too many refresh attempts. Please try again later.",
            details={"retry_after_seconds": refresh_limit.retry_after_seconds},
        )

    refresh_token = (payload.refresh_token if payload else None) or request.cookies.get(
        settings.auth_refresh_cookie_name
    )
    if not refresh_token:
        raise APIError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="missing_refresh_token",
            message="Refresh token is required",
        )

    issued, user = await rotate_refresh_token(
        refresh_token=refresh_token,
        refresh_repo=refresh_repo,
        user_repo=user_repo,
    )
    _set_access_cookie(response, issued.access_token)
    _set_refresh_cookie(response, issued.refresh_token)
    bind_context(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
        session_id=issued.session_id,
    )
    record_audit(
        session_id=issued.session_id,
        agent_name="auth",
        action="refresh",
        input_summary=f"Refresh requested for {user.username}",
        output_summary="Tokens rotated successfully",
        status="success",
        confidence=1.0,
    )
    return AuthTokenResponse(
        access_token=issued.access_token,
        expires_in=issued.access_expires_in,
        refresh_expires_in=issued.refresh_expires_in,
        role=issued.role,
    )


@router.post("/logout")
async def auth_logout(
    request: Request,
    response: Response,
    refresh_repo: RefreshTokenRepository = Depends(get_refresh_token_repo),
) -> Dict[str, Any]:
    refresh_token = request.cookies.get(settings.auth_refresh_cookie_name)
    if not refresh_token:
        try:
            body = await request.json()
            if isinstance(body, dict):
                refresh_token = body.get("refresh_token")
        except Exception:
            refresh_token = None
    if refresh_token:
        await revoke_refresh_token(
            refresh_token=refresh_token,
            refresh_repo=refresh_repo,
            revoke_family=True,
        )
    _clear_auth_cookies(response)
    return {"ok": True, "timestamp": now_iso()}


@router.get("/me", response_model=AuthMeResponse)
async def auth_me(
    user: AuthUser = Depends(get_current_user),
) -> AuthMeResponse:
    return AuthMeResponse(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
        is_admin=user.is_admin,
    )
