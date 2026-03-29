from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from jose import JWTError

from backend.auth.jwt import TOKEN_TYPE_REFRESH, create_access_token, create_refresh_token, decode_token
from backend.config.settings import settings
from backend.repositories.refresh_tokens import RefreshTokenRecord, RefreshTokenRepository
from backend.repositories.users import UserInDB, UserRepository
from backend.utils.helpers import generate_session_id, hash_token, utcnow
from backend.utils.logger import get_logger

logger = get_logger("auth_service")


class AuthServiceError(Exception):
    code = "auth_error"


class InvalidRefreshTokenError(AuthServiceError):
    code = "invalid_refresh_token"


class RefreshTokenReuseError(AuthServiceError):
    code = "refresh_token_reuse_detected"


@dataclass(frozen=True)
class IssuedTokenPair:
    access_token: str
    refresh_token: str
    access_expires_in: int
    refresh_expires_in: int
    role: str
    session_id: str
    family_id: str
    access_token_id: str
    refresh_token_id: str


async def issue_token_pair(
    *,
    user: UserInDB,
    refresh_repo: RefreshTokenRepository,
    session_id: Optional[str] = None,
    family_id: Optional[str] = None,
) -> IssuedTokenPair:
    active_session_id = session_id or generate_session_id()
    access = create_access_token(
        subject=user.user_id,
        username=user.username,
        role=user.role,
        session_id=active_session_id,
    )
    refresh = create_refresh_token(
        subject=user.user_id,
        username=user.username,
        role=user.role,
        session_id=active_session_id,
        family_id=family_id,
    )
    refresh_payload = refresh["payload"]
    await refresh_repo.create_token(
        RefreshTokenRecord(
            token_id=str(refresh_payload["jti"]),
            user_id=user.user_id,
            session_id=active_session_id,
            family_id=str(refresh_payload["family"]),
            token_hash=hash_token(refresh["token"]),
            expires_at=datetime.fromtimestamp(int(refresh_payload["exp"]), tz=timezone.utc),
            created_at=utcnow(),
            updated_at=utcnow(),
            revoked_at=None,
            rotated_at=None,
            replaced_by_token_id=None,
            reuse_detected=False,
        )
    )
    return IssuedTokenPair(
        access_token=access["token"],
        refresh_token=refresh["token"],
        access_expires_in=settings.auth_access_token_expire_seconds,
        refresh_expires_in=settings.auth_refresh_token_expire_seconds,
        role=user.role,
        session_id=active_session_id,
        family_id=str(refresh_payload["family"]),
        access_token_id=str(access["payload"]["jti"]),
        refresh_token_id=str(refresh_payload["jti"]),
    )


async def rotate_refresh_token(
    *,
    refresh_token: str,
    refresh_repo: RefreshTokenRepository,
    user_repo: UserRepository,
) -> tuple[IssuedTokenPair, UserInDB]:
    try:
        payload = decode_token(refresh_token, expected_type=TOKEN_TYPE_REFRESH)
    except JWTError as exc:
        raise InvalidRefreshTokenError("Refresh token verification failed") from exc

    token_id = str(payload.get("jti") or "").strip()
    family_id = str(payload.get("family") or "").strip()
    subject = str(payload.get("sub") or "").strip()
    if not token_id or not family_id or not subject:
        raise InvalidRefreshTokenError("Refresh token is missing required claims")

    record = await refresh_repo.get_token(token_id)
    if not record:
        raise InvalidRefreshTokenError("Refresh token is not recognized")

    if record.token_hash != hash_token(refresh_token):
        await refresh_repo.revoke_family(family_id=family_id)
        logger.warning("refresh_reuse_attack", token_id=token_id, family_id=family_id)
        raise RefreshTokenReuseError("Refresh token hash mismatch detected")

    if record.revoked_at is not None or record.rotated_at is not None or record.replaced_by_token_id:
        await refresh_repo.revoke_family(family_id=family_id)
        logger.warning("refresh_reuse_attack", token_id=token_id, family_id=family_id)
        raise RefreshTokenReuseError("Refresh token reuse detected")

    if record.expires_at <= utcnow():
        await refresh_repo.revoke_token(token_id=token_id)
        raise InvalidRefreshTokenError("Refresh token expired")

    user = await user_repo.get_by_id(subject)
    if not user:
        await refresh_repo.revoke_family(family_id=family_id)
        raise InvalidRefreshTokenError("User for refresh token no longer exists")

    issued = await issue_token_pair(
        user=user,
        refresh_repo=refresh_repo,
        session_id=record.session_id,
        family_id=record.family_id,
    )
    await refresh_repo.rotate_token(token_id=token_id, replacement_token_id=issued.refresh_token_id)
    return issued, user


async def revoke_refresh_token(
    *,
    refresh_token: str,
    refresh_repo: RefreshTokenRepository,
    revoke_family: bool = False,
) -> bool:
    try:
        payload = decode_token(refresh_token, expected_type=TOKEN_TYPE_REFRESH)
    except JWTError:
        return False

    token_id = str(payload.get("jti") or "").strip()
    family_id = str(payload.get("family") or "").strip()
    if revoke_family and family_id:
        await refresh_repo.revoke_family(family_id=family_id)
        return True
    if token_id:
        await refresh_repo.revoke_token(token_id=token_id)
        return True
    return False
