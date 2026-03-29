from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt

from backend.config.settings import settings
from backend.utils.helpers import generate_id, utcnow

TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


def _base_payload(
    *,
    subject: str,
    username: str,
    role: str,
    token_type: str,
    expires_delta: timedelta,
    session_id: Optional[str],
    token_id: Optional[str],
    family_id: Optional[str],
) -> Dict[str, Any]:
    now = utcnow()
    token_id = token_id or generate_id("tok")
    payload: Dict[str, Any] = {
        "sub": subject,
        "username": username,
        "role": role,
        "type": token_type,
        "jti": token_id,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "iss": settings.auth_issuer,
        "aud": settings.auth_audience,
    }
    if session_id:
        payload["sid"] = session_id
    if family_id:
        payload["family"] = family_id
    return payload


def create_access_token(
    *,
    subject: str,
    username: str,
    role: str,
    session_id: Optional[str] = None,
    token_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _base_payload(
        subject=subject,
        username=username,
        role=role,
        token_type=TOKEN_TYPE_ACCESS,
        expires_delta=timedelta(minutes=settings.auth_access_token_expire_minutes),
        session_id=session_id,
        token_id=token_id,
        family_id=None,
    )
    return {"token": jwt.encode(payload, settings.auth_secret_key, algorithm=settings.auth_algorithm), "payload": payload}


def create_refresh_token(
    *,
    subject: str,
    username: str,
    role: str,
    session_id: str,
    family_id: Optional[str] = None,
    token_id: Optional[str] = None,
) -> Dict[str, Any]:
    family_id = family_id or generate_id("fam")
    payload = _base_payload(
        subject=subject,
        username=username,
        role=role,
        token_type=TOKEN_TYPE_REFRESH,
        expires_delta=timedelta(days=settings.auth_refresh_token_expire_days),
        session_id=session_id,
        token_id=token_id,
        family_id=family_id,
    )
    return {"token": jwt.encode(payload, settings.auth_secret_key, algorithm=settings.auth_algorithm), "payload": payload}


def decode_token(token: str, *, expected_type: Optional[str] = None) -> Dict[str, Any]:
    payload = jwt.decode(
        token,
        settings.auth_secret_key,
        algorithms=[settings.auth_algorithm],
        audience=settings.auth_audience,
        issuer=settings.auth_issuer,
    )
    if expected_type and payload.get("type") != expected_type:
        raise JWTError(f"Unexpected token type: {payload.get('type')}")
    return payload
