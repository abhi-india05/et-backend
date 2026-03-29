from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from backend.auth.jwt import TOKEN_TYPE_ACCESS, decode_token
from backend.config.settings import settings
from backend.deps import get_user_repo
from backend.repositories.users import UserRepository
from backend.utils.logger import bind_context

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _extract_token(request: Request, credentials: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    if credentials and (credentials.scheme or "").lower() == "bearer":
        return credentials.credentials
    return request.cookies.get(settings.auth_cookie_name)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    user_repo: UserRepository = Depends(get_user_repo),
) -> AuthUser:
    if not settings.auth_enabled:
        user = AuthUser(user_id="anonymous", username="anonymous", role="user")
        request.state.user = user
        bind_context(user_id=user.user_id, username=user.username, role=user.role)
        return user

    token = _extract_token(request, credentials)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token")

    try:
        payload = decode_token(token, expected_type=TOKEN_TYPE_ACCESS)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed") from exc

    subject = str(payload.get("sub") or "").strip()
    if not subject:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    db_user = await user_repo.get_by_id(subject)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    user = AuthUser(user_id=db_user.user_id, username=db_user.username, role=db_user.role)
    request.state.user = user
    bind_context(user_id=user.user_id, username=user.username, role=user.role)
    return user


def require_role(*allowed_roles: str) -> Callable[[AuthUser], AuthUser]:
    allowed = {role.strip().lower() for role in allowed_roles if role.strip()}

    async def _dependency(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if allowed and user.role.lower() not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return _dependency
