from __future__ import annotations

import re

import bcrypt as _bcrypt

if not hasattr(_bcrypt, "__about__"):
    # Passlib 1.7.x expects bcrypt.__about__.__version__, but newer bcrypt
    # exposes __version__ directly.
    class _BcryptAbout:
        __version__ = getattr(_bcrypt, "__version__", "unknown")

    _bcrypt.__about__ = _BcryptAbout()  # type: ignore[attr-defined]

from passlib.context import CryptContext

from backend.config.settings import settings

pwd_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")


class PasswordValidationError(ValueError):
    pass


def validate_password_strength(password: str) -> None:
    requirements: list[str] = []
    if len(password) < settings.auth_password_min_length:
        requirements.append(f"at least {settings.auth_password_min_length} characters")
    if settings.auth_password_require_upper and not re.search(r"[A-Z]", password):
        requirements.append("one uppercase letter")
    if settings.auth_password_require_lower and not re.search(r"[a-z]", password):
        requirements.append("one lowercase letter")
    if settings.auth_password_require_digit and not re.search(r"\d", password):
        requirements.append("one number")
    if settings.auth_password_require_special and not re.search(r"[^A-Za-z0-9]", password):
        requirements.append("one special character")
    if requirements:
        raise PasswordValidationError("Password must contain " + ", ".join(requirements))


def hash_password(password: str) -> str:
    validate_password_strength(password)
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(password, password_hash)
    except (TypeError, ValueError):
        return False
