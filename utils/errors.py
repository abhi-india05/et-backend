"""Centralised error types and response formatting for the API layer."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from backend.utils.helpers import generate_request_id, now_iso


class APIError(Exception):
    """Application-level error raised anywhere in the stack.

    Caught by the global exception handler and rendered as a standard JSON
    error response.
    """

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def error_payload(
    *,
    request_id: str,
    code: str,
    message: str,
    details: Any = None,
) -> Dict[str, Any]:
    return {
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "details": jsonable_encoder(details),
            "request_id": request_id,
            "timestamp": now_iso(),
        },
        "meta": {
            "request_id": request_id,
            "timestamp": now_iso(),
        },
    }


def error_response(
    *,
    request_id: str,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code,
        content=error_payload(
            request_id=request_id,
            code=code,
            message=message,
            details=details,
        ),
    )
    response.headers["X-Request-ID"] = request_id
    for key, value in (headers or {}).items():
        response.headers[key] = value
    return response


def success_payload(
    *,
    data: Any,
    request_id: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_meta: Dict[str, Any] = {
        "request_id": request_id,
        "timestamp": now_iso(),
    }
    if meta:
        base_meta.update(meta)
    return {
        "data": data,
        "error": None,
        "meta": base_meta,
    }
