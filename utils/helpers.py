from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(UTC)


def now_iso() -> str:
    return utcnow().isoformat().replace("+00:00", "Z")


def days_ago(days: int) -> str:
    return (utcnow() - timedelta(days=days)).isoformat().replace("+00:00", "Z")


def generate_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:12]}"


def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def generate_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def truncate_text(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def sanitize_text(value: Any, *, max_len: int, allow_empty: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return "" if allow_empty else None
    return text[:max_len]


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_company_name(name: str) -> str:
    return hashlib.md5(name.lower().encode("utf-8")).hexdigest()[:8]


def safe_json_loads(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        candidate = text.strip()
        if candidate.startswith("```json"):
            candidate = candidate[7:]
        elif candidate.startswith("```"):
            candidate = candidate[3:]
        if candidate.endswith("```"):
            candidate = candidate[:-3]
        return json.loads(candidate.strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def extract_json_from_text(text: str) -> Optional[Any]:
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return safe_json_loads(match.group())
    return None


def compute_risk_score(factors: Dict[str, float]) -> float:
    if not factors:
        return 0.0
    values = list(factors.values())
    return round(min(1.0, sum(values) / len(values)), 4)


def format_email_body(template: str, variables: Dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    normalized = date_str.strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(normalized, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def days_since(date_str: str) -> int:
    parsed = parse_date(date_str)
    if not parsed:
        return 0
    return max(0, (utcnow() - parsed.astimezone(UTC)).days)


def clamp_page_size(page_size: int, *, default: int, max_value: int) -> int:
    if page_size <= 0:
        return default
    return max(1, min(page_size, max_value))


def build_agent_response(
    status: str,
    data: Dict[str, Any],
    reasoning: str,
    confidence: float,
    agent_name: str,
    error: Optional[str] = None,
    *,
    tools_used: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "data": data,
        "reasoning": reasoning,
        "confidence": round(float(confidence), 4),
        "agent_name": agent_name,
        "timestamp": now_iso(),
        "error": error,
        "tools_used": tools_used or [],
        "metadata": metadata or {},
    }
