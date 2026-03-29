import json
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger("gemini_client")

_key_rotation_lock = Lock()
_next_key_cursor = 0


def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][gemini_client][{level.upper()}] {message}")


def _normalize_model_id(model_id: str) -> str:
    cleaned = (model_id or "").strip()
    if cleaned.startswith("models/"):
        return cleaned[len("models/") :]
    return cleaned


def _configured_api_keys() -> List[str]:
    keys = settings.gemini_api_key_list
    if not keys:
        raise RuntimeError("GEMINI_API_KEY or GEMINI_API_KEYS is not configured.")
    return keys


def _rotated_keys(keys: List[str]) -> List[str]:
    global _next_key_cursor
    if len(keys) <= 1:
        return list(keys)

    with _key_rotation_lock:
        start = _next_key_cursor % len(keys)
        _next_key_cursor = (_next_key_cursor + 1) % len(keys)

    return [keys[(start + offset) % len(keys)] for offset in range(len(keys))]


def _post_with_key_failover(
    *,
    model: str,
    method: str,
    payload: Dict[str, Any],
    timeout_seconds: float,
) -> Tuple[Dict[str, Any], int, int]:
    keys = _rotated_keys(_configured_api_keys())
    total_keys = len(keys)

    for slot, api_key in enumerate(keys, start=1):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}?key={api_key}"
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json(), slot, total_keys
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            body = e.response.text
            logger.error(
                "gemini_http_error",
                model=model,
                method=method,
                status_code=status_code,
                key_slot=slot,
                total_keys=total_keys,
                text=body,
            )
            if status_code == 429 and slot < total_keys:
                _terminal_log(
                    "warning",
                    f"model={model} method={method} key_slot={slot}/{total_keys} rate_limited=429 switching_to_next_key=true",
                )
                continue
            if status_code == 429:
                raise RuntimeError(
                    f"Gemini HTTP Error 429: rate limit exceeded across all configured keys ({total_keys}). Last error: {body}"
                ) from e
            raise RuntimeError(f"Gemini HTTP Error {status_code}: {body}") from e
        except Exception as e:
            logger.error(
                "gemini_call_failed",
                model=model,
                method=method,
                key_slot=slot,
                total_keys=total_keys,
                error=str(e),
            )
            raise RuntimeError(f"Failed to call Gemini: {e}") from e

    raise RuntimeError("Gemini request failed before attempting any API key.")


def call_gemini(
    prompt: str,
    structured: bool = False,
    temperature: float = 0.2,
    system_instruction: Optional[str] = None
) -> Union[str, Dict[str, Any]]:
    """
    Calls Google Gemini directly via restricted REST API.
    Raises RuntimeError if key is missing or request fails.
    """
    model = _normalize_model_id(settings.resolved_gemini_model)

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": system_instruction}]
        }

    if structured:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    try:
        data, key_slot, total_keys = _post_with_key_failover(
            model=model,
            method="generateContent",
            payload=payload,
            timeout_seconds=30.0,
        )

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates returned from Gemini.")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = parts[0].get("text", "") if parts else ""
        _terminal_log("success", f"model={model} key_slot={key_slot}/{total_keys} raw_output={text}")

        if structured:
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.error("gemini_json_parse_failed", text=text, error=str(e))
                raise ValueError(f"Gemini returned invalid JSON: {text}") from e

        return text.strip()
    except Exception as e:
        logger.error("gemini_call_failed", error=str(e))
        raise RuntimeError(f"Failed to call Gemini: {e}") from e

def get_gemini_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using configured Gemini embedding model."""
    model = _normalize_model_id(settings.resolved_gemini_embedding_model)

    payload = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text[:8000]}]}
    }

    try:
        data, key_slot, total_keys = _post_with_key_failover(
            model=model,
            method="embedContent",
            payload=payload,
            timeout_seconds=10.0,
        )
        embedding = data.get("embedding", {}).get("values", [])
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Gemini embedding response did not include vector values.")
        _terminal_log("success", f"model={model} method=embedContent key_slot={key_slot}/{total_keys} dims={len(embedding)}")
        return embedding
    except Exception as e:
        logger.error("gemini_embedding_failed", error=str(e))
        raise RuntimeError(f"Failed to generate embedding: {e}") from e
