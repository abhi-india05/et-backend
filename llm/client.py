from __future__ import annotations

from typing import Optional

from openai import OpenAI

from backend.config.settings import settings


def get_llm_client(api_key: Optional[str] = None) -> OpenAI:
    key = (api_key or settings.llm_api_key or "").strip()
    if not key:
        raise RuntimeError(
            f"No API key configured for LLM provider '{settings.llm_provider}'. "
            "Set OPENAI_API_KEY or GEMINI_API_KEY."
        )

    base_url = settings.llm_base_url
    if base_url:
        return OpenAI(api_key=key, base_url=base_url)
    return OpenAI(api_key=key)
