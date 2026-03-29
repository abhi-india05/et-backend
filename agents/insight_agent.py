from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from backend.llm.gemini_client import call_gemini

def _norm(value: Optional[str]) -> str:
    return (value or "").strip()

def _load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_insights(lead: dict) -> dict:
    role = _norm(lead.get("role") or lead.get("title"))
    company = _norm(lead.get("company"))
    headline = _norm(lead.get("headline"))
    about = _norm(lead.get("about"))
    activity = _norm(lead.get("activity"))

    used_fields: List[str] = []
    for field_name, field_value in {
        "role": role, "company": company, "headline": headline, "about": about, "activity": activity,
    }.items():
        if field_value:
            used_fields.append(field_name)

    prompt_template = _load_prompt("insight_prompt.txt")
    
    extracted_data = {
        "role": role,
        "company": company,
        "headline": headline,
        "about": about,
        "activity": activity,
    }
    prompt = prompt_template.replace("{lead_data}", json.dumps(extracted_data))

    try:
        response = call_gemini(prompt, structured=True, temperature=0.1)
        return {
            "role": role,
            "explicit_signals": response.get("explicit_signals", []),
            "inferred_signals": response.get("inferred_signals", []),
            "pain_hypothesis": response.get("pain_hypothesis"),
            "confidence": float(response.get("confidence", 0.5)),
            "reasoning": response.get("reasoning", f"Analyzed {', '.join(used_fields)} via Gemini."),
        }
    except Exception as e:
        return {
            "role": role,
            "explicit_signals": [],
            "inferred_signals": [],
            "pain_hypothesis": None,
            "confidence": 0.0,
            "reasoning": f"Insight generation failed: {e}",
        }


def evaluate_product_fit(product: dict, insights: dict) -> dict:
    product_name = _norm((product or {}).get("name"))
    product_description = _norm((product or {}).get("description"))

    if not product_name:
        return {
            "is_relevant": False,
            "reason": "No product context provided.",
            "confidence": 0.0,
        }

    prompt = f"""You evaluate whether a product is relevant to a prospective lead's insights.
Product: {product_name} - {product_description}
Lead Insights: {json.dumps(insights)}

Determine if the product directly addresses the lead's inferred signals or pain hypothesis.
Output ONLY strict JSON:
{{
  "is_relevant": boolean,
  "reason": "string (why it fits or doesn't fit)",
  "confidence": 0.0 to 1.0
}}
"""
    try:
        response = call_gemini(prompt, structured=True, temperature=0.1)
        return {
            "is_relevant": bool(response.get("is_relevant")),
            "reason": str(response.get("reason", "")),
            "confidence": float(response.get("confidence", 0.5)),
        }
    except Exception as e:
        return {
            "is_relevant": False,
            "reason": f"Evaluation failed: {e}",
            "confidence": 0.0,
        }
