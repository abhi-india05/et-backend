from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from backend.agents.insight_agent import evaluate_product_fit, extract_insights
from backend.llm.gemini_client import call_gemini
from backend.memory.vector_store import get_vector_store
from backend.models.schemas import EmailSequenceResult
from backend.utils.helpers import build_agent_response, generate_id
from backend.utils.logger import get_logger, record_audit

logger = get_logger("outreach_agent")

_EXPLANATION_ALLOWED_KEYS = {"used_fields", "insight", "reasoning", "confidence"}

def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][outreach_agent][{level.upper()}] {message}")


def _safe_confidence(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _normalize_explanation(raw: Any, insights: Dict[str, Any], blended_confidence: float) -> Dict[str, Any]:
    if isinstance(raw, dict):
        source = dict(raw)
    else:
        source = {"reasoning": str(raw or "")}

    insight_text = str(
        source.get("insight")
        or source.get("insight_used")
        or insights.get("pain_hypothesis")
        or ""
    )

    used_fields = source.get("used_fields")
    if not isinstance(used_fields, list):
        used_fields = []

    normalized = {
        "used_fields": [str(item) for item in used_fields if str(item).strip()],
        "insight": insight_text,
        "reasoning": str(source.get("reasoning") or ""),
        "confidence": _safe_confidence(source.get("confidence"), blended_confidence),
    }

    # Hard whitelist to prevent strict schema failures from unknown LLM keys.
    return {key: value for key, value in normalized.items() if key in _EXPLANATION_ALLOWED_KEYS}

def _load_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "outreach_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_outreach_email(lead: Dict[str, Any], insights: Dict[str, Any], product_context: Dict[str, str], tone_hint: str) -> Dict[str, Any]:
    prompt_template = _load_prompt()
    prompt = prompt_template.replace("{lead}", json.dumps(lead))\
                            .replace("{insights}", json.dumps(insights))\
                            .replace("{product_context}", json.dumps(product_context))\
                            .replace("{tone_hint}", tone_hint)

    product_fit = evaluate_product_fit(product_context, insights)

    try:
        response = call_gemini(prompt, structured=True, temperature=0.7)
        # add product_fit for parity with old schema
        response["product_fit"] = product_fit
        
        # default structure
        if "subject" not in response:
            response["subject"] = "Quick check-in"
        if "body" not in response:
            response["body"] = "Hi there,\n\nI saw your profile and wanted to connect.\n\nBest,"

        explanation = response.get("explanation", {})
        if not isinstance(explanation, dict):
            explanation = {"reasoning": str(explanation)}
        
        confidence = float(insights.get("confidence") or 0.0)
        fit_confidence = float(product_fit.get("confidence") or 0.0)
        blended_confidence = confidence
        if product_fit.get("is_relevant", False):
            blended_confidence = min(0.98, (0.7 * confidence) + (0.3 * fit_confidence))
            
        response["explanation"] = _normalize_explanation(
            explanation,
            insights,
            round(blended_confidence, 2),
        )

        return response
    except Exception as e:
        logger.error("outreach_llm_failed", error=str(e))
        _terminal_log("error", f"Outreach LLM failed for {lead.get('name')}: {e}")
        return {
            "subject": f"Question for {lead.get('company', 'your team')}",
            "body": f"Hi {lead.get('name') or 'there'},\n\nI noticed your role at {lead.get('company')} and wanted to connect.\n\nWould you be open to a brief chat?",
            "explanation": {
                "used_fields": [],
                "insight": "",
                "reasoning": f"Fallback applied due to error: {e}",
                "confidence": 0.0,
            },
            "product_fit": product_fit,
        }

def run_outreach_agent(
    leads: List[Dict[str, Any]],
    twin_profiles: List[Dict[str, Any]],
    company: str,
    product_context: Dict[str, str],
    session_id: str,
    user_id: str,
) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")

    logger.info("outreach_agent_start", company=company, session_id=session_id)
    memory = get_vector_store()
    namespace = user_id
    sequences: List[Dict[str, Any]] = []

    for index, lead in enumerate(leads[:2]):
        twin = twin_profiles[index] if index < len(twin_profiles) else {}
        tone_hint = str(twin.get("recommended_tone") or "consultative")
        lead_id = str(lead.get("id") or lead.get("lead_id") or lead.get("source_profile") or "")
        lead_name = str(lead.get("name") or "there")

        insights = extract_insights(lead)
        generated = generate_outreach_email(lead, insights, product_context, tone_hint)

        sequence_id = generate_id("seq")
        confidence = float(generated["explanation"].get("confidence") or 0.0)
        predicted_open = round(min(0.65, max(0.05, 0.12 + (0.35 * confidence))), 4)
        
        product_fit = generated.get("product_fit", {})
        if product_fit.get("is_relevant", False):
            predicted_reply = round(min(0.35, max(0.01, 0.04 + (0.22 * confidence))), 4)
        else:
            predicted_reply = round(min(0.2, max(0.01, 0.02 + (0.1 * confidence))), 4)

        sequence = EmailSequenceResult(
            lead_id=lead_id or None,
            lead_name=lead_name,
            lead_email=str(lead.get("email") or ""),
            sequence_id=sequence_id,
            emails=[
                {
                    "step": 1,
                    "send_day": 1,
                    "subject": generated.get("subject", ""),
                    "body": generated.get("body", ""),
                    "email": generated.get("body", ""),
                    "cta": "See body",
                    "angle": f"insight-grounded/{tone_hint}",
                    "explanation": generated.get("explanation", {}),
                }
            ],
            sequence_strategy="Insight-first outreach grounded in LinkedIn profile fields using Gemini.",
            predicted_open_rate=predicted_open,
            predicted_reply_rate=predicted_reply,
        )
        sequences.append(sequence.model_dump())

    average_open = sum(item.get("predicted_open_rate", 0.0) for item in sequences) / max(len(sequences), 1)
    average_reply = sum(item.get("predicted_reply_rate", 0.0) for item in sequences) / max(len(sequences), 1)

    memory.add_document(
        doc_id=generate_id("outreach"),
        content=f"Grounded outreach generated for {company} using Gemini.",
        metadata={"company": company, "agent": "outreach", "session_id": session_id, "user_id": user_id or ""},
        namespace=namespace,
    )

    result = build_agent_response(
        status="success",
        data={
            "sequences": sequences,
            "total_sequences": len(sequences),
            "avg_predicted_open_rate": round(average_open, 4),
            "avg_predicted_reply_rate": round(average_reply, 4),
            "company": company,
            "product_context": product_context,
            "grounding_mode": "gemini_dynamic",
        },
        reasoning=(
            f"Generated {len(sequences)} grounded outreach sequence(s) for {company} using Gemini."
        ),
        confidence=0.82,
        agent_name="outreach_agent",
        tools_used=["vector_memory"],
    )
    _terminal_log("success", f"Generated {len(sequences)} grounded outreach sequence(s) for {company}")
    record_audit(
        session_id=session_id,
        agent_name="outreach_agent",
        action="generate_sequences",
        input_summary=f"Company: {company}, Leads: {len(leads)}",
        output_summary=f"Generated {len(sequences)} grounded sequences",
        status="success",
        reasoning=result["reasoning"],
        confidence=0.82,
    )
    return result
