from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from backend.llm.gemini_client import call_gemini
from backend.memory.vector_store import get_vector_store
from backend.models.schemas import DigitalTwinProfileOutput
from backend.utils.helpers import build_agent_response, generate_id
from backend.utils.logger import get_logger, record_audit

logger = get_logger("digital_twin_agent")

def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][digital_twin_agent][{level.upper()}] {message}")

def _load_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "digital_twin_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _build_grounded_twin(lead: Dict[str, Any], company: str) -> DigitalTwinProfileOutput:
    prompt_template = _load_prompt()
    prompt = prompt_template.replace("{lead_data}", json.dumps(lead)).replace("{company}", company)

    try:
        response = call_gemini(prompt, structured=True, temperature=0.1)
        return DigitalTwinProfileOutput(**response)
    except Exception as e:
        logger.error("digital_twin_llm_failed", error=str(e))
        _terminal_log("error", f"LLM generation failed for {lead.get('name')}: {e}")
        # fallback twin
        return DigitalTwinProfileOutput(
            buyer_name=lead.get("name", "Unknown"),
            buyer_title=lead.get("title", "Revenue Leader"),
            buying_style="evaluator",
            primary_motivations=["Profile-grounded relevance"],
            top_objections=[
                {
                    "objection": "Insufficient profile context to evaluate fit.",
                    "severity": "high",
                    "counter_strategy": "Use only explicitly observable profile details before proposing any value mapping.",
                }
            ],
            decision_criteria=["Grounded evidence", "Low-friction next step"],
            likely_questions=["What profile evidence are you using for this outreach?"],
            emotional_triggers=["Clarity"],
            risk_perception="high",
            estimated_decision_timeline="unknown",
            recommended_tone="consultative",
            opening_hook="I want to keep this grounded to what is visible on your profile.",
            confidence_score=0.35,
        )

def run_digital_twin_agent(
    leads: List[Dict[str, Any]],
    company: str,
    industry: str,
    session_id: str,
    user_id: str,
    product_context: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")

    logger.info("digital_twin_agent_start", company=company, session_id=session_id)
    memory = get_vector_store()
    namespace = user_id
    _ = industry
    _ = product_context

    twin_profiles: List[Dict[str, Any]] = []
    for lead in leads[:2]:
        twin = _build_grounded_twin(lead, company)
        twin_profiles.append(twin.model_dump())

    avg_confidence = sum(profile.get("confidence_score", 0.5) for profile in twin_profiles) / max(len(twin_profiles), 1)
    memory.add_document(
        doc_id=generate_id("twin"),
        content=f"Digital twin profiles for {company}: {[profile.get('buyer_name') for profile in twin_profiles]}",
        metadata={"company": company, "agent": "digital_twin", "session_id": session_id, "user_id": user_id or ""},
        namespace=namespace,
    )

    result = build_agent_response(
        status="success",
        data={"twin_profiles": twin_profiles, "company": company},
        reasoning=f"Simulated {len(twin_profiles)} buyer personas for {company}.",
        confidence=avg_confidence,
        agent_name="digital_twin_agent",
        tools_used=["vector_memory"],
    )
    _terminal_log("success", f"Generated {len(twin_profiles)} digital twin profiles for {company}")
    record_audit(
        session_id=session_id,
        agent_name="digital_twin_agent",
        action="simulate_buyers",
        input_summary=f"Company: {company}, Leads: {len(leads)}",
        output_summary=f"Simulated {len(twin_profiles)} buyer twins",
        status="success",
        reasoning=result["reasoning"],
        confidence=avg_confidence,
    )
    return result
