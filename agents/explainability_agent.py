from __future__ import annotations

import json
from typing import Any, Dict

from tenacity import stop_after_attempt, wait_exponential
from tenacity import retry as tenacity_retry

from backend.config.settings import settings
from backend.models.schemas import ExplainabilityOutput
from backend.utils.helpers import build_agent_response, now_iso
from backend.utils.logger import get_logger, record_audit

logger = get_logger("explainability_agent")


def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][explainability_agent][{level.upper()}] {message}")

def _extract_key_data(agent_name: str, output: Dict[str, Any]) -> Dict[str, Any]:
    data = output.get("data", {})
    if agent_name == "prospecting_agent":
        return {"leads_found": len(data.get("leads", []))}
    if agent_name == "digital_twin_agent":
        return {"twins_created": len(data.get("twin_profiles", []))}
    if agent_name == "outreach_agent":
        return {"sequences_generated": data.get("total_sequences", 0)}
    if agent_name == "deal_intelligence_agent":
        return {"risks_found": data.get("total_at_risk", 0), "critical": data.get("critical_count", 0)}
    if agent_name == "churn_agent":
        return {"churn_risks": len(data.get("top_churn_risks", [])), "arr_at_risk": data.get("total_arr_at_risk", 0)}
    if agent_name == "action_agent":
        return {"emails_sent": data.get("emails_sent", 0), "crm_updates": data.get("crm_updates", 0)}
    return {}

@tenacity_retry(stop=stop_after_attempt(settings.max_retries + 1), wait=wait_exponential(min=1, max=4))
def _call_llm(prompt: str) -> ExplainabilityOutput:
    from backend.llm.gemini_client import call_gemini

    try:
        response = call_gemini(prompt, structured=True, temperature=0.3)
        return ExplainabilityOutput(**response)
    except Exception as e:
        logger.error("explainability_llm_failed", error=str(e))
        raise RuntimeError(f"Explainability generation failed: {e}") from e

def run_explainability_agent(session_id: str, agent_outputs: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    logger.info("explainability_agent_start", session_id=session_id, task_type=task_type)
    summaries = {}
    for agent_name, output in agent_outputs.items():
        if isinstance(output, dict):
            summaries[agent_name] = {
                "status": output.get("status"),
                "reasoning": output.get("reasoning", "")[:300],
                "confidence": output.get("confidence", 0),
                "key_data": _extract_key_data(agent_name, output),
            }

    try:
        prompt = f"""You are an AI explainability specialist. Generate a clear explanation of what the RevOps AI system did and why.

Task Type: {task_type}
Session ID: {session_id}
Agent Summaries:
{json.dumps(summaries, indent=2)}

Return ONLY valid JSON with:
{{
  "executive_summary": "summary",
  "decision_chain": [{{"step": 1, "agent": "agent_name", "decision": "decision", "why": "why", "confidence": 0.8, "impact": "impact"}}],
  "key_insights": ["insight"],
  "data_sources_used": ["source"],
  "overall_confidence": 0.82,
  "limitations": ["limitation"],
  "human_review_recommended": false,
  "human_review_reasons": [],
  "impact_metrics": {{"actions_taken": 0}}
}}"""
        explanation = _call_llm(prompt)
    except Exception as exc:
        logger.warning("explainability_llm_failed", error=str(exc))
        _terminal_log("failure", f"LLM explanation generation failed: {exc}")
        raise RuntimeError(f"Explainability generation failed: {exc}") from exc

    payload = explanation.model_dump()
    low_confidence_agents = [name for name, item in summaries.items() if item.get("confidence", 1.0) < 0.5]
    if low_confidence_agents:
        payload["human_review_recommended"] = True
        payload["human_review_reasons"] = payload.get("human_review_reasons", []) + [
            f"Low confidence from: {', '.join(low_confidence_agents)}"
        ]
    full_explanation = {
        **payload,
        "session_id": session_id,
        "task_type": task_type,
        "generated_at": now_iso(),
        "agent_count": len(agent_outputs),
        "failed_agents": [name for name, item in summaries.items() if item.get("status") == "failure"],
    }
    result = build_agent_response(
        status="success",
        data=full_explanation,
        reasoning=payload.get("executive_summary", "Explanation generated."),
        confidence=payload.get("overall_confidence", 0.8),
        agent_name="explainability_agent",
        tools_used=["llm"],
    )
    _terminal_log("success", f"Generated explainability output for task '{task_type}'")
    record_audit(
        session_id=session_id,
        agent_name="explainability_agent",
        action="generate_explanation",
        input_summary=f"Task: {task_type}, Agents: {list(agent_outputs.keys())}",
        output_summary=payload.get("executive_summary", "Explanation complete")[:200],
        status="success",
        reasoning=result["reasoning"],
        confidence=result["confidence"],
    )
    return result
