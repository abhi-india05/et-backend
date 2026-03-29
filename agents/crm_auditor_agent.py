from __future__ import annotations

import json
from typing import Any, Dict, List

from tenacity import stop_after_attempt, wait_exponential
from tenacity import retry as tenacity_retry

from backend.config.settings import settings
from pydantic import Field

from backend.models.schemas import StrictBaseModel
from backend.tools.crm_tool import get_all_accounts, get_pipeline_stats
from backend.utils.helpers import build_agent_response, days_since
from backend.utils.logger import get_logger, record_audit

logger = get_logger("crm_auditor_agent")

def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][crm_auditor_agent][{level.upper()}] {message}")

STUCK_STAGE_THRESHOLDS = {
    "Discovery": 21,
    "Proposal": 14,
    "Negotiation": 21,
    "Prospecting": 30,
}

class CRMAuditRecommendation(StrictBaseModel):
    audit_score: int
    health_rating: str
    top_priorities: List[Dict[str, Any]] = Field(default_factory=list)
    process_gaps: List[str] = Field(default_factory=list)
    immediate_actions: List[str] = Field(default_factory=list)
    revenue_recovery_potential: float = 0.0
    recommendations_summary: str

def detect_crm_anomalies(accounts: List[Dict[str, Any]]) -> Dict[str, Any]:
    missed_followups = []
    stuck_deals = []
    data_quality_issues = []
    revenue_at_risk = 0.0
    for account in accounts:
        stage = account.get("stage", "")
        days_in_stage = account.get("days_in_stage", 0)
        if stage in ["Closed Won", "Closed Lost"]:
            continue
        days_inactive = days_since(account.get("last_activity", "")) if account.get("last_activity") else days_in_stage
        threshold = STUCK_STAGE_THRESHOLDS.get(stage, 30)
        if days_in_stage > threshold:
            stuck_deals.append(
                {
                    "account_id": account.get("account_id"),
                    "company": account.get("company"),
                    "stage": stage,
                    "days_in_stage": days_in_stage,
                    "threshold": threshold,
                    "overage_days": days_in_stage - threshold,
                    "deal_value": account.get("deal_value", 0),
                    "severity": "critical" if days_in_stage > threshold * 2 else "high",
                }
            )
            revenue_at_risk += float(account.get("deal_value", 0))
        if days_inactive >= 7:
            missed_followups.append(
                {
                    "account_id": account.get("account_id"),
                    "company": account.get("company"),
                    "contact_name": account.get("contact_name"),
                    "email": account.get("email"),
                    "stage": stage,
                    "days_since_contact": days_inactive,
                    "deal_value": account.get("deal_value", 0),
                    "urgency": "high" if days_inactive >= 14 else "medium",
                }
            )
        if not account.get("email") or not account.get("contact_name"):
            data_quality_issues.append(
                {
                    "account_id": account.get("account_id"),
                    "company": account.get("company"),
                    "issue": "Missing contact email or name",
                }
            )
    return {
        "missed_followups": sorted(missed_followups, key=lambda item: item["days_since_contact"], reverse=True),
        "stuck_deals": sorted(stuck_deals, key=lambda item: item["days_in_stage"], reverse=True),
        "data_quality_issues": data_quality_issues,
        "revenue_at_risk": revenue_at_risk,
    }

@tenacity_retry(stop=stop_after_attempt(settings.max_retries + 1), wait=wait_exponential(min=1, max=4))
def _call_llm(prompt: str) -> CRMAuditRecommendation:
    from backend.llm.gemini_client import call_gemini
    
    try:
        response = call_gemini(prompt, structured=True, temperature=0.3)
        return CRMAuditRecommendation(**response)
    except Exception as e:
        logger.error("crm_auditor_llm_failed", error=str(e))
        raise RuntimeError(f"CRM auditor recommendation generation failed: {e}") from e

def run_crm_auditor_agent(session_id: str, user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")
        
    logger.info("crm_auditor_agent_start", session_id=session_id, user_id=user_id)
    accounts = get_all_accounts(user_id=user_id)
    anomalies = detect_crm_anomalies(accounts)
    pipeline_stats = get_pipeline_stats(user_id=user_id)
    
    prompt = f"""You are a CRM audit specialist. Analyze this CRM audit data and provide strategic recommendations.

Pipeline Stats: {json.dumps(pipeline_stats, indent=2)}
Audit Findings: {json.dumps(anomalies, indent=2)}

Return ONLY valid JSON with:
{{
  "audit_score": 72,
  "health_rating": "Fair | Good | Poor | Critical",
  "top_priorities": [{{"priority": 1, "action": "Specific action", "impact": "Expected impact", "deals_affected": 3}}],
  "process_gaps": ["gap 1"],
  "immediate_actions": ["action 1"],
  "revenue_recovery_potential": 450000,
  "recommendations_summary": "Brief assessment"
}}"""

    try:
        recommendations = _call_llm(prompt).model_dump()
    except Exception as exc:
        logger.warning("crm_auditor_llm_failed", error=str(exc))
        _terminal_log("failure", f"LLM CRM audit recommendation failed: {exc}")
        recommendations = {
            "audit_score": 0,
            "health_rating": "Poor",
            "top_priorities": [],
            "process_gaps": ["Error processing audit"],
            "immediate_actions": [],
            "revenue_recovery_potential": 0.0,
            "recommendations_summary": "Failed to generate recommendations",
        }

    data = {
        **anomalies,
        "pipeline_stats": pipeline_stats,
        "recommendations": recommendations,
        "total_accounts_audited": len(accounts),
    }
    result = build_agent_response(
        status="success",
        data=data,
        reasoning=f"CRM audit complete. Found {len(anomalies['missed_followups'])} missed follow-ups and {len(anomalies['stuck_deals'])} stuck deals.",
        confidence=0.9,
        agent_name="crm_auditor_agent",
        tools_used=["crm_tool", "llm"],
    )
    _terminal_log("success", f"CRM audit completed for {len(accounts)} accounts")
    record_audit(
        session_id=session_id,
        agent_name="crm_auditor_agent",
        action="audit_crm",
        input_summary=f"Audited {len(accounts)} accounts",
        output_summary=f"{len(anomalies['missed_followups'])} missed followups, {len(anomalies['stuck_deals'])} stuck deals",
        status="success",
        reasoning=result["reasoning"],
        confidence=0.9,
    )
    return result
