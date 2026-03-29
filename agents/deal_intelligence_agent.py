from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config.settings import settings
from backend.memory.vector_store import get_vector_store
from backend.models.schemas import DealRisk
from backend.tools.crm_tool import get_at_risk_deals
from backend.tools.scraping_tool import detect_intent_signals
from backend.utils.helpers import build_agent_response, days_since, generate_id
from backend.utils.logger import get_logger, record_audit

logger = get_logger("deal_intelligence_agent")


def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][deal_intelligence_agent][{level.upper()}] {message}")


def _load_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "deal_risk_prompt.txt")
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _normalize_company(value: str) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _build_customer_signal_indexes(
    customer_engagement_signals: Optional[List[Dict[str, Any]]],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_company: Dict[str, Dict[str, Any]] = {}
    by_email: Dict[str, Dict[str, Any]] = {}

    for signal in customer_engagement_signals or []:
        if not isinstance(signal, dict):
            continue
        marked_at = str(signal.get("marked_as_customer_at") or "").strip()
        if not marked_at:
            continue

        company_key = _normalize_company(signal.get("company_name", ""))
        email_key = str(signal.get("contact_email") or "").strip().lower()

        if company_key:
            existing = by_company.get(company_key)
            existing_marked = str((existing or {}).get("marked_as_customer_at") or "")
            if not existing or days_since(marked_at) <= days_since(existing_marked):
                by_company[company_key] = signal

        if email_key:
            existing = by_email.get(email_key)
            existing_marked = str((existing or {}).get("marked_as_customer_at") or "")
            if not existing or days_since(marked_at) <= days_since(existing_marked):
                by_email[email_key] = signal

    return by_company, by_email


def _match_customer_signal(
    account: Dict[str, Any],
    *,
    by_company: Dict[str, Dict[str, Any]],
    by_email: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    email_key = str(account.get("email") or "").strip().lower()
    if email_key and email_key in by_email:
        return by_email[email_key]

    company_key = _normalize_company(account.get("company", ""))
    if company_key and company_key in by_company:
        return by_company[company_key]

    return None


def _build_deal_risk_prompt(
    *,
    prompt_template: str,
    account: Dict[str, Any],
    external_signals: List[Dict[str, Any]],
    customer_signal: Dict[str, Any],
    all_client_responses: List[Dict[str, Any]],
) -> str:
    marked_at = str(customer_signal.get("marked_as_customer_at") or "").strip()
    current_client_response = {
        "account_id": account.get("account_id"),
        "company": account.get("company"),
        "marked_as_customer_at": marked_at,
        "days_since_marked_as_customer": days_since(marked_at) if marked_at else None,
        "account_profile": account,
        "customer_signal": customer_signal,
        "external_signals": external_signals,
    }

    return (
        f"{prompt_template}\n\n"
        f"All Client Responses (JSON):\n{json.dumps(all_client_responses, ensure_ascii=True, default=str)}\n\n"
        f"Current Deal Client Response (JSON):\n{json.dumps(current_client_response, ensure_ascii=True, default=str)}\n\n"
        "Return ONLY strict JSON with this exact schema and no extra keys:\n"
        "{\n"
        f"  \"deal_id\": \"{account.get('account_id', '')}\",\n"
        f"  \"company\": \"{account.get('company', '')}\",\n"
        "  \"risk_level\": \"critical|high|medium|low\",\n"
        "  \"risk_score\": 0.0,\n"
        "  \"risk_signals\": [\"string\"],\n"
        "  \"competitor_threat\": false,\n"
        "  \"competitor_name\": null,\n"
        "  \"deal_velocity\": \"stalled|slow|on_track|accelerating\",\n"
        f"  \"days_inactive\": {int(account.get('days_inactive') or 0)},\n"
        "  \"recovery_strategy\": \"string\",\n"
        "  \"recommended_actions\": [\"string\"],\n"
        "  \"escalate_to_manager\": false,\n"
        "  \"predicted_close_probability\": 0.0,\n"
        "  \"reasoning\": \"string\"\n"
        "}"
    )


@retry(stop=stop_after_attempt(settings.max_retries + 1), wait=wait_exponential(min=1, max=4))
def analyze_deal_risk(prompt: str) -> DealRisk:
    from backend.llm.gemini_client import call_gemini

    try:
        response = call_gemini(prompt, structured=True, temperature=0.3)
        return DealRisk(**response)
    except Exception as e:
        logger.error("deal_intelligence_llm_failed", error=str(e))
        raise RuntimeError(f"Deal intelligence LLM failed: {e}") from e


def run_deal_intelligence_agent(
    deal_ids: Optional[List[str]],
    inactivity_threshold: int,
    customer_engagement_signals: Optional[List[Dict[str, Any]]],
    session_id: str,
    user_id: str,
) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")
        
    logger.info("deal_intelligence_agent_start", session_id=session_id)
    memory = get_vector_store()
    namespace = user_id
    prompt_template = _load_prompt()
    by_company, by_email = _build_customer_signal_indexes(customer_engagement_signals)
    at_risk = get_at_risk_deals(inactivity_days=inactivity_threshold, user_id=user_id)
    if deal_ids:
        at_risk = [account for account in at_risk if account.get("account_id") in deal_ids]

    client_scoped_accounts: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for account in at_risk:
        signal = _match_customer_signal(account, by_company=by_company, by_email=by_email)
        marked_at = str((signal or {}).get("marked_as_customer_at") or "").strip()
        if signal and marked_at:
            client_scoped_accounts.append((account, signal))

    if not client_scoped_accounts:
        result = build_agent_response(
            status="success",
            data={"risks": [], "total_at_risk": 0, "critical_count": 0, "customer_signal_matches": 0},
            reasoning="No client-linked deals with marked_as_customer_at found for risk analysis.",
            confidence=1.0,
            agent_name="deal_intelligence_agent",
            tools_used=["crm_tool", "scraping_tool"],
        )
        record_audit(
            session_id=session_id,
            agent_name="deal_intelligence_agent",
            action="detect_risks",
            input_summary=f"Threshold: {inactivity_threshold} days",
            output_summary="No eligible client-linked deals found",
            status="success",
            confidence=1.0,
        )
        return result

    all_client_responses: List[Dict[str, Any]] = []
    for account, signal in client_scoped_accounts:
        marked_at = str(signal.get("marked_as_customer_at") or "").strip()
        all_client_responses.append(
            {
                "account_id": account.get("account_id"),
                "company": account.get("company"),
                "marked_as_customer_at": marked_at,
                "days_since_marked_as_customer": days_since(marked_at) if marked_at else None,
                "account_profile": account,
                "customer_signal": signal,
            }
        )

    analyzed: List[Dict[str, Any]] = []
    customer_signal_matches = 0
    for account, signal in client_scoped_accounts:
        try:
            customer_signal_matches += 1
            external_signals = detect_intent_signals(account.get("company", ""), account)
            prompt = _build_deal_risk_prompt(
                prompt_template=prompt_template,
                account=account,
                external_signals=external_signals,
                customer_signal=signal,
                all_client_responses=all_client_responses,
            )
            analyzed_item = analyze_deal_risk(prompt).model_dump()
            analyzed.append(analyzed_item)
        except Exception as exc:
            logger.warning("deal_risk_llm_failed", account_id=account.get("account_id"), error=str(exc))
            _terminal_log("failure", f"LLM risk analysis failed for account {account.get('account_id', 'unknown')}: {exc}")
            raise RuntimeError(
                f"Deal intelligence analysis failed for account {account.get('account_id', 'unknown')}: {exc}"
            ) from exc

    analyzed.sort(key=lambda item: item.get("risk_score", 0.0), reverse=True)
    critical_count = len([item for item in analyzed if item.get("risk_level") in {"critical", "high"}])
    memory.add_document(
        doc_id=generate_id("deal_intel"),
        content=f"Deal risk analysis completed for {len(analyzed)} accounts.",
        metadata={"agent": "deal_intelligence", "session_id": session_id, "user_id": user_id or ""},
        namespace=namespace,
    )

    result = build_agent_response(
        status="success",
        data={
            "risks": analyzed,
            "total_at_risk": len(analyzed),
            "critical_count": critical_count,
            "customer_signal_matches": customer_signal_matches,
            "requires_escalation": any(item.get("escalate_to_manager") for item in analyzed),
        },
        reasoning=f"Analyzed {len(analyzed)} at-risk deals. {critical_count} classified as critical/high risk.",
        confidence=0.84,
        agent_name="deal_intelligence_agent",
        tools_used=["crm_tool", "scraping_tool", "vector_memory", "llm"],
    )
    _terminal_log("success", f"Analyzed {len(analyzed)} at-risk deals")
    record_audit(
        session_id=session_id,
        agent_name="deal_intelligence_agent",
        action="detect_risks",
        input_summary=f"Threshold: {inactivity_threshold} days, Checked: {len(client_scoped_accounts)} client-linked deals",
        output_summary=f"Found {len(analyzed)} at-risk deals, {critical_count} critical/high",
        status="success",
        reasoning=result["reasoning"],
        confidence=0.84,
    )
    return result
