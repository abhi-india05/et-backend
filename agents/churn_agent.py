from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.config.settings import settings
from backend.llm.gemini_client import call_gemini
from backend.memory.vector_store import get_vector_store
from backend.models.schemas import ChurnRisk
from backend.tools.crm_tool import get_all_accounts
from backend.utils.helpers import build_agent_response, days_since, generate_id
from backend.utils.logger import get_logger, record_audit

logger = get_logger("churn_agent")


def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][churn_agent][{level.upper()}] {message}")


def _load_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "churn_risk_prompt.txt")
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


def _build_churn_ranking_prompt(
    *,
    prompt_template: str,
    all_client_responses: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    top_n: int,
) -> str:
    return (
        f"{prompt_template}\n\n"
        f"All Client Responses (JSON):\n{json.dumps(all_client_responses, ensure_ascii=True, default=str)}\n\n"
        f"Candidate Clients For Ranking (JSON):\n{json.dumps(candidates, ensure_ascii=True, default=str)}\n\n"
        f"Return ONLY strict JSON with this exact shape and no extra keys. Return exactly top {top_n} clients if possible:\n"
        "{\n"
        "  \"top_clients\": [\n"
        "    {\n"
        "      \"account_id\": \"string\",\n"
        "      \"company\": \"string\",\n"
        "      \"churn_probability\": 0.0,\n"
        "      \"risk_factors\": [\"string\"],\n"
        "      \"retention_strategy\": \"string\",\n"
        "      \"urgency\": \"critical|high|medium|low\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )


@retry(
    stop=stop_after_attempt(settings.max_retries + 1),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((RuntimeError, ValueError)),
    reraise=True,
)
def analyze_churn_risks(prompt: str, expected_top_n: int) -> List[ChurnRisk]:
    try:
        response = call_gemini(prompt, structured=True, temperature=0.25)
        _terminal_log("success", f"LLM raw output: {json.dumps(response, ensure_ascii=True)}")
        logger.info("churn_llm_output", output=response)

        if not isinstance(response, dict):
            raise ValueError("LLM churn response is not a JSON object")

        raw_items = response.get("top_clients")
        if not isinstance(raw_items, list):
            raise ValueError("LLM churn response missing top_clients list")

        risks: List[ChurnRisk] = []
        for item in raw_items[: max(1, expected_top_n)]:
            risks.append(ChurnRisk(**item))

        if not risks:
            raise ValueError("LLM returned no ranked churn clients")
        return risks
    except Exception as exc:
        logger.warning("churn_llm_failed", error=str(exc))
        _terminal_log("failure", f"LLM churn ranking failed: {exc}")
        raise RuntimeError(f"Churn risk ranking failed: {exc}") from exc


def run_churn_agent(
    account_ids: Optional[List[str]],
    top_n: int,
    customer_engagement_signals: Optional[List[Dict[str, Any]]],
    session_id: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")

    logger.info("churn_agent_start", session_id=session_id)
    memory = get_vector_store()
    namespace = user_id
    prompt_template = _load_prompt()
    by_company, by_email = _build_customer_signal_indexes(customer_engagement_signals)
    accounts = get_all_accounts(user_id=user_id)

    if account_ids:
        accounts = [account for account in accounts if account.get("account_id") in account_ids]

    active_accounts = [account for account in accounts if account.get("stage") not in ["Closed Lost", "Prospecting"]]
    client_scoped_accounts: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for account in active_accounts:
        signal = _match_customer_signal(account, by_company=by_company, by_email=by_email)
        marked_at = str((signal or {}).get("marked_as_customer_at") or "").strip()
        if signal and marked_at:
            client_scoped_accounts.append((account, signal))

    if not client_scoped_accounts:
        result = build_agent_response(
            status="success",
            data={"top_churn_risks": [], "total_analyzed": 0, "total_arr_at_risk": 0, "customer_signal_matches": 0},
            reasoning="No client accounts with marked_as_customer_at were found for churn analysis.",
            confidence=1.0,
            agent_name="churn_agent",
            tools_used=["crm_tool"],
        )
        record_audit(
            session_id=session_id,
            agent_name="churn_agent",
            action="predict_churn",
            input_summary="No client-linked accounts with marked_as_customer_at",
            output_summary="No churn risks found",
            status="success",
            confidence=1.0,
        )
        return result

    customer_signal_matches = len(client_scoped_accounts)
    effective_top_n = min(3, len(client_scoped_accounts))

    all_client_responses: List[Dict[str, Any]] = []
    ranking_candidates: List[Dict[str, Any]] = []
    context_by_account_id: Dict[str, Dict[str, Any]] = {}
    context_by_company: Dict[str, Dict[str, Any]] = {}

    for account, signal in client_scoped_accounts:
        account_id = str(account.get("account_id") or "").strip()
        company = str(account.get("company") or "").strip()
        marked_at = str(signal.get("marked_as_customer_at") or "").strip()
        all_client_responses.append(
            {
                "account_id": account_id,
                "company": company,
                "marked_as_customer_at": marked_at,
                "days_since_marked_as_customer": days_since(marked_at) if marked_at else None,
                "account_profile": account,
                "customer_signal": signal,
            }
        )
        candidate = {
            "account_id": account_id,
            "company": company,
            "marked_as_customer_at": marked_at,
            "days_since_marked_as_customer": days_since(marked_at) if marked_at else None,
            "account_profile": account,
            "customer_signal": signal,
        }
        ranking_candidates.append(candidate)

        context = {
            "arr": account.get("arr", 0),
            "health_score": account.get("health_score"),
            "contact_name": account.get("contact_name"),
            "contact_email": account.get("email"),
            "industry": account.get("industry"),
            "stage": account.get("stage"),
            "marked_as_customer_at": marked_at or None,
            "customer_days_since_reply": days_since(marked_at) if marked_at else None,
        }
        if account_id:
            context_by_account_id[account_id] = context
        company_key = _normalize_company(company)
        if company_key:
            context_by_company[company_key] = context

    ranking_prompt = _build_churn_ranking_prompt(
        prompt_template=prompt_template,
        all_client_responses=all_client_responses,
        candidates=ranking_candidates,
        top_n=effective_top_n,
    )
    ranked_churn_risks = analyze_churn_risks(ranking_prompt, expected_top_n=effective_top_n)

    top_risks: List[Dict[str, Any]] = []
    for churn_risk in ranked_churn_risks:
        payload = churn_risk.model_dump()
        account_context = context_by_account_id.get(payload.get("account_id", ""))
        if not account_context:
            account_context = context_by_company.get(_normalize_company(payload.get("company", "")), {})
        top_risks.append({**payload, **(account_context or {})})

    total_arr_at_risk = sum(item["arr"] for item in top_risks)
    average_probability = sum(item["churn_probability"] for item in top_risks) / max(len(top_risks), 1)

    result = build_agent_response(
        status="success",
        data={
            "top_churn_risks": top_risks,
            "total_analyzed": len(client_scoped_accounts),
            "top_n": effective_top_n,
            "total_arr_at_risk": total_arr_at_risk,
            "avg_churn_probability": round(average_probability, 4),
            "critical_count": len([risk for risk in top_risks if risk["urgency"] == "critical"]),
            "customer_signal_matches": customer_signal_matches,
        },
        reasoning=(
            f"LLM ranked churn risk for {len(client_scoped_accounts)} client accounts using marked_as_customer_at recency. "
            f"Returned top {effective_top_n} potential churn clients."
        ),
        confidence=0.87,
        agent_name="churn_agent",
        tools_used=["crm_tool", "llm"],
    )
    _terminal_log("success", f"Analyzed {len(client_scoped_accounts)} client accounts; top churn risks: {len(top_risks)}")
    memory.add_document(
        doc_id=generate_id("churn"),
        content=f"Churn ranking completed for {len(client_scoped_accounts)} client accounts.",
        metadata={"agent": "churn", "session_id": session_id, "user_id": user_id},
        namespace=namespace,
    )
    record_audit(
        session_id=session_id,
        agent_name="churn_agent",
        action="predict_churn",
        input_summary=(
            f"Analyzed {len(client_scoped_accounts)} client accounts with marked_as_customer_at "
            f"(requested_top_n={top_n}, enforced_top_n={effective_top_n})"
        ),
        output_summary=f"Top {effective_top_n} risks, ${total_arr_at_risk:,.0f} ARR at risk",
        status="success",
        reasoning=result["reasoning"],
        confidence=0.87,
    )
    return result
