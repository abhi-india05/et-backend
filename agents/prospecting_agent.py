from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from backend.llm.gemini_client import call_gemini
from backend.memory.vector_store import get_vector_store
from backend.models.schemas import ProspectingOutput
from backend.tools.scraping_tool import enrich_company
from backend.utils.helpers import build_agent_response, generate_id
from backend.utils.logger import get_logger, record_audit

logger = get_logger("prospecting_agent")

MAX_PROMPT_LEADS = 30
MAX_RANKED_LEADS = 2

def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][prospecting_agent][{level.upper()}] {message}")

def _load_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "prospecting_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _lead_identity(lead: Dict[str, Any]) -> str:
    return str(
        lead.get("id")
        or lead.get("lead_id")
        or lead.get("source_profile")
        or lead.get("linkedin_url")
        or lead.get("name")
        or ""
    ).strip().lower()


def _dedupe_leads(leads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for lead in leads:
        identity = _lead_identity(lead)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        deduped.append(lead)
    return deduped


def _lead_data_richness(lead: Dict[str, Any]) -> int:
    fields = ["name", "title", "role", "company", "headline", "about", "activity", "linkedin_url", "email"]
    return sum(bool(str(lead.get(field) or "").strip()) for field in fields)


def _resolve_ranked_source(raw_leads: List[Dict[str, Any]], lead_identifier: Any) -> Optional[Dict[str, Any]]:
    token = str(lead_identifier or "").strip().lower()
    if not token:
        return None

    for lead in raw_leads:
        if token == _lead_identity(lead):
            return lead
        if token == str(lead.get("name") or "").strip().lower():
            return lead
        if token == str(lead.get("source_profile") or "").strip().lower():
            return lead

    for lead in raw_leads:
        name = str(lead.get("name") or "").strip().lower()
        if token and name and token in name:
            return lead
    return None


def _map_ranked_lead(
    original_lead: Dict[str, Any],
    company: str,
    ranking_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ranking = ranking_data or {}
    lead_id = str(original_lead.get("id") or original_lead.get("lead_id") or generate_id("lead"))
    role = str(original_lead.get("role") or original_lead.get("title") or "").strip()
    base_score = float(original_lead.get("score") or 0.0)
    ranked_score = ranking.get("score")
    merged_score = float(ranked_score) if ranked_score is not None else base_score

    return {
        "id": lead_id,
        "name": original_lead.get("name", "Unknown Lead"),
        "title": role,
        "role": role,
        "company": original_lead.get("company") or company,
        "email": original_lead.get("email") or "",
        "linkedin": original_lead.get("linkedin") or "",
        "linkedin_url": original_lead.get("linkedin_url") or original_lead.get("linkedin") or "",
        "headline": original_lead.get("headline") or "",
        "about": original_lead.get("about") or "",
        "activity": original_lead.get("activity") or "",
        "source_profile": original_lead.get("source_profile") or "",
        "raw_data": original_lead.get("raw_data"),
        "score": max(0.0, min(1.0, merged_score)),
        "signals": ranking.get("signals", []) if isinstance(ranking.get("signals"), list) else [],
        "pain_points": ranking.get("pain_points", []) if isinstance(ranking.get("pain_points"), list) else [],
        "why_prioritized": ranking.get("why_prioritized", "Selected using LinkedIn dataset relevance and profile completeness."),
    }


def _deterministic_rank(raw_leads: List[Dict[str, Any]], company: str, limit: int = MAX_RANKED_LEADS) -> List[Dict[str, Any]]:
    ranked = sorted(
        raw_leads,
        key=lambda lead: (float(lead.get("score") or 0.0), _lead_data_richness(lead)),
        reverse=True,
    )
    return [_map_ranked_lead(lead, company) for lead in ranked[:limit]]

def _build_grounded_prospecting(company: str, enriched: Dict[str, Any]) -> ProspectingOutput:
    raw_leads = _dedupe_leads(list(enriched.get("leads", [])))
    if not raw_leads:
        return ProspectingOutput(
            leads=[],
            company_summary=f"No matching LinkedIn profiles were found for '{company}' in the available dataset.",
            recommended_approach="Skip outreach generation until a grounded profile match exists.",
            icp_fit_score=0.0,
        )

    prompt_template = _load_prompt()
    prompt_leads = raw_leads[:MAX_PROMPT_LEADS]
    prompt = prompt_template.replace("{leads}", json.dumps(prompt_leads)).replace("{company}", company)

    try:
        response = call_gemini(prompt, structured=True, temperature=0.1)
        mapped_leads = []
        used_identities = set()
        for lead_data in response.get("ranked_leads", []):
            if not isinstance(lead_data, dict):
                continue
            source = _resolve_ranked_source(raw_leads, lead_data.get("lead_id"))
            if not source:
                continue
            identity = _lead_identity(source)
            if not identity or identity in used_identities:
                continue
            used_identities.add(identity)
            mapped_leads.append(_map_ranked_lead(source, company, lead_data))
            if len(mapped_leads) >= MAX_RANKED_LEADS:
                break

        if len(mapped_leads) < MAX_RANKED_LEADS:
            ranked_candidates = sorted(
                raw_leads,
                key=lambda lead: (float(lead.get("score") or 0.0), _lead_data_richness(lead)),
                reverse=True,
            )
            for candidate in ranked_candidates:
                identity = _lead_identity(candidate)
                if not identity or identity in used_identities:
                    continue
                used_identities.add(identity)
                mapped_leads.append(_map_ranked_lead(candidate, company))
                if len(mapped_leads) >= MAX_RANKED_LEADS:
                    break

        if not mapped_leads:
            mapped_leads = _deterministic_rank(raw_leads, company, limit=MAX_RANKED_LEADS)

        fit_score = float(response.get("icp_fit_score") or 0.0)
        if fit_score <= 0.0 and mapped_leads:
            fit_score = sum(float(item.get("score") or 0.0) for item in mapped_leads) / len(mapped_leads)

        return ProspectingOutput(
            leads=mapped_leads,
            company_summary=response.get("company_summary", f"Matched {len(raw_leads)} LinkedIn leads for {company}; ranked top {len(mapped_leads)}."),
            recommended_approach=response.get("recommended_approach", "Reference only observed profile evidence."),
            icp_fit_score=max(0.0, min(1.0, fit_score)),
        )
    except Exception as e:
        logger.error("prospecting_llm_failed", error=str(e))
        _terminal_log("error", f"LLM generation failed: {e}")
        fallback_leads = _deterministic_rank(raw_leads, company, limit=MAX_RANKED_LEADS)
        fallback_fit = (
            sum(float(item.get("score") or 0.0) for item in fallback_leads) / max(len(fallback_leads), 1)
            if fallback_leads
            else 0.0
        )
        return ProspectingOutput(
            leads=fallback_leads,
            company_summary=f"Processed {len(raw_leads)} LinkedIn leads for {company} using deterministic ranking after LLM error.",
            recommended_approach="Use the selected top leads while retrying LLM ranking in parallel.",
            icp_fit_score=max(0.0, min(1.0, fallback_fit)),
        )

def run_prospecting_agent(
    company: str,
    industry: str,
    company_size: str,
    session_id: str,
    notes: str = "",
    product_context: Optional[Dict[str, str]] = None,
    user_id: str = None,
) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")

    logger.info("prospecting_agent_start", company=company, session_id=session_id)
    memory = get_vector_store()
    namespace = user_id

    enriched = enrich_company(company, industry)
    parsed = _build_grounded_prospecting(company, enriched)

    memory.add_document(
        doc_id=generate_id("prospect"),
        content=f"Prospected {company}: {parsed.company_summary}",
        metadata={"company": company, "agent": "prospecting", "session_id": session_id, "user_id": user_id or ""},
        namespace=namespace,
    )

    confidence = float(parsed.icp_fit_score or 0.0)
    result = build_agent_response(
        status="success",
        data=parsed.model_dump(),
        reasoning=f"Identified {len(parsed.leads)} grounded lead(s) for {company}. ICP fit: {confidence:.0%}.",
        confidence=confidence,
        agent_name="prospecting_agent",
        tools_used=["scraping_tool", "vector_memory"],
    )
    _terminal_log(
        "success",
        f"Generated {len(parsed.leads)} grounded lead(s) for {company} using Gemini"
    )
    record_audit(
        session_id=session_id,
        agent_name="prospecting_agent",
        action="identify_leads",
        input_summary=f"Company: {company}, Industry: {industry}",
        output_summary=f"Found {len(parsed.leads)} leads, confidence={confidence:.2f}",
        status="success",
        reasoning=result["reasoning"],
        confidence=confidence,
    )
    return result
