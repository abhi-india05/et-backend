"""Outreach tracking routes — entries and metrics."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends

from backend.agents.guardrails import parse_llm_json
from backend.auth.deps import AuthUser, get_current_user
from backend.config.settings import settings
from backend.deps import get_customer_repo, get_outreach_entry_repo, get_session_repo
from backend.llm.gemini_client import call_gemini
from backend.models.schemas import (
    Customer,
    CustomerCreateFromEntryRequest,
    CustomerCreateRequest,
    OutreachEntryStatus,
    OutreachEntryStatusUpdate,
    RefineEmailRequest,
    StrictBaseModel,
)
from backend.repositories.customers import CustomerRepository
from backend.repositories.outreach_entries import OutreachEntryRepository
from backend.repositories.sessions import SessionRepository
from backend.utils.errors import APIError
from backend.utils.helpers import clamp_page_size, generate_id, now_iso, utcnow
from backend.utils.logger import get_logger, record_audit

router = APIRouter(tags=["outreach"])
logger = get_logger("outreach_routes")


class RefineEmailLLMResponse(StrictBaseModel):
    refined_email: str
    explanation: str = ""


def _terminal_log(level: str, message: str) -> None:
    print(f"[BACKEND][outreach_routes][{level.upper()}] {message}")


def _extract_product_name(input_data: Dict[str, Any]) -> str:
    direct_name = input_data.get("product_name")
    if isinstance(direct_name, str) and direct_name.strip():
        return direct_name.strip()
    product_context = input_data.get("product_context")
    if isinstance(product_context, dict):
        ctx_name = product_context.get("name")
        if isinstance(ctx_name, str):
            return ctx_name.strip()
    return ""


@router.get("/sessions")
async def get_outreach_sessions(
    page: int = 1,
    page_size: Optional[int] = None,
    user: AuthUser = Depends(get_current_user),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> Dict[str, Any]:
    normalized_size = clamp_page_size(page_size or 50, default=50, max_value=200)
    sessions, total, _running = await session_repo.list_sessions(
        owner_user_id=user.user_id,
        page=max(1, page),
        page_size=normalized_size,
        task_type="outreach",
    )
    items = [
        {
            "session_id": item.session_id,
            "company": str(item.input_data.get("company", "")),
            "industry": str(item.input_data.get("industry", "")),
            "product_name": _extract_product_name(item.input_data),
            "created_at": item.created_at,
            "status": item.status,
        }
        for item in sessions
    ]
    return {
        "items": items,
        "total": total,
        "page": max(1, page),
        "page_size": normalized_size,
        "timestamp": now_iso(),
    }


@router.get("/sessions/{session_id}")
async def get_outreach_session(
    session_id: str,
    user: AuthUser = Depends(get_current_user),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> Dict[str, Any]:
    session = await session_repo.get_session(session_id=session_id)
    if not session:
        raise APIError(status_code=404, code="not_found", message="Session not found")
    if session.owner_user_id != user.user_id:
        raise APIError(status_code=403, code="forbidden", message="Cannot access another user's session")
    if session.task_type not in {"outreach", "cold_outreach"}:
        raise APIError(status_code=404, code="not_found", message="Outreach session not found")

    final_output = session.final_output if isinstance(session.final_output, dict) else {}
    return {
        "session_id": session.session_id,
        "input_data": session.input_data,
        "plan": session.plan,
        "status": session.status,
        "agent_outputs": final_output.get("agent_outputs", {}),
        "final_output": final_output,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "completed_at": session.completed_at,
        "timestamp": now_iso(),
    }


@router.get("/entries")
async def get_outreach_entries(
    page: int = 1,
    page_size: Optional[int] = None,
    status: Optional[str] = None,
    company: Optional[str] = None,
    user: AuthUser = Depends(get_current_user),
    repo: OutreachEntryRepository = Depends(get_outreach_entry_repo),
) -> Dict[str, Any]:
    normalized_size = clamp_page_size(page_size or 50, default=50, max_value=200)
    entries, total = await repo.list_entries(
        user_id=user.user_id,
        page=max(1, page),
        page_size=normalized_size,
        status=status,
        company=company,
    )
    return {
        "entries": [entry.model_dump() for entry in entries],
        "total": total,
        "page": max(1, page),
        "page_size": normalized_size,
        "timestamp": now_iso(),
    }


@router.patch("/entries/{entry_id}/status")
async def update_outreach_status(
    entry_id: str,
    update_data: OutreachEntryStatusUpdate,
    user: AuthUser = Depends(get_current_user),
    repo: OutreachEntryRepository = Depends(get_outreach_entry_repo),
) -> Dict[str, Any]:
    entry = await repo.get_entry(entry_id)
    if not entry:
        raise APIError(status_code=404, code="not_found", message="Outreach entry not found")
        
    if entry.user_id != user.user_id:
        raise APIError(status_code=403, code="forbidden", message="Cannot update entry for another user")
        
    updated = await repo.update_status(entry_id, update_data.status)
    if not updated:
        raise APIError(status_code=500, code="update_failed", message="Failed to update entry status")
        
    return {
        "entry": updated.model_dump(),
        "timestamp": now_iso()
    }


@router.get("/customers")
async def list_customers(
    page: int = 1,
    page_size: Optional[int] = None,
    query: Optional[str] = None,
    user: AuthUser = Depends(get_current_user),
    repo: CustomerRepository = Depends(get_customer_repo),
) -> Dict[str, Any]:
    normalized_size = clamp_page_size(page_size or 50, default=50, max_value=200)
    customers, total = await repo.list_customers(
        user_id=user.user_id,
        page=max(1, page),
        page_size=normalized_size,
        query=query,
    )
    return {
        "customers": [customer.model_dump() for customer in customers],
        "total": total,
        "page": max(1, page),
        "page_size": normalized_size,
        "timestamp": now_iso(),
    }


@router.get("/customers/{customer_id}")
async def get_customer(
    customer_id: str,
    user: AuthUser = Depends(get_current_user),
    repo: CustomerRepository = Depends(get_customer_repo),
) -> Dict[str, Any]:
    customer = await repo.get_customer(customer_id)
    if not customer:
        raise APIError(status_code=404, code="not_found", message="Customer not found")
    if customer.user_id != user.user_id:
        raise APIError(status_code=403, code="forbidden", message="Cannot access another user's customer")

    return {
        "customer": customer.model_dump(),
        "timestamp": now_iso(),
    }


@router.patch("/customers/{customer_id}/mark-replied")
async def mark_customer_replied(
    customer_id: str,
    user: AuthUser = Depends(get_current_user),
    repo: CustomerRepository = Depends(get_customer_repo),
) -> Dict[str, Any]:
    customer = await repo.get_customer(customer_id)
    if not customer:
        raise APIError(status_code=404, code="not_found", message="Customer not found")
    if customer.user_id != user.user_id:
        raise APIError(status_code=403, code="forbidden", message="Cannot access another user's customer")

    marked_at = utcnow()
    updated = await repo.update_marked_as_customer_at(
        customer_id=customer_id,
        marked_as_customer_at=marked_at,
    )
    if not updated:
        raise APIError(status_code=500, code="update_failed", message="Failed to mark customer as replied")

    record_audit(
        session_id="customer_mark_replied",
        agent_name="customer_manager",
        action="mark_customer_replied",
        input_summary=f"Customer: {customer_id}",
        output_summary=f"Updated replied timestamp for customer {customer_id}",
        status="success",
        reasoning="User marked follow-up as replied; refreshed timestamp for risk and churn analytics.",
        confidence=0.95,
        extra={
            "customer_id": customer_id,
            "user_id": user.user_id,
            "marked_as_customer_at": marked_at.isoformat(),
        },
    )

    return {
        "customer": updated.model_dump(),
        "timestamp": now_iso(),
    }


@router.post("/customers")
async def create_customer(
    payload: CustomerCreateRequest,
    user: AuthUser = Depends(get_current_user),
    repo: CustomerRepository = Depends(get_customer_repo),
) -> Dict[str, Any]:
    source_entry_id = (payload.source_entry_id or "").strip()
    if source_entry_id:
        existing = await repo.get_by_source_entry(user_id=user.user_id, source_entry_id=source_entry_id)
        if existing:
            return {
                "customer": existing.model_dump(),
                "created": False,
                "timestamp": now_iso(),
            }

    customer = Customer(
        id=generate_id("cust"),
        user_id=user.user_id,
        company_name=payload.company_name,
        company_domain=payload.company_domain,
        contact_name=payload.contact_name,
        contact_email=payload.contact_email,
        notes=payload.notes,
        source_entry_id=payload.source_entry_id,
        source_outreach_status=payload.source_outreach_status,
        marked_as_customer_at=utcnow(),
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    created = await repo.create_customer(customer)
    record_audit(
        session_id="customer_manual",
        agent_name="customer_manager",
        action="create_customer",
        input_summary=f"Company: {created.company_name}",
        output_summary=f"Created customer {created.id}",
        status="success",
        reasoning="Manual customer add from outreach workflow.",
        confidence=0.93,
        extra={"customer_id": created.id, "user_id": user.user_id},
    )
    return {
        "customer": created.model_dump(),
        "created": True,
        "timestamp": now_iso(),
    }


@router.post("/customers/from-entry/{entry_id}")
async def create_customer_from_entry(
    entry_id: str,
    payload: CustomerCreateFromEntryRequest,
    user: AuthUser = Depends(get_current_user),
    entry_repo: OutreachEntryRepository = Depends(get_outreach_entry_repo),
    customer_repo: CustomerRepository = Depends(get_customer_repo),
) -> Dict[str, Any]:
    entry = await entry_repo.get_entry(entry_id)
    if not entry:
        raise APIError(status_code=404, code="not_found", message="Outreach entry not found")
    if entry.user_id != user.user_id:
        raise APIError(status_code=403, code="forbidden", message="Cannot use another user's outreach entry")
    if entry.status != OutreachEntryStatus.REPLIED:
        raise APIError(
            status_code=400,
            code="invalid_status",
            message="Only entries with status 'replied' can be added as customers.",
        )

    existing = await customer_repo.get_by_source_entry(user_id=user.user_id, source_entry_id=entry.id)
    if existing:
        return {
            "customer": existing.model_dump(),
            "created": False,
            "timestamp": now_iso(),
        }

    customer = Customer(
        id=generate_id("cust"),
        user_id=user.user_id,
        company_name=entry.company_name,
        company_domain=entry.company_domain,
        contact_name=payload.contact_name,
        contact_email=payload.contact_email,
        notes=payload.notes,
        source_entry_id=entry.id,
        source_outreach_status=entry.status,
        marked_as_customer_at=utcnow(),
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    created = await customer_repo.create_customer(customer)
    record_audit(
        session_id="customer_from_reply",
        agent_name="customer_manager",
        action="create_customer_from_entry",
        input_summary=f"Entry: {entry.id} ({entry.company_name})",
        output_summary=f"Created customer {created.id} from replied entry",
        status="success",
        reasoning="User manually converted replied outreach entry into customer.",
        confidence=0.95,
        extra={"customer_id": created.id, "entry_id": entry.id, "user_id": user.user_id},
    )
    return {
        "customer": created.model_dump(),
        "created": True,
        "timestamp": now_iso(),
    }


@router.post("/refine-email")
async def refine_email(
    payload: RefineEmailRequest,
    user: AuthUser = Depends(get_current_user),
) -> Dict[str, Any]:
    lead_context = payload.lead_context if isinstance(payload.lead_context, dict) else {}
    insights = payload.insights if isinstance(payload.insights, dict) else {}

    try:
        prompt = (
            "You refine outbound sales emails. You MUST revise the provided original email, "
            "not regenerate from scratch. Keep factual grounding limited to the provided "
            "lead_context and insights. Return only valid JSON.\n\n"
            f"Lead ID: {payload.lead_id}\n"
            f"User Prompt: {payload.prompt}\n\n"
            f"Original Email:\n{payload.original_email}\n\n"
            f"Lead Context (JSON): {json.dumps(lead_context, ensure_ascii=True)}\n"
            f"Insights (JSON): {json.dumps(insights, ensure_ascii=True)}\n\n"
            "Return JSON with this exact shape:\n"
            "{\"refined_email\": \"...\", \"explanation\": \"...\"}"
        )

        response = call_gemini(prompt, structured=True, temperature=0.25)
        
        refined_email = response.get("refined_email", "")
        explanation = response.get("explanation", "Successfully refined email via Gemini.")
        
        if not refined_email.strip():
            raise ValueError("LLM returned empty refined email")

        session_id = str(lead_context.get("session_id") or "manual_refine")
        record_audit(
            session_id=session_id,
            agent_name="outreach_refiner",
            action="refine_email",
            input_summary=f"Refine email for lead {payload.lead_id}",
            output_summary="Refined outreach email with user prompt and twin context",
            status="success",
            reasoning=explanation,
            confidence=0.84,
            extra={"lead_id": payload.lead_id, "user_id": user.user_id},
        )
        return {
            "refined_email": refined_email,
            "explanation": explanation,
            "timestamp": now_iso(),
        }
    except Exception as exc:
        logger.warning("refine_email_failed", lead_id=payload.lead_id, error=str(exc), user_id=user.user_id)
        _terminal_log("failure", f"LLM refine failed for lead {payload.lead_id}: {exc}")
        session_id = str(lead_context.get("session_id") or "manual_refine")
        record_audit(
            session_id=session_id,
            agent_name="outreach_refiner",
            action="refine_email",
            input_summary=f"Refine email for lead {payload.lead_id}",
            output_summary="LLM refinement failed",
            status="failure",
            reasoning=f"LLM refinement failed: {exc}",
            confidence=0.0,
            extra={"lead_id": payload.lead_id, "user_id": user.user_id},
        )
        raise APIError(
            status_code=502,
            code="llm_refinement_failed",
            message=(
                "Email refinement failed because the configured LLM request could not be completed. "
                "Verify provider, base URL, model, and API key configuration."
            ),
            details={
                "provider": "gemini",
                "lead_id": payload.lead_id,
                "error": str(exc),
            },
        ) from exc
