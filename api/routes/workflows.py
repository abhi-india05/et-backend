"""Workflow routes — outreach, risk detection, churn prediction."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request, Response

from backend.agents.orchestrator import run_orchestrator
from backend.auth.deps import AuthUser, get_current_user
from backend.deps import get_customer_repo, get_outreach_entry_repo, get_session_repo
from backend.models.schemas import (
    ChurnPredictionRequest,
    OutreachRequest,
    RiskDetectionRequest,
    OutreachEntry,
    OutreachEntryStatus
)
from backend.repositories.customers import CustomerRepository
from backend.repositories.sessions import SessionRepository
from backend.repositories.outreach_entries import OutreachEntryRepository
from backend.utils.errors import APIError
from backend.utils.helpers import generate_session_id, now_iso, utcnow
from backend.utils.logger import bind_context

router = APIRouter(tags=["workflows"])


async def _load_customer_engagement_signals(
    *,
    user_id: str,
    customer_repo: CustomerRepository,
    max_customers: int = 1000,
) -> list[Dict[str, Any]]:
    page = 1
    page_size = 200
    signals: list[Dict[str, Any]] = []

    while len(signals) < max_customers:
        items, _total = await customer_repo.list_customers(
            user_id=user_id,
            page=page,
            page_size=page_size,
            query=None,
        )
        if not items:
            break

        for customer in items:
            serialized = customer.model_dump(mode="json")
            signals.append(
                {
                    "customer_id": serialized.get("id"),
                    "company_name": serialized.get("company_name"),
                    "contact_email": serialized.get("contact_email"),
                    "marked_as_customer_at": serialized.get("marked_as_customer_at") or serialized.get("created_at"),
                    "source_outreach_status": serialized.get("source_outreach_status"),
                    "client_response": serialized,
                }
            )
            if len(signals) >= max_customers:
                break

        if len(items) < page_size:
            break
        page += 1

    return signals

async def _run_workflow(
    *,
    task_type: str,
    orchestrator_task_type: Optional[str] = None,
    workflow_input: Dict[str, Any],
    request: Request,
    user: AuthUser,
    response: Response,
    session_repo: SessionRepository,
    session_id_override: Optional[str] = None,
) -> Dict[str, Any]:
    session_id = session_id_override or generate_session_id()
    bind_context(session_id=session_id, user_id=user.user_id, username=user.username, role=user.role)
    await session_repo.create_session(
        session_id=session_id,
        owner_user_id=user.user_id,
        task_type=task_type,
        input_data=workflow_input,
        plan={"task_type": task_type, "status": "running"},
        request_id=request.state.request_id,
        status="running",
    )
    try:
        result = await run_orchestrator(
            task_type=orchestrator_task_type or task_type,
            input_data=workflow_input,
            session_id=session_id,
            user_id=user.user_id,
        )
        await session_repo.update_session(
            session_id=session_id,
            status=result.get("status", "completed"),
            plan=result.get("plan", {}),
            final_output=result,
        )
        response.headers["X-Session-ID"] = session_id
        return {
            "session_id": session_id,
            "task_type": task_type,
            "status": result.get("status", "completed"),
            "data": result,
            "timestamp": now_iso(),
        }
    except Exception as exc:
        await session_repo.update_session(
            session_id=session_id,
            status="failed",
            error=str(exc),
        )
        raise

@router.post("/run-outreach")
async def run_outreach(
    payload: OutreachRequest,
    request: Request,
    response: Response,
    user: AuthUser = Depends(get_current_user),
    session_repo: SessionRepository = Depends(get_session_repo),
    entry_repo: OutreachEntryRepository = Depends(get_outreach_entry_repo),
) -> Dict[str, Any]:
    product_context = {
        "name": payload.product_name or "",
        "description": payload.product_description or ""
    }
    workflow_input = {
        "company": payload.company,
        "industry": payload.industry,
        "size": payload.size,
        "website": payload.website,
        "notes": payload.notes,
        "product_context": product_context,
        "auto_send": payload.auto_send,
    }
    result = await _run_workflow(
        task_type="outreach",
        orchestrator_task_type="cold_outreach",
        workflow_input=workflow_input,
        request=request,
        user=user,
        response=response,
        session_repo=session_repo,
        session_id_override=payload.session_id,
    )

    if result.get("status") in ("completed", "completed_with_errors") and "outreach_agent" in result.get("data", {}).get("agent_outputs", {}):
        sequences = result["data"]["agent_outputs"]["outreach_agent"].get("data", {}).get("sequences", [])
        for seq in sequences:
            status = OutreachEntryStatus.SENT if payload.auto_send else OutreachEntryStatus.DRAFT
            await entry_repo.create_entry(
                OutreachEntry(
                    id=seq.get("sequence_id", generate_session_id()),
                    user_id=user.user_id,
                    company_name=payload.company,
                    company_domain=payload.website,
                    outreach_type="email",
                    message=seq.get("sequence_strategy"),
                    status=status
                )
            )

    return result


@router.post("/detect-risk")
async def detect_risk(
    payload: RiskDetectionRequest,
    request: Request,
    response: Response,
    user: AuthUser = Depends(get_current_user),
    session_repo: SessionRepository = Depends(get_session_repo),
    customer_repo: CustomerRepository = Depends(get_customer_repo),
) -> Dict[str, Any]:
    customer_engagement_signals = await _load_customer_engagement_signals(
        user_id=user.user_id,
        customer_repo=customer_repo,
    )
    workflow_input = {
        "deal_ids": payload.deal_ids,
        "check_all": payload.check_all,
        "inactivity_threshold_days": payload.inactivity_threshold_days,
        "customer_engagement_signals": customer_engagement_signals,
    }
    return await _run_workflow(
        task_type="risk_detection",
        workflow_input=workflow_input,
        request=request,
        user=user,
        response=response,
        session_repo=session_repo,
    )


@router.post("/predict-churn")
async def predict_churn(
    payload: ChurnPredictionRequest,
    request: Request,
    response: Response,
    user: AuthUser = Depends(get_current_user),
    session_repo: SessionRepository = Depends(get_session_repo),
    customer_repo: CustomerRepository = Depends(get_customer_repo),
) -> Dict[str, Any]:
    customer_engagement_signals = await _load_customer_engagement_signals(
        user_id=user.user_id,
        customer_repo=customer_repo,
    )
    workflow_input = {
        "account_ids": payload.account_ids,
        "top_n": payload.top_n,
        "customer_engagement_signals": customer_engagement_signals,
    }
    return await _run_workflow(
        task_type="churn_prediction",
        workflow_input=workflow_input,
        request=request,
        user=user,
        response=response,
        session_repo=session_repo,
    )
