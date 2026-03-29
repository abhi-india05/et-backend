"""Admin / operational routes — logs, sessions, pipeline, emails, memory, metrics."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request, Response, status

from backend.agents.failure_recovery import get_recovery_engine
from backend.auth.deps import AuthUser, get_current_user, require_role
from backend.config.settings import settings
from backend.deps import get_session_repo, get_outreach_entry_repo
from backend.memory.vector_store import get_vector_store
from backend.models.schemas import (
    SendEmailRequest,
    SendSequencesRequest,
    SingleLeadSendRequest,
    ReviewedSequence,
    OutreachEntryStatus,
)
from backend.repositories.sessions import SessionRepository
from backend.repositories.outreach_entries import OutreachEntryRepository
from backend.services.observability import get_metrics_registry
from backend.tools.crm_tool import get_all_accounts, get_pipeline_stats
from backend.tools.email_tool import get_email_client, get_email_stats, get_sent_emails
from backend.utils.errors import APIError
from backend.utils.helpers import clamp_page_size, generate_session_id, now_iso
from backend.utils.logger import query_audit_logs, delete_audit_log, clear_audit_logs

router = APIRouter(tags=["admin"])


def _normalized_page_size(page_size: Optional[int]) -> int:
    desired = page_size if page_size is not None else settings.default_page_size
    return clamp_page_size(desired, default=settings.default_page_size, max_value=settings.max_page_size)


def _apply_pagination_headers(
    response: Response,
    *,
    page: int,
    page_size: int,
    total: int,
) -> None:
    response.headers["X-Page"] = str(page)
    response.headers["X-Page-Size"] = str(page_size)
    response.headers["X-Total-Count"] = str(total)


def _session_payload(item: Any) -> Dict[str, Any]:
    return {
        "session_id": item.session_id,
        "owner_user_id": item.owner_user_id,
        "task_type": item.task_type,
        "status": item.status,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "completed_at": item.completed_at,
        "error": item.error,
        "request_id": item.request_id,
    }


@router.get("/logs")
async def get_logs(
    response: Response,
    session_id: Optional[str] = None,
    page: int = 1,
    page_size: Optional[int] = None,
    user_id: Optional[str] = None,
    user: AuthUser = Depends(get_current_user),
) -> Dict[str, Any]:
    normalized = _normalized_page_size(page_size)
    scoped_user_id = user_id if user.is_admin and user_id else user.user_id
    logs = query_audit_logs(
        session_id=session_id,
        user_id=scoped_user_id,
        page=max(1, page),
        page_size=normalized,
    )
    _apply_pagination_headers(response, page=max(1, page), page_size=normalized, total=logs["total"])
    return {
        "logs": logs["items"],
        "total": logs["total"],
        "page": max(1, page),
        "page_size": normalized,
        "timestamp": now_iso(),
    }


@router.get("/sessions")
async def get_sessions(
    response: Response,
    page: int = 1,
    page_size: Optional[int] = None,
    status_filter: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
    owner_user_id: Optional[str] = None,
    user: AuthUser = Depends(get_current_user),
    repo: SessionRepository = Depends(get_session_repo),
) -> Dict[str, Any]:
    normalized = _normalized_page_size(page_size)
    scoped_owner_id = owner_user_id if user.is_admin and owner_user_id else user.user_id
    sessions, total, running = await repo.list_sessions(
        owner_user_id=scoped_owner_id,
        page=max(1, page),
        page_size=normalized,
        status=status_filter,
        created_from=created_from,
        created_to=created_to,
    )
    _apply_pagination_headers(response, page=max(1, page), page_size=normalized, total=total)
    return {
        "sessions": [_session_payload(item) for item in sessions],
        "total": total,
        "running": running,
        "page": max(1, page),
        "page_size": normalized,
        "timestamp": now_iso(),
    }


@router.get("/pipeline")
async def get_pipeline(user: AuthUser = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "stats": get_pipeline_stats(user_id=user.user_id),
        "accounts": get_all_accounts(user_id=user.user_id)[:20],
        "timestamp": now_iso(),
    }


@router.get("/emails")
async def get_emails(
    limit: int = 50,
    to_email: Optional[str] = None,
    sequence_id: Optional[str] = None,
    user: AuthUser = Depends(get_current_user),
) -> Dict[str, Any]:
    return {
        "emails": get_sent_emails(user_id=user.user_id, to_email=to_email, sequence_id=sequence_id)[: max(1, limit)],
        "stats": get_email_stats(user_id=user.user_id),
        "timestamp": now_iso(),
    }


@router.post("/send-email")
async def send_email(
    req: SendEmailRequest,
    user: AuthUser = Depends(get_current_user),
) -> Dict[str, Any]:
    client = get_email_client()
    result = await asyncio.to_thread(
        client.send_email,
        to_email=req.to_email,
        to_name=req.to_name or "",
        subject=req.subject,
        body_text=req.body_text,
        body_html=req.body_html,
        user_id=user.user_id,
    )
    if not result.get("success"):
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="email_failed",
            message=result.get("error", "Email failed"),
        )
    return {"result": result, "timestamp": now_iso()}


@router.post("/send-sequences")
async def send_sequences(
    req: SendSequencesRequest | SingleLeadSendRequest,
    user: AuthUser = Depends(get_current_user),
    entry_repo: OutreachEntryRepository = Depends(get_outreach_entry_repo),
) -> Dict[str, Any]:
    if isinstance(req, SingleLeadSendRequest):
        single_email_payload = [
            {
                "subject": req.subject or "Outreach message",
                "body": req.content,
                "from_email": req.from_email,
                "from_name": req.from_name,
            }
        ]
        req = SendSequencesRequest(
            sequences=[
                ReviewedSequence(
                    lead_id=req.lead_id,
                    lead_name=req.lead_name or "",
                    lead_email=req.email,
                    email=req.email,
                    sequence_id=req.sequence_id,
                    content=req.content,
                    emails=single_email_payload,
                )
            ]
        )

    client = get_email_client()
    results: List[Dict[str, Any]] = []
    total_sent = 0
    total_failed = 0
    for sequence in req.sequences:
        payload = [
            {
                "subject": email.subject,
                "body": email.body,
                "from_email": email.from_email,
                "from_name": email.from_name,
            }
            for email in sequence.emails
        ]
        if not payload and (sequence.content or "").strip():
            payload = [
                {
                    "subject": "Outreach message",
                    "body": sequence.content.strip(),
                    "from_email": None,
                    "from_name": None,
                }
            ]

        recipient_email = (sequence.email or sequence.lead_email or "").strip()
        if not recipient_email:
            results.append(
                {
                    "success": False,
                    "error": "Missing recipient email",
                    "sequence_id": sequence.sequence_id,
                    "sent": 0,
                    "failed": max(1, len(payload)),
                }
            )
            total_failed += max(1, len(payload))
            continue

        seq_id = sequence.sequence_id or sequence.lead_id or generate_session_id()
        result = await asyncio.to_thread(
            client.send_sequence,
            to_email=recipient_email,
            to_name=sequence.lead_name or "",
            emails=payload,
            sequence_id=seq_id,
            user_id=user.user_id,
        )
        results.append(result)
        sent_in_seq = int(result.get("sent", 0))
        total_sent += sent_in_seq
        total_failed += int(result.get("failed", 0))
        
        if sent_in_seq > 0 and sequence.sequence_id:
            await entry_repo.update_status(sequence.sequence_id, OutreachEntryStatus.SENT)
            
    return {
        "results": results,
        "summary": {
            "total_sequences": len(results),
            "sent": total_sent,
            "failed": total_failed,
        },
        "timestamp": now_iso(),
    }


@router.delete("/logs/{log_id}")
async def delete_log(
    log_id: str,
    user: AuthUser = Depends(get_current_user),
) -> Dict[str, Any]:
    deleted = delete_audit_log(
        log_id=log_id,
        user_id=None if user.is_admin else user.user_id,
    )
    if not deleted:
        raise APIError(status_code=404, code="not_found", message="Log not found")
    return {"deleted": 1, "log_id": log_id, "timestamp": now_iso()}


@router.delete("/logs")
async def clear_logs(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user: AuthUser = Depends(get_current_user),
) -> Dict[str, Any]:
    scoped_user_id = user_id if user.is_admin and user_id else user.user_id
    deleted = clear_audit_logs(user_id=scoped_user_id, session_id=session_id)
    return {
        "deleted": deleted,
        "session_id": session_id,
        "timestamp": now_iso(),
    }


@router.get("/memory/stats")
async def get_memory_stats(user: AuthUser = Depends(get_current_user)) -> Dict[str, Any]:
    stats = get_vector_store().stats()
    if user.is_admin:
        return stats
    return {
        "namespace": user.user_id,
        "documents": stats.get("namespaces", {}).get(user.user_id, 0),
        "dimension": stats.get("dimension"),
        "faiss_available": stats.get("faiss_available"),
        "ttl_seconds": settings.memory_ttl_seconds,
        "max_documents_per_user": settings.memory_max_documents_per_user,
    }


@router.delete("/memory/clear")
async def clear_memory(
    namespace: Optional[str] = None,
    _user: AuthUser = Depends(require_role("admin")),
) -> Dict[str, Any]:
    if settings.is_production:
        raise APIError(
            status_code=status.HTTP_403_FORBIDDEN,
            code="not_allowed",
            message="Not available in production",
        )
    get_vector_store().clear(namespace=namespace)
    return {"status": "cleared", "namespace": namespace or "all", "timestamp": now_iso()}


@router.get("/recovery-report")
async def recovery_report(
    _user: AuthUser = Depends(require_role("admin")),
) -> Dict[str, Any]:
    return get_recovery_engine().get_recovery_report()


@router.get("/metrics")
async def metrics(
    _user: AuthUser = Depends(require_role("admin")),
) -> Dict[str, Any]:
    return get_metrics_registry().snapshot()
