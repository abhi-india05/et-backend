from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection

from backend.config.settings import settings
from backend.email_service import send_email as smtp_send_email
from backend.db.mongo import get_sync_database
from backend.utils.helpers import generate_id, now_iso
from backend.utils.logger import get_logger

logger = get_logger("email_tool")


def _get_emails_col() -> Collection:
    return get_sync_database()["sent_emails"]


class EmailClient:
    def __init__(self):
        self.transport = settings.email_transport
        logger.info(
            "SMTP email client initialized (live mode)",
            mail_server=settings.mail_server,
            mail_port=settings.mail_port,
            sender_email=settings.resolved_mail_from,
        )

    def _build_html_fallback(self, body_text: str) -> str:
        escaped = html.escape(body_text or "")
        escaped = escaped.replace("\r\n", "\n").replace("\r", "\n")
        return "<p>" + escaped.replace("\n", "<br>") + "</p>"

    def _send_via_smtp(
        self,
        *,
        to_email: str,
        to_name: str,
        subject: str,
        body_text: str,
        body_html: str,
        from_email: Optional[str],
        from_name: str,
    ) -> Dict[str, Any]:
        return smtp_send_email(
            to_email=to_email,
            to_name=to_name or None,
            subject=subject,
            content=body_text or "",
            html_content=body_html or None,
            from_email=from_email,
            from_name=from_name,
        )

    def send_email(
        self,
        to_email: str,
        to_name: str,
        subject: str,
        body_text: str,
        user_id: str,
        body_html: Optional[str] = None,
        from_email: Optional[str] = None,
        from_name: str = "RevOps AI",
        sequence_id: Optional[str] = None,
        sequence_step: int = 1,
    ) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required")

        effective_from_email = (from_email or settings.resolved_mail_from or settings.mail_username or "").strip()
        if not effective_from_email:
            raise RuntimeError("MAIL_FROM (or SENDER_EMAIL) must be configured before sending email")

        email_record = {
            "user_id": user_id,
            "email_id": generate_id("email"),
            "to_email": to_email,
            "to_name": to_name,
            "from_email": effective_from_email,
            "from_name": from_name,
            "subject": subject,
            "body_text": body_text,
            "body_html": body_html or self._build_html_fallback(body_text),
            "sequence_id": sequence_id,
            "sequence_step": sequence_step,
            "sent_at": now_iso(),
            "status": "pending",
            "transport": self.transport,
        }

        try:
            send_result = self._send_via_smtp(
                to_email=to_email,
                to_name=to_name,
                subject=subject,
                body_text=body_text,
                body_html=email_record["body_html"],
                from_email=effective_from_email,
                from_name=from_name,
            )

            email_record["status"] = "sent"
            email_record["provider_response_code"] = send_result.get("code")
            _get_emails_col().insert_one(email_record.copy())
            logger.info("Email sent", to=to_email, user_id=user_id, transport=self.transport)
            return {
                "success": True,
                "email_id": email_record["email_id"],
                "status": "sent",
                "code": send_result.get("code"),
            }
        except Exception as e:
            email_record["status"] = "failed"
            email_record["error"] = str(e)
            _get_emails_col().insert_one(email_record.copy())
            logger.error("Email send failed", to=to_email, error=str(e), user_id=user_id)
            return {
                "success": False,
                "email_id": email_record["email_id"],
                "status": "failed",
                "error": str(e),
            }

    def send_sequence(
        self,
        to_email: str,
        to_name: str,
        emails: List[Dict[str, str]],
        sequence_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required")
            
        results = []
        failed = 0

        for i, email in enumerate(emails, 1):
            result = self.send_email(
                to_email=to_email,
                to_name=to_name,
                subject=email.get("subject", f"Follow-up #{i}"),
                body_text=email.get("body", ""),
                user_id=user_id,
                from_email=email.get("from_email"),
                from_name=email.get("from_name", "RevOps AI"),
                sequence_id=sequence_id,
                sequence_step=i,
            )
            results.append(result)
            if not result.get("success"):
                failed += 1

        return {
            "sequence_id": sequence_id,
            "total_emails": len(emails),
            "sent": len(emails) - failed,
            "failed": failed,
            "results": results,
            "timestamp": now_iso(),
        }


def get_sent_emails(
    user_id: str,
    to_email: Optional[str] = None,
    sequence_id: Optional[str] = None,
) -> List[Dict]:
    if not user_id:
        raise ValueError("user_id is required")
        
    query: Dict[str, Any] = {"user_id": user_id}
    if to_email:
        query["to_email"] = to_email
    if sequence_id:
        query["sequence_id"] = sequence_id
        
    return list(_get_emails_col().find(query, {"_id": 0}))


def get_email_stats(user_id: str) -> Dict[str, Any]:
    if not user_id:
        raise ValueError("user_id is required")
        
    col = _get_emails_col()
    total = col.count_documents({"user_id": user_id})
    sent = col.count_documents({"user_id": user_id, "status": "sent"})
    failed = col.count_documents({"user_id": user_id, "status": "failed"})
    
    return {
        "total": total,
        "sent": sent,
        "failed": failed,
        "success_rate": sent / total if total > 0 else 0.0,
    }


_email_client_instance: Optional[EmailClient] = None


def get_email_client() -> EmailClient:
    global _email_client_instance
    if _email_client_instance is None:
        _email_client_instance = EmailClient()
    return _email_client_instance
