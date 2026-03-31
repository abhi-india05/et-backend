from __future__ import annotations

from typing import Any, Dict, Optional

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Email, Mail, To

from backend.config.settings import settings


def send_email(
    *,
    to_email: str,
    subject: str,
    content: str,
    html_content: Optional[str] = None,
    from_email: Optional[str] = None,
    from_name: str = "RevOps AI",
    to_name: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = (settings.sendgrid_api_key or "").strip()
    sender_email = (from_email or settings.sender_email or "").strip()

    if not api_key:
        raise RuntimeError("SENDGRID_API_KEY must be configured")
    if not sender_email:
        raise RuntimeError("SENDER_EMAIL must be configured")

    message_kwargs: Dict[str, Any] = {
        "from_email": Email(sender_email, from_name),
        "to_emails": To(to_email, to_name) if to_name else to_email,
        "subject": subject,
        "plain_text_content": content or "",
    }
    if html_content:
        message_kwargs["html_content"] = html_content

    message = Mail(**message_kwargs)

    response = SendGridAPIClient(api_key).send(message)
    return {
        "status": "success",
        "code": response.status_code,
    }