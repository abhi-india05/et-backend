from __future__ import annotations

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Any, Dict, Optional

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
    username = (settings.mail_username or "").strip()
    password = (settings.mail_password or "").strip()
    sender_email = (from_email or settings.resolved_mail_from or "").strip()
    smtp_server = (settings.mail_server or "").strip()
    smtp_port = int(settings.mail_port)

    if not username:
        raise RuntimeError("MAIL_USERNAME must be configured")
    if not password:
        raise RuntimeError("MAIL_PASSWORD must be configured")
    if not sender_email:
        raise RuntimeError("MAIL_FROM (or SENDER_EMAIL) must be configured")
    if not smtp_server:
        raise RuntimeError("MAIL_SERVER must be configured")

    if html_content:
        message: MIMEText | MIMEMultipart = MIMEMultipart("alternative")
        message.attach(MIMEText(content or "", "plain", "utf-8"))
        message.attach(MIMEText(html_content, "html", "utf-8"))
    else:
        message = MIMEText(content or "", "plain", "utf-8")

    message["Subject"] = subject
    message["From"] = formataddr((from_name, sender_email))
    message["To"] = formataddr((to_name, to_email)) if to_name else to_email

    if settings.mail_ssl:
        smtp_client: smtplib.SMTP | smtplib.SMTP_SSL = smtplib.SMTP_SSL(
            smtp_server,
            smtp_port,
            timeout=settings.mail_timeout_seconds,
        )
    else:
        smtp_client = smtplib.SMTP(
            smtp_server,
            smtp_port,
            timeout=settings.mail_timeout_seconds,
        )

    with smtp_client as client:
        client.ehlo()
        if settings.mail_tls and not settings.mail_ssl:
            client.starttls(context=ssl.create_default_context())
            client.ehlo()
        client.login(username, password)
        client.send_message(message, from_addr=sender_email, to_addrs=[to_email])

    return {
        "status": "success",
        "code": 250,
    }