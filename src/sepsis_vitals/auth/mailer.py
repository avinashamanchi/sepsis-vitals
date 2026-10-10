"""
sepsis_vitals.auth.mailer
~~~~~~~~~~~~~~~~~~~~~~~~~
Transactional email (password reset) over SMTP using the standard library.

Configuration (all optional; without SMTP_HOST nothing is sent):

* ``SMTP_HOST``, ``SMTP_PORT`` (default 587; 465 uses implicit TLS)
* ``SMTP_USERNAME``, ``SMTP_PASSWORD``, ``SMTP_FROM``
* ``SMTP_STARTTLS`` (default ``true``; set ``false`` only for a local mail catcher)
* ``SEPSIS_APP_URL`` — public URL of the web app, including any base path

The reset token travels in the URL fragment (``#reset_token=``), which
browsers never send to servers, so it stays out of proxy and access logs.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from urllib.parse import quote

logger = logging.getLogger(__name__)


def reset_link(token: str) -> str:
    """Return the web-app URL that opens the set-new-password form."""
    app_url = os.getenv("SEPSIS_APP_URL", "http://localhost:3000").rstrip("/")
    return f"{app_url}/login#reset_token={quote(token, safe='')}"


def send_password_reset(to_email: str, token: str) -> bool:
    """Email a password-reset link. Returns False when SMTP is not configured.

    Never logs the recipient address or the token.
    """
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        logger.warning("Password reset requested but SMTP_HOST is not set; email not sent")
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_FROM") or username or "no-reply@localhost"

    msg = EmailMessage()
    msg["Subject"] = "Sepsis Vitals password reset"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(
        "A password reset was requested for your Sepsis Vitals research account.\n\n"
        f"Set a new password within one hour:\n{reset_link(token)}\n\n"
        "The link works once. If you did not request this, ignore this email; "
        "your password is unchanged.\n"
    )

    context = ssl.create_default_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=10, context=context) as smtp:
                if username:
                    smtp.login(username, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                if os.getenv("SMTP_STARTTLS", "true").lower() != "false":
                    smtp.starttls(context=context)
                if username:
                    smtp.login(username, password)
                smtp.send_message(msg)
    except (OSError, smtplib.SMTPException) as exc:
        logger.error("Password-reset email could not be sent: %s", type(exc).__name__)
        return False
    logger.info("Password-reset email sent")
    return True
