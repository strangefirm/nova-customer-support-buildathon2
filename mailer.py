"""Outgoing email for password-reset and password-changed notices.

Backends (EMAIL_BACKEND):
  smtp    real delivery (STARTTLS on 587 by default, implicit TLS on 465). Used automatically when SMTP_HOST is set.
  outbox  writes the message to data/outbox/ instead of sending it. For demos and local development ONLY:
          anyone with read access to that folder can read reset links.
  off     sends nothing.

send_async() delivers in a background thread so that the time the page takes to respond does not reveal
whether an email address belongs to an account.
"""
from __future__ import annotations

import hashlib
import os
import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

import audit
from config import Settings, get_settings


def _build(to: str, subject: str, body: str, sender: str, in_reply_to: str = "") -> EmailMessage:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    if in_reply_to:  # keeps the answer in the customer's original thread
        msg["In-Reply-To"] = msg["References"] = in_reply_to
    msg["Auto-Submitted"] = "auto-replied" if in_reply_to else "auto-generated"  # stops mail loops with other robots
    msg["Date"], msg["Message-ID"] = formatdate(localtime=True), make_msgid()
    msg.set_content(body)
    return msg


def send_email(to: str, subject: str, body: str, settings: Settings | None = None, in_reply_to: str = "") -> None:
    """Deliver one message. Raises on failure (send_async catches and audits it)."""
    s = settings or get_settings()
    if s.email_backend == "off":
        return
    if any(c in v for v in (to, subject, in_reply_to) for c in "\r\n"):  # header injection guard
        raise ValueError("invalid header value")
    msg = _build(to, subject, body, s.smtp_from, in_reply_to)
    if s.email_backend == "outbox":
        s.outbox_dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{hashlib.sha256((to + subject + str(time.time_ns())).encode()).hexdigest()[:8]}.eml"
        path = s.outbox_dir / name
        path.write_bytes(bytes(msg))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return
    if not s.smtp_host:
        raise RuntimeError("EMAIL_BACKEND=smtp but SMTP_HOST is not set")
    if s.smtp_port == 465:
        server = smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, timeout=10, context=ssl.create_default_context())
    else:
        server = smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=10)
    with server:
        if s.smtp_port != 465 and s.smtp_starttls:
            server.starttls(context=ssl.create_default_context())
        if s.smtp_user:
            server.login(s.smtp_user, s.smtp_password)
        server.send_message(msg)


def send_async(to: str, subject: str, body: str, settings: Settings | None = None, audit_actor: str = "system") -> threading.Thread:
    """Fire-and-forget delivery. Failures are recorded in the audit trail (never with the address or link)."""
    s = settings or get_settings()

    def run() -> None:
        try:
            send_email(to, subject, body, s)
        except Exception as exc:  # noqa: BLE001 - delivery problems must never reach the requester
            audit.record(audit_actor, "", "email_failed", detail=type(exc).__name__, path=s.db_path)

    t = threading.Thread(target=run, daemon=True, name="send-email")
    t.start()
    return t
