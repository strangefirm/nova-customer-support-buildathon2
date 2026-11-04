"""Inbound email: parsing, robot/duplicate filtering, auto-reply for resolved, acknowledgement + HITL reply for escalated."""
import email
from email import policy
from email.message import EmailMessage

import audit
import auth
import inbox
import logstore
from config import get_settings

from conftest import scripted_client

SHIP = dict(category="Shipping", urgency="Low", sentiment="Neutral", language="English", summary="Late order.")
GOOD_REPLY = "Please check Track Order in My Orders. Orders over 3 business days late can be investigated [1]. NovaCart Support"


def raw(sender="Priya <priya@example.com>", subject="Late order", body="Where is my order?", **headers) -> bytes:
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = sender, "support@novacart.example", subject
    m["Message-ID"] = headers.pop("mid", "<abc123@example.com>")
    for k, v in headers.items():
        m[k.replace("_", "-")] = v
    m.set_content(body)
    return bytes(m)


def sent_mails(_=None):
    files = sorted(get_settings().outbox_dir.glob("*.eml"), key=lambda f: f.stat().st_mtime_ns)  # send order
    return [email.message_from_bytes(f.read_bytes(), policy=policy.default) for f in files]


def test_parse_strips_quotes_and_signature():
    body = "My blender is broken.\n\n> old quoted text\nOn Mon, 1 Jan 2026, Support wrote:\n> earlier\n"
    m, why = inbox.parse_email(raw(body=body), get_settings())
    assert why == "" and m.body == "My blender is broken."
    assert m.sender == "priya@example.com" and m.sender_name == "Priya" and m.message_id == "<abc123@example.com>"


def test_robots_and_own_address_are_ignored():
    s = get_settings()
    for kw in (dict(sender="noreply@shop.com"), dict(Auto_Submitted="auto-generated"), dict(Precedence="bulk"),
               dict(List_Id="<news.example.com>"), dict(sender="no-reply@novacart.example")):
        m, why = inbox.parse_email(raw(**kw), s)
        assert m is None and why, kw


async def test_resolved_email_is_answered_in_thread():
    s = get_settings()
    client = scripted_client(SHIP, "late order tracking", GOOD_REPLY, 0.9, ["shipping_policy.pdf"])
    mail, _ = inbox.parse_email(raw(), s)
    row = await inbox.process_mail(mail, s, client)
    assert row["status"] == "Resolved" and row["channel"] == "email" and row["sender_email"] == "priya@example.com"
    out = sent_mails(None)
    assert len(out) == 1 and out[0]["To"] == "priya@example.com" and out[0]["Subject"] == "Re: Late order"
    assert out[0]["In-Reply-To"] == "<abc123@example.com>" and row["ticket_id"] in out[0].get_content()


async def test_duplicate_message_id_is_not_reprocessed():
    s = get_settings()
    mail, _ = inbox.parse_email(raw(), s)
    await inbox.process_mail(mail, s, scripted_client(SHIP, "late order", GOOD_REPLY, 0.9, ["shipping_policy.pdf"]))
    assert await inbox.process_mail(mail, s, scripted_client(SHIP, "late order", GOOD_REPLY, 0.9, ["shipping_policy.pdf"])) is None
    assert len(logstore.read_tickets()) == 1 and len(sent_mails(None)) == 1


async def test_escalated_email_gets_ack_then_human_reply():
    s = get_settings()
    client = scripted_client(
        dict(category="Billing", urgency="High", sentiment="Angry", language="English", summary="Triple charge."),
        "duplicate charge", "A specialist will review this. NovaCart Support", 0.7, ["billing_and_payments.pdf"])
    mail, _ = inbox.parse_email(raw(subject="Charged 3 times", body="You charged my card three times! I'm calling my lawyer.", mid="<x1@e.com>"), s)
    row = await inbox.process_mail(mail, s, client)
    assert row["status"] == "Escalated" and row["review_status"] == "Pending"
    ack = sent_mails(None)
    assert len(ack) == 1 and "specialist" in ack[0].get_content() and "AI" not in ack[0].get_content()
    assert logstore.update_review(row["ticket_id"], "Approved", "We refunded the duplicate charges.", "agent1")
    stored = logstore.read_tickets().iloc[0].to_dict()
    assert inbox.send_review_reply(stored, "We refunded the duplicate charges.", s)
    final = sent_mails(None)[-1]
    assert final["To"] == "priya@example.com" and "refunded" in final.get_content() and final["In-Reply-To"] == "<x1@e.com>"


async def test_sender_with_account_owns_the_ticket():
    s = get_settings()
    auth.create_user("priya", "Str0ng-Passw0rd!", "customer", "Priya", s, email="priya@example.com")
    mail, _ = inbox.parse_email(raw(), s)
    row = await inbox.process_mail(mail, s, scripted_client(SHIP, "late order", GOOD_REPLY, 0.9, ["shipping_policy.pdf"]))
    assert row["owner"] == "priya"


async def test_poll_once_reads_folder_and_moves_files(tmp_path):
    s = get_settings()
    inbox.write_demo_email(s, "ravi@example.com", "Late order", "Where is my order?")
    assert len(list(s.inbox_dir.glob("*.eml"))) == 1
    stats = await inbox.poll_once(s, scripted_client(SHIP, "late order", GOOD_REPLY, 0.9, ["shipping_policy.pdf"]))
    assert stats["fetched"] == 1 and stats["tickets"] == 1 and stats["resolved"] == 1 and not stats["error"]
    assert not list(s.inbox_dir.glob("*.eml")) and len(list((s.inbox_dir / "processed").glob("*.eml"))) == 1


def test_reply_headers_cannot_be_injected():
    import pytest, mailer
    with pytest.raises(ValueError):
        mailer.send_email("a@b.com", "hi", "x", get_settings(), in_reply_to="<a>\r\nBcc: evil@x.com")


def _triage_client(*verdicts):
    from autogen_ext.models.replay import ReplayChatCompletionClient
    from conftest import MODEL_INFO
    return ReplayChatCompletionClient(list(verdicts), model_info=MODEL_INFO)


async def test_non_support_email_gets_autoreply_and_no_ticket(monkeypatch):
    monkeypatch.setenv("MAIL_TRIAGE", "true")
    s = get_settings()
    inbox.write_demo_email(s, "sales@vendor.com", "Partnership", "We offer SEO services, want a call?")
    stats = await inbox.poll_once(s, model_client=None, triage_client=_triage_client("PROMOTIONAL"))
    assert stats["offtopic"] == 1 and stats["labels"] == {"PROMOTIONAL": 1} and stats["tickets"] == 0 and len(logstore.read_tickets()) == 0
    out = sent_mails()
    assert len(out) == 1 and out[0]["To"] == "sales@vendor.com" and "support" in out[0].get_content().lower()


async def test_offtopic_autoreply_is_limited_to_once_a_day(monkeypatch):
    monkeypatch.setenv("MAIL_TRIAGE", "true")
    s = get_settings()
    for i in range(2):
        inbox.write_demo_email(s, "chatty@example.com", f"hello {i}", "just saying hi")
    await inbox.poll_once(s, triage_client=_triage_client("GENERAL", "GENERAL"))
    assert len(sent_mails()) == 1


async def test_spam_is_dropped_without_a_reply_or_ticket(monkeypatch):
    monkeypatch.setenv("MAIL_TRIAGE", "true")
    s = get_settings()
    inbox.write_demo_email(s, "win@lottery.biz", "You won!!!", "claim your prize now")
    stats = await inbox.poll_once(s, triage_client=_triage_client("SPAM"))
    assert stats["labels"] == {"SPAM": 1} and not sent_mails() and len(logstore.read_tickets()) == 0


async def test_labels_parse_and_unclear_or_failed_triage_fails_open_to_support(monkeypatch):
    monkeypatch.setenv("MAIL_TRIAGE", "true")
    s = get_settings()
    mail, _ = inbox.parse_email(raw(), s)
    assert await inbox.classify_email(mail, s, _triage_client("business.")) == "BUSINESS"
    assert await inbox.classify_email(mail, s, _triage_client("SUPPORT")) == "SUPPORT"
    assert await inbox.classify_email(mail, s, _triage_client("not sure")) == "SUPPORT"
    assert await inbox.classify_email(mail, s, _triage_client()) == "SUPPORT"  # model raises -> still support


async def test_overlapping_polls_are_skipped(monkeypatch):
    s = get_settings()
    assert inbox._poll_lock.acquire(blocking=False)
    try:
        assert (await inbox.poll_once(s))["busy"] is True  # a poll already running: the second one does nothing
    finally:
        inbox._poll_lock.release()


def test_auto_poll_starts_once_and_respects_the_switch(monkeypatch):
    started = []
    monkeypatch.setattr(inbox.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: started.append(kw["name"])})())
    monkeypatch.setattr(inbox, "_auto_started", False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert not inbox.start_auto_poll(get_settings())  # MAIL_AUTO=false in tests
    monkeypatch.setenv("MAIL_AUTO", "true")
    assert inbox.start_auto_poll(get_settings()) and not inbox.start_auto_poll(get_settings())
    assert started == ["mail-auto-poll"]
