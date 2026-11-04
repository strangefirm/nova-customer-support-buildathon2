"""Authentication, authorisation and data isolation."""
import time

import pytest
from streamlit.testing.v1 import AppTest

import audit
import auth
import db
import logstore
from config import get_settings

from test_app import APP, _patch, signed_in

PW = "correct horse battery"


def _row(owner, complaint, status="Resolved", **kw):
    base = {"owner": owner, "complaint": complaint, "category": "Shipping", "urgency": "Low", "sentiment": "Neutral",
            "language": "English", "kb_sources": "shipping_policy.pdf", "response": f"AI reply to {complaint}", "confidence": "0.90",
            "status": status, "review_status": "N/A" if status == "Resolved" else "Pending",
            "escalation_reason": "" if status == "Resolved" else "High urgency"}
    base.update(kw)
    return logstore.append_ticket(base)


# ---------- passwords and accounts ----------
def test_password_is_hashed_salted_and_never_stored_plain():
    auth.create_user("alice", PW, "staff")
    auth.create_user("bob.smith", PW, "customer")
    rows = {r["username"]: dict(r) for r in db.connect().execute("SELECT * FROM users")}
    assert PW not in " ".join(str(v) for r in rows.values() for v in r.values())  # nowhere in the stored data
    assert rows["alice"]["salt"] != rows["bob.smith"]["salt"]
    assert rows["alice"]["hash"] != rows["bob.smith"]["hash"]  # same password, different salt => different hash


@pytest.mark.parametrize("pw", ["short", "x" * 129, "password123", "aaaaaaaaaaaa", "xx-alice-xx-9"])
def test_weak_passwords_rejected(pw):
    with pytest.raises(ValueError):
        auth.create_user("alice", pw, "customer")


@pytest.mark.parametrize("name", ["ab", "has space", "UPPER!", "../etc", "x" * 40, ""])
def test_bad_usernames_rejected(name):
    with pytest.raises(ValueError):
        auth.create_user(name, PW, "customer")


def test_duplicate_username_rejected_case_insensitively():
    auth.create_user("alice", PW, "customer")
    with pytest.raises(ValueError):
        auth.create_user("ALICE", PW, "customer")


def test_signup_can_only_create_customers():
    u = auth.signup_customer("newbie", PW, "New B", "newbie@example.com")
    assert u.role == "customer"
    with pytest.raises(TypeError):
        auth.signup_customer("evil", PW, "x", "evil@example.com", role="staff")  # there is no way to request a role
    assert [x["role"] for x in auth.list_users()] == ["customer"]


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        auth.create_user("root.user", PW, "admin")


def test_correct_password_authenticates_wrong_does_not():
    auth.create_user("alice", PW, "staff", "Alice A")
    ok = auth.authenticate("alice", PW)
    assert ok and ok.role == "staff" and ok.display_name == "Alice A"
    assert auth.authenticate("alice", "wrong password!") is None
    assert auth.authenticate(" ALICE ", PW)  # username is case/space-insensitive


def test_unknown_user_and_wrong_password_are_indistinguishable():
    auth.create_user("alice", PW, "staff")
    assert auth.authenticate("alice", "nope-nope-nope") is None
    assert auth.authenticate("ghost", "nope-nope-nope") is None  # same result; the UI shows one generic message


def test_account_locks_after_repeated_failures_even_with_right_password():
    auth.create_user("alice", PW, "staff")
    for _ in range(get_settings().max_failed_logins):
        assert auth.authenticate("alice", "bad password 1") is None
    assert auth.authenticate("alice", PW) is None  # locked
    auth.unlock("alice")
    assert auth.authenticate("alice", PW)


def test_unknown_usernames_are_throttled_too():
    s = get_settings()
    assert not auth.is_locked("ghost")
    for _ in range(s.max_failed_logins + 2):
        auth.authenticate("ghost", "whatever pass")
    assert auth.is_locked("ghost")  # an account that does not exist is throttled exactly like one that does


def test_lockout_persists_across_restart():
    auth.create_user("alice", PW, "staff")
    for _ in range(get_settings().max_failed_logins):
        auth.authenticate("alice", "bad password 1")
    db.close_all()  # simulate a new process: only the database file remains
    assert auth.authenticate("alice", PW) is None


def test_malformed_and_oversized_credentials_do_not_crash():
    for u, p in [("", ""), ("a" * 500, "x"), ("alice", "x" * 10_000), ("'; DROP TABLE--", "x"), (None, None)]:
        assert auth.authenticate(u, p) is None


def test_failed_logins_and_lockouts_are_audited_without_passwords():
    auth.create_user("alice", PW, "staff")
    for _ in range(get_settings().max_failed_logins):
        auth.authenticate("alice", "secret-guess-123")
    auth.authenticate("alice", PW)
    text = audit.read().to_string()
    assert "login_fail" in text and "lockout" in text
    assert "secret-guess-123" not in text and PW not in text


# ---------- isolation ----------
def _tickets():
    _row("cust.a", "A late order")
    _row("cust.b", "B refund")
    _row("cust.a", "A billing", status="Escalated")
    return logstore.read_tickets()


def test_customer_sees_only_own_tickets():
    df = _tickets()
    a = auth.scope_tickets(df, auth.User("cust.a", "A", "customer"))
    b = auth.scope_tickets(df, auth.User("cust.b", "B", "customer"))
    assert set(a["owner"]) == {"cust.a"} and len(a) == 2
    assert set(b["owner"]) == {"cust.b"} and len(b) == 1


def test_staff_see_all_and_default_deny_for_everything_else():
    df = _tickets()
    assert len(auth.scope_tickets(df, auth.User("s", "S", "staff"))) == 3
    assert auth.scope_tickets(df, None).empty
    assert auth.scope_tickets(df, auth.User("x", "X", "admin")).empty  # unknown role: nothing
    assert auth.scope_tickets(df.drop(columns=["owner"]), auth.User("cust.a", "A", "customer")).empty  # legacy rows: nobody


def test_username_lookalikes_do_not_leak():
    _row("cust.a", "mine")
    df = logstore.read_tickets()
    assert auth.scope_tickets(df, auth.User("cust.", "x", "customer")).empty
    assert auth.scope_tickets(df, auth.User("cust.a ", "x", "customer")).empty


def test_require_staff():
    auth.require_staff(auth.User("s", "S", "staff"))
    for u in (None, auth.User("c", "C", "customer")):
        with pytest.raises(PermissionError):
            auth.require_staff(u)


def test_customer_view_hides_internal_fields_and_unapproved_drafts():
    t = _row("cust.a", "triple charge", status="Escalated", security_flags="prompt_injection", escalation_reason="Legal threat")
    v = auth.customer_view(logstore.read_tickets().iloc[0])
    assert v["reply"] == "" and v["status_label"] == "With our support team"
    assert set(v) == {"ticket_id", "timestamp", "category", "complaint", "status_label", "reply", "citations"}  # nothing internal
    assert v["citations"] == ""  # no citations either until a human has approved the reply
    logstore.update_review(t["ticket_id"], "Edited", "Final human answer", "alice")
    v2 = auth.customer_view(logstore.read_tickets().iloc[0])
    assert v2["reply"] == "Final human answer" and v2["status_label"] == "Answered by our team"
    logstore.append_ticket({"owner": "cust.a", "status": "Escalated", "review_status": "Pending", "response": "draft"})
    assert auth.customer_view(logstore.read_tickets().iloc[-1])["reply"] == ""


# ---------- the app itself ----------
def _all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.subheader]
    parts += [i.value for i in at.info] + [t.value for t in at.text]
    for df in at.dataframe:
        parts.append(df.value.to_string())
    return " ".join(parts)


def test_anonymous_visitor_sees_only_the_sign_in_form():
    _tickets()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    text = _all_text(at)
    assert "A late order" not in text and "B refund" not in text
    assert [t.label for t in at.tabs][0].endswith("Sign in")
    assert not at.dataframe and not [b for b in at.button if b.label == "Sign out"]


def test_sign_in_flow_and_generic_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    auth.create_user("alice", PW, "staff", "Alice A")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].set_value("alice")
    at.text_input[1].set_value("wrong password!")
    at.button[0].click().run()
    assert any(auth.GENERIC_FAILURE in e.value for e in at.error)
    at.text_input[1].set_value(PW)
    at.button[0].click().run()
    assert not at.exception and at.session_state["user"]["role"] == "staff"
    assert any(b.label == "Sign out" for b in at.button)


def test_self_signup_creates_customer_session():
    at = AppTest.from_file(APP, default_timeout=60).run()
    f = list(at.text_input)
    f[2].set_value("Casey Q")
    f[3].set_value("casey.q")
    f[4].set_value("casey@example.com")
    f[5].set_value(PW)
    f[6].set_value(PW)
    next(b for b in at.button if b.label == "Create account").click().run()
    assert not at.exception
    assert at.session_state["user"] == {"username": "casey.q", "display_name": "Casey Q", "role": "customer"}


def test_signup_password_mismatch_rejected():
    at = AppTest.from_file(APP, default_timeout=60).run()
    f = list(at.text_input)
    f[2].set_value("Casey")
    f[3].set_value("casey.q")
    f[4].set_value("casey@example.com")
    f[5].set_value(PW)
    f[6].set_value(PW + "x")
    next(b for b in at.button if b.label == "Create account").click().run()
    assert any("do not match" in e.value for e in at.error) and not auth.list_users()


def test_customer_sees_only_own_requests_and_no_staff_tabs():
    _tickets()
    at = signed_in("cust.b", "customer").run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert any("My requests (1)" in x for x in labels)
    assert not any(k in " ".join(labels) for k in ("Review", "Ticket log", "Dashboard", "Audit"))
    text = _all_text(at)
    assert "B refund" in text and "A late order" not in text and "A billing" not in text
    assert not at.dataframe  # no raw table for customers


def test_customer_never_sees_internal_details_of_own_escalated_ticket():
    _row("cust.a", "my escalated issue", status="Escalated", escalation_reason="Legal threat", security_flags="prompt_injection",
         response="SECRET INTERNAL DRAFT")
    text = _all_text(signed_in("cust.a", "customer").run())
    assert "my escalated issue" in text and "specialist is reviewing" in text
    assert "SECRET INTERNAL DRAFT" not in text and "Legal threat" not in text and "prompt_injection" not in text


def test_staff_see_everything_and_all_tabs():
    _tickets()
    at = signed_in("alice", "staff").run()
    assert not at.exception
    labels = " ".join(t.label for t in at.tabs)
    for k in ("Log a ticket for customer", "Review queue (1)", "Ticket log", "Dashboard", "Audit trail"):
        assert k in labels
    assert {m.label: m.value for m in at.metric}["Tickets"] == "3"


def test_customer_submission_is_owned_by_their_session_and_hidden_from_others(monkeypatch):
    _patch(monkeypatch, dict(category="Shipping", urgency="Low", sentiment="Neutral", language="English", summary="Late."),
           "Please check Track Order [1]. NovaCart Support", 0.9, [])
    at = signed_in("cust.a", "customer", "Cara A").run()
    at.text_area(key="complaint").set_value("Where is my parcel")
    next(b for b in at.button if b.label == "Send request").click().run()
    assert not at.exception
    row = logstore.read_tickets().iloc[0]
    assert row["owner"] == "cust.a" and row["customer_name"] == "Cara A"  # name from the account, not the form
    assert "Where is my parcel" not in _all_text(signed_in("cust.b", "customer").run())
    audit_text = audit.read().to_string()
    assert "ticket_submit" in audit_text and "cust.a" in audit_text


def test_session_expires_after_idle_timeout():
    at = signed_in("alice", "staff")
    at.session_state["last_active"] = time.time() - 31 * 60
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs][0].endswith("Sign in")
    assert "inactivity" in _all_text(at)


def test_sign_out_clears_the_session():
    _tickets()
    at = signed_in("alice", "staff").run()
    next(b for b in at.button if b.label == "Sign out").click().run()
    assert "user" not in at.session_state and not at.dataframe
    assert [t.label for t in at.tabs][0].endswith("Sign in")


def test_staff_review_records_authenticated_reviewer_in_audit():
    t = _row("cust.a", "needs a human", status="Escalated")
    at = signed_in("alice", "staff").run()
    at.button(key=f"ok_{t['ticket_id']}").click().run()
    assert logstore.read_tickets().iloc[0]["reviewed_by"] == "alice"
    assert "review_approved" in audit.read().to_string()
