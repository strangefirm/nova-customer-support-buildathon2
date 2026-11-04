"""Drive the real Streamlit script headlessly: submit, review queue, log and dashboard."""
import time
from pathlib import Path

import logstore
import ui
from streamlit.testing.v1 import AppTest

from conftest import scripted_client

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _patch(monkeypatch, analysis, reply, conf, sources):
    orig = ui.run_ticket

    def patched(complaint, name="", order_id="", settings=None, model_client=None, owner="", **extra):
        return orig(complaint, name, order_id, settings, scripted_client(analysis, "q", reply, conf, sources), owner=owner, **extra)

    monkeypatch.setattr(ui, "run_ticket", patched)


def signed_in(username="alice", role="staff", name=None) -> AppTest:
    """An AppTest whose session is already authenticated (the login form itself is tested in test_auth)."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["user"] = {"username": username, "display_name": name or username.title(), "role": role}
    at.session_state["last_active"] = time.time()
    return at


def _fresh(username="alice", role="staff") -> AppTest:
    at = signed_in(username, role).run()
    assert not at.exception, at.exception
    return at


def _submit(text, username="alice", role="staff"):
    at = _fresh(username, role)
    at.text_area(key="complaint").set_value(text)
    next(b for b in at.button if b.label in ("Submit complaint", "Send request")).click()
    return at.run()


def _final(at):
    return next(e for e in at.session_state["last_run"] if e.kind == "final").data


def _seed(n_resolved=3, n_escalated=2):
    for i in range(n_resolved):
        logstore.append_ticket({"complaint": f"where is order {i}", "category": "Shipping", "urgency": "Low", "sentiment": "Neutral",
                                "language": "English", "kb_sources": "shipping_policy.md", "response": "Track it. $4.99 fee", "confidence": "0.90",
                                "status": "Resolved", "review_status": "N/A"})
    for i in range(n_escalated):
        logstore.append_ticket({"complaint": f"charged 3 times {i}", "category": "Billing", "urgency": "High", "sentiment": "Angry",
                                "language": "English", "response": "Sorry, a specialist will call.", "confidence": "0.80",
                                "status": "Escalated", "escalation_reason": "High urgency + Angry customer", "review_status": "Pending"})


def test_resolved_ticket_is_logged_and_replayed_after_rerun(monkeypatch):
    _patch(monkeypatch, dict(category="Account", urgency="Low", sentiment="Neutral", language="English", summary="Forgot password."),
           "Click Forgot password on the login page [1]. NovaCart Support", 0.9, [])
    at = _submit("I forgot my password")
    assert not at.exception
    assert _final(at)["status"] == "Resolved" and len(logstore.read_tickets()) == 1
    # result survives an unrelated rerun (e.g. switching tabs / touching a widget)
    at.run()
    assert not at.exception and _final(at)["ticket_id"] == "TKT-0001"
    assert any("Ticket TKT-0001" in s.value for s in at.subheader)
    assert logstore.read_tickets().iloc[0]["owner"] == "alice"  # owner comes from the session, not the form


def test_escalated_ticket_enters_review_queue_and_tab_count_updates(monkeypatch):
    _patch(monkeypatch, dict(category="Billing", urgency="High", sentiment="Angry", language="English", summary="Triple charge."),
           "We are sorry. A specialist will review this. NovaCart Support", 0.8, [])
    at = _submit("You charged me three times, I am calling my lawyer")
    assert not at.exception
    row = _final(at)
    assert row["status"] == "Escalated" and row["review_status"] == "Pending"
    assert "AI pipeline error" not in row["escalation_reason"]  # real escalation rules, not the error fallback
    assert "Review queue (1)" in [t.label for t in at.tabs][1]


def test_empty_complaint_is_rejected_in_ui():
    at = _submit("   ")
    assert any("enter a complaint" in w.value for w in at.warning)
    assert len(logstore.read_tickets()) == 0


def test_review_queue_approve_writes_decision():
    _seed(0, 1)
    at = _fresh()
    assert "Review queue (1)" in [t.label for t in at.tabs][1]
    at.text_area(key="final_TKT-0001").set_value("Edited reply for the customer")
    at.button(key="ok_TKT-0001").click()
    at.run()
    assert not at.exception
    r = logstore.read_tickets().iloc[0]
    assert (r["review_status"], r["final_response"]) == ("Edited", "Edited reply for the customer")
    assert "Review queue (0)" in [t.label for t in at.tabs][1]


def test_review_queue_reject():
    _seed(0, 1)
    at = _fresh()
    at.button(key="no_TKT-0001").click()
    at.run()
    assert logstore.read_tickets().iloc[0]["review_status"] == "Rejected"


def test_dashboard_and_log_render_with_data():
    _seed(3, 2)
    at = _fresh()  # renders log + dashboard code paths with real rows
    labels = {m.label: m.value for m in at.metric}
    assert labels["Tickets"] == "5" and labels["Resolved by AI"] == "60%" and labels["Escalated"] == "2"
    assert labels["Avg urgency"].startswith("1.8")  # (3*1 + 2*3) / 5 = 1.8


def test_log_filters_do_not_break_with_data():
    _seed(2, 2)
    at = _fresh()
    at.multiselect[0].select("Billing").run()
    assert not at.exception
    at.text_input[-1].set_value("nomatch-xyz").run()
    assert not at.exception
