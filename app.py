"""NovaCart AI customer support desk - Streamlit UI.  Run with:  streamlit run app.py
Staff-only tabs (review queue, ticket log, dashboard, audit trail, email inbox) live in staff_ui.py."""
from __future__ import annotations

import asyncio
import time

import pandas as pd
import streamlit as st

import audit
import auth
import account_ui
import auth_ui
import inbox
import media
import staff_ui
import voice_ui
from config import get_settings
from kb.store import get_kb
from ui import render_customer_run, render_customer_ticket, render_run, safe_md, stream_run, stream_run_customer

st.set_page_config(page_title="NovaCart support desk", page_icon=":material/support_agent:", layout="wide")

PIPELINE = [
    (":material/psychology:", "Customer query agent", "Category, urgency, sentiment"),
    (":material/menu_book:", "Knowledge base agent", "Finds the right policy"),
    (":material/edit_note:", "Resolution agent", "Writes a grounded reply"),
    (":material/fact_check:", "Log agent", "Records and escalates"),
]

s = get_settings()
inbox.start_auto_poll(s)  # incoming mail is triaged on its own; the button below is only for an immediate check

# ---------- sign-in gate: nothing below runs for anonymous visitors ----------
user = auth_ui.require_login()
st.session_state.setdefault("last_run", None)
st.session_state.setdefault("submit_times", [])


MAX_ROWS = 20000  # newest tickets shown in the log and dashboard


def load_tickets(for_user: auth.User) -> pd.DataFrame:
    """Customers' rows are filtered inside the database; auth.scope_tickets re-checks the result."""
    return auth.tickets_for_user(for_user, s, limit=MAX_ROWS)


def rate_limited() -> bool:
    now = time.monotonic()
    recent = [t for t in st.session_state["submit_times"] if now - t < 60]
    st.session_state["submit_times"] = recent
    if len(recent) >= s.rate_limit_per_min:
        return True
    recent.append(now)
    return False


tickets = load_tickets(user)
pending = (
    tickets[(tickets["status"] == "Escalated") & (tickets["review_status"] == "Pending")]
    if (user.is_staff and len(tickets)) else tickets.iloc[0:0]
)

# ---------- sidebar ----------
with st.sidebar:
    st.title("NovaCart", icon=":material/support_agent:")
    st.caption("AI support desk")
    with st.container(border=True):
        st.markdown(f"**{safe_md(user.display_name)}**")
        st.badge("Staff" if user.is_staff else "Customer", icon=":material/badge:" if user.is_staff else ":material/person:",
                 color="violet" if user.is_staff else "blue")
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            auth_ui.sign_out()
            st.rerun()
    if user.is_staff:
        st.subheader("System")
        with st.container(border=True):
            if s.has_key:
                st.badge("API key configured", icon=":material/key:", color="green")
            else:
                st.badge("API key missing", icon=":material/key_off:", color="red")
            try:
                kb = get_kb(s)
                st.badge(kb.retrieval_label, icon=":material/manage_search:", color="orange" if (kb.index and kb.index.note) else "blue")
                if kb.index and kb.index.note:
                    st.caption(f":material/warning: {kb.index.note}")
                st.caption(f"{len(kb.sources)} documents · {len(kb.chunks)} passages")
            except Exception:  # noqa: BLE001 - the sidebar must never take the page down
                st.badge("Knowledge base unavailable", icon=":material/error:", color="red")
            st.caption(f"Model `{s.model}` · {s.team_mode}")
        st.subheader("Pipeline")
        for icon, name, what in PIPELINE:
            st.markdown(f"{icon} **{name}**  \n{what}")

st.title("Customer support desk" if user.is_staff else "How can we help?", icon=":material/support_agent:")
st.caption(
    "AI agents triage each complaint, answer from company policy and hand the hard cases to your team."
    if user.is_staff else "Tell us what went wrong. You can follow every request you send under My requests."
)
if msg := st.session_state.pop("toast", None):
    st.toast(msg, icon=":material/check_circle:")

if user.is_staff:
    labels = [":material/edit_note: Log a ticket for customer", f":material/rate_review: Review queue ({len(pending)})",
              ":material/table_rows: Ticket log", ":material/monitoring: Dashboard", ":material/policy: Audit trail",
              ":material/manage_accounts: Account", ":material/inbox: Email inbox"]
    tab_submit, tab_review, tab_log, tab_dash, tab_audit, tab_account, tab_mail = st.tabs(labels)
    tab_talk = None  # the voice agent is for customers; staff do not raise complaints about themselves
else:
    tab_submit, tab_mine, tab_talk, tab_account = st.tabs(
        [":material/edit_note: New request", f":material/inbox: My requests ({len(tickets)})", ":material/mic: Talk to us", ":material/manage_accounts: Account"])
    tab_review = tab_log = tab_dash = tab_audit = None

with tab_account:
    account_ui.render_account(user)

if tab_talk is not None:
    with tab_talk:
        voice_ui.render_talk(user, s)

# ---------- submit (both roles) ----------
with tab_submit:
    left, right = st.columns([2, 3], gap="large") if user.is_staff else st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Log a ticket for a customer" if user.is_staff else "Describe the problem", icon=":material/mail:")
            if not s.has_key:
                st.error("The assistant is not available right now. Please try again later." if not user.is_staff
                         else "No API key found. Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, then restart.", icon=":material/key_off:")
            with st.form("ticket", border=False, clear_on_submit=True):
                if user.is_staff:
                    c1, c2 = st.columns(2)
                    name = c1.text_input("Customer name", max_chars=60, placeholder="Optional")
                    order = c2.text_input("Order ID", max_chars=24, placeholder="Optional")
                else:
                    name = user.display_name  # taken from the signed-in account, not from the form
                    order = st.text_input("Order ID", max_chars=24, placeholder="Optional")
                complaint = st.text_area(
                    "Complaint" if user.is_staff else "Your message", key="complaint", height=170, max_chars=s.max_complaint_chars,
                    placeholder="Describe the problem in the customer's own words…" if user.is_staff else "What happened?",
                )
                if s.media_enabled:
                    voice = st.audio_input("Or record a voice message", sample_rate=16000)
                    photos = st.file_uploader("Attach photos (optional)", type=list(media.IMAGE_TYPES), accept_multiple_files=True,
                                              help=f"Up to {s.max_images} photos, {s.max_image_mb} MB each")
                else:
                    voice, photos = None, []
                submitted = st.form_submit_button(
                    "Submit complaint" if user.is_staff else "Send request", type="primary", icon=":material/send:",
                    width="stretch", disabled=not s.has_key,
                )
            st.caption("Card numbers, emails and phone numbers are masked before any processing. Voice and photos are turned into text first.")

    with right:
        if submitted:
            if not complaint.strip() and not voice and not photos:
                st.warning("Please enter a complaint, record a voice message or attach a photo.", icon=":material/edit:")
            elif rate_limited():
                st.warning("Too many requests in the last minute. Please wait a moment.", icon=":material/hourglass_top:")
            else:
                files = ([voice.getvalue()] if voice else []) + [f.getvalue() for f in (photos or [])]
                with st.spinner("Reading your voice message and photos…" if files else "Starting…"):
                    prepared = asyncio.run(media.build_complaint(complaint, files, s))
                extra = dict(flags=prepared.flags, extra_chars=prepared.extra_chars)
                events = (stream_run(prepared.text, name, order, owner=user.username, **extra) if user.is_staff
                          else stream_run_customer(prepared.text, order, owner=user.username, name=name, **extra))
                st.session_state["last_run"] = events
                final = next((e for e in events if e.kind == "final"), None)
                if final:
                    audit.record(user.username, user.role, "ticket_submit", final.data["ticket_id"], final.data["status"])
                    st.rerun()  # refresh counts, log and dashboard
                else:
                    render_run(events) if user.is_staff else render_customer_run(events)
        elif st.session_state["last_run"]:
            (render_run if user.is_staff else render_customer_run)(st.session_state["last_run"])
        elif user.is_staff:
            with st.container(border=True):
                st.subheader("Agent activity", icon=":material/account_tree:")
                st.caption("Submit a complaint and watch the four agents work through it step by step.")

# ---------- customer: my requests (only their own tickets) ----------
if not user.is_staff:
    with tab_mine:
        st.subheader("My requests", icon=":material/inbox:")
        if tickets.empty:
            st.info("You have not sent any requests yet.", icon=":material/inbox:")
        for _, row in tickets.iloc[::-1].head(50).iterrows():
            render_customer_ticket(row)

# ---------- staff only from here (staff_ui.py) ----------
if user.is_staff:
    staff_ui.render_staff(user, s, tickets, pending, tab_review, tab_log, tab_dash, tab_audit, tab_mail)
