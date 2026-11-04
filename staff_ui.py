"""Staff-only tabs of the support desk: review queue, ticket log, dashboard, audit trail, email inbox. Called from app.py."""
from __future__ import annotations

import asyncio
import io
from datetime import datetime

import pandas as pd
import streamlit as st

import audit
import auth
import inbox
import logstore
import telegram_bot
from models import URGENCY_RANK
from ui import SENTIMENT_COLOR, URGENCY_COLOR, citation_badges, safe_md, sources_line, status_badge


def render_staff(user, s, tickets, pending, tab_review, tab_log, tab_dash, tab_audit, tab_mail) -> None:
    # ---------- review queue (human in the loop) ----------
    with tab_review:
        st.subheader("Awaiting human review", icon=":material/rate_review:")
        st.caption("Escalated tickets keep an AI draft. Edit it if needed, then approve or reject.")
        if pending.empty:
            st.success("All caught up. No escalated tickets are waiting.", icon=":material/task_alt:")
        for _, t in pending.iloc[::-1].iterrows():
            tid = t["ticket_id"]
            with st.container(border=True):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"**{tid}**")
                    if t["channel"] == "email":
                        st.badge("Email", icon=":material/mail:", color="violet")
                    elif t["channel"] == "voice":
                        st.badge("Voice call", icon=":material/call:", color="violet")
                    elif t["channel"] == "telegram":
                        st.badge("Telegram", icon=":material/send:", color="blue")
                    st.badge(t["category"], color="blue")
                    st.badge(t["urgency"], color=URGENCY_COLOR.get(t["urgency"], "gray"))
                    st.badge(t["sentiment"], color=SENTIMENT_COLOR.get(t["sentiment"], "gray"))
                    st.caption(t["timestamp"])
                st.markdown(" ".join(f":orange-badge[{r}]" for r in t["escalation_reason"].split(" + ") if r))
                who = f"{t['customer_name'] or 'Unknown customer'} · {t['sender_email'] or t['caller_phone'] or t['order_id'] or 'no order ID'}"
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.caption(f"Customer complaint · {who}")
                    st.markdown(safe_md(t["complaint"]))
                    st.markdown(("**Cited:** " + citation_badges(t["citations"])) if t["citations"] else sources_line(t["kb_sources"].split("; ")))
                with col_b:
                    final = st.text_area("Reply to send", value=t["response"], key=f"final_{tid}", height=180)
                    with st.container(horizontal=True):
                        approve = st.button("Approve and email" if t["channel"] == "email" else ("Approve and send on Telegram" if t["channel"] == "telegram" else ("Approve (then follow up)" if t["channel"] == "voice" else "Approve and send")), key=f"ok_{tid}", type="primary", icon=":material/check:")
                        reject = st.button("Reject", key=f"no_{tid}", icon=":material/close:")
                decision = "Rejected" if reject else (("Edited" if final.strip() != t["response"].strip() else "Approved") if approve else None)
                if decision:
                    auth.require_staff(user)
                    if logstore.update_review(tid, decision, "" if decision == "Rejected" else final, user.username):
                        audit.record(user.username, user.role, f"review_{decision.lower()}", tid)
                        if decision != "Rejected" and t["channel"] == "telegram":  # answer in the customer's chat
                            sent = telegram_bot.send_review_reply(dict(t), final, s)
                            st.session_state["toast"] = f"{tid} {decision.lower()}" + (" and sent on Telegram" if sent else " (Telegram message could not be sent)")
                            st.rerun()
                        if decision != "Rejected" and t["channel"] == "email":  # answer the customer's email
                            sent = inbox.send_review_reply(dict(t), final, s)
                            st.session_state["toast"] = f"{tid} {decision.lower()}" + (" and emailed" if sent else " (email could not be sent)")
                            st.rerun()
                        st.session_state["toast"] = f"{tid} {decision.lower()}"
                    else:
                        st.session_state["toast"] = f"{tid} was already reviewed by someone else"
                    st.rerun()
        reviewed = tickets[tickets["review_status"].isin(["Approved", "Edited", "Rejected"])] if len(tickets) else tickets
        if len(reviewed):
            with st.expander(f"Reviewed tickets ({len(reviewed)})", icon=":material/history:"):
                st.dataframe(
                    reviewed.iloc[::-1][["ticket_id", "review_status", "reviewed_by", "reviewed_at", "category"]],
                    hide_index=True, width="stretch", alt="Reviewed tickets",
                    column_config={"ticket_id": "Ticket", "review_status": "Decision", "reviewed_by": "Reviewer",
                                   "reviewed_at": "Reviewed at", "category": "Category"},
                )

    # ---------- ticket log ----------
    with tab_log:
        st.subheader("Ticket log", icon=":material/table_rows:")
        if tickets.empty:
            st.info("No tickets yet. Submit a complaint to create the first one.", icon=":material/inbox:")
        else:
            with st.container(horizontal=True, vertical_alignment="bottom"):
                status_f = st.segmented_control("Status", ["All", "Resolved", "Escalated"], default="All", required=True)
                cat_f = st.multiselect("Category", sorted(tickets["category"].unique()), placeholder="All categories")
                query = st.text_input("Search", placeholder="Search complaints, replies, IDs…", icon=":material/search:")
            view = tickets
            if status_f != "All":
                view = view[view["status"] == status_f]
            if cat_f:
                view = view[view["category"].isin(cat_f)]
            if query:
                q = query.lower()
                view = view[view[["ticket_id", "owner", "customer_name", "order_id", "complaint", "response"]].apply(
                    lambda col: col.str.lower().str.contains(q, regex=False)).any(axis=1)]
            view = view.iloc[::-1].reset_index(drop=True)
            shown = view.assign(confidence=pd.to_numeric(view["confidence"], errors="coerce"))
            event = st.dataframe(
                shown, hide_index=True, width="stretch", height=380, on_select="rerun", selection_mode="single-row",
                alt="Support tickets, newest first. Select a row to see its details.",
                column_order=["ticket_id", "timestamp", "channel", "status", "category", "urgency", "sentiment", "confidence", "owner",
                              "customer_name", "order_id", "complaint", "citations", "escalation_reason", "review_status"],
                column_config={
                    "ticket_id": st.column_config.TextColumn("Ticket", width="small"),
                    "timestamp": st.column_config.TextColumn("Time", width="small"),
                    "channel": st.column_config.TextColumn("Channel", width="small"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "category": st.column_config.TextColumn("Category", width="small"),
                    "urgency": st.column_config.TextColumn("Urgency", width="small"),
                    "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="percent", width="small"),
                    "owner": st.column_config.TextColumn("Account", width="small"),
                    "customer_name": st.column_config.TextColumn("Customer", width="small"),
                    "order_id": st.column_config.TextColumn("Order", width="small"),
                    "complaint": st.column_config.TextColumn("Complaint", width="large"),
                    "citations": st.column_config.TextColumn("Citations", width="large"),
                    "escalation_reason": st.column_config.TextColumn("Escalation reason", width="large"),
                    "review_status": st.column_config.TextColumn("Review", width="small"),
                },
            )
            st.caption(f"{len(view)} of {len(tickets)} tickets · select a row for details")
            with st.container(horizontal=True):
                st.download_button("Download CSV", logstore.export_frame(view).to_csv(index=False).encode("utf-8-sig"), "tickets.csv", "text/csv",
                                   icon=":material/download:", on_click=audit.record, args=(user.username, user.role, "download_csv", "", f"{len(view)} rows"))
                try:
                    buf = io.BytesIO()
                    logstore.export_frame(view).to_excel(buf, index=False, engine="openpyxl")
                    st.download_button("Download Excel", buf.getvalue(), "tickets.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/table:",
                                       on_click=audit.record, args=(user.username, user.role, "download_xlsx", "", f"{len(view)} rows"))
                except ImportError:
                    st.caption("Install openpyxl to enable Excel export.")

            if event.selection.rows:
                r = view.iloc[event.selection.rows[0]]
                seen = st.session_state.setdefault("viewed", set())
                if r["ticket_id"] not in seen:  # audit each ticket opened once per session
                    seen.add(r["ticket_id"])
                    audit.record(user.username, user.role, "view_ticket", r["ticket_id"])
                with st.container(border=True):
                    with st.container(horizontal=True, vertical_alignment="center"):
                        st.subheader(f"{r['ticket_id']}")
                        status_badge(r["status"])
                        if r["review_status"] not in ("N/A", "Pending", ""):
                            st.badge(f"{r['review_status']} by {r['reviewed_by']}", icon=":material/verified:", color="violet")
                    if r["escalation_reason"]:
                        st.markdown(" ".join(f":orange-badge[{x}]" for x in r["escalation_reason"].split(" + ") if x))
                    d1, d2 = st.columns(2, gap="large")
                    d1.caption("Complaint")
                    d1.markdown(safe_md(r["complaint"]))
                    d2.caption("Final reply" if r["final_response"] else "AI reply")
                    d2.markdown(safe_md(r["final_response"] or r["response"]))
                    st.markdown(("**Cited:** " + citation_badges(r["citations"])) if r["citations"] else sources_line(r["kb_sources"].split("; ")))

    # ---------- dashboard ----------
    with tab_dash:
        st.subheader("Support analytics", icon=":material/monitoring:")
        if tickets.empty:
            st.info("The dashboard fills up as tickets arrive.", icon=":material/insights:")
        else:
            df = tickets.assign(day=pd.to_datetime(tickets["timestamp"], errors="coerce").dt.date)
            total = len(df)
            esc_n = int((df["status"] == "Escalated").sum())
            daily = df.groupby("day").size().sort_index()
            conf = pd.to_numeric(df["confidence"], errors="coerce").mean()
            avg_u = df["urgency"].map(URGENCY_RANK).dropna().mean()
            with st.container(horizontal=True):
                st.metric("Tickets", total, border=True, chart_data=daily.tolist() if len(daily) > 1 else None, chart_type="bar")
                st.metric("Resolved by AI", f"{(total - esc_n) / total:.0%}", f"{total - esc_n} of {total}", border=True, delta_color="off")
                st.metric("Escalated", esc_n, f"{len(pending)} awaiting review", border=True, delta_color="off")
                st.metric("Avg urgency", f"{avg_u:.1f} / 3" if pd.notna(avg_u) else "—", border=True)
                st.metric("Avg confidence", f"{conf:.0%}" if pd.notna(conf) else "—", border=True)

            c1, c2 = st.columns(2)
            with c1.container(border=True):
                st.markdown("**Tickets by category**")
                by_cat = df.pivot_table(index="category", columns="status", values="ticket_id", aggfunc="count", fill_value=0)
                by_cat = by_cat.reindex(columns=["Resolved", "Escalated"], fill_value=0)
                st.bar_chart(by_cat, color=["#15803d", "#b91c1c"], alt="Tickets per category, split into resolved and escalated")
            with c2.container(border=True):
                st.markdown("**Why tickets escalate**")
                reasons = df.loc[df["status"] == "Escalated", "escalation_reason"].str.split(r" \+ ").explode().value_counts()
                if reasons.empty:
                    st.caption("No escalations yet.")
                else:
                    st.bar_chart(reasons, horizontal=True, alt="Most common escalation reasons")

            c3, c4 = st.columns(2)
            with c3.container(border=True):
                st.markdown("**Urgency mix**")
                st.bar_chart(df["urgency"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0), alt="Tickets by urgency")
            with c4.container(border=True):
                st.markdown("**Sentiment mix**")
                st.bar_chart(df["sentiment"].value_counts().reindex(["Calm", "Neutral", "Frustrated", "Angry"], fill_value=0),
                             alt="Tickets by customer sentiment")
            with st.container(border=True):
                st.markdown("**Daily volume**")
                st.area_chart(daily, alt="Tickets per day")

    # ---------- audit trail ----------
    with tab_audit:
        st.subheader("Audit trail", icon=":material/policy:")
        st.caption("Sign-ins, failed attempts, lockouts, ticket reviews and exports. The latest 500 events, newest first.")
        trail = audit.read()
        if trail.empty:
            st.info("No events recorded yet.", icon=":material/policy:")
        else:
            st.dataframe(trail, hide_index=True, width="stretch", height=420, alt="Security audit events, newest first",
                         column_config={"timestamp": "Time", "actor": "User", "role": "Role", "action": "Action",
                                        "target": "Target", "detail": "Detail"})

    # ---------- email inbox ----------
    with tab_mail:
        st.subheader("Email inbox", icon=":material/inbox:")
        backend = {"imap": f"IMAP · {s.imap_host} · {s.imap_folder}", "folder": f"Local folder · {s.inbox_dir.name}/ (demo)", "off": "Off"}.get(s.mail_backend, s.mail_backend)
        st.caption(f"Source: {backend}. Support requests run through the four agents; promotional, spam, business and general emails are labelled, get a polite auto-reply (spam: none) and no ticket; resolved ones are answered automatically, "
                   "escalated ones are acknowledged and wait in the review queue for your approval.")
        if s.mail_auto and s.mail_backend != "off" and s.has_key:
            seen = inbox.last_poll["at"]
            st.caption(f"Checked automatically every {s.mail_poll_seconds}s" + (f" · last check {datetime.fromtimestamp(seen):%H:%M:%S}" if seen else " · first check is running"))
        if st.button("Check mail now", icon=":material/sync:", type="primary", disabled=s.mail_backend == "off" or not s.has_key):
            with st.spinner("Reading the mailbox and running the agents…"):
                st.session_state["mail_stats"] = asyncio.run(inbox.poll_once(s))
            audit.record(user.username, user.role, "mail_poll", "", str(st.session_state["mail_stats"]))
            st.rerun()
        if stats := st.session_state.get("mail_stats"):
            if stats["error"]:
                st.error(stats["error"], icon=":material/error:")
            with st.container(horizontal=True):
                st.metric("Emails read", stats["fetched"], border=True)
                st.metric("Auto-resolved", stats["resolved"], border=True)
                st.metric("Escalated", stats["escalated"], border=True)
                st.metric("Not support", stats["offtopic"], border=True, help=", ".join(f"{k.title()}: {v}" for k, v in stats.get("labels", {}).items()) or "Promotional, spam, business and general emails")
                st.metric("Ignored", stats["skipped"], border=True)
        if s.mail_backend == "folder":
            with st.expander("Simulate an incoming email", icon=":material/mark_email_unread:"), st.form("sim_mail", border=False):
                f_from = st.text_input("From", value="customer@example.com")
                f_subj = st.text_input("Subject", value="Where is my order?")
                f_body = st.text_area("Body", value="Hi, my order ORD-4512 was supposed to arrive yesterday. Where is it?", height=120)
                if st.form_submit_button("Drop into inbox", icon=":material/send:"):
                    inbox.write_demo_email(s, f_from.strip(), f_subj.strip(), f_body)
                    st.toast("Email added. It is picked up automatically, or click Check mail now.", icon=":material/mail:")
        mail_tickets = tickets[tickets["channel"] == "email"] if len(tickets) else tickets
        st.markdown(f"**Tickets from email ({len(mail_tickets)})**")
        if mail_tickets.empty:
            st.caption("No emails processed yet.")
        else:
            st.dataframe(
                mail_tickets.iloc[::-1][["ticket_id", "timestamp", "sender_email", "email_subject", "status", "review_status", "escalation_reason"]],
                hide_index=True, width="stretch", alt="Tickets that arrived by email, newest first",
                column_config={"ticket_id": "Ticket", "timestamp": "Time", "sender_email": "From", "email_subject": "Subject",
                               "status": "Status", "review_status": "Review", "escalation_reason": "Escalation reason"},
            )
