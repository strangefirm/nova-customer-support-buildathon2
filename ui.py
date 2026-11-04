"""Reusable Streamlit rendering for the support desk (native elements only; theme lives in config.toml)."""
from __future__ import annotations

import asyncio
import re

import streamlit as st

import auth
from team import StepEvent, run_ticket

STAGES = {
    "query": ("Customer query agent", ":material/psychology:"),
    "knowledge": ("Knowledge base agent", ":material/menu_book:"),
    "resolution": ("Resolution agent", ":material/edit_note:"),
    "log": ("Log agent", ":material/fact_check:"),
}
URGENCY_COLOR = {"Low": "green", "Medium": "orange", "High": "red"}
SENTIMENT_COLOR = {"Calm": "green", "Neutral": "gray", "Frustrated": "orange", "Angry": "red"}

# Everything that can alter Streamlit markdown: emphasis, links/images, headings, LaTeX, :emoji: and :badge[...]
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-!|~$:])")


def safe_md(text: object) -> str:
    """Render untrusted text (customer input, model output) as plain text inside st.markdown.

    Without this a complaint like `![x](https://attacker/pixel)` or `:red-badge[...]` would be
    interpreted as markdown and could load remote content or spoof UI.
    """
    return _MD_SPECIAL.sub(r"\\\1", str(text)).replace("\n", "  \n")


def error_text(ev: StepEvent) -> str:
    """Staff-facing error message: the friendly text plus the (scrubbed) cause, as plain text."""
    detail = (ev.data or {}).get("detail")
    return f"{ev.text}  \n{safe_md(detail)}" if detail else ev.text


def status_badge(status: str) -> None:
    if status == "Resolved":
        st.badge("Resolved", icon=":material/check_circle:", color="green")
    else:
        st.badge("Escalated", icon=":material/priority_high:", color="red")


def sources_line(sources: list[str]) -> str:
    return " ".join(f":blue-badge[:material/description: {s}]" for s in sources if s) or "_No source documents_"


def citation_badges(citations: str) -> str:
    """'[1] refund_policy.pdf \u203a Return window; [2] ...' as badges. Brackets are dropped because they would break the badge syntax."""
    out = []
    for part in (citations or "").split("; "):
        m = re.match(r"^\[(\d+)\]\s*(.+)$", part.strip())
        if m:
            out.append(f":blue-badge[:material/format_quote: {m.group(1)} \u00b7 {m.group(2).replace('[', '').replace(']', '')}]")
    return " ".join(out)


# ---------- agent steps ----------
def _step_label(ev: StepEvent, running: bool = False) -> str:
    name, icon = STAGES[ev.stage]
    suffix = " · working…" if running else f" · {ev.elapsed:.1f}s"
    return f"{icon} {name}{suffix}"


def step_details(ev: StepEvent) -> None:
    d = ev.data
    if ev.stage == "query" and d:
        c = st.columns(4)
        c[0].metric("Category", d["category"])
        c[1].metric("Urgency", d["urgency"])
        c[2].metric("Sentiment", d["sentiment"])
        c[3].metric("Language", d["language"])
        st.caption(safe_md(d["summary"]))
    elif ev.stage == "knowledge":
        if not d.get("hits"):
            st.warning("No matching company document found.", icon=":material/search_off:")
        for h in d.get("hits", []):
            st.markdown(
                f":blue-badge[:material/description: {h['source']}] **{safe_md(h['heading'])}** · relevance {h['score']:.2f}"
            )
            st.caption(safe_md(h["passage"]))
    elif ev.stage == "resolution" and d:
        st.progress(float(d["confidence"]), text=f"Confidence {d['confidence']:.0%}")
        st.markdown(citation_badges("; ".join(f"[{c['n']}] {c['source']} \u203a {c['section']}" for c in d.get("citations", []))) or sources_line(d["used_sources"]))
        if d.get("citation_issues"):
            st.warning(f"Citation problems: {', '.join(d['citation_issues'])}", icon=":material/format_quote:")
        if d.get("grounding_issues"):
            st.warning(
                f"Figures not found in the knowledge base: {', '.join(d['grounding_issues'])}",
                icon=":material/rule:",
            )
    else:
        st.caption(safe_md(ev.text))


def _step(ev: StepEvent, box=None) -> None:
    box = box or st.status(_step_label(ev), type="step", state="complete")
    box.update(label=_step_label(ev), state="complete")
    with box:
        step_details(ev)


# ---------- final result ----------
def render_result(row: dict, elapsed: float = 0.0) -> None:
    escalated = row["status"] == "Escalated"
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(f"Ticket {row['ticket_id']}")
            status_badge(row["status"])
            if row.get("language") and row["language"] not in ("English", "Unknown"):
                st.badge(row["language"], icon=":material/translate:", color="violet")
        c = st.columns(4)
        c[0].metric("Category", row["category"])
        c[1].metric("Urgency", row["urgency"])
        c[2].metric("Sentiment", row["sentiment"])
        c[3].metric("Confidence", f"{float(row['confidence']):.0%}" if row["confidence"] else "—")

        if escalated:
            reasons = [r for r in row["escalation_reason"].split(" + ") if r]
            st.markdown("**Escalated to human support** because: " + " ".join(f":orange-badge[{r}]" for r in reasons))
        if row.get("security_flags"):
            st.markdown("**Security flags:** " + " ".join(f":red-badge[{f}]" for f in row["security_flags"].split("; ") if f))

        st.markdown("**Draft reply, pending human review**" if escalated else "**Reply to customer**")
        with st.chat_message("assistant", avatar=":material/support_agent:"):
            st.markdown(safe_md(row["response"] or "No reply was drafted."))
        st.markdown(("**Cited:** " + citation_badges(row.get("citations", ""))) if row.get("citations") else sources_line(row["kb_sources"].split("; ")))
        if escalated:
            st.caption("This AI draft waits in the Review queue and should not be sent as-is.")
        if elapsed:
            st.caption(f"Processed in {elapsed:.1f}s")


# ---------- run orchestration ----------
async def _stream(complaint: str, name: str, order: str, owner: str = "", **extra) -> list[StepEvent]:
    events: list[StepEvent] = []
    boxes: dict = {}
    with st.container(border=True):
        st.subheader("Agent activity", icon=":material/account_tree:")
        async for ev in run_ticket(complaint, name, order, owner=owner, **extra):
            events.append(ev)
            if ev.kind == "start" and ev.stage in STAGES:
                boxes[ev.stage] = st.status(_step_label(ev, running=True), type="step", state="running")
            elif ev.kind == "done" and ev.stage in STAGES:
                _step(ev, boxes.get(ev.stage))
            elif ev.kind == "error":
                st.error(error_text(ev), icon=":material/error:")
    return events


def stream_run(complaint: str, name: str, order: str, owner: str = "", **extra) -> list[StepEvent]:
    return asyncio.run(_stream(complaint, name, order, owner, **extra))


def render_run(events: list[StepEvent]) -> None:
    """Replay a finished run from stored events (survives reruns and tab switches)."""
    with st.container(border=True):
        st.subheader("Agent activity", icon=":material/account_tree:")
        for ev in events:
            if ev.kind == "done" and ev.stage in STAGES:
                _step(ev)
            elif ev.kind == "error":
                st.error(error_text(ev), icon=":material/error:")
    final = next((e for e in events if e.kind == "final"), None)
    if final:
        render_result(final.data, final.elapsed)


# ---------- customer-facing views (no internal reasons, flags, confidence, agent steps or unapproved drafts) ----------
CUSTOMER_PROGRESS = {
    "query": "Understanding your request…",
    "knowledge": "Checking our policies…",
    "resolution": "Preparing your reply…",
    "log": "Recording your ticket…",
}


async def _stream_customer(complaint: str, order: str, owner: str, name: str, **extra) -> list[StepEvent]:
    events: list[StepEvent] = []
    with st.status("Working on your request…", expanded=False) as box:
        async for ev in run_ticket(complaint, name, order, owner=owner, **extra):
            events.append(ev)
            if ev.kind == "start" and ev.stage in CUSTOMER_PROGRESS:
                box.update(label=CUSTOMER_PROGRESS[ev.stage], state="running")
        box.update(label="Done", state="complete")
    return events


def stream_run_customer(complaint: str, order: str, owner: str, name: str, **extra) -> list[StepEvent]:
    return asyncio.run(_stream_customer(complaint, order, owner, name, **extra))


def customer_status_badge(label: str) -> None:
    if label == "With our support team":
        st.badge(label, icon=":material/support_agent:", color="orange")
    else:
        st.badge(label, icon=":material/check_circle:", color="green")


def render_customer_ticket(row: dict, heading: str | None = None) -> None:
    """Render one ticket from auth.customer_view() only."""
    v = auth.customer_view(row)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(heading or f"Request {v['ticket_id']}")
            customer_status_badge(v["status_label"])
            st.caption(v["timestamp"])
        st.caption("Your message")
        st.markdown(safe_md(v["complaint"]))
        if v["reply"]:
            with st.chat_message("assistant", avatar=":material/support_agent:"):
                st.markdown(safe_md(v["reply"]))
            if v.get("citations"):
                st.markdown("**Based on:** " + citation_badges(v["citations"]))
        else:
            st.info("A support specialist is reviewing your request and will reply here.", icon=":material/hourglass_top:")


def render_customer_run(events: list[StepEvent]) -> None:
    for ev in events:
        if ev.kind == "error":
            st.error(ev.text, icon=":material/error:")
    final = next((e for e in events if e.kind == "final"), None)
    if final:
        render_customer_ticket(final.data, heading=f"Your request {final.data['ticket_id']}")
