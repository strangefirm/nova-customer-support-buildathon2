"""Log Agent: records the ticket and applies the deterministic escalation rule."""
from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent

import logstore
from config import Settings, get_settings
from guardrails.escalation import decide, format_reasons
from guardrails.input_guard import mask_pii
from models import TicketContext

NAME = "LogAgent"
DESCRIPTION = "Records the ticket to the log file and marks it Escalated for human support when escalation rules fire."
DONE_MARKER = "LOGGED"

SYSTEM_MESSAGE = """You are the Log Agent of an online store's support desk.
Your ONLY job: call log_ticket exactly once (it takes no arguments; it reads the finished ticket itself and decides whether it must be escalated to a human). Then stop.
The tool output is your entire message. Never edit or add to it."""


def build_row(ctx: TicketContext, settings: Settings) -> dict:
    """Assemble the log row from the shared context (never from LLM-typed text)."""
    status, reasons = decide(ctx, settings.confidence_threshold)
    q, r = ctx.query, ctx.resolution
    response = r.reply if r else ""
    if settings.redact_pii:
        response = mask_pii(response)
    sources = r.used_sources if (r and r.used_sources) else sorted({h.source for h in ctx.hits})
    return {
        "owner": ctx.owner,
        "customer_name": ctx.customer_name,
        "order_id": ctx.order_id,
        "complaint": ctx.complaint,
        "category": q.category if q else "Unknown",
        "urgency": q.urgency if q else "Unknown",
        "sentiment": q.sentiment if q else "Unknown",
        "language": q.language if q else "Unknown",
        "kb_sources": "; ".join(sources),
        "citations": "; ".join(c.label() for c in r.citations) if r else "",
        "response": response,
        "confidence": f"{r.confidence:.2f}" if r else "",
        "status": status,
        "escalation_reason": format_reasons(reasons),
        "security_flags": "; ".join(ctx.security_flags),
        "review_status": "Pending" if status == "Escalated" else "N/A",
        "channel": ctx.channel,
        "sender_email": ctx.sender_email,
        "email_subject": ctx.email_subject,
        "message_id": ctx.message_id,
        "caller_phone": ctx.caller_phone,
    }


def write_ticket(ctx: TicketContext, settings: Settings) -> dict:
    ctx.row = logstore.append_ticket(build_row(ctx, settings), settings.db_path)
    return ctx.row


def build(model_client, ctx: TicketContext, settings: Settings | None = None) -> AssistantAgent:
    s = settings or get_settings()

    def log_ticket() -> str:
        """Save the ticket to the log file and decide Resolved vs Escalated."""
        if ctx.row is None:
            try:
                write_ticket(ctx, s)
            except Exception:  # noqa: BLE001 - surfaced to the pipeline's fail-closed fallback
                ctx.errors.append("log_write_failed")
                return "LOG FAILED: the ticket could not be saved."
        row = ctx.row
        reason = row["escalation_reason"] or "no escalation triggers"
        return f"TICKET {row['ticket_id']} | Status: {row['status']} | Reason: {reason} | {ctx.done_marker}"

    return AssistantAgent(
        NAME,
        model_client=model_client,
        tools=[log_ticket],
        description=DESCRIPTION,
        system_message=SYSTEM_MESSAGE,
        reflect_on_tool_use=False,
    )
