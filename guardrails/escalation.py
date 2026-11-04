"""Deterministic escalation rule (Task 5).

Escalate when ANY trigger is true. This runs in plain Python inside the log tool, so the
decision can never be talked out of by the customer's text or by an LLM slip.
"""
from __future__ import annotations

import re

from models import TicketContext

LEGAL = re.compile(r"\b(?:lawyers?|attorneys?|solicitors?|sue|suing|lawsuits?|legal action|court|small claims|legal)\b", re.I)
FRAUD = re.compile(r"\b(?:fraud\w*|scam\w*|stolen|unauthori[sz]ed|identity theft|hacked|phishing)\b", re.I)
PAYMENT_DISPUTE = re.compile(
    r"\b(?:charge ?backs?|disput\w+|double[- ]charged|triple[- ]charged|overcharged|"
    r"(?:contact|call|report|tell)\w* (?:\w+ ){0,2}(?:my )?bank|"
    r"charged (?:\w+ ){0,3}(?:twice|2 times|two times|three times|3 times|multiple times|again and again))\b",
    re.I,
)


def decide(ctx: TicketContext, confidence_threshold: float = 0.6) -> tuple[str, list[str]]:
    """Return ("Resolved" | "Escalated", reasons)."""
    reasons: list[str] = []
    q = ctx.query

    if q is None:
        reasons.append("Incomplete pipeline: query analysis missing")
    else:
        if q.urgency == "High":
            reasons.append("High urgency")
        if q.sentiment == "Angry":
            reasons.append("Angry customer")

    if not ctx.kb_searched:
        reasons.append("Incomplete pipeline: knowledge search missing")
    elif not ctx.hits:
        reasons.append("No matching knowledge-base document")

    text = ctx.complaint
    if LEGAL.search(text):
        reasons.append("Legal threat")
    if FRAUD.search(text):
        reasons.append("Possible fraud")
    if PAYMENT_DISPUTE.search(text):
        reasons.append("Payment dispute")

    r = ctx.resolution
    if r is None:
        reasons.append("Incomplete pipeline: no resolution drafted")
    else:
        if not r.answered:
            reasons.append("Question not fully answered from the knowledge base")
        if r.confidence < confidence_threshold:
            reasons.append(f"Low resolution confidence ({r.confidence:.2f})")

    if ctx.grounding_issues:
        reasons.append("Reply contains ungrounded figures")
    if ctx.citation_issues:
        reasons.append("Reply has missing or invalid citations")
    if "prompt_injection" in ctx.security_flags:
        reasons.append("Possible prompt injection")
    if "media_unreadable" in ctx.security_flags:
        reasons.append("Attachment could not be read")
    if "guard_unsafe" in ctx.security_flags:
        reasons.append("Flagged unsafe by Llama Guard")
    if "guard_unavailable" in ctx.security_flags:
        reasons.append("Safety screen unavailable")
    if "pipeline_error" in ctx.errors:
        reasons.append("AI pipeline error")

    return ("Escalated" if reasons else "Resolved"), reasons


def format_reasons(reasons: list[str]) -> str:
    return " + ".join(reasons)
