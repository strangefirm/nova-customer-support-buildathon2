"""Typed data contracts shared by agents, guardrails, logging and the UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["Shipping", "Refund", "Account", "Billing", "Product", "Other"]
Urgency = Literal["Low", "Medium", "High"]
Sentiment = Literal["Calm", "Neutral", "Frustrated", "Angry"]

URGENCY_RANK = {"Low": 1, "Medium": 2, "High": 3}


class QueryResult(BaseModel):
    category: Category
    urgency: Urgency
    sentiment: Sentiment
    language: str = Field(default="English", max_length=40)
    summary: str = Field(max_length=300)


class KBHit(BaseModel):
    source: str
    heading: str
    passage: str
    score: float


class Citation(BaseModel):
    """A numbered marker in the reply ([n]) resolved to the passage it points to."""

    n: int
    source: str
    section: str

    def label(self) -> str:
        return f"[{self.n}] {self.source} \u203a {self.section}"


class ResolutionResult(BaseModel):
    reply: str
    confidence: float = Field(ge=0.0, le=1.0)
    used_sources: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answered: bool = True  # the model's yes/no decision: do the passages fully answer the question?


@dataclass
class TicketContext:
    """Per-run shared state. Tools write into it; the Log tool reads from it.

    This is the single source of truth for the ticket row, so the Log Agent never has to
    re-type (and possibly hallucinate) facts produced by the other agents.
    """

    complaint: str
    customer_name: str = ""
    owner: str = ""  # authenticated username; set server-side, never from the form
    order_id: str = ""
    security_flags: list[str] = field(default_factory=list)
    query: QueryResult | None = None
    kb_searched: bool = False
    hits: list[KBHit] = field(default_factory=list)
    resolution: ResolutionResult | None = None
    grounding_issues: list[str] = field(default_factory=list)
    citation_issues: list[str] = field(default_factory=list)
    row: dict | None = None
    errors: list[str] = field(default_factory=list)
    # Set only for tickets that arrive by email (see inbox.py)
    channel: str = "web"
    sender_email: str = ""
    email_subject: str = ""
    message_id: str = ""
    caller_phone: str = ""
    # Per-run random suffix on the termination word, so customer text can never end the run early.
    nonce: str = ""

    @property
    def done_marker(self) -> str:
        return f"LOGGED-{self.nonce}" if self.nonce else "LOGGED"
