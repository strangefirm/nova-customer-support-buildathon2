"""Knowledge Base Agent: retrieves relevant passages (with source file names) via search_kb."""
from __future__ import annotations

from typing import Annotated

from autogen_agentchat.agents import AssistantAgent

from config import Settings, get_settings
from kb.store import get_kb
from models import TicketContext

NAME = "KnowledgeBaseAgent"
DESCRIPTION = "Searches the company documents (policies, FAQs) and returns only the relevant passages with their source file."

SYSTEM_MESSAGE = """You are the Knowledge Base Agent of an online store's support desk.
Your ONLY job: call search_kb exactly once with a short keyword-rich query IN ENGLISH describing the customer's problem (use the category and summary from the previous message; translate if the customer wrote in another language, because the documents are in English), then stop.
If the complaint contains two separate problems (for example a refund AND a late delivery), put the second one in second_query; the tool searches both and joins the passages. Still one call only.
Do not answer the customer. Do not add facts of your own. The tool output is your entire message."""


def build(model_client, ctx: TicketContext, settings: Settings | None = None) -> AssistantAgent:
    s = settings or get_settings()

    def search_kb(
        query: Annotated[str, "Short keyword query describing the customer's problem"],
        second_query: Annotated[str, "Only if the complaint has a second, separate problem (e.g. a refund AND a late delivery): its own short query. Otherwise empty."] = "",
    ) -> str:
        """Search the company knowledge base. Returns matching passages with the source file name."""
        extra = f" {ctx.query.category} {ctx.query.summary}" if ctx.query else ""
        ctx.kb_searched = True
        kb = get_kb(s)
        if second_query.strip():
            # each problem is searched on its own, then the results are joined (best score per passage, at most 5)
            merged = {}
            for q in (f"{query}{extra}", second_query.strip()):
                for h in kb.search(q, k=3):
                    key = (h.source, h.heading)
                    if key not in merged or h.score > merged[key].score:
                        merged[key] = h
            ctx.hits = sorted(merged.values(), key=lambda h: -h.score)[:5]
        else:
            ctx.hits = kb.search(f"{query}{extra}", k=3)
        if not ctx.hits:
            return "KNOWLEDGE SEARCH | 0 matching documents. NO RELEVANT COMPANY DOCUMENT WAS FOUND."
        lines = [f"KNOWLEDGE SEARCH | {len(ctx.hits)} passage(s) found"]
        for i, h in enumerate(ctx.hits, 1):
            lines.append(f"\n[{i}] SOURCE: {h.source} | SECTION: {h.heading} | RELEVANCE: {h.score:.2f}\n{h.passage}")
        return "\n".join(lines)

    return AssistantAgent(
        NAME,
        model_client=model_client,
        tools=[search_kb],
        description=DESCRIPTION,
        system_message=SYSTEM_MESSAGE,
        reflect_on_tool_use=False,
    )
