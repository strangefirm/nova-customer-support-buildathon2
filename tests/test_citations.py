"""Inline citations: every [n] must point at a passage that was really retrieved, and is stored with the ticket."""
import pytest

import auth
import db
import logstore
from guardrails.citations import markers, resolve, strip_markers
from guardrails.output_guard import ungrounded_figures
from models import Citation, KBHit
from team import run_ticket

from conftest import collect, scripted_client

HITS = [
    KBHit(source="refund_policy.pdf", heading="Return window", passage="Items can be returned within 30 days.", score=0.9),
    KBHit(source="shipping_policy.pdf", heading="Late or delayed orders", passage="Wait 3 business days.", score=0.5),
]
ANALYSIS = dict(category="Refund", urgency="Low", sentiment="Neutral", language="English", summary="Return window question.")
QUERY = "return refund window days"


# ---------- the validator ----------
def test_markers_and_stripping():
    assert markers("Within 30 days [1]. See also [2][10].") == [1, 2, 10]
    assert strip_markers("Within 30 days [1]. Done.") == "Within 30 days . Done."
    assert markers("array[abc] and (1) and [123] and [] and [ 1 ]") == []  # only [1]-[99] counts


def test_valid_citations_resolve_to_document_and_section():
    cites, issues = resolve("Returns within 30 days [1]. Late orders: wait [2]. Again [1].", HITS)
    assert issues == [] and [c.n for c in cites] == [1, 2]  # unique, in order of first use
    assert cites[0] == Citation(n=1, source="refund_policy.pdf", section="Return window")
    assert cites[0].label() == "[1] refund_policy.pdf › Return window"


@pytest.mark.parametrize("reply,expected", [
    ("Claim [3].", ["invalid citation [3]"]),
    ("Claim [0].", ["invalid citation [0]"]),
    ("Claim [99] and [1].", ["invalid citation [99]"]),
])
def test_markers_that_do_not_match_a_retrieved_passage_are_invalid(reply, expected):
    _, issues = resolve(reply, HITS)
    assert issues == expected


def test_a_confident_answer_without_any_citation_is_flagged():
    assert resolve("Returns take 30 days.", HITS)[1] == ["no citations"]


def test_citations_not_required_when_disabled_or_nothing_was_retrieved():
    assert resolve("Returns take 30 days.", HITS, required=False) == ([], [])
    assert resolve("I do not have that information.", [], required=True) == ([], [])
    assert resolve("Claim [1].", [], required=True)[1] == ["invalid citation [1]"]  # but you can never cite what was not retrieved


def test_marker_digits_are_not_mistaken_for_factual_figures():
    reply = "Items can be returned within 30 days [1]."
    assert ungrounded_figures(strip_markers(reply), [HITS[0].passage]) == []
    assert ungrounded_figures(strip_markers("Within 45 days [1]."), [HITS[0].passage]) == ["45"]  # real invented figures still caught


# ---------- through the whole pipeline ----------
async def _run(reply, confidence=0.9, query=QUERY):
    client = scripted_client(ANALYSIS, query, reply, confidence, ["refund_policy.pdf"])
    return (await collect(run_ticket("How many days do I have to return an item?", model_client=client)))[-1].data


async def test_cited_answer_is_resolved_and_citations_are_stored_with_the_ticket():
    row = await _run("Items can be returned within 30 days of delivery [1].")
    assert row["status"] == "Resolved" and row["escalation_reason"] == ""
    assert row["citations"].startswith("[1] refund_policy.pdf › ")
    assert logstore.read_tickets().iloc[0]["citations"] == row["citations"]  # persisted, not only shown


async def test_invalid_citation_forces_escalation():
    row = await _run("Items can be returned within 30 days of delivery [9].")
    assert row["status"] == "Escalated" and "missing or invalid citations" in row["escalation_reason"]
    assert row["citations"] == ""  # nothing is claimed as a source that was not retrieved


async def test_uncited_confident_answer_forces_escalation():
    row = await _run("Items can be returned within 30 days of delivery.")
    assert row["status"] == "Escalated" and "missing or invalid citations" in row["escalation_reason"]


async def test_low_confidence_honest_reply_needs_no_citation_for_the_citation_rule():
    row = await _run("Sorry, I do not have information on that; a specialist will follow up.", confidence=0.2)
    assert "citations" not in row["escalation_reason"] and "Low resolution confidence" in row["escalation_reason"]


async def test_requirement_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("REQUIRE_CITATIONS", "false")
    row = await _run("Items can be returned within 30 days of delivery.")
    assert row["status"] == "Resolved"


async def test_invented_figure_is_still_caught_when_the_sentence_is_cited():
    row = await _run("Items can be returned within 90 days of delivery [1].")
    assert row["status"] == "Escalated" and "ungrounded" in row["escalation_reason"]


async def test_multiple_citations_are_all_recorded():
    row = await _run("Return within 30 days [1]. If your parcel is late, wait three business days first [2].", query="return refund late delayed orders days")
    assert row["citations"].count("›") >= 1 and "[1]" in row["citations"]


# ---------- what each audience sees ----------
def test_customers_see_citations_only_for_answers_they_are_allowed_to_see():
    logstore.append_ticket({"owner": "c", "status": "Resolved", "response": "30 days [1].", "citations": "[1] refund_policy.pdf › Return window"})
    assert auth.customer_view(logstore.read_tickets().iloc[0])["citations"] == "[1] refund_policy.pdf › Return window"
    e = logstore.append_ticket({"owner": "c", "status": "Escalated", "response": "draft [1]", "citations": "[1] refund_policy.pdf › Return window"})
    v = auth.customer_view(logstore.read_tickets().iloc[1])
    assert v["reply"] == "" and v["citations"] == ""
    logstore.update_review(e["ticket_id"], "Approved", "Final human answer", "staff")
    assert auth.customer_view(logstore.read_tickets().iloc[1])["citations"] == ""  # a human-approved answer carries no AI citations


def test_citation_badges_render_and_cannot_break_the_markup():
    from ui import citation_badges

    out = citation_badges("[1] refund_policy.pdf › Return window; [2] shipping_policy.pdf › Late [or] delayed")
    assert out.count(":blue-badge[") == 2 and "[or]" not in out and "1 · refund_policy.pdf" in out
    assert citation_badges("") == "" and citation_badges("garbage without markers") == ""


def test_citations_column_round_trips_and_defaults_empty():
    logstore.append_ticket({"complaint": "x", "status": "Resolved", "citations": "[1] a.pdf › b"})
    logstore.append_ticket({"complaint": "y", "status": "Resolved"})
    df = logstore.read_tickets()
    assert list(df["citations"]) == ["[1] a.pdf › b", ""]
    assert db.connect().execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


# ---------- structured citations (the cited_passages tool argument) ----------
def test_list_numbering_is_not_mistaken_for_a_factual_figure():
    p = ["Items can be returned within 30 days. Shoes must be unworn."]
    assert ungrounded_figures("Steps:\n1. Return the shoes\n2) Place a new order\nStep 3: wait.\n  4: done. Within 30 days.", p) == []
    assert ungrounded_figures("Steps:\n1. Return the shoes\nYou have 45 days.", p) == ["45"]  # a real invented figure is still caught
    assert ungrounded_figures("Pay $4.99 within 3 days", p) == ["3", "4.99"]


async def _run_declared(reply, declared, confidence=0.9):
    client = scripted_client(ANALYSIS, QUERY, reply, confidence, ["refund_policy.pdf"], cited=declared)
    return (await collect(run_ticket("How many days do I have to return an item?", model_client=client)))[-1].data


async def test_declared_citations_satisfy_the_requirement_without_inline_markers():
    row = await _run_declared("Items can be returned within 30 days of delivery.", [1])
    assert row["status"] == "Resolved" and row["citations"].startswith("[1] refund_policy.pdf \u203a ")


async def test_declared_citation_pointing_at_nothing_escalates():
    row = await _run_declared("Items can be returned within 30 days of delivery.", [9])
    assert row["status"] == "Escalated" and "missing or invalid citations" in row["escalation_reason"]


async def test_empty_declaration_and_no_markers_escalates_when_confident():
    row = await _run_declared("Items can be returned within 30 days of delivery.", [])
    assert row["status"] == "Escalated" and "missing or invalid citations" in row["escalation_reason"]


async def test_declared_and_inline_citations_are_merged_without_duplicates():
    row = await _run_declared("Return within 30 days [1].", [1, 1])
    assert row["status"] == "Resolved" and row["citations"].count("[1]") == 1


def test_cited_passages_is_a_required_tool_argument():
    """Models follow required tool arguments far more reliably than prose instructions."""
    import asyncio

    from autogen_ext.models.replay import ReplayChatCompletionClient

    from agents import resolution_agent
    from models import TicketContext

    from conftest import MODEL_INFO

    agent = resolution_agent.build(ReplayChatCompletionClient([], model_info=MODEL_INFO), TicketContext(complaint="x"))
    tool = asyncio.run(agent._workbench[0].list_tools())[0]  # exactly what the model is shown
    schema = tool["parameters"]
    assert tool["name"] == "submit_resolution" and "cited_passages" in schema["required"]
    assert schema["properties"]["cited_passages"]["items"] == {"type": "integer"}


# ---------- the answered decision ----------
async def _run_answered(reply, confidence, answered, cited=(1,)):
    client = scripted_client(ANALYSIS, QUERY, reply, confidence, ["refund_policy.pdf"], cited=list(cited), answered=answered)
    return (await collect(run_ticket("How many days do I have to return an item?", model_client=client)))[-1].data


async def test_an_answer_that_claims_to_answer_cannot_carry_a_contradicting_low_score():
    """Seen live: a correct, cited, grounded reply reported confidence 0.20 and was needlessly escalated."""
    row = await _run_answered("Items can be returned within 30 days of delivery.", confidence=0.2, answered=True)
    assert row["status"] == "Resolved" and float(row["confidence"]) >= 0.6


async def test_not_answered_escalates_with_an_explicit_reason_even_with_a_high_self_score():
    row = await _run_answered("Items can be returned within 30 days of delivery.", confidence=0.95, answered=False)
    assert row["status"] == "Escalated"
    assert "Question not fully answered from the knowledge base" in row["escalation_reason"]
    assert float(row["confidence"]) < 0.6


async def test_answered_true_does_not_override_the_grounding_and_citation_guards():
    invented = await _run_answered("Items can be returned within 90 days of delivery.", confidence=0.9, answered=True)
    assert invented["status"] == "Escalated" and "ungrounded" in invented["escalation_reason"]
    uncited = await _run_answered("Items can be returned within 30 days of delivery.", confidence=0.9, answered=True, cited=())
    assert uncited["status"] == "Escalated" and "citations" in uncited["escalation_reason"]


def test_answered_is_a_required_tool_argument():
    import asyncio

    from autogen_ext.models.replay import ReplayChatCompletionClient

    from agents import resolution_agent
    from models import TicketContext

    from conftest import MODEL_INFO

    agent = resolution_agent.build(ReplayChatCompletionClient([], model_info=MODEL_INFO), TicketContext(complaint="x"))
    schema = asyncio.run(agent._workbench[0].list_tools())[0]["parameters"]
    assert "answered" in schema["required"] and schema["properties"]["answered"]["type"] == "boolean"
