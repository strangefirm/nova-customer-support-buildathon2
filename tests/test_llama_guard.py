"""Llama Guard screen: strict verdict parsing, fail-closed behaviour, and escalation through the pipeline."""
import pytest

from config import get_settings
from guardrails import llama_guard as lg
from team import run_ticket

from conftest import collect, scripted_client
from test_citations import ANALYSIS, QUERY


def test_parse_verdicts():
    assert lg.parse_verdict("safe").safe
    v = lg.parse_verdict("unsafe\nS10,S1")
    assert not v.safe and v.categories == ["S1", "S10"] and not v.error


@pytest.mark.parametrize("bad", ["", "maybe", "Sure! This is safe", "safe-ish"])
def test_anything_unclear_is_not_treated_as_safe(bad):
    v = lg.parse_verdict(bad)
    assert not v.safe and v.error


class _Fake:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc = reply, exc
        self.chat = type("C", (), {"completions": self})()

    async def create(self, **_):
        if self.exc:
            raise self.exc
        return type("R", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": self.reply})()})()]})()


async def test_classify_uses_the_client_and_fails_closed(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", "llama-guard3")
    s = get_settings()
    assert (await lg.classify("hi", s, _Fake("safe"))).safe
    assert not (await lg.classify("hi", s, _Fake("unsafe\nS11"))).safe
    down = await lg.classify("hi", s, _Fake(exc=TimeoutError()))
    assert not down.safe and down.error == "TimeoutError"


def test_guard_is_off_by_default():
    assert get_settings().guard_model == ""


async def _run(monkeypatch, verdict):
    monkeypatch.setenv("GUARD_MODEL", "llama-guard3")

    async def fake(text, s, client=None):
        return verdict

    monkeypatch.setattr("guardrails.llama_guard.classify", fake)
    client = scripted_client(ANALYSIS, QUERY, "Items can be returned within 30 days of delivery [1].", 0.9, ["refund_policy.pdf"], cited=[1])
    return (await collect(run_ticket("How many days do I have to return an item?", model_client=client)))[-1].data


async def test_safe_message_is_unaffected(monkeypatch):
    row = await _run(monkeypatch, lg.Verdict(True))
    assert row["status"] == "Resolved" and "guard" not in row["security_flags"]


async def test_unsafe_message_escalates_with_categories_logged(monkeypatch):
    row = await _run(monkeypatch, lg.Verdict(False, ["S10"]))
    assert row["status"] == "Escalated" and "Llama Guard" in row["escalation_reason"]
    assert "guard_unsafe" in row["security_flags"] and "guard_S10" in row["security_flags"]


async def test_unreachable_guard_fails_closed(monkeypatch):
    row = await _run(monkeypatch, lg.Verdict(False, error="TimeoutError"))
    assert row["status"] == "Escalated" and "Safety screen unavailable" in row["escalation_reason"]
