"""Forced tool use, plus the failure modes only a real model produces."""
import json

import pytest
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

import audit
import auth
import config
from llm_client import RequiredToolClient
from team import run_ticket

from conftest import MODEL_INFO, collect, scripted_client


class Recorder:
    """Stands in for a real model client and records how it was called."""

    model_info = {"function_calling": True}
    capabilities = {"function_calling": True}

    def __init__(self):
        self.calls = []

    async def create(self, messages, **kw):
        self.calls.append(kw)
        return "result"

    def create_stream(self, messages, **kw):
        self.calls.append(kw)
        return "stream"

    def actual_usage(self):
        return "usage"

    total_usage = actual_usage

    def count_tokens(self, messages, *, tools=()):
        return 7

    remaining_tokens = count_tokens

    async def close(self):
        self.calls.append("closed")


@pytest.mark.asyncio
async def test_forces_tool_use_when_tools_are_offered():
    inner = Recorder()
    out = await RequiredToolClient(inner).create(["m"], tools=[object()])
    assert out == "result" and inner.calls[0]["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_leaves_other_choices_and_toolless_calls_alone():
    inner, client = Recorder(), RequiredToolClient(Recorder())
    client._inner = inner
    await client.create(["m"], tools=[])  # no tools: nothing to force
    await client.create(["m"], tools=[object()], tool_choice="none")  # an explicit choice is respected
    assert [c["tool_choice"] for c in inner.calls] == ["auto", "none"]


def test_stream_and_delegation():
    inner = Recorder()
    client = RequiredToolClient(inner)
    assert client.create_stream(["m"], tools=[object()]) == "stream" and inner.calls[0]["tool_choice"] == "required"
    assert client.model_info == inner.model_info and client.count_tokens([]) == 7 and client.total_usage() == "usage"


def test_factory_wraps_by_default_and_not_when_auto(monkeypatch):
    monkeypatch.setattr(config, "_client", lambda *a: Recorder())
    assert isinstance(config.build_model_client(), RequiredToolClient)
    monkeypatch.setenv("TOOL_CHOICE", "auto")
    assert isinstance(config.build_model_client(), Recorder)


# ---------- failure modes a real model produces ----------
def _text(msg):
    return CreateResult(finish_reason="stop", content=msg, usage=RequestUsage(prompt_tokens=0, completion_tokens=0), cached=False)


def _call(name, **args):
    return CreateResult(finish_reason="function_calls", content=[FunctionCall(id="c", name=name, arguments=json.dumps(args))],
                        usage=RequestUsage(prompt_tokens=0, completion_tokens=0), cached=False)


@pytest.mark.asyncio
async def test_agent_answering_in_plain_text_instead_of_calling_its_tool_fails_closed():
    """The bug seen with a real model: the Resolution agent chatted instead of calling submit_resolution."""
    client = ReplayChatCompletionClient([
        _call("submit_analysis", category="Account", urgency="Low", sentiment="Calm", language="English", summary="Forgot password."),
        _call("search_kb", query="password reset"),
        _text("Sure! Go to the login page and click Forgot password."),  # plain text, no tool call
        _call("log_ticket"),
    ], model_info=MODEL_INFO)
    row = (await collect(run_ticket("I forgot my password", model_client=client)))[-1].data
    assert row["status"] == "Escalated" and "no resolution drafted" in row["escalation_reason"]  # safe, never silently "Resolved"


@pytest.mark.asyncio
async def test_pipeline_error_detail_is_surfaced_to_staff_and_audited():
    class Boom:
        pass

    events = await collect(run_ticket("Where is my order?", model_client=Boom(), owner="suhaila"))
    err = next(e for e in events if e.kind == "error")
    assert "escalated to a human" in err.text and err.data["detail"]  # staff can see the cause
    row = events[-1].data
    trail = audit.read()
    hit = trail[trail["action"] == "pipeline_error"].iloc[0]
    assert hit["actor"] == "suhaila" and hit["target"] == row["ticket_id"] and hit["detail"]


@pytest.mark.asyncio
async def test_error_detail_never_contains_the_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecretkeyvalue1234567890")

    class Leaky:
        def __getattr__(self, name):
            raise RuntimeError("auth failed for key sk-supersecretkeyvalue1234567890")

    events = await collect(run_ticket("Where is my order?", model_client=Leaky()))
    err = next(e for e in events if e.kind == "error")
    assert "supersecretkey" not in err.data["detail"] and "supersecretkey" not in audit.read().to_string()


# ---------- small UX fixes ----------
def test_username_error_names_the_offending_characters():
    with pytest.raises(ValueError, match=r"Not allowed: @"):
        auth.normalise_username("krish@123")
    with pytest.raises(ValueError, match="Not allowed"):
        auth.normalise_username("has space")
    assert auth.normalise_username("krish.123") == "krish.123"
