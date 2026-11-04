"""Regression: 'RuntimeError: Event loop is closed' on the second and later tickets in the Streamlit app.

The app runs every ticket in its own asyncio.run() loop. A model client cached across tickets keeps an HTTP
pool bound to the first (closed) loop, so later tickets failed and were escalated as pipeline errors.
"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from autogen_core.models import UserMessage

import config
import team
from team import run_ticket

from conftest import scripted_client

BODY = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_llm(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    yield
    srv.shutdown()
    srv.server_close()


def test_real_client_works_across_many_event_loops(stub_llm):
    """Exactly what the app does: a new client and a fresh asyncio.run() for every ticket."""
    async def one_ticket():
        client = config.build_model_client()
        try:
            return await client.create([UserMessage(content="hello", source="user")])
        finally:
            await client.close()  # closed inside the loop that used it, as team.run_ticket does

    for i in range(6):
        assert asyncio.run(one_ticket()).content == "hi", f"ticket {i + 1} failed"


def test_each_call_builds_a_new_client(stub_llm):
    a, b = config.build_model_client(), config.build_model_client()
    assert a is not b and a._inner is not b._inner  # never shared across event loops


def _tracked(client):
    closed = []
    original = client.close

    async def close():
        closed.append(True)
        await original()

    client.close = close
    return client, closed


ANALYSIS = dict(category="Shipping", urgency="Low", sentiment="Neutral", language="English", summary="Late.")


@pytest.mark.asyncio
async def test_pipeline_closes_the_client_it_created(monkeypatch):
    client, closed = _tracked(scripted_client(ANALYSIS, "late order", "Please check Track Order [1]. NovaCart Support", 0.9, []))
    monkeypatch.setattr(team, "build_model_client", lambda s=None: client)
    events = [e async for e in run_ticket("Where is my order?")]
    assert events[-1].kind == "final" and closed == [True]


@pytest.mark.asyncio
async def test_pipeline_does_not_close_a_client_the_caller_owns():
    client, closed = _tracked(scripted_client(ANALYSIS, "late order", "Please check Track Order [1]. NovaCart Support", 0.9, []))
    _ = [e async for e in run_ticket("Where is my order?", model_client=client)]
    assert closed == []


@pytest.mark.asyncio
async def test_client_is_closed_even_when_the_pipeline_fails(monkeypatch):
    class Boom:
        closed = False

        async def close(self):
            Boom.closed = True

    monkeypatch.setattr(team, "build_model_client", lambda s=None: Boom())
    events = [e async for e in run_ticket("Where is my order?")]
    assert events[-1].data["status"] == "Escalated" and Boom.closed


def test_consecutive_tickets_in_separate_loops_all_complete(monkeypatch):
    """The app's pattern end to end: ticket after ticket, each in its own asyncio.run()."""
    monkeypatch.setattr(team, "build_model_client", lambda s=None: scripted_client(ANALYSIS, "late order", "Please check Track Order [1]. NovaCart Support", 0.9, []))

    async def one():
        return [e async for e in run_ticket("Where is my order?")]

    for _ in range(4):
        events = asyncio.run(one())
        assert events[-1].kind == "final" and events[-1].data["status"] == "Resolved"
