import json
from pathlib import Path

import pytest
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

MODEL_INFO = {"vision": False, "function_calling": True, "json_output": True, "family": "unknown", "structured_output": False}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own SQLite database and runs offline (no embeddings, fake key)."""
    import db

    monkeypatch.setenv("DB_PATH", str(tmp_path / "support.db"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SESSION_TIMEOUT_MIN", "30")
    monkeypatch.setenv("EMBEDDINGS", "off")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("TEAM_MODE", "roundrobin")
    for var in ("SMTP_HOST", "IMAP_HOST", "IMAP_USER", "SMTP_USER", "DEEPSEEK_API_KEY", "FAST_API_KEY", "GROQ_API_KEY", "STT_MODEL", "STT_BASE_URL", "STT_API_KEY"):  # never touch a real mailbox or the real fast model from a test
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("EMAIL_BACKEND", "outbox")
    monkeypatch.setenv("MAIL_BACKEND", "folder")
    monkeypatch.setenv("MAIL_AUTO", "false")  # no background poller in tests
    monkeypatch.setenv("MAIL_TRIAGE", "false")  # tests that exercise triage pass their own scripted client
    monkeypatch.setenv("OUTBOX_DIR", str(tmp_path / "outbox"))
    monkeypatch.setenv("INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.setenv("KB_DIR", str(Path(__file__).resolve().parents[1] / "knowledge_base_pdf"))  # never the dev's .env

    def no_network(*_a, **_k):
        raise RuntimeError("tests must pass a scripted model_client; real LLM calls are blocked")

    monkeypatch.setattr("team.build_model_client", no_network)
    yield tmp_path
    db.close_all()  # release the SQLite files so Windows can delete the temp directory


def _call(name: str, **args) -> CreateResult:
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id="call_1", name=name, arguments=json.dumps(args))],
        usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
        cached=False,
    )


def scripted_client(analysis: dict, kb_query: str, reply: str, confidence: float, sources: list[str], cited=None, answered=None):
    """A model client that plays the four agents' tool calls in order. `cited` = the structured cited_passages argument."""
    return ReplayChatCompletionClient(
        [
            _call("submit_analysis", **analysis),
            _call("search_kb", query=kb_query),
            _call("submit_resolution", reply=reply, confidence=confidence, used_sources=sources,
                  answered=(confidence >= 0.6) if answered is None else answered,
                  cited_passages=[] if cited is None else cited),
            _call("log_ticket"),
        ],
        model_info=MODEL_INFO,
    )


async def collect(gen):
    return [ev async for ev in gen]
