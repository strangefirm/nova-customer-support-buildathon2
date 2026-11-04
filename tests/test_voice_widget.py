"""'Talk to us' web widget: the access token, the page that carries it, and the tab in the app."""
import pytest

import voice_token
import voice_ui
from config import get_settings

from test_app import signed_in


def _livekit(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://demo.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "APIkey123")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s" * 40)
    return get_settings()


def test_token_joins_one_room_and_dispatches_the_voice_agent(monkeypatch):
    jwt = pytest.importorskip("jwt")  # optional voice dependencies (requirements-voice.txt)
    pytest.importorskip("livekit.api")
    s = _livekit(monkeypatch)
    token, room = voice_token.make_token(s, "alice", "Alice")
    c = jwt.decode(token, "s" * 40, algorithms=["HS256"], options={"verify_aud": False})
    assert c["video"]["room"] == room and c["video"]["roomJoin"] is True
    assert c["roomConfig"]["agents"] == [{"agentName": s.voice_agent_name}]
    assert c["attributes"] == {"username": "alice"}  # signed by the server, so the browser cannot change whose ticket it is
    assert c["sub"].startswith("web-alice-") and c["exp"] - c["nbf"] <= voice_token.TOKEN_MINUTES * 60
    assert voice_token.make_token(s, "alice")[1] != room  # a new room for every call


def test_token_needs_livekit_settings(monkeypatch):
    for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        voice_token.make_token(get_settings(), "alice")


def test_widget_page_never_contains_the_secret_and_cannot_break_out_of_its_script(monkeypatch):
    s = _livekit(monkeypatch)
    html = voice_ui.widget_html("wss://demo.livekit.cloud", "tok</script><script>alert(1)")
    assert "</script><script>alert" not in html and "\u003c/script" in html
    assert "s" * 40 not in html and s.livekit_api_secret not in html


def test_talk_tab_is_for_customers_only(monkeypatch):
    _livekit(monkeypatch)
    for role, expected in (("customer", True), ("staff", False)):
        at = signed_in("alice", role).run()
        assert not at.exception, at.exception
        assert any("Talk to us" in t.label for t in at.tabs) is expected


def test_talk_tab_explains_when_voice_is_not_set_up(monkeypatch):
    for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(k, raising=False)
    at = signed_in("bob", "customer").run()
    assert not at.exception and any("not set up" in i.value for i in at.info)


def test_interim_transcripts_update_one_line_instead_of_adding_many():
    html = voice_ui.widget_html("wss://demo.livekit.cloud", "tok")
    assert "lk.segment_id" in html and "function upsert" in html and "addLine" not in html
