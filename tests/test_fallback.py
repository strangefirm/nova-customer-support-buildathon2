"""Runtime fallbacks: a failing fast model, Groq speech-to-text and the live voice models all fall back instead of failing the ticket."""
import asyncio

import pytest

import media
from config import get_settings
from llm_client import FallbackClient


class _Client:
    def __init__(self, name, fail=False, stream_fail_after=None):
        self.name, self.fail, self.stream_fail_after, self.calls, self.closed = name, fail, stream_fail_after, 0, False

    async def create(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise ConnectionError("provider down")
        return self.name

    async def create_stream(self, messages, **kw):
        self.calls += 1
        for i in range(3):
            if self.stream_fail_after is not None and i == self.stream_fail_after:
                raise ConnectionError("dropped")
            yield f"{self.name}{i}"

    async def close(self):
        self.closed = True


async def test_failing_primary_is_retried_on_the_fallback():
    primary, fallback = _Client("fast", fail=True), _Client("main")
    assert await FallbackClient(primary, fallback).create([]) == "main" and primary.calls == fallback.calls == 1
    assert await FallbackClient(_Client("fast"), fallback).create([]) == "fast"  # healthy primary: the fallback is not used


async def test_stream_falls_back_only_before_anything_was_delivered():
    got = [x async for x in FallbackClient(_Client("fast", stream_fail_after=0), _Client("main")).create_stream([])]
    assert got == ["main0", "main1", "main2"]
    with pytest.raises(ConnectionError):  # a half-delivered answer is not restarted on another model
        [x async for x in FallbackClient(_Client("fast", stream_fail_after=1), _Client("main")).create_stream([])]


async def test_cancellation_is_not_swallowed():
    class Slow(_Client):
        async def create(self, messages, **kw):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await FallbackClient(Slow("fast"), _Client("main")).create([])


async def test_close_closes_the_fallback_only_when_owned():
    a, b = _Client("a"), _Client("b")
    await FallbackClient(a, b).close()
    assert a.closed and not b.closed
    a2, b2 = _Client("a"), _Client("b")
    await FallbackClient(a2, b2, own_fallback=True).close()
    assert a2.closed and b2.closed


async def test_uploaded_audio_falls_back_to_openai_whisper(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s, calls = get_settings(), []

    async def fake(data, ext, model, key, base, timeout):
        calls.append((model, base))
        if "groq" in base:
            raise ConnectionError("groq down")
        return "my blender is broken"

    monkeypatch.setattr(media, "_transcribe_with", fake)
    assert await media.transcribe(b"x", "wav", s) == "my blender is broken"
    assert [c[0] for c in calls] == ["whisper-large-v3-turbo", "whisper-1"]


async def test_no_fallback_when_already_on_openai(monkeypatch):
    s = get_settings()  # conftest: no Groq key, so the main speech-to-text is OpenAI whisper-1 already

    async def boom(*a):
        raise ConnectionError("down")

    monkeypatch.setattr(media, "_transcribe_with", boom)
    with pytest.raises(ConnectionError):
        await media.transcribe(b"x", "wav", s)


def test_live_voice_models_are_fallback_chains(monkeypatch):
    pytest.importorskip("livekit.agents")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    silero = pytest.importorskip("livekit.plugins.silero")
    import voice_agent
    stt_, llm_, tts_ = voice_agent.build_voice_models(get_settings(), silero.VAD.load())
    assert [type(x).__name__ for x in (stt_, llm_, tts_)] == ["FallbackAdapter"] * 3
