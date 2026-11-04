"""Fast (DeepSeek) model wiring, and parallel processing of a burst."""
import asyncio

import inbox
import team
import telegram_bot
from config import build_fast_client, get_settings

from conftest import scripted_client
from test_telegram import GOOD, SHIP, Outbox, msg, settings as tg_settings


def test_fast_model_off_without_a_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("FAST_API_KEY", raising=False)
    assert not get_settings().fast_enabled


def test_fast_model_defaults_to_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    s = get_settings()
    assert s.fast_enabled and s.fast_model == "deepseek-chat" and s.fast_base_url == "https://api.deepseek.com"
    assert build_fast_client(s) is not None


def test_only_the_reply_is_written_by_the_main_model(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, name): self.name = name

    import agents
    for mod in (team.query_agent, team.knowledge_agent, team.resolution_agent, team.log_agent):
        monkeypatch.setattr(mod, "build", lambda client, *a, _m=mod, **k: seen.setdefault(_m.NAME, client) or object())
    monkeypatch.setattr(team, "RoundRobinGroupChat", lambda parts, **k: parts)
    from models import TicketContext
    main, fast = Fake("main"), Fake("fast")
    team.build_team(TicketContext(complaint="x"), main, get_settings(), fast)
    assert seen[team.resolution_agent.NAME] is main
    assert all(seen[n] is fast for n in (team.query_agent.NAME, team.knowledge_agent.NAME, team.log_agent.NAME))


async def test_a_burst_of_chat_messages_runs_in_parallel(monkeypatch):
    monkeypatch.setenv("MAX_PARALLEL", "3")
    s, out = tg_settings(monkeypatch), Outbox()
    running, peak = 0, 0
    real = telegram_bot.process_message

    async def slow(m, *a, **k):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        return None

    monkeypatch.setattr(telegram_bot, "process_message", slow)
    updates = {"result": [{"update_id": i, "message": msg(mid=i, chat=100 + i, text="Where is my order, it is late")} for i in range(1, 7)]}
    offset = await telegram_bot.poll_once(s, 0, call=lambda *a, **k: updates, send=out)
    assert offset == 7 and 1 < peak <= 3  # several at once, never more than MAX_PARALLEL


async def test_same_sender_gets_one_auto_reply_even_in_a_burst(monkeypatch):
    s = get_settings()
    sent = []
    monkeypatch.setattr(inbox, "_send", lambda *a: sent.append(a) or True)
    mail = inbox.InboundMail("promo@example.com", "", "Sale", "big sale", "<1@x>")
    await asyncio.gather(*(inbox.handle_non_support(mail, s, "PROMOTIONAL") for _ in range(3)))
    assert len(sent) == 1


def test_speech_to_text_uses_groq_when_a_groq_key_exists(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    s = get_settings()
    assert s.stt_model == "whisper-large-v3-turbo" and s.stt_base_url == "https://api.groq.com/openai/v1" and s.stt_api_key == "gsk_test"
    monkeypatch.setenv("STT_MODEL", "whisper-large-v3")  # an explicit setting always wins
    assert get_settings().stt_model == "whisper-large-v3"


def test_speech_to_text_stays_on_openai_without_a_groq_key():
    s = get_settings()
    assert s.stt_model == "whisper-1" and not s.stt_base_url
