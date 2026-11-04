"""Telegram bot: text/voice/photo messages, replies, polling offsets, guards."""
import logstore
import media
import telegram_bot
from config import get_settings

from conftest import scripted_client
from test_inbox import _triage_client

OGG = b"OggS\x00" + b"0" * 64
SHIP = dict(category="Shipping", urgency="Low", sentiment="Neutral", language="English", summary="Late order.")
GOOD = "Please check Track Order in My Orders. Orders over 3 business days late can be investigated [1]. NovaCart Support"


def settings(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    return get_settings()


class Outbox:
    def __init__(self):
        self.sent = []

    def __call__(self, chat, text, s):
        self.sent.append((chat, text))


def msg(mid=1, chat=555, kind="private", **body):
    return {"message_id": mid, "chat": {"id": chat, "type": kind}, "from": {"first_name": "Priya"}, **(body or {"text": "Where is my order? It was due yesterday."})}


async def test_text_message_resolved_reply_in_chat(monkeypatch):
    s, out = settings(monkeypatch), Outbox()
    row = await telegram_bot.process_message(msg(), s, scripted_client(SHIP, "late order", GOOD, 0.9, ["shipping_policy.pdf"]), send=out)
    assert row["status"] == "Resolved" and row["channel"] == "telegram" and row["caller_phone"] == "telegram:555"
    assert out.sent[0][0] == "555" and "Track Order" in out.sent[0][1] and row["ticket_id"] in out.sent[0][1]


async def test_voice_note_is_transcribed(monkeypatch):
    async def stt(data, ext, s):
        return "my blender arrived broken"

    monkeypatch.setattr(media, "transcribe", stt)
    s, out = settings(monkeypatch), Outbox()
    row = await telegram_bot.process_message(msg(voice={"file_id": "v1"}), s, scripted_client(SHIP, "blender", GOOD, 0.9, ["shipping_policy.pdf"]),
                                             fetch=lambda fid, st: OGG, send=out)
    assert "my blender arrived broken" in row["complaint"]


async def test_escalated_gets_ack_then_review_reply(monkeypatch):
    s, out = settings(monkeypatch), Outbox()
    client = scripted_client(dict(category="Billing", urgency="High", sentiment="Angry", language="English", summary="x"),
                             "charge", "A specialist will look. NovaCart Support", 0.7, ["billing_and_payments.pdf"])
    row = await telegram_bot.process_message(msg(text="You charged me three times, calling my lawyer"), s, client, send=out)
    assert row["status"] == "Escalated" and "specialist" in out.sent[0][1] and "will look" not in out.sent[0][1]
    stored = logstore.read_tickets().iloc[0].to_dict()
    assert telegram_bot.send_review_reply(stored, "We refunded the extra charges.", s, send=out)
    assert out.sent[-1][0] == "555" and "refunded" in out.sent[-1][1]


async def test_retried_message_never_creates_a_second_ticket(monkeypatch):
    s, out = settings(monkeypatch), Outbox()
    c = lambda: scripted_client(SHIP, "q", GOOD, 0.9, ["shipping_policy.pdf"])  # noqa: E731
    await telegram_bot.process_message(msg(), s, c(), send=out)
    assert await telegram_bot.process_message(msg(), s, c(), send=out) is None
    assert len(logstore.read_tickets()) == 1 and len(out.sent) == 1


async def test_failed_download_escalates(monkeypatch):
    s, out = settings(monkeypatch), Outbox()

    def boom(fid, st):
        raise RuntimeError("download failed")

    client = scripted_client(SHIP, "q", "Sorry. NovaCart Support", 0.2, [], answered=False)
    row = await telegram_bot.process_message(msg(voice={"file_id": "v"}), s, client, fetch=boom, send=out)
    assert row["status"] == "Escalated" and "Attachment could not be read" in row["escalation_reason"]


async def test_start_greeting_and_groups(monkeypatch):
    s, out = settings(monkeypatch), Outbox()
    assert await telegram_bot.process_message(msg(text="/start"), s, send=out) is None and "support bot" in out.sent[0][1]
    assert await telegram_bot.process_message(msg(mid=2, kind="group", text="Where is my order, it is late"), s, send=out) is None
    assert len(out.sent) == 1 and len(logstore.read_tickets()) == 0  # groups are ignored


async def test_small_talk_gets_friendly_reply_and_spam_none(monkeypatch):
    monkeypatch.setenv("MAIL_TRIAGE", "true")
    s, out = settings(monkeypatch), Outbox()
    assert await telegram_bot.process_message(msg(text="hi"), s, triage_client=_triage_client("GENERAL"), send=out) is None
    assert "support" in out.sent[0][1].lower() and len(logstore.read_tickets()) == 0
    assert await telegram_bot.process_message(msg(mid=2, text="win a prize"), s, triage_client=_triage_client("SPAM"), send=out) is None
    assert len(out.sent) == 1


async def test_poll_once_advances_offset_and_survives_a_bad_message(monkeypatch):
    s, out = settings(monkeypatch), Outbox()
    updates = {"result": [{"update_id": 10, "message": msg(mid=1, text="/start")}, {"update_id": 11, "edited_message": {}},
                          {"update_id": 12, "message": msg(mid=2, text="/start")}]}
    calls = []

    def call(st, method, payload=None, timeout=20):
        calls.append((method, payload))
        return updates

    offset = await telegram_bot.poll_once(s, 0, call=call, send=out)
    assert offset == 13 and calls[0][0] == "getUpdates" and len(out.sent) == 2


def test_chat_rate_limit():
    lim = telegram_bot.Limiter()
    assert not any(lim.hit("1") for _ in range(telegram_bot.MSGS_PER_HOUR)) and lim.hit("1") and not lim.hit("2")
