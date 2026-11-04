"""Telegram complaints: customers message your bot with text, a voice note or a photo and the same four agents handle it.
Free, and no public URL: the bot polls Telegram (getUpdates), so it runs on a laptop behind any router.

  Resolved  -> the grounded answer is sent back in the chat, with the reference number
  Escalated -> an acknowledgement is sent now; staff approve or edit the draft in the Review queue and it is sent then
  Not a support message (greeting, promo, spam) -> a short friendly reply (spam: none), no ticket

Setup: talk to @BotFather on Telegram, send /newbot, copy the token into TELEGRAM_BOT_TOKEN. Run:  python telegram_bot.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request

import audit
import db
import inbox
import media
from config import Settings, get_settings
from team import run_ticket

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
ACK = "Thanks for contacting NovaCart. Your request {tid} needs a specialist. Our team will review it and reply to you here."
START = "Hi! I'm the NovaCart support bot. Describe your problem in a message, a voice note or a photo and I'll help."
MSGS_PER_HOUR = 20


# ---------- Bot API (injectable in tests) ----------
def _call(s: Settings, method: str, payload: dict | None = None, timeout: int = 20) -> dict:
    req = urllib.request.Request(f"{API}/bot{s.telegram_token}/{method}", data=json.dumps(payload or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https host
        return json.loads(r.read())


def fetch_file(file_id: str, s: Settings) -> bytes:
    info = _call(s, "getFile", {"file_id": file_id})["result"]
    path = urllib.parse.quote(info["file_path"], safe="/")
    limit = max(s.max_audio_mb, s.max_image_mb) * 1024 * 1024
    with urllib.request.urlopen(f"{API}/file/bot{s.telegram_token}/{path}", timeout=30) as r:  # noqa: S310 - fixed https host
        return r.read(limit + 1)


def send_text(chat_id: str, text: str, s: Settings) -> None:
    _call(s, "sendMessage", {"chat_id": chat_id, "text": text[:3500]})


def send_review_reply(row: dict, final_text: str, s: Settings | None = None, send=send_text) -> bool:
    """After staff approve or edit an escalated Telegram ticket, send the final reply in the chat."""
    s = s or get_settings()
    if row.get("channel") != "telegram" or not final_text.strip():
        return False
    try:
        send(row["caller_phone"].split(":", 1)[1], f"{final_text.strip()}\n\nReference: {row['ticket_id']}", s)
        audit.record("telegram", "", "telegram_reply_sent", row["ticket_id"], "after human review", path=s.db_path)
        return True
    except Exception as exc:  # noqa: BLE001
        audit.record("telegram", "", "telegram_failed", row["ticket_id"], type(exc).__name__, path=s.db_path)
        return False


# ---------- one message -> one ticket ----------
def _known(s: Settings, mid: str) -> bool:
    return db.connect(s.db_path).execute("SELECT 1 FROM tickets WHERE message_id = ?", (mid,)).fetchone() is not None


async def process_message(msg: dict, s: Settings, model_client=None, triage_client=None, fetch=fetch_file, send=send_text) -> dict | None:
    chat = msg.get("chat") or {}
    if chat.get("type") != "private" or not chat.get("id"):  # groups and channels are ignored
        return None
    chat_id, mid = str(chat["id"]), f"tg:{chat['id']}:{msg.get('message_id', '')}"[:80]
    if not msg.get("message_id") or _known(s, mid):
        return None
    name = re.sub(r"[^\w .'-]", "", (msg.get("from") or {}).get("first_name", ""))[:60]
    text = msg.get("text") or msg.get("caption") or ""
    if text.strip().split("@")[0] == "/start":
        await asyncio.to_thread(send, chat_id, START, s)
        return None
    file_id = (msg.get("voice") or msg.get("audio") or {}).get("file_id") or (msg["photo"][-1].get("file_id") if msg.get("photo") else "")
    if not text and not file_id:
        await asyncio.to_thread(send, chat_id, "Sorry, I can read text, voice messages and photos. Please send one of those.", s)
        return None
    files, flags_extra = [], []
    if file_id:
        try:
            files.append(await asyncio.to_thread(fetch, file_id, s))
        except Exception as exc:  # noqa: BLE001 - a failed download is escalated below, not dropped
            log.warning("telegram media download failed: %s", type(exc).__name__)
            text = text or "[The customer sent a voice message or photo that could not be downloaded.]"
            flags_extra = ["media_unreadable"]
    prepared = await media.build_complaint(text, files, s)
    prepared.flags += flags_extra
    # triage: greetings, promotions and spam do not become tickets
    label = await inbox.classify_email(inbox.InboundMail(chat_id, name, "", prepared.text, mid), s, triage_client)
    if label != "SUPPORT":
        if label != "SPAM":
            await asyncio.to_thread(send, chat_id, inbox.AUTO_REPLIES[label].format(name=name or "there").split("\n\n", 1)[1].rsplit("\n\n", 1)[0], s)
        audit.record("telegram", "", "telegram_non_support", label, "chat …" + chat_id[-3:], path=s.db_path)
        return None
    final = None
    async for ev in run_ticket(
        prepared.text, customer_name=name, settings=s, model_client=model_client, owner=f"telegram:{chat_id}"[:60],
        call={"channel": "telegram", "phone": f"telegram:{chat_id}", "message_id": mid}, flags=prepared.flags, extra_chars=prepared.extra_chars,
    ):
        if ev.kind == "final":
            final = ev.data
    if not final:
        return None
    audit.record("telegram", "", "telegram_ticket", final["ticket_id"], final["status"], path=s.db_path)
    reply = (f"{final['response'].strip()}\n\nReference: {final['ticket_id']}" if final["status"] == "Resolved"
             else ACK.format(tid=final["ticket_id"]))
    try:
        await asyncio.to_thread(send, chat_id, reply, s)
    except Exception as exc:  # noqa: BLE001 - the ticket is saved; staff can still follow up
        audit.record("telegram", "", "telegram_failed", final["ticket_id"], type(exc).__name__, path=s.db_path)
    return final


# ---------- polling ----------
class Limiter:
    def __init__(self):
        self.seen: dict[str, list[float]] = {}

    def hit(self, who: str) -> bool:
        now = time.monotonic()
        recent = [t for t in self.seen.get(who, []) if now - t < 3600]
        limited = len(recent) >= MSGS_PER_HOUR
        if not limited:
            recent.append(now)
        self.seen[who] = recent
        return limited


async def poll_once(s: Settings, offset: int = 0, model_client=None, triage_client=None, call=_call, fetch=fetch_file, send=send_text,
                    limiter: Limiter | None = None, wait: int = 0) -> int:
    """Fetch pending updates, handle each message, return the next offset (Telegram forgets updates below it)."""
    limiter = limiter or Limiter()
    updates = (await asyncio.to_thread(call, s, "getUpdates", {"offset": offset, "timeout": wait, "allowed_updates": ["message"]}, wait + 20)).get("result", [])
    gate = asyncio.Semaphore(s.max_parallel)

    async def handle(m: dict) -> None:
        async with gate:
            try:
                await process_message(m, s, model_client, triage_client, fetch, send)
            except Exception as exc:  # noqa: BLE001 - one bad message must not stop the bot
                log.exception("telegram message failed")
                audit.record("telegram", "", "telegram_failed", "", type(exc).__name__, path=s.db_path)

    todo = []
    for u in updates:
        offset = max(offset, u["update_id"] + 1)
        m = u.get("message")
        if m and not limiter.hit(str((m.get("chat") or {}).get("id"))):
            todo.append(handle(m))
    await asyncio.gather(*todo)  # different chats are answered at the same time, a few at once
    return offset


def serve(s: Settings | None = None) -> None:
    s = s or get_settings()
    if not s.telegram_token or not s.has_key:
        sys.exit("Set in .env: " + ", ".join(n for n, v in (("TELEGRAM_BOT_TOKEN", s.telegram_token), ("OPENAI_API_KEY", s.has_key)) if not v))
    me = _call(s, "getMe").get("result", {})
    print(f"Telegram bot @{me.get('username', '?')} is listening. Open it in Telegram and send a message. Ctrl+C to stop.")
    offset, limiter = 0, Limiter()
    while True:
        try:
            offset = asyncio.run(poll_once(s, offset, limiter=limiter, wait=30))
        except KeyboardInterrupt:
            return
        except Exception as exc:  # noqa: BLE001 - network hiccup: wait and retry
            log.warning("telegram poll failed: %s", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
