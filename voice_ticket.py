"""Voice complaints: the part of the voice desk that has nothing to do with audio. A caller's complaint (already turned into text by
LiveKit's speech-to-text) goes through the same four agents as every other channel, and the result is turned into something that can be
spoken aloud. voice_agent.py is the thin LiveKit wrapper around this.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import audit
from config import Settings
from team import run_ticket

MAX_TICKETS_PER_CALL = 3
TONES = {"calm": "calm", "neutral": "neutral", "frustrated": "frustrated", "angry": "angry"}
_CITATION = re.compile(r"\s*\[\d+\]")
_MARKUP = re.compile(r"[*_`#>]+")


def spell_ticket(ticket_id: str) -> str:
    """'TKT-20260514-0007' -> 'T K T, 2 0 2 6 0 5 1 4, 0 0 0 7' so it can be read out clearly."""
    return ", ".join(" ".join(part) for part in re.split(r"[-\s]+", ticket_id) if part)


_SALUTATION = re.compile(r"^(dear|hi|hello|hey)\b[^\n]{0,60}$", re.I)
_SIGNOFF = re.compile(r"^(best regards|kind regards|warm regards|regards|sincerely|best|thanks|thank you|cheers|novacart support( team)?)[,.!]?\s*$", re.I)


def speakable(reply: str) -> str:
    """Spoken text has no citation markers, markdown, 'Dear X' greeting or sign-off lines."""
    text = _MARKUP.sub("", _CITATION.sub("", reply))
    lines = [ln.strip() for ln in text.split("\n")]
    while lines and (not lines[0] or _SALUTATION.match(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1] or _SIGNOFF.match(lines[-1])):
        lines.pop()
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return re.sub(r"\s*NovaCart Support\.?\s*$", "", text).strip()


def _clean(value: str, limit: int) -> str:
    return re.sub(r"[\r\n\t]+", " ", value or "").strip()[:limit]


@dataclass
class VoiceCall:
    """State of one phone-style session in a LiveKit room."""

    room: str
    caller: str  # the participant identity LiveKit gave the caller
    owner: str = ""  # the signed-in customer's username, so the ticket shows under My requests
    name: str = ""  # their display name
    tickets: int = 0
    words_from: int = 0  # how many of the caller's turns earlier tickets already used
    seen: dict[str, str] = field(default_factory=dict)  # complaint fingerprint -> what was said, so a repeated tool call never makes a second ticket


def _script(text: str) -> str:
    n_lat = len(re.findall(r"[A-Za-z]", text))
    n_cyr = len(re.findall(r"[Ѐ-ӿ]", text))
    n_other = sum(1 for ch in text if ch.isalpha()) - n_lat - n_cyr
    return max((n_lat, "latin"), (n_cyr, "cyrillic"), (n_other, "other"))[1] if (n_lat or n_cyr or n_other) else ""


def drop_off_language(turns: list[str], lang: str) -> list[str]:
    """Speech-to-text sometimes turns breath or noise into a one-word fragment in another language ('Спасибо.') that would make the
    reply come out in that language. English calls: turns with no Latin letters are dropped. Auto-detect calls ('multi'): a short
    fragment (up to 3 words) in a different script from the rest of the call is dropped. A single language other than English is left
    alone. Nothing is dropped if that would leave the caller with nothing."""
    if lang in ("", "en"):
        return [t for t in turns if re.search(r"[A-Za-z]", t)] or turns
    if lang != "multi" or len(turns) < 2:
        return turns
    weight: dict[str, int] = {}
    for t in turns:
        weight[_script(t)] = weight.get(_script(t), 0) + len(t)
    main = max(weight, key=weight.get)
    return [t for t in turns if _script(t) == main or len(t.split()) > 3] or turns


def build_complaint(summary: str, caller_words: str, tone: str) -> str:
    """The text the four agents analyse. The assistant's summary is calm by nature, so the caller's own transcribed words come first when
    we have them, and a heard-tone note (a signal speech-to-text text cannot carry) is added in front."""
    body = caller_words.strip() if len(caller_words.strip()) >= 8 else summary.strip()
    heard = TONES.get(re.sub(r"\W+", "", tone).lower())
    return f"[Voice call: the caller sounded {heard}.] {body}" if heard in ("frustrated", "angry") else body


async def raise_voice_ticket(call: VoiceCall, complaint: str, s: Settings, customer_name: str = "", order_id: str = "",
                             contact: str = "", model_client=None, caller_words: str = "", tone: str = "") -> str:
    """Run the pipeline and return the words the voice agent should say. Raises ValueError for things the caller should hear about."""
    complaint = complaint.strip()
    if len(complaint) < 8:
        raise ValueError("I didn't catch the problem. Please ask the customer to describe it in a sentence or two.")
    key = hashlib.sha1(re.sub(r"\W+", " ", complaint.lower()).encode()).hexdigest()
    if key in call.seen:
        return call.seen[key]
    if call.tickets >= MAX_TICKETS_PER_CALL:
        raise ValueError("This call has reached its limit of requests. Tell the customer to call again or use the website.")
    call.tickets += 1
    who = _clean(contact, 60) or call.caller
    final = None
    async for ev in run_ticket(
        build_complaint(complaint, caller_words, tone), customer_name=_clean(customer_name, 60) or _clean(call.name, 60), order_id=order_id, settings=s, model_client=model_client,
        owner=call.owner or f"voice:{call.caller}"[:60],
        call={"channel": "voice", "phone": f"voice:{who}", "message_id": f"voice:{call.room}:{call.tickets}"[:80]},
    ):
        if ev.kind == "final":
            final = ev.data
    if not final:
        raise ValueError("The request could not be saved. Apologise and ask the customer to try again or use the website.")
    audit.record("voice", "", "voice_ticket", final["ticket_id"], final["status"], path=s.db_path)
    ref = spell_ticket(final["ticket_id"])
    lang = final.get("language") or "the caller's language"
    head = f"REPLY LANGUAGE: {lang}. Say everything below in {lang}, never in English unless that is the language. "
    if final["status"] == "Resolved":
        said = f"{head}ANSWER: {speakable(final['response'])} REFERENCE NUMBER, spell it slowly: {ref}."
    else:
        said = (f"{head}ANSWER: this needs a specialist. Tell the customer the team will review it and follow up with them. "
                f"Do not guess a solution. REFERENCE NUMBER, spell it slowly: {ref}.")
    call.seen[key] = said
    return said
