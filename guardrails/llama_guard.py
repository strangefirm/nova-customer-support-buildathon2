"""Optional Llama Guard screen for the customer's message (defence in depth beside the regex input guard).

Llama Guard is a safety classifier model. It answers "safe", or "unsafe" followed by the violated
categories ("S1,S10"). It is served by any OpenAI-compatible endpoint: Groq (llama-guard-4-12b),
Ollama (llama-guard3), Together, vLLM.

It adds a flag; it never decides the outcome alone. An unsafe verdict escalates the ticket to a human.
If the guard is configured but unreachable, the ticket is flagged and escalated too (fail closed).
Off unless GUARD_MODEL is set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import Settings

CATEGORIES = {
    "S1": "violent crimes", "S2": "non-violent crimes", "S3": "sex crimes", "S4": "child exploitation",
    "S5": "defamation", "S6": "specialised advice", "S7": "privacy", "S8": "intellectual property",
    "S9": "indiscriminate weapons", "S10": "hate", "S11": "self-harm", "S12": "sexual content",
    "S13": "elections", "S14": "code interpreter abuse",
}
_CODES = re.compile(r"\bS(?:1[0-4]|[1-9])\b")


@dataclass
class Verdict:
    safe: bool
    categories: list[str] = field(default_factory=list)
    error: str = ""


def parse_verdict(text: str) -> Verdict:
    """Strict parse: anything that is not a clear 'safe' or 'unsafe' is treated as an error, not as safe."""
    first, _, rest = (text or "").strip().partition("\n")
    word = first.strip().lower()
    if word == "safe":
        return Verdict(True)
    if word == "unsafe":
        return Verdict(False, sorted(set(_CODES.findall(rest)), key=lambda c: int(c[1:])))
    return Verdict(False, error="unrecognised guard output")


async def classify(text: str, s: Settings, client=None) -> Verdict:
    """Screen one message. `client` is injectable for tests; otherwise an OpenAI-compatible client is built."""
    own = client is None
    try:
        if own:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=s.guard_api_key or s.api_key or "none", base_url=s.guard_base_url or s.base_url or None,
                                 timeout=s.guard_timeout, max_retries=1)
        resp = await client.chat.completions.create(
            model=s.guard_model, temperature=0, max_tokens=20, messages=[{"role": "user", "content": text}],
        )
        return parse_verdict(resp.choices[0].message.content or "")
    except Exception as exc:  # noqa: BLE001 - any failure must fail closed, never crash the ticket
        return Verdict(False, error=type(exc).__name__)
    finally:
        if own and client is not None:
            await client.close()


def describe(v: Verdict) -> str:
    if v.error:
        return "Llama Guard unavailable"
    names = ", ".join(f"{c} {CATEGORIES.get(c, '')}".strip() for c in v.categories)
    return f"Flagged unsafe by Llama Guard ({names})" if names else "Flagged unsafe by Llama Guard"
