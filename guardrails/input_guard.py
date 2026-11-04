"""Input hardening: normalisation, length cap, PII masking and prompt-injection detection.

The complaint is untrusted user text. Everything downstream (LLM, log, UI) only ever sees
the sanitised version.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")
_DELIMS = re.compile(r"<{3,}|>{3,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier|your) (?:instructions|rules|prompts?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier|your)",
    r"forget (?:all |everything |your )?(?:previous |prior )?(?:instructions|rules)",
    r"(?:reveal|show|print|repeat|leak|output) (?:me )?(?:your |the )?(?:system )?(?:prompt|instructions|api[ _-]?key|secret|credentials)",
    r"you are now\b",
    r"\bact as (?:an? )?(?:admin|root|developer|system|dan)\b",
    r"\b(?:jailbreak|developer mode|dan mode)\b",
    r"(?:mark|set|flag) (?:this |the )?ticket (?:as )?(?:resolved|closed)",
    r"do not escalate|don't escalate|never escalate",
    r"(?:approve|issue|grant) (?:a |the |my )?(?:full )?refund (?:immediately|now|without)",
    r"</?(?:system|assistant|tool|function)[^>]*>",
    r"\bnew (?:system )?instructions?:",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


class InputRejected(ValueError):
    """Raised for input that must not enter the pipeline (empty after sanitising)."""


@dataclass
class Sanitized:
    text: str
    flags: list[str] = field(default_factory=list)
    truncated: bool = False


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def mask_pii(text: str) -> str:
    """Mask card numbers (Luhn-valid), emails and phone numbers."""

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        return "[CARD-REDACTED]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group()

    text = _CARD.sub(card, text)
    text = _EMAIL.sub("[EMAIL-REDACTED]", text)
    return _PHONE.sub("[PHONE-REDACTED]", text)


def sanitize_complaint(raw: str, max_chars: int = 2000, redact: bool = True) -> Sanitized:
    if not isinstance(raw, str):
        raise InputRejected("Complaint must be text.")
    text = unicodedata.normalize("NFKC", raw)
    text = _CONTROL.sub("", text)
    text = _DELIMS.sub(" ", text)  # cannot break out of our untrusted-data delimiters
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise InputRejected("Please describe your issue so we can help.")
    flags: list[str] = []
    truncated = False
    if len(text) > max_chars:
        text, truncated = text[:max_chars], True
        flags.append("truncated_input")
    if _INJECTION.search(text):
        flags.append("prompt_injection")
    if redact:
        masked = mask_pii(text)
        if masked != text:
            flags.append("pii_masked")
        text = masked
    return Sanitized(text=text, flags=flags, truncated=truncated)


def clean_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = _CONTROL.sub("", value)
    value = re.sub(r"[^\w .'\-]", "", value).strip()
    return value[:60]


def clean_order_id(value: str) -> str:
    value = (value or "").strip().upper()
    return value if re.fullmatch(r"[A-Z0-9][A-Z0-9\-_]{0,23}", value) else ""
