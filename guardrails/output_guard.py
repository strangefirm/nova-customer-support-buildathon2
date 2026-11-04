"""Output checks: every figure in a reply must be traceable to retrieved passages, and no secrets leak."""
from __future__ import annotations

import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_SECRET = re.compile(
    r"sk-[A-Za-z0-9_\-]{16,}|gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{20,}|-----BEGIN [A-Z ]*KEY-----"
)


def _numbers(text: str) -> set[str]:
    return {n.rstrip(".,").replace(",", "") for n in _NUMBER.findall(text)}


_ORDINAL = re.compile(r"(?im)^\s*(?:step\s+)?\d{1,2}\s*[.):]\s+|\bstep\s+\d{1,2}\b")


def ungrounded_figures(reply: str, passages: list[str], extra_context: str = "") -> list[str]:
    """Numbers in the reply that appear neither in the passages nor in the customer's own input.

    List numbering ("1. Go to..." / "Step 2") is structure, not a factual figure, so it is ignored; "30 days" is not.
    """
    allowed = _numbers(" ".join(passages)) | _numbers(extra_context)
    return sorted(n for n in _numbers(_ORDINAL.sub(" ", reply)) if n not in allowed)


def contains_secret(text: str, known_key: str = "") -> bool:
    if _SECRET.search(text):
        return True
    return bool(known_key and len(known_key) >= 8 and known_key in text)


def scrub_secrets(text: str, known_key: str = "") -> str:
    text = _SECRET.sub("[REDACTED]", text)
    if known_key and len(known_key) >= 8:
        text = text.replace(known_key, "[REDACTED]")
    return text
