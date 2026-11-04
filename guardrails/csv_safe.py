"""Neutralise spreadsheet formula injection (CSV/Excel injection, CWE-1236)."""
from __future__ import annotations

_DANGEROUS = ("=", "+", "-", "@", "\t", "\r")


def neutralise(value: object) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(_DANGEROUS):
        return "'" + text
    return text
