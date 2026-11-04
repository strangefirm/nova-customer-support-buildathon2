"""Inline citations: validate that every [n] marker in a reply points at a passage that was really retrieved."""
from __future__ import annotations

import re

from models import Citation, KBHit

MARKER = re.compile(r"\[(\d{1,2})\]")


def markers(reply: str) -> list[int]:
    return [int(m) for m in MARKER.findall(reply)]


def strip_markers(reply: str) -> str:
    """The reply without [n] markers (so the digits in a marker are not mistaken for factual figures)."""
    return MARKER.sub("", reply)


def resolve(reply: str, hits: list[KBHit], required: bool = True, declared: list[int] | None = None) -> tuple[list[Citation], list[str]]:
    """Map citations to passages. Returns (citations, issues).

    `declared` is the structured list of passage numbers the model gave in the tool call (reliable); inline [n]
    markers in the reply text are merged in (a bonus the model often forgets). Both are validated the same way.

    Issues: a marker that does not correspond to a retrieved passage ("invalid citation [k]"), and, when
    `required` and passages were retrieved, an answer that carries no citation at all ("no citations").
    """
    cited = list(dict.fromkeys([int(n) for n in (declared or []) if isinstance(n, (int, float)) and not isinstance(n, bool)] + markers(reply)))  # unique
    issues = [f"invalid citation [{n}]" for n in cited if not 1 <= n <= len(hits)]
    if hits and required and not cited:
        issues.append("no citations")
    citations = [Citation(n=n, source=hits[n - 1].source, section=hits[n - 1].heading) for n in cited if 1 <= n <= len(hits)]
    return citations, issues
