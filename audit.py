"""Append-only audit trail of security-relevant events (who did what, when).

Stored in the `audit` table, whose triggers reject UPDATE and DELETE. Never holds passwords or complaint text.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

import db

COLUMNS = ["timestamp", "actor", "role", "action", "target", "detail"]


def record(actor: str, role: str, action: str, target: str = "", detail: str = "", path: Path | None = None) -> None:
    """Best effort: an audit failure must never block the user, but it is reported on stderr."""
    try:
        with db.transaction(path) as conn:
            conn.execute(
                "INSERT INTO audit (timestamp, actor, role, action, target, detail) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(actor)[:64], str(role)[:16], str(action)[:48], str(target)[:64], str(detail)[:200]),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"audit write failed: {type(exc).__name__}", flush=True)


def read(path: Path | None = None, limit: int = 500) -> pd.DataFrame:
    """The newest `limit` events, newest first."""
    rows = db.connect(path).execute(
        f"SELECT {', '.join(COLUMNS)} FROM audit ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    return pd.DataFrame([tuple(r) for r in rows], columns=COLUMNS)
