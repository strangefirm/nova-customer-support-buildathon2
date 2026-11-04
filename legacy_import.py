"""One-time import of the old file-based storage (users.json, tickets.csv, audit.csv) into SQLite.

    python -m db migrate-legacy

Idempotent and non-destructive: rows that already exist are skipped, ticket IDs are preserved, and the
source files are renamed to *.migrated (never deleted) once they have been imported.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import db
from config import ROOT

ESCAPED = ("=", "+", "-", "@", "\t", "\r")
STATUSES = ("Resolved", "Escalated")
REVIEWS = ("N/A", "Pending", "Approved", "Edited", "Rejected")


def _unescape(v: str) -> str:
    """The old CSV writer prefixed a quote to formula-looking cells; the database stores text as-is."""
    return v[1:] if len(v) > 1 and v[0] == "'" and v[1] in ESCAPED else v


def _conf(v: str):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 1.0 else None


def _mark_done(path: Path) -> None:
    path.replace(path.with_name(path.name + ".migrated"))


def import_legacy(users_json: Path | None = None, tickets_csv: Path | None = None, audit_csv: Path | None = None,
                  path: Path | None = None) -> dict:
    users_json = Path(users_json) if users_json else ROOT / "data" / "users.json"
    tickets_csv = Path(tickets_csv) if tickets_csv else ROOT / "logs" / "tickets.csv"
    audit_csv = Path(audit_csv) if audit_csv else ROOT / "logs" / "audit.csv"
    report = {"users": 0, "tickets": 0, "audit": 0, "skipped_files": []}

    with db.transaction(path) as conn:
        if users_json.exists():
            for name, u in json.loads(users_json.read_text(encoding="utf-8")).get("users", {}).items():
                cur = conn.execute(
                    "INSERT OR IGNORE INTO users (username, display_name, role, salt, hash, params, created_at) VALUES (?,?,?,?,?,?,?)",
                    (name, u["display_name"], u["role"], u["salt"], u["hash"], json.dumps(u.get("params") or {}), u.get("created_at", "")),
                )
                report["users"] += cur.rowcount
        else:
            report["skipped_files"].append(str(users_json))

        if tickets_csv.exists():
            with tickets_csv.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    tid = r.get("ticket_id", "")
                    if not (tid.startswith("TKT-") and tid[4:].isdigit()):
                        continue
                    status = r.get("status") if r.get("status") in STATUSES else "Escalated"
                    review = r.get("review_status") if r.get("review_status") in REVIEWS else ("Pending" if status == "Escalated" else "N/A")
                    g = lambda k: _unescape(r.get(k, "") or "")  # noqa: E731
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO tickets (id, timestamp, owner, customer_name, order_id, complaint, category, urgency, sentiment,"
                        " language, kb_sources, response, confidence, status, escalation_reason, security_flags, review_status,"
                        " final_response, reviewed_by, reviewed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (int(tid[4:]), g("timestamp"), g("owner"), g("customer_name"), g("order_id"), g("complaint"),
                         g("category") or "Unknown", g("urgency") or "Unknown", g("sentiment") or "Unknown", g("language"),
                         g("kb_sources"), g("response"), _conf(r.get("confidence", "")), status, g("escalation_reason"),
                         g("security_flags"), review, g("final_response"), g("reviewed_by"), g("reviewed_at")),
                    )
                    report["tickets"] += cur.rowcount
        else:
            report["skipped_files"].append(str(tickets_csv))

        if audit_csv.exists():
            with audit_csv.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    cur = conn.execute(
                        "INSERT INTO audit (timestamp, actor, role, action, target, detail) VALUES (?,?,?,?,?,?)",
                        tuple(_unescape(r.get(k, "")) for k in ("timestamp", "actor", "role", "action", "target", "detail")),
                    )
                    report["audit"] += cur.rowcount
        else:
            report["skipped_files"].append(str(audit_csv))

    # only after the transaction committed do we retire the originals
    for p in (users_json, tickets_csv, audit_csv):
        if p.exists():
            _mark_done(p)
    return report
