"""SQLite storage layer: one database file holds accounts, tickets and the audit trail.
(Embeddings are deliberately NOT here: they live inside the FAISS index, see kb/vector_index.py.)

Production properties
  * WAL journal: readers never block the writer; durable commits (synchronous=FULL)
  * foreign keys ON, STRICT tables and CHECK constraints: bad data is rejected by the database itself
  * every write is a short BEGIN IMMEDIATE transaction (safe across threads AND processes)
  * schema versioning via PRAGMA user_version with ordered, transactional migrations
  * tamper resistance: the audit table is append-only (triggers) and ticket identity fields are immutable
  * parameterised queries only; one connection per thread (sqlite3 connections are not thread-safe)
  * online backup (sqlite3 backup API) and an integrity check, both available from the CLI

CLI:  python -m db init | check | backup DEST | migrate-legacy
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path

from config import get_settings

SCHEMA_VERSION = 5

# Each migration is a list of single statements (triggers contain ';', so scripts cannot be split safely).
MIGRATIONS: dict[int, list[str]] = {
    1: [
        """CREATE TABLE users (
            username     TEXT PRIMARY KEY CHECK (length(username) BETWEEN 3 AND 32),
            display_name TEXT NOT NULL,
            role         TEXT NOT NULL CHECK (role IN ('customer', 'staff')),
            salt         TEXT NOT NULL,
            hash         TEXT NOT NULL,
            params       TEXT NOT NULL,
            created_at   TEXT NOT NULL
        ) STRICT""",
        """CREATE TABLE login_failures (
            username TEXT NOT NULL,
            ts       REAL NOT NULL
        ) STRICT""",
        "CREATE INDEX idx_login_failures ON login_failures (username, ts)",
        """CREATE TABLE tickets (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id         TEXT GENERATED ALWAYS AS ('TKT-' || printf('%04d', id)) STORED UNIQUE,
            timestamp         TEXT NOT NULL,
            owner             TEXT NOT NULL DEFAULT '',
            customer_name     TEXT NOT NULL DEFAULT '',
            order_id          TEXT NOT NULL DEFAULT '',
            complaint         TEXT NOT NULL DEFAULT '',
            category          TEXT NOT NULL DEFAULT 'Unknown',
            urgency           TEXT NOT NULL DEFAULT 'Unknown',
            sentiment         TEXT NOT NULL DEFAULT 'Unknown',
            language          TEXT NOT NULL DEFAULT '',
            kb_sources        TEXT NOT NULL DEFAULT '',
            response          TEXT NOT NULL DEFAULT '',
            confidence        REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
            status            TEXT NOT NULL DEFAULT 'Escalated' CHECK (status IN ('Resolved', 'Escalated')),
            escalation_reason TEXT NOT NULL DEFAULT '',
            security_flags    TEXT NOT NULL DEFAULT '',
            review_status     TEXT NOT NULL DEFAULT 'N/A' CHECK (review_status IN ('N/A', 'Pending', 'Approved', 'Edited', 'Rejected')),
            final_response    TEXT NOT NULL DEFAULT '',
            reviewed_by       TEXT NOT NULL DEFAULT '',
            reviewed_at       TEXT NOT NULL DEFAULT ''
        ) STRICT""",
        "CREATE INDEX idx_tickets_owner ON tickets (owner, id)",
        "CREATE INDEX idx_tickets_review ON tickets (status, review_status)",
        """CREATE TRIGGER tickets_identity_immutable
           BEFORE UPDATE OF timestamp, owner, customer_name, order_id, complaint ON tickets
           BEGIN SELECT RAISE(ABORT, 'ticket identity fields are immutable'); END""",
        """CREATE TABLE audit (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            actor     TEXT NOT NULL,
            role      TEXT NOT NULL DEFAULT '',
            action    TEXT NOT NULL,
            target    TEXT NOT NULL DEFAULT '',
            detail    TEXT NOT NULL DEFAULT ''
        ) STRICT""",
        """CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit
           BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END""",
        """CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit
           BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END""",
    ],
    2: [
        # email for password-reset links; pw_version lets a password change sign out every other session
        "ALTER TABLE users ADD COLUMN email TEXT CHECK (email IS NULL OR (length(email) <= 254 AND email = lower(email)))",
        "ALTER TABLE users ADD COLUMN pw_version INTEGER NOT NULL DEFAULT 0",
        "CREATE UNIQUE INDEX idx_users_email ON users (email) WHERE email IS NOT NULL",
        # only the SHA-256 of a reset token is stored, so a database leak never yields a usable link
        """CREATE TABLE password_resets (
            token_hash TEXT PRIMARY KEY,
            username   TEXT NOT NULL REFERENCES users (username) ON DELETE CASCADE,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            used_at    REAL
        ) STRICT""",
        "CREATE INDEX idx_resets_user ON password_resets (username, created_at)",
    ],
    3: [
        # which numbered passage ([n] marker) each cited fact in the reply points to: "[1] file > section; [2] ..."
        "ALTER TABLE tickets ADD COLUMN citations TEXT NOT NULL DEFAULT ''",
    ],
    4: [
        # tickets that arrive by email: where they came from, who to answer, and the Message-ID (dedupe + reply threading)
        "ALTER TABLE tickets ADD COLUMN channel TEXT NOT NULL DEFAULT 'web'",
        "ALTER TABLE tickets ADD COLUMN sender_email TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE tickets ADD COLUMN email_subject TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE tickets ADD COLUMN message_id TEXT NOT NULL DEFAULT ''",
        "CREATE UNIQUE INDEX idx_tickets_message_id ON tickets (message_id) WHERE message_id <> ''",
    ],
    5: [
        # chat channels: the sender's chat identity, e.g. "telegram:<chat id>" (message_id is used for dedupe)
        "ALTER TABLE tickets ADD COLUMN caller_phone TEXT NOT NULL DEFAULT ''",
    ],
}

_local = threading.local()
_init_lock = threading.Lock()
_initialised: set[str] = set()


def db_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else get_settings().db_path


def _set_wal(conn: sqlite3.Connection, timeout: float = 10.0) -> None:
    """Switch to WAL. The one-time conversion needs an exclusive lock and SQLite does NOT apply the busy
    timeout to it, so several threads/processes opening a brand-new database at once can get
    'database is locked'. Retry until the competing opener has finished converting."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def _open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)  # autocommit; we manage transactions explicitly
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    _set_wal(conn)
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"Database schema v{version} is newer than this app (v{SCHEMA_VERSION}). Upgrade the app.")
        for v in sorted(MIGRATIONS):
            if v > version:
                for stmt in MIGRATIONS[v]:
                    conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {int(v)}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """This thread's connection to the database, created (and the schema upgraded) on first use."""
    p = db_path(path).resolve()
    key = str(p)
    conns: dict[str, sqlite3.Connection] = _local.__dict__.setdefault("conns", {})
    conn = conns.get(key)
    if conn is None:
        with _init_lock:  # opening (WAL switch + migration) is serialised within the process; _set_wal handles other processes
            conn = _open(p)
            if key not in _initialised:
                _migrate(conn)
                try:
                    os.chmod(p, 0o600)  # owner-only on POSIX; Windows relies on folder permissions
                except OSError:
                    pass
                _initialised.add(key)
        conns[key] = conn
    return conn


@contextmanager
def transaction(path: str | Path | None = None):
    """A write transaction: BEGIN IMMEDIATE takes the write lock up front, so it cannot deadlock or be starved."""
    conn = connect(path)
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def close_all() -> None:
    """Close this thread's connections and forget initialisation (used by tests and graceful shutdown)."""
    conns: dict[str, sqlite3.Connection] = _local.__dict__.get("conns", {})
    for c in conns.values():
        try:
            c.close()
        except sqlite3.Error:
            pass
    conns.clear()
    with _init_lock:
        _initialised.clear()


# ---------- operations ----------
def check(path: str | Path | None = None) -> dict:
    conn = connect(path)
    integrity = conn.execute("PRAGMA integrity_check").fetchall()
    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    return {
        "integrity": [r[0] for r in integrity],
        "foreign_key_violations": len(fk),
        "schema_version": conn.execute("PRAGMA user_version").fetchone()[0],
        "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
        "rows": {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("users", "tickets", "audit", "password_resets")},
        "ok": [r[0] for r in integrity] == ["ok"] and not fk,
    }


def backup(dest: str | Path, path: str | Path | None = None) -> Path:
    """Consistent online backup (safe while the app is running)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; refusing to overwrite a backup")
    target = sqlite3.connect(dest)
    try:
        connect(path).backup(target)
    finally:
        target.close()
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass
    return dest


def _cli(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "init":
        connect()
        print(f"Database ready: {db_path()} (schema v{SCHEMA_VERSION})")
    elif cmd == "check":
        report = check()
        for k, v in report.items():
            print(f"{k}: {v}")
        return 0 if report["ok"] else 1
    elif cmd == "backup" and len(argv) == 2:
        print(f"Backup written to {backup(argv[1])}")
    elif cmd == "migrate-legacy":
        from legacy_import import import_legacy

        print(import_legacy())
    else:
        print("usage: python -m db init | check | backup DEST | migrate-legacy")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
