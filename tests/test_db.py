"""SQLite storage layer: durability settings, constraints, tamper protection, concurrency, backup, migration."""
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import audit
import auth
import db
import legacy_import
import logstore
from config import ROOT, get_settings

PW = "correct horse battery"


def _ticket(**kw):
    return logstore.append_ticket({"complaint": "c", "category": "Shipping", "urgency": "Low", "sentiment": "Neutral",
                                   "status": "Resolved", **kw})


# ---------- settings and schema ----------
def test_wal_foreign_keys_and_schema_version():
    c = db.connect()
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert c.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


def test_reopening_keeps_data_and_does_not_remigrate():
    _ticket(complaint="keep me")
    db.close_all()
    assert logstore.read_tickets().iloc[0]["complaint"] == "keep me"


def test_database_from_a_newer_app_is_refused(tmp_path):
    p = tmp_path / "future.db"
    raw = sqlite3.connect(p)
    raw.execute("PRAGMA user_version = 99")
    raw.close()
    with pytest.raises(RuntimeError, match="newer"):
        db.connect(p)


def test_ticket_id_is_generated_and_zero_padded():
    assert [_ticket()["ticket_id"] for _ in range(3)] == ["TKT-0001", "TKT-0002", "TKT-0003"]
    c = db.connect()
    c.execute("INSERT INTO tickets (id, timestamp, status) VALUES (12345, 't', 'Resolved')")
    assert c.execute("SELECT ticket_id FROM tickets WHERE id = 12345").fetchone()[0] == "TKT-12345"


# ---------- constraints enforced by the database itself ----------
@pytest.mark.parametrize("sql", [
    "INSERT INTO tickets (timestamp, status) VALUES ('t', 'Bogus')",
    "INSERT INTO tickets (timestamp, status, review_status) VALUES ('t', 'Resolved', 'Maybe')",
    "INSERT INTO tickets (timestamp, status, confidence) VALUES ('t', 'Resolved', 5)",
    "INSERT INTO tickets (timestamp, status, confidence) VALUES ('t', 'Resolved', 'abc')",
    "INSERT INTO users (username, display_name, role, salt, hash, params, created_at) VALUES ('rooty', 'R', 'admin', 's', 'h', '{}', 'now')",
    "INSERT INTO users (username, display_name, role, salt, hash, params, created_at) VALUES ('ab', 'R', 'staff', 's', 'h', '{}', 'now')",
    "INSERT INTO users (username, display_name, role, salt, hash, params, created_at, email) VALUES ('mixedcase', 'R', 'staff', 's', 'h', '{}', 'now', 'A@B.COM')",
])
def test_invalid_rows_are_rejected_by_the_database(sql):
    with pytest.raises(sqlite3.IntegrityError):
        db.connect().execute(sql)


def test_duplicate_username_is_rejected_by_the_database():
    auth.create_user("alice", PW, "staff")
    with pytest.raises(sqlite3.IntegrityError):
        db.connect().execute("INSERT INTO users (username, display_name, role, salt, hash, params, created_at) VALUES ('alice', 'X', 'staff', 's', 'h', '{}', 'now')")


# ---------- tamper protection ----------
@pytest.mark.parametrize("col", ["complaint", "owner", "customer_name", "order_id", "timestamp"])
def test_ticket_identity_fields_are_immutable(col):
    _ticket(owner="cust.a")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.connect().execute(f"UPDATE tickets SET {col} = 'tampered'")


def test_ticket_id_cannot_be_edited():
    _ticket()
    with pytest.raises(sqlite3.OperationalError):
        db.connect().execute("UPDATE tickets SET ticket_id = 'TKT-9999'")


def test_audit_log_is_append_only():
    audit.record("alice", "staff", "login_ok")
    c = db.connect()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        c.execute("UPDATE audit SET actor = 'mallory'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        c.execute("DELETE FROM audit")
    assert len(audit.read()) == 1


# ---------- injection ----------
def test_sql_injection_payloads_are_stored_literally_and_harmless():
    evil = "'); DROP TABLE tickets;--"
    _ticket(complaint=evil, customer_name=evil, owner=evil)
    assert db.connect().execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1  # table survived
    assert logstore.read_tickets().iloc[0]["complaint"] == evil
    assert logstore.read_tickets(owner="x' OR '1'='1").empty  # a crafted owner matches nothing
    assert auth.authenticate("admin' --", "x") is None


# ---------- concurrency ----------
def test_parallel_processes_get_unique_sequential_ids(isolated_env):
    code = (
        "import logstore\n"
        "for _ in range(8):\n"
        "    logstore.append_ticket({'complaint': 'p', 'status': 'Resolved'})\n"
    )
    db.connect()  # create the schema once, as the real app does at startup
    procs = [subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=os.environ.copy(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(4)]
    errors = [p.communicate()[1].decode() for p in procs]
    assert all(p.returncode == 0 for p in procs), errors
    ids = list(logstore.read_tickets()["ticket_id"])
    assert len(ids) == 32 and len(set(ids)) == 32
    assert sorted(ids) == [f"TKT-{i:04d}" for i in range(1, 33)]  # no gaps, no duplicates


def test_readers_are_not_blocked_by_an_open_write_transaction():
    _ticket()
    with db.transaction() as conn:
        conn.execute("INSERT INTO audit (timestamp, actor, action) VALUES ('t', 'a', 'x')")
        other = sqlite3.connect(get_settings().db_path, timeout=0.2)  # a separate connection, like another process
        assert other.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1  # WAL: reads proceed during a write
        other.close()


def test_failed_transaction_rolls_back_completely():
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction() as conn:
            conn.execute("INSERT INTO audit (timestamp, actor, action) VALUES ('t', 'a', 'x')")
            conn.execute("INSERT INTO tickets (timestamp, status) VALUES ('t', 'Bogus')")
    assert audit.read().empty  # the first insert was rolled back with the failed second one


# ---------- operations ----------
def test_check_reports_healthy_database():
    _ticket()
    report = db.check()
    assert report["ok"] and report["integrity"] == ["ok"] and report["rows"]["tickets"] == 1
    assert db._cli(["check"]) == 0


def test_online_backup_is_a_consistent_copy(tmp_path):
    auth.create_user("alice", PW, "staff")
    _ticket(complaint="back me up")
    dest = db.backup(tmp_path / "backups" / "copy.db")
    copy = sqlite3.connect(dest)
    assert copy.execute("SELECT complaint FROM tickets").fetchone()[0] == "back me up"
    assert copy.execute("SELECT username FROM users").fetchone()[0] == "alice"
    assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    copy.close()
    with pytest.raises(FileExistsError):
        db.backup(dest)  # never silently overwrite an existing backup


# ---------- legacy import ----------
def _legacy_files(tmp_path):
    salt = bytes(range(16))
    digest = hashlib.scrypt(PW.encode(), salt=salt, n=2**14, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=32)
    users = tmp_path / "users.json"
    users.write_text(json.dumps({"users": {"olduser": {
        "display_name": "Old User", "role": "staff", "salt": salt.hex(), "hash": digest.hex(),
        "params": {"n": 2**14, "r": 8, "p": 1}, "failed": 0, "locked_until": 0, "created_at": "2026-01-01T00:00:00"}}}), encoding="utf-8")
    tickets = tmp_path / "tickets.csv"
    with tickets.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticket_id", "timestamp", "customer_name", "order_id", "complaint", "category", "urgency", "sentiment",
                    "language", "kb_sources", "response", "confidence", "status", "escalation_reason", "security_flags",
                    "review_status", "final_response", "reviewed_by", "reviewed_at"])
        w.writerow(["TKT-0007", "2026-01-02 10:00:00", "Priya", "ORD-1", "'=1+1 where is it", "Shipping", "Low", "Calm", "English",
                    "shipping_policy.pdf", "Track it", "0.90", "Resolved", "", "", "N/A", "", "", ""])
        w.writerow(["TKT-0008", "2026-01-02 11:00:00", "", "", "angry!", "Billing", "High", "Angry", "English", "",
                    "Sorry", "0.30", "Escalated", "High urgency", "", "Pending", "", "", ""])
        w.writerow(["garbage", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
    aud = tmp_path / "audit.csv"
    aud.write_text("timestamp,actor,role,action,target,detail\r\n2026-01-02 09:00:00,olduser,staff,login_ok,,\r\n", encoding="utf-8")
    return users, tickets, aud


def test_legacy_import_preserves_ids_credentials_and_unescapes_formulas(tmp_path):
    users, tickets, aud = _legacy_files(tmp_path)
    report = legacy_import.import_legacy(users, tickets, aud)
    assert (report["users"], report["tickets"], report["audit"]) == (1, 2, 1)  # the malformed row is skipped
    df = logstore.read_tickets()
    assert list(df["ticket_id"]) == ["TKT-0007", "TKT-0008"] and df.iloc[0]["complaint"] == "=1+1 where is it"
    assert df.iloc[0]["confidence"] == "0.90" and df.iloc[1]["review_status"] == "Pending" and df.iloc[0]["owner"] == ""
    assert auth.authenticate("olduser", PW).role == "staff"  # the old password hash still works
    assert audit.read().iloc[0]["action"] == "login_ok"
    assert logstore.append_ticket({"complaint": "new", "status": "Resolved"})["ticket_id"] == "TKT-0009"  # numbering continues


def test_legacy_import_retires_sources_and_is_idempotent(tmp_path):
    users, tickets, aud = _legacy_files(tmp_path)
    legacy_import.import_legacy(users, tickets, aud)
    assert not users.exists() and Path(str(users) + ".migrated").exists() and Path(str(tickets) + ".migrated").exists()
    again = legacy_import.import_legacy(users, tickets, aud)  # sources are gone: nothing to do
    assert (again["users"], again["tickets"], again["audit"]) == (0, 0, 0) and len(again["skipped_files"]) == 3
    users2, tickets2, _ = _legacy_files(tmp_path)  # same data re-supplied: existing rows are skipped, never duplicated
    assert legacy_import.import_legacy(users2, tickets2, None)["tickets"] == 0
    assert len(logstore.read_tickets()) == 2 and len(auth.list_users()) == 1


def test_failed_import_leaves_sources_untouched(tmp_path):
    users, tickets, aud = _legacy_files(tmp_path)
    users.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        legacy_import.import_legacy(users, tickets, aud)
    assert tickets.exists() and aud.exists() and logstore.read_tickets().empty  # all-or-nothing


def test_many_threads_opening_a_brand_new_database_at_once(tmp_path):
    """Regression: the one-time WAL switch used to fail with 'database is locked' under a simultaneous first open."""
    import threading

    errors, ids = [], []
    p = tmp_path / "fresh.db"

    def work():
        try:
            for _ in range(3):
                ids.append(logstore.append_ticket({"complaint": "t", "status": "Resolved"}, p)["ticket_id"])
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
        finally:
            db.close_all()

    threads = [threading.Thread(target=work) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(ids) == 48 and len(set(ids)) == 48


def test_parallel_processes_opening_a_brand_new_database_at_once(tmp_path):
    """Same race across processes: nobody pre-creates the schema."""
    code = (
        "import sys, logstore\n"
        "from pathlib import Path\n"
        "for _ in range(5):\n"
        "    logstore.append_ticket({'complaint': 'p', 'status': 'Resolved'}, Path(sys.argv[1]))\n"
    )
    target = str(tmp_path / "fresh_mp.db")
    procs = [subprocess.Popen([sys.executable, "-c", code, target], cwd=ROOT, env=os.environ.copy(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(5)]
    errors = [p.communicate()[1].decode() for p in procs]
    assert all(p.returncode == 0 for p in procs), errors
    ids = list(logstore.read_tickets(Path(target))["ticket_id"])
    assert len(ids) == 25 and len(set(ids)) == 25


def test_upgrade_from_schema_v1_preserves_data_and_adds_reset_support(tmp_path):
    """An existing v1 database (accounts, tickets, audit) must upgrade in place without losing anything."""
    p = tmp_path / "v1.db"
    raw = sqlite3.connect(p)
    for stmt in db.MIGRATIONS[1]:
        raw.execute(stmt)
    raw.execute("PRAGMA user_version = 1")
    raw.execute("INSERT INTO users (username, display_name, role, salt, hash, params, created_at) VALUES ('old.user', 'Old', 'staff', 'aa', 'bb', '{}', 'then')")
    raw.execute("INSERT INTO tickets (timestamp, owner, complaint, status, review_status) VALUES ('t', 'old.user', 'kept', 'Escalated', 'Pending')")
    raw.execute("INSERT INTO audit (timestamp, actor, action) VALUES ('t', 'old.user', 'login_ok')")
    raw.commit()
    raw.close()

    c = db.connect(p)
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert c.execute("SELECT citations FROM tickets").fetchone()[0] == ""  # v3 column added, existing rows defaulted
    user = c.execute("SELECT * FROM users WHERE username = 'old.user'").fetchone()
    assert user["email"] is None and user["pw_version"] == 0 and user["hash"] == "bb"  # untouched, new columns defaulted
    assert c.execute("SELECT complaint FROM tickets").fetchone()[0] == "kept"
    assert c.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1
    c.execute("UPDATE users SET email = 'old@example.com' WHERE username = 'old.user'")  # new column is usable
    assert db.check(p)["ok"]
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO password_resets (token_hash, username, created_at, expires_at) VALUES ('h', 'nobody', 0, 1)")  # FK enforced


def test_email_uniqueness_is_enforced_by_the_database():
    c = db.connect()
    base = "INSERT INTO users (username, display_name, role, salt, hash, params, created_at, email) VALUES (?, 'x', 'staff', 's', 'h', '{}', 'n', ?)"
    c.execute(base, ("user.one", "same@example.com"))
    with pytest.raises(sqlite3.IntegrityError):
        c.execute(base, ("user.two", "same@example.com"))
    c.execute(base, ("user.three", None))
    c.execute(base, ("user.four", None))  # many accounts may have no email
