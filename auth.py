"""Local accounts for the support desk (stored in SQLite).

Security properties:
  * passwords: salted scrypt (n=2^14, r=8, p=1), constant-time comparison, never logged or stored in plain text
  * roles: "customer" (self sign-up) and "staff" (created ONLY by an admin through the CLI, so a
    sign-up form can never mint a privileged account); the database also CHECKs the role value
  * brute force: failed attempts are counted in the database in a sliding window. Each attempt is
    counted BEFORE the password is hashed, inside one transaction, so a burst of parallel guesses
    cannot slip past the limit. Unknown usernames are throttled identically, the failure message is
    generic, and unknown users cost the same hashing work, so accounts cannot be enumerated
  * data isolation: tickets_for_user() filters in SQL, and scope_tickets() re-checks the result
  * default deny: any role other than staff/customer sees no tickets
  * password recovery: an emailed link whose token is single-use, expires, and is stored only as a SHA-256 hash.
    The reply to a reset request is identical whether or not the email exists (no account enumeration), emails
    are sent in the background (no timing difference), and any password change or reset invalidates every other
    session of that account (pw_version)

Admin CLI:  python -m auth create|passwd|unlock|list|seed-demo
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import audit
import db
import logstore
import mailer
from config import Settings, get_settings

ROLES = ("customer", "staff")
_USERNAME = re.compile(r"[a-z0-9][a-z0-9_.-]{2,31}")
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
_MAXMEM = 64 * 1024 * 1024
MIN_PASSWORD, MAX_PASSWORD = 10, 128
_COMMON = {"password123", "1234567890", "qwertyuiop", "letmein123", "welcome123", "password1234", "iloveyou123", "admin12345"}
GENERIC_FAILURE = "Invalid username or password, or the account is temporarily locked."


@dataclass(frozen=True)
class User:
    username: str
    display_name: str
    role: str

    def as_session(self) -> dict:
        return {"username": self.username, "display_name": self.display_name, "role": self.role}

    @staticmethod
    def from_session(d: dict | None) -> "User | None":
        if not d or d.get("role") not in ROLES or not d.get("username"):
            return None
        return User(d["username"], d.get("display_name") or d["username"], d["role"])

    @property
    def is_staff(self) -> bool:
        return self.role == "staff"


# ---------- hashing ----------
def _norm(password: str) -> bytes:
    return unicodedata.normalize("NFKC", password).encode("utf-8")


def _hash(password: str, salt: bytes, params: dict | None = None) -> bytes:
    prm = params or _SCRYPT
    return hashlib.scrypt(_norm(password), salt=salt, n=prm["n"], r=prm["r"], p=prm["p"], maxmem=_MAXMEM, dklen=32)


_DUMMY_SALT = secrets.token_bytes(16)


# ---------- validation ----------
def normalise_username(username: str) -> str:
    u = (username or "").strip().lower()
    if not _USERNAME.fullmatch(u):
        bad = sorted({c for c in u if not re.fullmatch(r"[a-z0-9_.-]", c)})
        hint = f" Not allowed: {' '.join(repr(c) if c == ' ' else c for c in bad)}." if bad else ""
        raise ValueError("Username must be 3-32 characters: letters, digits, dot, dash or underscore, starting with a letter or digit." + hint)
    return u


def validate_password(password: str, username: str = "") -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
    if len(password) > MAX_PASSWORD:
        raise ValueError(f"Password must be at most {MAX_PASSWORD} characters.")
    low = password.lower()
    if low in _COMMON or (username and username.lower() in low):
        raise ValueError("That password is too easy to guess.")
    if len(set(password)) < 5:
        raise ValueError("Password needs more variety.")


_EMAIL = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)
RESET_INVALID = "This reset link is invalid or has expired. Request a new one."
RESET_ACK = "If an account uses that email address, we have sent a password reset link. It expires shortly and can be used once."


def normalise_email(email: str) -> str:
    e = unicodedata.normalize("NFKC", (email or "").strip()).lower()
    if len(e) > 254 or not _EMAIL.fullmatch(e):
        raise ValueError("Enter a valid email address.")
    return e


def mask_email(email: str | None) -> str:
    if not email or "@" not in email:
        return "not set"
    local, domain = email.split("@", 1)
    return f"{local[:1]}{'*' * max(2, len(local) - 1)}@{domain}"


def _clean_display(name: str, fallback: str) -> str:
    name = re.sub(r"[^\w .'\-]", "", unicodedata.normalize("NFKC", name or "")).strip()[:60]
    return name or fallback


def _db(settings: Settings | None):
    return (settings or get_settings()).db_path


# ---------- accounts ----------
def create_user(username: str, password: str, role: str, display_name: str = "", settings: Settings | None = None,
                *, email: str = "") -> User:
    if role not in ROLES:
        raise ValueError("Unknown role.")
    uname = normalise_username(username)
    validate_password(password, uname)
    addr = normalise_email(email) if email and email.strip() else None
    salt = secrets.token_bytes(16)
    digest = _hash(password, salt)
    shown = _clean_display(display_name, uname)
    try:
        with db.transaction(_db(settings)) as conn:
            conn.execute(
                "INSERT INTO users (username, display_name, role, salt, hash, params, created_at, email) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (uname, shown, role, salt.hex(), digest.hex(), json.dumps(_SCRYPT), datetime.now().isoformat(timespec="seconds"), addr),
            )
    except sqlite3.IntegrityError as exc:
        if "email" in str(exc).lower():
            raise ValueError("That email address cannot be used for a new account.") from None
        raise ValueError("That username is already taken.") from None
    return User(uname, shown, role)


def signup_customer(username: str, password: str, display_name: str = "", email: str = "",
                    settings: Settings | None = None) -> User:
    """Self-service sign-up. The role is fixed here; callers cannot choose it. An email is required so the password can be reset."""
    if not (email or "").strip():
        raise ValueError("Enter your email address; it is used to reset your password.")
    user = create_user(username, password, "customer", display_name, settings, email=email)
    audit.record(user.username, "customer", "signup", path=_db(settings))
    return user


def set_password(username: str, password: str, settings: Settings | None = None) -> None:
    uname = normalise_username(username)
    validate_password(password, uname)
    salt = secrets.token_bytes(16)
    digest = _hash(password, salt)
    with db.transaction(_db(settings)) as conn:
        cur = conn.execute("UPDATE users SET salt = ?, hash = ?, params = ?, pw_version = pw_version + 1 WHERE username = ?",
                           (salt.hex(), digest.hex(), json.dumps(_SCRYPT), uname))
        if cur.rowcount != 1:
            raise ValueError("No such user.")
        conn.execute("DELETE FROM login_failures WHERE username = ?", (uname,))
        conn.execute("DELETE FROM password_resets WHERE username = ?", (uname,))


def unlock(username: str, settings: Settings | None = None) -> None:
    uname = normalise_username(username)
    with db.transaction(_db(settings)) as conn:
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (uname,)).fetchone() is None:
            raise ValueError("No such user.")
        conn.execute("DELETE FROM login_failures WHERE username = ?", (uname,))


def list_users(settings: Settings | None = None) -> list[dict]:
    s = settings or get_settings()
    conn = db.connect(s.db_path)
    window = time.time() - s.lockout_minutes * 60
    out = []
    for r in conn.execute("SELECT username, role, display_name, created_at, email FROM users ORDER BY username"):
        n = conn.execute("SELECT COUNT(*) FROM login_failures WHERE username = ? AND ts > ?", (r["username"], window)).fetchone()[0]
        out.append({"username": r["username"], "role": r["role"], "display_name": r["display_name"],
                    "created_at": r["created_at"], "email": r["email"], "locked": n >= s.max_failed_logins})
    return out


# ---------- authentication ----------
def is_locked(username: str, settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    n = db.connect(s.db_path).execute(
        "SELECT COUNT(*) FROM login_failures WHERE username = ? AND ts > ?",
        ((username or "").strip().lower(), time.time() - s.lockout_minutes * 60),
    ).fetchone()[0]
    return n >= s.max_failed_logins


def _reserve_attempt(uname: str, s: Settings) -> tuple[bool, int]:
    """Atomically count this attempt. Returns (allowed, attempts_in_window_including_this_one)."""
    now = time.time()
    with db.transaction(s.db_path) as conn:
        conn.execute("DELETE FROM login_failures WHERE ts < ?", (now - s.lockout_minutes * 60,))  # also bounds table growth
        n = conn.execute("SELECT COUNT(*) FROM login_failures WHERE username = ?", (uname,)).fetchone()[0]
        if n >= s.max_failed_logins:
            return False, n
        conn.execute("INSERT INTO login_failures (username, ts) VALUES (?, ?)", (uname, now))
        return True, n + 1


def authenticate(username: str, password: str, settings: Settings | None = None, *, purpose: str = "login") -> User | None:
    """Return the User on success, else None. Every failure looks the same to the caller.

    purpose labels the audit events ("login" or "reauth" when an already signed-in user re-enters their password).
    """
    s = settings or get_settings()
    uname = (username or "").strip().lower()
    password = password if isinstance(password, str) else ""
    if not _USERNAME.fullmatch(uname) or len(password) > MAX_PASSWORD:
        _hash("x", _DUMMY_SALT)  # equalise work
        audit.record(uname[:32] or "?", "", f"{purpose}_fail", detail="malformed", path=s.db_path)
        return None

    allowed, attempts = _reserve_attempt(uname, s)
    if not allowed:
        _hash(password, _DUMMY_SALT)
        audit.record(uname, "", f"{purpose}_fail", detail="locked", path=s.db_path)
        return None

    rec = db.connect(s.db_path).execute("SELECT * FROM users WHERE username = ?", (uname,)).fetchone()
    if rec is None:
        _hash(password, _DUMMY_SALT)
        audit.record(uname, "", f"{purpose}_fail", detail="bad_credentials", path=s.db_path)
        return None
    ok = hmac.compare_digest(
        _hash(password, bytes.fromhex(rec["salt"]), json.loads(rec["params"])), bytes.fromhex(rec["hash"])
    )
    if ok:
        with db.transaction(s.db_path) as conn:
            conn.execute("DELETE FROM login_failures WHERE username = ?", (uname,))
        audit.record(uname, rec["role"], f"{purpose}_ok", path=s.db_path)
        return User(uname, rec["display_name"], rec["role"])
    audit.record(uname, rec["role"], "lockout" if attempts >= s.max_failed_logins else f"{purpose}_fail", detail="bad_credentials", path=s.db_path)
    return None


# ---------- sessions, password change, email, reset ----------
def session_version(username: str, settings: Settings | None = None) -> int | None:
    """The account's current password version; a session whose stored version differs must be signed out."""
    row = db.connect(_db(settings)).execute("SELECT pw_version FROM users WHERE username = ?", (username,)).fetchone()
    return None if row is None else int(row["pw_version"])


def get_email(username: str, settings: Settings | None = None) -> str | None:
    row = db.connect(_db(settings)).execute("SELECT email FROM users WHERE username = ?", (username,)).fetchone()
    return row["email"] if row else None


def _store_new_password(uname: str, password: str, settings: Settings | None) -> int:
    """Hash and save a new password, sign out every other session, cancel pending reset links, clear lockout."""
    salt = secrets.token_bytes(16)
    digest = _hash(password, salt)
    with db.transaction(_db(settings)) as conn:
        cur = conn.execute("UPDATE users SET salt = ?, hash = ?, params = ?, pw_version = pw_version + 1 WHERE username = ?",
                           (salt.hex(), digest.hex(), json.dumps(_SCRYPT), uname))
        if cur.rowcount != 1:
            raise ValueError("No such user.")
        conn.execute("DELETE FROM login_failures WHERE username = ?", (uname,))
        conn.execute("DELETE FROM password_resets WHERE username = ?", (uname,))
        return int(conn.execute("SELECT pw_version FROM users WHERE username = ?", (uname,)).fetchone()[0])


def _notify(uname: str, subject: str, body: str, settings: Settings | None) -> None:
    addr = get_email(uname, settings)
    if addr:
        mailer.send_async(addr, subject, body, settings or get_settings())


def change_password(user: User, current: str, new: str, settings: Settings | None = None) -> int:
    """Change the signed-in user's password. Returns the new password version (the caller's session must adopt it).

    The current password is re-checked with the same lockout counting as sign-in, so a hijacked, unattended session
    cannot be used to guess it.
    """
    validate_password(new, user.username)
    if authenticate(user.username, current, settings, purpose="reauth") is None:
        raise ValueError("Your current password is incorrect, or the account is temporarily locked.")
    if hmac.compare_digest(_norm(new), _norm(current)):
        raise ValueError("Choose a password different from your current one.")
    version = _store_new_password(user.username, new, settings)
    audit.record(user.username, user.role, "password_changed", path=_db(settings))
    _notify(user.username, "Your NovaCart password was changed",
            "The password for your NovaCart support account was just changed.\n\nIf this was you, no action is needed. "
            "If it was not, reset your password immediately and contact support.\n", settings)
    return version


def set_email(user: User, current: str, new_email: str, settings: Settings | None = None) -> str:
    addr = normalise_email(new_email)
    if authenticate(user.username, current, settings, purpose="reauth") is None:
        raise ValueError("Your current password is incorrect, or the account is temporarily locked.")
    old = get_email(user.username, settings)
    try:
        with db.transaction(_db(settings)) as conn:
            conn.execute("UPDATE users SET email = ? WHERE username = ?", (addr, user.username))
    except sqlite3.IntegrityError:
        raise ValueError("That email address cannot be used.") from None
    audit.record(user.username, user.role, "email_changed", path=_db(settings))
    if old and old != addr:  # tell the previous address, so a hijacker cannot silently redirect resets
        mailer.send_async(old, "Your NovaCart email address was changed",
                          "The email address on your NovaCart support account was just changed.\n\n"
                          "If this was not you, contact support immediately.\n", settings or get_settings())
    return addr


def _token_hash(token: str) -> str:
    return hashlib.sha256((token or "").strip()[:200].encode()).hexdigest()


def request_reset(email: str, settings: Settings | None = None) -> None:
    """Start a password reset. Returns nothing and behaves the same for unknown addresses (no account enumeration)."""
    s = settings or get_settings()
    try:
        addr = normalise_email(email)
    except ValueError:
        return
    conn = db.connect(s.db_path)
    now = time.time()
    row = conn.execute("SELECT username, role FROM users WHERE email = ?", (addr,)).fetchone()
    if row is None:
        audit.record("anonymous", "", "reset_requested", detail="no_match", path=s.db_path)
        return
    uname = row["username"]
    recent = conn.execute("SELECT COUNT(*) FROM password_resets WHERE username = ? AND created_at > ?", (uname, now - 3600)).fetchone()[0]
    if recent >= s.reset_requests_per_hour:
        audit.record(uname, row["role"], "reset_throttled", path=s.db_path)
        return
    token = secrets.token_urlsafe(32)  # 256 bits; only its hash is stored
    with db.transaction(s.db_path) as c:
        c.execute("DELETE FROM password_resets WHERE expires_at < ?", (now - 86400,))
        c.execute("UPDATE password_resets SET used_at = ? WHERE username = ? AND used_at IS NULL", (now, uname))  # newest link wins
        c.execute("INSERT INTO password_resets (token_hash, username, created_at, expires_at) VALUES (?, ?, ?, ?)",
                  (_token_hash(token), uname, now, now + s.reset_token_minutes * 60))
    mailer.send_async(
        addr, "Reset your NovaCart password",
        f"We received a request to reset the password for your NovaCart support account.\n\n"
        f"Open this link to choose a new password (valid for {s.reset_token_minutes} minutes, one use):\n\n"
        f"{s.app_base_url}/?reset={token}\n\n"
        f"If you did not ask for this, ignore this email; your password has not changed.\n", s)
    audit.record(uname, row["role"], "reset_requested", detail=s.email_backend, path=s.db_path)


def peek_reset(token: str, settings: Settings | None = None) -> bool:
    """Is this token currently usable? (Does not consume it.)"""
    row = db.connect(_db(settings)).execute(
        "SELECT 1 FROM password_resets WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?", (_token_hash(token), time.time())).fetchone()
    return row is not None


def complete_reset(token: str, new_password: str, settings: Settings | None = None) -> User:
    """Use a reset token to set a new password. Single-use and atomic; a weak password does not burn the token."""
    s = settings or get_settings()
    th, now = _token_hash(token), time.time()
    row = db.connect(s.db_path).execute(
        "SELECT r.username, u.role, u.display_name FROM password_resets r JOIN users u ON u.username = r.username "
        "WHERE r.token_hash = ? AND r.used_at IS NULL AND r.expires_at > ?", (th, now)).fetchone()
    if row is None:
        audit.record("anonymous", "", "reset_invalid", path=s.db_path)
        raise ValueError(RESET_INVALID)
    uname = row["username"]
    validate_password(new_password, uname)  # raises ValueError; the token stays valid so the user can try again
    with db.transaction(s.db_path) as conn:
        cur = conn.execute("UPDATE password_resets SET used_at = ? WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?", (now, th, now))
        if cur.rowcount != 1:  # lost a race with another use of the same link
            raise ValueError(RESET_INVALID)
    _store_new_password(uname, new_password, s)
    audit.record(uname, row["role"], "password_reset_completed", path=s.db_path)
    _notify(uname, "Your NovaCart password was changed",
            "The password for your NovaCart support account was just reset using an emailed link.\n\n"
            "If this was not you, request another reset and contact support immediately.\n", s)
    return User(uname, row["display_name"], row["role"])


# ---------- data isolation ----------
def tickets_for_user(user: User | None, settings: Settings | None = None, limit: int | None = None) -> pd.DataFrame:
    """The tickets this user may see. Customers' rows are filtered INSIDE the database, so other
    users' tickets are never even loaded into memory; scope_tickets() then re-checks the result."""
    s = settings or get_settings()
    if user is None or user.role not in ROLES:
        return logstore.read_tickets(s.db_path).iloc[0:0]
    df = logstore.read_tickets(s.db_path, owner=None if user.role == "staff" else user.username, limit=limit)
    return scope_tickets(df, user)


def scope_tickets(df: pd.DataFrame, user: User | None) -> pd.DataFrame:
    """The ONLY place that decides which tickets a user may see. Default deny.

    staff    -> all tickets (they work the review queue and must see every escalation)
    customer -> only tickets they own
    anything else (including None) -> nothing
    """
    if user is None or df.empty:
        return df.iloc[0:0]
    if user.role == "staff":
        return df
    if user.role == "customer" and "owner" in df.columns:
        return df[df["owner"] == user.username]
    return df.iloc[0:0]


def require_staff(user: User | None) -> None:
    if user is None or user.role != "staff":
        raise PermissionError("Staff access required.")


# ---------- customer-safe projection ----------
def customer_view(row: "pd.Series | dict") -> dict:
    """What a customer may see about their own ticket: no internal reasons, flags, confidence or unapproved AI drafts."""
    status, review = row["status"], row.get("review_status", "")
    if status == "Resolved":
        label, reply = "Answered", row["response"]
    elif review in ("Approved", "Edited") and row.get("final_response"):
        label, reply = "Answered by our team", row["final_response"]
    else:
        label, reply = "With our support team", ""
    return {"ticket_id": row["ticket_id"], "timestamp": row["timestamp"], "category": row["category"],
            "complaint": row["complaint"], "status_label": label, "reply": reply,
            "citations": row.get("citations", "") if reply and label == "Answered" else ""}


# ---------- CLI ----------
def _cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m auth", description="Manage support desk accounts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="create an account")
    c.add_argument("--username", required=True)
    c.add_argument("--role", choices=ROLES, default="staff")
    c.add_argument("--name", default="")
    c.add_argument("--email", default="", help="needed for emailed password resets")
    e = sub.add_parser("set-email", help="set or change an account's email address")
    e.add_argument("--username", required=True)
    e.add_argument("--email", required=True)
    for n in ("passwd", "unlock"):
        sub.add_parser(n).add_argument("--username", required=True)
    sub.add_parser("list")
    sub.add_parser("seed-demo", help="create one demo staff and one demo customer with random one-time passwords")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "create":
            pw = getpass.getpass("Password: ")
            if pw != getpass.getpass("Repeat password: "):
                print("Passwords do not match.")
                return 1
            u = create_user(a.username, pw, a.role, a.name, email=a.email)
            audit.record("cli", "admin", "create_user", u.username, a.role)
            print(f"Created {u.role} '{u.username}'.")
        elif a.cmd == "set-email":
            addr = normalise_email(a.email)
            with db.transaction() as conn:
                if conn.execute("UPDATE users SET email = ? WHERE username = ?", (addr, normalise_username(a.username))).rowcount != 1:
                    raise ValueError("No such user.")
            audit.record("cli", "admin", "set_email", a.username)
            print("Email updated.")
        elif a.cmd == "passwd":
            set_password(a.username, getpass.getpass("New password: "))
            audit.record("cli", "admin", "set_password", a.username)
            print("Password updated.")
        elif a.cmd == "unlock":
            unlock(a.username)
            audit.record("cli", "admin", "unlock", a.username)
            print("Account unlocked.")
        elif a.cmd == "list":
            for u in list_users():
                print(f"{u['username']:<20} {u['role']:<9} {'LOCKED' if u['locked'] else '':<7} {mask_email(u['email']):<28} {u['display_name']}")
        elif a.cmd == "seed-demo":
            existing = {u["username"] for u in list_users()}
            for uname, role, name in (("staff.demo", "staff", "Demo Staff"), ("customer.demo", "customer", "Demo Customer")):
                if uname in existing:
                    print(f"{uname}: already exists (unchanged)")
                    continue
                pw = secrets.token_urlsafe(12)
                create_user(uname, pw, role, name)
                audit.record("cli", "admin", "create_user", uname, role)
                print(f"{uname}  password: {pw}   (shown once; change it with 'passwd')")
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
