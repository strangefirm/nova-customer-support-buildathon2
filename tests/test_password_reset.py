"""Change password, emailed reset links, the mailer backends and the UI flows."""
import re
import smtplib
import threading
import time

import pytest
from streamlit.testing.v1 import AppTest

import audit
import auth
import db
import mailer
from config import get_settings

from test_app import APP, signed_in

PW = "correct horse battery"
NEW = "a brand new passphrase"
LINK = re.compile(r"\?reset=([A-Za-z0-9_-]+)")


@pytest.fixture
def sent(monkeypatch):
    """Capture outgoing mail instead of sending it."""
    box = []
    monkeypatch.setattr(mailer, "send_async", lambda to, subject, body, settings=None, audit_actor="system":
                        box.append({"to": to, "subject": subject, "body": body}))
    return box


def _token(box):
    return LINK.search(box[-1]["body"]).group(1)


def _user(name="alice", role="customer", email="alice@example.com"):
    return auth.create_user(name, PW, role, name.title(), email=email)


def _dump():
    """Every stored value in the whole database, as one string."""
    c = db.connect()
    parts = []
    for t in ("users", "password_resets", "audit", "login_failures"):
        parts += [" ".join(str(v) for v in r) for r in c.execute(f"SELECT * FROM {t}")]
    return "\n".join(parts)


# ---------- email handling ----------
@pytest.mark.parametrize("bad", ["", "plain", "a@b", "@b.com", "a b@c.com", "a@@b.com", "a@b..com", "x" * 250 + "@b.com", None])
def test_invalid_emails_rejected(bad):
    with pytest.raises(ValueError):
        auth.normalise_email(bad)


def test_email_is_normalised_and_masked():
    assert auth.normalise_email("  Alice.Smith+tag@Example.COM ") == "alice.smith+tag@example.com"
    assert auth.mask_email("alice@example.com") == "a****@example.com" and auth.mask_email(None) == "not set"


def test_signup_requires_an_email_and_emails_are_unique():
    with pytest.raises(ValueError, match="email"):
        auth.signup_customer("nomail", PW, "N", "")
    auth.signup_customer("first", PW, "F", "Shared@Example.com")
    with pytest.raises(ValueError, match="cannot be used"):
        auth.signup_customer("second", PW, "S", "shared@example.COM")  # case-insensitive uniqueness


def test_staff_can_exist_without_an_email_but_cannot_be_reset_by_mail(sent):
    auth.create_user("boss", PW, "staff")
    auth.request_reset("boss@example.com")
    assert sent == []


# ---------- change password (signed in) ----------
def test_change_password_success(sent):
    u = _user()
    before = auth.session_version("alice")
    version = auth.change_password(u, PW, NEW)
    assert version == before + 1 == auth.session_version("alice")
    assert auth.authenticate("alice", NEW) and auth.authenticate("alice", PW) is None
    assert "password_changed" in audit.read()["action"].tolist()
    assert sent[-1]["to"] == "alice@example.com" and "changed" in sent[-1]["subject"]  # the owner is told
    assert NEW not in sent[-1]["body"] and PW not in sent[-1]["body"]


def test_wrong_current_password_is_refused_and_counts_toward_lockout(sent):
    u = _user()
    for _ in range(get_settings().max_failed_logins):
        with pytest.raises(ValueError, match="current password is incorrect"):
            auth.change_password(u, "not the password", NEW)
    with pytest.raises(ValueError):
        auth.change_password(u, PW, NEW)  # locked: even the right current password no longer works
    assert auth.authenticate("alice", PW) is None and sent == []
    actions = audit.read()["action"].tolist()
    assert actions.count("reauth_fail") >= get_settings().max_failed_logins - 1  # change-password attempts are labelled re-authentication
    assert actions.count("login_fail") == 1  # only the one explicit sign-in attempt above


@pytest.mark.parametrize("weak", ["short", "password123", "alice-alice-1"])
def test_weak_new_password_rejected_without_burning_attempts(weak):
    u = _user()
    with pytest.raises(ValueError):
        auth.change_password(u, PW, weak)
    assert auth.authenticate("alice", PW) and not auth.is_locked("alice")


def test_new_password_must_differ_from_current():
    u = _user()
    with pytest.raises(ValueError, match="different"):
        auth.change_password(u, PW, PW)


def test_changing_password_invalidates_other_sessions_and_pending_reset_links(sent):
    u = _user()
    auth.request_reset("alice@example.com")
    token = _token(sent)
    assert auth.peek_reset(token)
    auth.change_password(u, PW, NEW)
    assert not auth.peek_reset(token)  # an older reset link cannot override a deliberate change


def test_set_email_needs_password_and_notifies_old_address(sent):
    u = _user()
    with pytest.raises(ValueError):
        auth.set_email(u, "wrong password", "new@example.com")
    assert auth.get_email("alice") == "alice@example.com"
    auth.set_email(u, PW, "New@Example.com")
    assert auth.get_email("alice") == "new@example.com"
    assert sent[-1]["to"] == "alice@example.com" and "email address was changed" in sent[-1]["subject"]


# ---------- requesting a reset ----------
def test_token_is_stored_only_as_a_hash(sent):
    _user()
    auth.request_reset("Alice@Example.com")
    token = _token(sent)
    row = db.connect().execute("SELECT * FROM password_resets").fetchone()
    assert row["token_hash"] != token and len(row["token_hash"]) == 64
    assert token not in _dump()  # nothing in the database (or audit trail) can be used as a reset link
    assert sent[0]["to"] == "alice@example.com" and sent[0]["body"].count("?reset=") == 1
    assert get_settings().app_base_url in sent[0]["body"]


def test_unknown_and_malformed_addresses_behave_identically_and_send_nothing(sent):
    _user()
    results = [auth.request_reset(x) for x in ("nobody@example.com", "not an email", "", "alice@example.com.evil.com")]
    assert results == [None] * 4 and sent == []
    assert db.connect().execute("SELECT COUNT(*) FROM password_resets").fetchone()[0] == 0


def test_requests_are_rate_limited_per_account(sent):
    _user()
    for _ in range(6):
        auth.request_reset("alice@example.com")
    assert len(sent) == get_settings().reset_requests_per_hour
    assert "reset_throttled" in audit.read()["action"].tolist()


def test_a_newer_link_cancels_older_ones(sent):
    _user()
    auth.request_reset("alice@example.com")
    first = _token(sent)
    auth.request_reset("alice@example.com")
    second = _token(sent)
    assert first != second and not auth.peek_reset(first) and auth.peek_reset(second)


def test_expired_link_is_rejected(sent):
    _user()
    auth.request_reset("alice@example.com")
    token = _token(sent)
    db.connect().execute("UPDATE password_resets SET expires_at = ?", (time.time() - 1,))
    assert not auth.peek_reset(token)
    with pytest.raises(ValueError, match="invalid or has expired"):
        auth.complete_reset(token, NEW)
    assert auth.authenticate("alice", PW)  # password untouched


# ---------- completing a reset ----------
def test_reset_changes_the_password_once(sent):
    _user()
    auth.request_reset("alice@example.com")
    token = _token(sent)
    before = auth.session_version("alice")
    user = auth.complete_reset(token, NEW)
    assert user.username == "alice" and auth.session_version("alice") == before + 1
    assert auth.authenticate("alice", NEW) and auth.authenticate("alice", PW) is None
    with pytest.raises(ValueError, match="invalid or has expired"):
        auth.complete_reset(token, "another passphrase here")  # single use
    assert auth.authenticate("alice", NEW)
    assert sent[-1]["to"] == "alice@example.com" and "changed" in sent[-1]["subject"]
    trail = audit.read().to_string()
    assert "password_reset_completed" in trail and token not in trail and NEW not in trail


def test_weak_password_does_not_burn_the_link(sent):
    _user()
    auth.request_reset("alice@example.com")
    token = _token(sent)
    with pytest.raises(ValueError, match="at least"):
        auth.complete_reset(token, "short")
    assert auth.peek_reset(token)
    auth.complete_reset(token, NEW)
    assert auth.authenticate("alice", NEW)


def test_reset_unlocks_a_locked_account(sent):
    _user()
    for _ in range(get_settings().max_failed_logins):
        auth.authenticate("alice", "wrong password!")
    assert auth.is_locked("alice")
    auth.request_reset("alice@example.com")
    auth.complete_reset(_token(sent), NEW)
    assert not auth.is_locked("alice") and auth.authenticate("alice", NEW)


@pytest.mark.parametrize("bad", ["", "   ", "x", "a" * 5000, "' OR '1'='1", "../../etc/passwd", None])
def test_garbage_tokens_are_rejected(sent, bad):
    _user()
    auth.request_reset("alice@example.com")
    assert not auth.peek_reset(bad)
    with pytest.raises(ValueError, match="invalid or has expired"):
        auth.complete_reset(bad, NEW)
    assert auth.authenticate("alice", PW)


def test_a_tampered_token_is_rejected(sent):
    _user()
    auth.request_reset("alice@example.com")
    token = _token(sent)
    flipped = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(ValueError):
        auth.complete_reset(flipped, NEW)
    assert auth.peek_reset(token)


def test_concurrent_use_of_one_link_succeeds_exactly_once(sent):
    _user()
    auth.request_reset("alice@example.com")
    token = _token(sent)
    results, gate = [], threading.Barrier(8)

    def use(i):
        gate.wait()
        try:
            auth.complete_reset(token, f"concurrent passphrase {i}")
            results.append("ok")
        except ValueError:
            results.append("refused")
        finally:
            db.close_all()

    threads = [threading.Thread(target=use, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") == 1 and results.count("refused") == 7


def test_deleting_a_user_removes_their_reset_links(sent):
    _user()
    auth.request_reset("alice@example.com")
    db.connect().execute("DELETE FROM users WHERE username = 'alice'")
    assert db.connect().execute("SELECT COUNT(*) FROM password_resets").fetchone()[0] == 0


# ---------- mailer ----------
def _settings(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return get_settings()


def test_outbox_backend_writes_the_message_to_disk(monkeypatch, tmp_path):
    s = _settings(monkeypatch, EMAIL_BACKEND="outbox", OUTBOX_DIR=tmp_path / "out")
    mailer.send_email("alice@example.com", "Hello", "Body with link", s)
    files = list((tmp_path / "out").glob("*.eml"))
    assert len(files) == 1 and "alice@example.com" not in files[0].name  # the filename does not leak the address
    text = files[0].read_text(encoding="utf-8")
    assert "To: alice@example.com" in text and "Body with link" in text


def test_header_injection_is_blocked(monkeypatch, tmp_path):
    s = _settings(monkeypatch, EMAIL_BACKEND="outbox", OUTBOX_DIR=tmp_path)
    with pytest.raises(ValueError):
        mailer.send_email("a@example.com\nBcc: attacker@evil.com", "S", "b", s)
    with pytest.raises(ValueError):
        mailer.send_email("a@example.com", "S\r\nBcc: attacker@evil.com", "b", s)
    assert not list(tmp_path.iterdir())


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.calls = host, port, []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg):
        self.calls.append(("send", msg["To"], msg["Subject"]))


def test_smtp_backend_uses_starttls_and_login(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    s = _settings(monkeypatch, EMAIL_BACKEND="smtp", SMTP_HOST="mail.example.com", SMTP_PORT=587, SMTP_USER="u", SMTP_PASSWORD="p")
    mailer.send_email("alice@example.com", "Subj", "Body", s)
    assert FakeSMTP.instances[0].calls == ["starttls", ("login", "u", "p"), ("send", "alice@example.com", "Subj")]


def test_port_465_uses_implicit_tls(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    s = _settings(monkeypatch, EMAIL_BACKEND="smtp", SMTP_HOST="mail.example.com", SMTP_PORT=465)
    mailer.send_email("alice@example.com", "Subj", "Body", s)
    assert "starttls" not in FakeSMTP.instances[0].calls and ("send", "alice@example.com", "Subj") in FakeSMTP.instances[0].calls


def test_smtp_without_host_fails_loudly(monkeypatch):
    s = _settings(monkeypatch, EMAIL_BACKEND="smtp")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    with pytest.raises(RuntimeError):
        mailer.send_email("a@example.com", "S", "b", get_settings())
    assert s.email_backend == "smtp"


def test_off_backend_sends_nothing(monkeypatch, tmp_path):
    s = _settings(monkeypatch, EMAIL_BACKEND="off", OUTBOX_DIR=tmp_path)
    mailer.send_email("a@example.com", "S", "b", s)
    assert not list(tmp_path.iterdir())


def test_delivery_failure_is_audited_but_never_reaches_the_caller(monkeypatch):
    class Boom(FakeSMTP):
        def send_message(self, msg):
            raise ConnectionRefusedError("mail server down")

    monkeypatch.setattr(smtplib, "SMTP", Boom)
    s = _settings(monkeypatch, EMAIL_BACKEND="smtp", SMTP_HOST="mail.example.com", SMTP_PORT=587)
    mailer.send_async("alice@example.com", "S", "b", s).join(5)  # no exception escapes
    text = audit.read().to_string()
    assert "email_failed" in text and "ConnectionRefusedError" in text and "alice@example.com" not in text


def test_default_backend_is_outbox_without_smtp_and_smtp_with_it(monkeypatch):
    monkeypatch.delenv("EMAIL_BACKEND", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert get_settings().email_backend == "outbox"
    monkeypatch.setenv("SMTP_HOST", "mail.example.com")
    assert get_settings().email_backend == "smtp"


def test_full_flow_through_the_real_outbox(monkeypatch, tmp_path):
    """No mocks: request a reset, read the saved email, follow its link."""
    monkeypatch.setenv("EMAIL_BACKEND", "outbox")
    monkeypatch.setenv("OUTBOX_DIR", str(tmp_path / "out"))
    _user()
    auth.request_reset("alice@example.com")
    deadline = time.time() + 5
    while not list((tmp_path / "out").glob("*.eml")) and time.time() < deadline:
        time.sleep(0.05)  # delivery happens on a background thread
    body = next((tmp_path / "out").glob("*.eml")).read_text(encoding="utf-8")
    token = LINK.search(body.replace("=\n", "")).group(1)
    auth.complete_reset(token, NEW)
    assert auth.authenticate("alice", NEW)


# ---------- CLI ----------
def test_cli_set_email_and_masked_listing(capsys):
    auth.create_user("boss", PW, "staff")
    assert auth._cli(["set-email", "--username", "boss", "--email", "Boss@Example.com"]) == 0
    assert auth.get_email("boss") == "boss@example.com"
    auth._cli(["list"])
    out = capsys.readouterr().out
    assert "b***@example.com" in out and "boss@example.com" not in out
    assert auth._cli(["set-email", "--username", "ghost", "--email", "g@example.com"]) == 1


# ---------- the UI ----------
def _all_text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.success] +
                    [e.value for e in at.error] + [i.value for i in at.info] + [t.value for t in at.title])


def test_sign_in_page_offers_forgot_password():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("Forgot password" in t.label for t in at.tabs)


def test_forgot_password_gives_the_same_answer_for_known_and_unknown_emails(sent):
    _user()
    answers = []
    for addr in ("alice@example.com", "nobody@example.com"):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at.text_input(key="forgot_email").set_value(addr)
        next(b for b in at.button if b.label == "Send reset link").click().run()
        assert not at.exception
        answers.append(sorted(s.value for s in at.success))
    assert answers[0] == answers[1] == [auth.RESET_ACK]  # nothing in the page reveals whether the email exists
    assert len(sent) == 1 and sent[0]["to"] == "alice@example.com"  # but only the real account got mail


def test_reset_link_page_sets_a_new_password(sent):
    _user()
    auth.request_reset("alice@example.com")
    token = _token(sent)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["reset"] = token
    at.run()
    assert not at.exception and "Choose a new password" in _all_text(at)
    inputs = list(at.text_input)
    inputs[0].set_value(NEW)
    inputs[1].set_value(NEW)
    next(b for b in at.button if b.label == "Set new password").click().run()
    assert not at.exception and auth.authenticate("alice", NEW) and auth.authenticate("alice", PW) is None
    assert "reset" not in at.query_params  # the token is removed from the address bar
    assert "Your password was changed" in _all_text(at)


def test_reset_link_page_rejects_bad_tokens_and_mismatched_passwords(sent):
    _user()
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["reset"] = "bogus-token"
    at.run()
    assert not at.exception and auth.RESET_INVALID in _all_text(at)
    auth.request_reset("alice@example.com")
    at2 = AppTest.from_file(APP, default_timeout=60)
    at2.query_params["reset"] = _token(sent)
    at2.run()
    inputs = list(at2.text_input)
    inputs[0].set_value(NEW)
    inputs[1].set_value(NEW + "x")
    next(b for b in at2.button if b.label == "Set new password").click().run()
    assert "do not match" in _all_text(at2) and auth.authenticate("alice", PW)


def test_account_tab_is_available_to_staff_and_customers():
    for role in ("staff", "customer"):
        at = signed_in("alice", role).run()
        assert not at.exception and any("Account" in t.label for t in at.tabs)


def test_change_password_through_the_ui_keeps_this_session_and_signs_out_others(sent, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _user()
    mine = AppTest.from_file(APP, default_timeout=60)
    mine.session_state["user"] = {"username": "alice", "display_name": "Alice", "role": "customer"}
    mine.session_state["last_active"] = time.time()
    mine.session_state["pwv"] = auth.session_version("alice")
    mine.run()
    other = AppTest.from_file(APP, default_timeout=60)  # the same account signed in on another device
    other.session_state["user"] = dict(mine.session_state["user"])
    other.session_state["last_active"] = time.time()
    other.session_state["pwv"] = auth.session_version("alice")
    other.run()
    assert not other.exception and any("Sign out" == b.label for b in other.button)

    mine.text_input(key="cp_current").set_value(PW)
    mine.text_input(key="cp_new1").set_value(NEW)
    mine.text_input(key="cp_new2").set_value(NEW)
    next(b for b in mine.button if b.label == "Change password").click().run()
    assert not mine.exception and auth.authenticate("alice", NEW)
    assert any("Password changed" in s.value for s in mine.success)
    mine.run()
    assert any("Sign out" == b.label for b in mine.button)  # still signed in here

    other.run()  # the other device is kicked out
    assert [t.label for t in other.tabs][0].endswith("Sign in") and "signed out" in _all_text(other)
    assert "session_revoked" in audit.read()["action"].tolist()


def test_change_password_ui_rejects_wrong_current_and_mismatch(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    _user()
    at = signed_in("alice", "customer")
    at.session_state["pwv"] = auth.session_version("alice")
    at.run()
    at.text_input(key="cp_current").set_value("wrong password!")
    at.text_input(key="cp_new1").set_value(NEW)
    at.text_input(key="cp_new2").set_value(NEW)
    next(b for b in at.button if b.label == "Change password").click().run()
    assert any("current password is incorrect" in e.value for e in at.error) and auth.authenticate("alice", PW)
    at.text_input(key="cp_current").set_value(PW)
    at.text_input(key="cp_new1").set_value(NEW)
    at.text_input(key="cp_new2").set_value(NEW + "x")
    next(b for b in at.button if b.label == "Change password").click().run()
    assert any("do not match" in e.value for e in at.error) and auth.authenticate("alice", PW)


def test_deleted_account_session_is_signed_out():
    _user()
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["user"] = {"username": "alice", "display_name": "Alice", "role": "customer"}
    at.session_state["last_active"] = time.time()
    at.session_state["pwv"] = auth.session_version("alice")
    at.run()
    assert any("Sign out" == b.label for b in at.button)
    db.connect().execute("DELETE FROM users WHERE username = 'alice'")
    at.run()
    assert [t.label for t in at.tabs][0].endswith("Sign in")
