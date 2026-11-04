"""Sign-in gate for the Streamlit app. Nothing else in the app renders until require_login() returns a user."""
from __future__ import annotations

import time

import streamlit as st

import audit
import auth
from config import get_settings

SIGNUP_LIMIT_PER_HOUR = 30
RESET_LIMIT_PER_HOUR = 30


@st.cache_resource
def _hourly_log(name: str) -> list[float]:
    return []  # process-wide, so opening new browser sessions does not bypass a limit


def _allowed(name: str, limit: int) -> bool:
    now = time.time()
    times = _hourly_log(name)
    times[:] = [t for t in times if now - t < 3600]
    if len(times) >= limit:
        return False
    times.append(now)
    return True


def current_user() -> auth.User | None:
    """The signed-in user, or None. Expires the session after the idle timeout or when the password changed elsewhere."""
    user = auth.User.from_session(st.session_state.get("user"))
    if user is None:
        return None
    timeout = get_settings().session_timeout_min * 60
    last = st.session_state.get("last_active", 0)
    if timeout and time.time() - last > timeout:
        audit.record(user.username, user.role, "session_expired")
        st.session_state.clear()
        st.session_state["notice"] = "You were signed out after a period of inactivity."
        return None
    if auth.session_version(user.username) != st.session_state.get("pwv"):
        # the password was changed or reset (possibly from another device): this session is no longer trusted
        audit.record(user.username, user.role, "session_revoked")
        st.session_state.clear()
        st.session_state["notice"] = "Your password was changed, so you were signed out. Please sign in again."
        return None
    st.session_state["last_active"] = time.time()
    return user


def sign_in_session(user: auth.User) -> None:
    st.session_state.clear()  # never inherit anything from a previous user in this browser session
    st.session_state["user"] = user.as_session()
    st.session_state["last_active"] = time.time()
    st.session_state["pwv"] = auth.session_version(user.username)


def sign_out() -> None:
    user = auth.User.from_session(st.session_state.get("user"))
    if user:
        audit.record(user.username, user.role, "logout")
    st.session_state.clear()


def _reset_page(token: str) -> None:
    """Shown when the app is opened from an emailed reset link (?reset=TOKEN)."""
    st.title("Choose a new password", icon=":material/lock_reset:")
    if not auth.peek_reset(token):
        st.error(auth.RESET_INVALID, icon=":material/link_off:")
        if st.button("Back to sign in", icon=":material/arrow_back:"):
            st.query_params.clear()
            st.rerun()
        return
    with st.form("reset"):
        pw1 = st.text_input("New password", type="password", autocomplete="new-password", help=f"At least {auth.MIN_PASSWORD} characters")
        pw2 = st.text_input("Repeat new password", type="password", autocomplete="new-password")
        go = st.form_submit_button("Set new password", type="primary", width="stretch")
    if go:
        try:
            if pw1 != pw2:
                raise ValueError("The two passwords do not match.")
            auth.complete_reset(token, pw1)
        except ValueError as exc:
            st.error(str(exc), icon=":material/error:")
        else:
            st.query_params.clear()  # the token must not linger in the address bar or history
            st.session_state.clear()
            st.session_state["notice"] = "Your password was changed. Sign in with your new password."
            st.rerun()
    st.caption("This link works once and expires shortly.")


def require_login() -> auth.User:
    user = current_user()
    if user:
        return user
    _, centre, _ = st.columns([1, 1.4, 1])
    with centre:
        st.space("large")
        token = st.query_params.get("reset")
        if token:
            _reset_page(str(token))
            st.stop()
        st.title("NovaCart support", icon=":material/support_agent:")
        st.caption("Sign in to continue.")
        if notice := st.session_state.pop("notice", None):
            st.info(notice, icon=":material/info:")
        tab_in, tab_up, tab_forgot = st.tabs(
            [":material/login: Sign in", ":material/person_add: Create customer account", ":material/lock_reset: Forgot password"]
        )
        with tab_in:
            with st.form("signin"):
                username = st.text_input("Username", autocomplete="username")
                password = st.text_input("Password", type="password", autocomplete="current-password")
                go = st.form_submit_button("Sign in", type="primary", width="stretch")
            if go:
                found = auth.authenticate(username, password)
                if found:
                    sign_in_session(found)
                    st.rerun()
                time.sleep(0.6)  # slows scripted guessing; identical message for every failure
                st.error(auth.GENERIC_FAILURE, icon=":material/lock:")
            st.caption("Support staff accounts are issued by an administrator.")
        with tab_up:
            with st.form("signup"):
                name = st.text_input("Your name")
                new_user = st.text_input("Choose a username", help="3-32 characters: letters, digits, dot, dash, underscore")
                email = st.text_input("Email address", help="Used only to reset your password", autocomplete="email")
                pw1 = st.text_input("Password", type="password", autocomplete="new-password", help=f"At least {auth.MIN_PASSWORD} characters")
                pw2 = st.text_input("Repeat password", type="password", autocomplete="new-password")
                up = st.form_submit_button("Create account", width="stretch")
            if up:
                try:
                    if pw1 != pw2:
                        raise ValueError("The two passwords do not match.")
                    if not _allowed("signup", SIGNUP_LIMIT_PER_HOUR):
                        raise ValueError("Too many sign-ups right now. Please try again later.")
                    created = auth.signup_customer(new_user, pw1, name, email)
                    sign_in_session(created)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc), icon=":material/error:")
        with tab_forgot:
            st.caption("Enter the email address on your account and we will send you a link to choose a new password.")
            with st.form("forgot"):
                addr = st.text_input("Email address", autocomplete="email", key="forgot_email")
                ask = st.form_submit_button("Send reset link", width="stretch")
            if ask:
                if _allowed("reset", RESET_LIMIT_PER_HOUR):
                    auth.request_reset(addr)
                time.sleep(0.4)  # constant-ish response time; the message is identical for every address
                st.success(auth.RESET_ACK, icon=":material/mark_email_read:")
                if get_settings().email_backend == "outbox":
                    st.info("Demo mode: no mail server is configured, so the email was saved to the `data/outbox` folder instead of being sent.",
                            icon=":material/inbox:")
    st.stop()
