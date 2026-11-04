"""Account tab shown to every signed-in user: change password and set the email used for resets."""
from __future__ import annotations

import time

import streamlit as st

import auth


def render_account(user: auth.User) -> None:
    st.subheader("Account", icon=":material/manage_accounts:")
    st.caption(f"Signed in as **{user.username}**")

    with st.container(border=True):
        st.markdown("**Change password**")
        with st.form("change_password", clear_on_submit=True):
            current = st.text_input("Current password", type="password", autocomplete="current-password", key="cp_current")
            new1 = st.text_input("New password", type="password", autocomplete="new-password", help=f"At least {auth.MIN_PASSWORD} characters", key="cp_new1")
            new2 = st.text_input("Repeat new password", type="password", autocomplete="new-password", key="cp_new2")
            go = st.form_submit_button("Change password", type="primary", icon=":material/lock_reset:")
        if go:
            try:
                if new1 != new2:
                    raise ValueError("The two new passwords do not match.")
                version = auth.change_password(user, current, new1)
                st.session_state["pwv"] = version  # this session stays signed in; every other session is signed out
                st.success("Password changed. Your other sessions have been signed out.", icon=":material/check_circle:")
            except ValueError as exc:
                time.sleep(0.4)
                st.error(str(exc), icon=":material/error:")

    with st.container(border=True):
        st.markdown("**Email for password resets**")
        st.caption(f"Current address: {auth.mask_email(auth.get_email(user.username))}")
        with st.form("change_email", clear_on_submit=True):
            addr = st.text_input("New email address", autocomplete="email", key="em_new")
            pw = st.text_input("Current password", type="password", autocomplete="current-password", key="email_pw")
            go = st.form_submit_button("Save email", icon=":material/mail:")
        if go:
            try:
                auth.set_email(user, pw, addr)
                st.success("Email address saved.", icon=":material/check_circle:")
                st.rerun()
            except ValueError as exc:
                time.sleep(0.4)
                st.error(str(exc), icon=":material/error:")
