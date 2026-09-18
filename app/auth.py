"""Clinician login for the worklist tab only — public assessment stays open.

Credentials and cookie settings come from st.secrets, never from a committed
file. See app/.streamlit/secrets.toml.example for the expected shape, and
scripts/hash_password.py for generating a new clinician login's hash.
"""
import streamlit as st
import streamlit_authenticator as stauth


def get_authenticator():
    # st.secrets values are read-only AttrDicts, even nested ones — the
    # authenticator library mutates the credentials dict in place (e.g. while
    # hashing), so a shallow dict() copy isn't enough; to_dict() deep-copies
    # into plain dicts.
    credentials = st.secrets["credentials"].to_dict()
    cookie = st.secrets["cookie"]
    return stauth.Authenticate(
        credentials,
        cookie["name"],
        cookie["key"],
        cookie.get("expiry_days", 30),
    )
