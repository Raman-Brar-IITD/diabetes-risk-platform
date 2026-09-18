"""One-off helper: generate a bcrypt hash for a new clinician login.

Usage: python scripts/hash_password.py "the-plaintext-password"
Paste the printed hash into app/.streamlit/secrets.toml (or the Streamlit
Cloud Secrets UI) under credentials.usernames.<username>.password.
"""
import sys

import streamlit_authenticator as stauth

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/hash_password.py <plaintext-password>")
        raise SystemExit(1)
    print(stauth.Hasher.hash(sys.argv[1]))
