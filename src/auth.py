"""Optional shared-passcode gate.

Disabled by default (settings.access_code is empty — see src/config.py's
APP_ACCESS_CODE) so this doesn't silently lock out any existing
deployment. When set, every page requires the correct passcode once per
browser session before anything else renders.

This is deliberately the simplest possible control: one shared secret,
not per-user accounts, sessions, or password hashing — it exists to close
the "anyone on the internet can upload to and query the knowledge base"
gap found in a security audit of this app, not to provide real multi-user
access control. If that's needed later, this is the one place to replace.
"""

import hmac

import streamlit as st

from src.config import settings


def require_access() -> None:
    """Call once near the top of every page, after st.set_page_config().

    Renders a passcode prompt and calls st.stop() until the correct code
    has been entered for this session; returns immediately (no-op) if
    settings.access_code is empty.
    """
    if not settings.access_code:
        return

    if st.session_state.get("_authenticated"):
        return

    st.markdown("### Access code required")
    code = st.text_input(
        "Enter the access code to continue", type="password", key="_access_code_input"
    )
    if code:
        # Constant-time comparison — a shared secret compared with `==`
        # leaks timing information about how many leading characters
        # matched, which a patient attacker can exploit to recover it
        # character-by-character instead of needing the whole value.
        if hmac.compare_digest(code, settings.access_code):
            st.session_state._authenticated = True
            st.rerun()
        else:
            st.error("Incorrect access code.")

    st.stop()
