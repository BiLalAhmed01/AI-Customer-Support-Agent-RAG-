"""Tests for src/auth.py. Only the disabled-by-default fast path is
covered here — the interactive passcode-prompt flow calls real Streamlit
widgets (st.text_input, st.stop, st.rerun) that need a full Streamlit
AppTest harness to exercise meaningfully, which is out of scope for this
pass. The one thing that must never regress silently is that an unset
access_code is a true no-op — that's what's asserted below.
"""

from types import SimpleNamespace

import src.auth as auth_module


def test_require_access_is_a_no_op_when_access_code_unset(monkeypatch):
    # settings is a frozen dataclass instance — the module-level name is
    # replaced rather than mutating a field on it.
    monkeypatch.setattr(auth_module, "settings", SimpleNamespace(access_code=""))

    # If this touched st.session_state, st.markdown, st.text_input, or
    # st.stop, it would raise (st is not mocked here) — reaching the end
    # of the call with no error is the assertion.
    auth_module.require_access()
