"""Orchis — Conversations.

Chat history in this app lives only in Streamlit session state (see
app.py) — there's no database persisting past sessions, so this page
honestly shows exactly one conversation: the current one. It does not
invent a conversation list or history that doesn't exist.
"""

import html

import streamlit as st

from src.auth import require_access
from src.branding import FAVICON_DIR, inject_favicon_metadata, inject_theme_css
from src.ui import render_header, render_sidebar

st.set_page_config(
    page_title="Orchis — Conversations",
    page_icon=str(FAVICON_DIR / "favicon-32.png"),
    layout="centered",
)

inject_theme_css()
inject_favicon_metadata("conversations")
require_access()

render_sidebar(active="conversations")
render_header("Conversations", "Current session")

messages = st.session_state.get("messages", [])

if not messages:
    st.markdown(
        """
        <div class="kb-empty">
            <div class="kb-empty-icon">🗂</div>
            <h3>No conversation yet</h3>
            <p>Start a chat and it will appear here for the rest of this session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    user_turns = sum(1 for m in messages if m["role"] == "user")
    first_question = next((m["content"] for m in messages if m["role"] == "user"), "")
    preview = first_question if len(first_question) <= 72 else first_question[:69] + "…"

    st.markdown(
        f"""
        <div class="doc-card">
            <div class="doc-main">
                <div class="doc-icon">💬</div>
                <div class="doc-info">
                    <div class="doc-name">{html.escape(preview)}</div>
                    <div class="doc-meta">{user_turns} message{'s' if user_turns != 1 else ''} &middot; this session only</div>
                </div>
            </div>
            <div class="doc-side">
                <span class="doc-status indexed">● Active</span>
            </div>
        </div>
        <p class="dashboard-note">Conversation history isn't persisted to a database in this
        build — refreshing the page or closing the tab starts a new one. Multi-conversation
        history would need a persistence layer added to the backend (see Settings).</p>
        """,
        unsafe_allow_html=True,
    )
