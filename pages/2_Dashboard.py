"""Orchis — Dashboard.

A premium SaaS overview built entirely from data the app already has
live access to: real knowledge-base counts (src/retrieval.py), the real
active model configuration (src/config.py), and this session's real
conversation/feedback state. Nothing here is a placeholder metric —
where no real number exists yet (cross-session analytics), that's
reported honestly rather than invented.
"""

import streamlit as st

from src.branding import FAVICON_DIR, inject_favicon_metadata, inject_theme_css
from src.retrieval import collection_count, collection_is_empty, list_indexed_sources
from src.ui import active_model, render_header, render_sidebar

st.set_page_config(
    page_title="Orchis — Dashboard",
    page_icon=str(FAVICON_DIR / "favicon-32.png"),
    layout="centered",
)

inject_theme_css()
inject_favicon_metadata("dashboard")

kb_empty = collection_is_empty()
doc_count = len(list_indexed_sources())

render_sidebar(active="dashboard", kb_empty=kb_empty)
render_header("Dashboard", "Live system overview")
chunk_count = collection_count() if not kb_empty else 0

messages = st.session_state.get("messages", [])
user_turns = sum(1 for m in messages if m["role"] == "user")
assistant_turns = [m for m in messages if m["role"] == "assistant" and not m.get("error")]
helpful = sum(1 for m in assistant_turns if m.get("feedback") == "up")
not_helpful = sum(1 for m in assistant_turns if m.get("feedback") == "down")
grounded_replies = sum(1 for m in assistant_turns if m.get("used_retrieval"))

st.markdown(
    f"""
    <div class="kb-stat-row">
        <div class="kb-stat">
            <div class="kb-stat-label">Knowledge Base</div>
            <div class="kb-stat-value">{"Ready" if not kb_empty else "Empty"}</div>
        </div>
        <div class="kb-stat">
            <div class="kb-stat-label">Indexed Chunks</div>
            <div class="kb-stat-value">{chunk_count}</div>
        </div>
        <div class="kb-stat">
            <div class="kb-stat-label">Documents</div>
            <div class="kb-stat-value">{doc_count}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">This session</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kb-stat-row">
        <div class="kb-stat">
            <div class="kb-stat-label">Messages Sent</div>
            <div class="kb-stat-value">{user_turns}</div>
        </div>
        <div class="kb-stat">
            <div class="kb-stat-label">Grounded Replies</div>
            <div class="kb-stat-value">{grounded_replies}/{len(assistant_turns)}</div>
        </div>
        <div class="kb-stat">
            <div class="kb-stat-label">Feedback</div>
            <div class="kb-stat-value">👍 {helpful} &nbsp; 👎 {not_helpful}</div>
        </div>
    </div>
    <p class="dashboard-note">Session metrics reset when the browser tab closes —
    conversations aren't persisted to a database yet (see Settings).</p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">System status</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="status-card">
        <div class="label">Generation model</div>
        <div class="value">{active_model()}</div>
    </div>
    <div class="status-card">
        <div class="label">Retrieval</div>
        <div class="value">Cross-encoder reranked &middot; intent-routed</div>
    </div>
    """,
    unsafe_allow_html=True,
)
