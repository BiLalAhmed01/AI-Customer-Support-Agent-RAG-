"""Orchis — Analytics.

Honest by design: this app has no event-logging or metrics-persistence
layer, so there is no real usage/quality data to chart across sessions.
Rather than fabricate numbers, this page says so plainly and states what
would need to exist for real analytics to appear here.
"""

import streamlit as st

from src.auth import require_access
from src.branding import FAVICON_DIR, inject_favicon_metadata, inject_theme_css
from src.ui import render_header, render_sidebar

st.set_page_config(
    page_title="Orchis — Analytics",
    page_icon=str(FAVICON_DIR / "favicon-32.png"),
    layout="centered",
)

inject_theme_css()
inject_favicon_metadata("analytics")
require_access()

render_sidebar(active="analytics")
render_header("Analytics", "Usage & quality trends")

st.markdown(
    """
    <div class="kb-empty">
        <div class="kb-empty-icon">◔</div>
        <h3>No analytics data yet</h3>
        <p>This build doesn't log conversations to a persistent store, so there's nothing
        real to chart here across sessions — showing invented numbers would be worse than
        showing nothing. Per-session totals (messages, grounded replies, feedback) are on
        the Dashboard, since those are genuinely available right now.</p>
    </div>
    <p class="dashboard-note">To power this page for real: add an events table (or
    lightweight logging) that records each turn's intent, retrieval outcome, and feedback,
    then aggregate it here — no change to the retrieval or generation pipeline required.</p>
    """,
    unsafe_allow_html=True,
)
