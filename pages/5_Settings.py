"""Orchis — Settings.

Read-only view of the real active configuration (src/config.py) — every
value shown here is the one actually in effect for this running process.
API keys are deliberately never read or displayed, not even redacted.
"""

import html

import streamlit as st

from src.branding import FAVICON_DIR, get_theme, inject_favicon_metadata, inject_theme_css
from src.config import settings
from src.ui import active_model, render_header, render_sidebar

st.set_page_config(
    page_title="Orchis — Settings",
    page_icon=str(FAVICON_DIR / "favicon-32.png"),
    layout="centered",
)

inject_theme_css()
inject_favicon_metadata("settings")

render_sidebar(active="settings")
render_header("Settings", "Active configuration (read-only)")

st.markdown('<div class="section-heading">Model</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="status-card">
        <div class="label">Provider</div>
        <div class="value">{html.escape(settings.llm_provider.title())}</div>
    </div>
    <div class="status-card">
        <div class="label">Active model</div>
        <div class="value">{active_model()}</div>
    </div>
    <div class="status-card">
        <div class="label">Embedding model</div>
        <div class="value">{html.escape(settings.embedding_model)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Retrieval</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="status-card">
        <div class="label">Chunks retrieved per query</div>
        <div class="value">{settings.retrieval_k}</div>
    </div>
    <div class="status-card">
        <div class="label">Candidate pool before reranking</div>
        <div class="value">{settings.fetch_k}</div>
    </div>
    <div class="status-card">
        <div class="label">Relevance threshold (cross-encoder)</div>
        <div class="value">{settings.rerank_score_threshold}</div>
    </div>
    <div class="status-card">
        <div class="label">Chunk size / overlap</div>
        <div class="value">{settings.chunk_size} / {settings.chunk_overlap} characters</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Interface</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="status-card">
        <div class="label">Theme</div>
        <div class="value">{get_theme().title()}</div>
    </div>
    <div class="status-card">
        <div class="label">Vector store</div>
        <div class="value">Chroma &middot; collection "{html.escape(settings.collection_name)}"</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="dashboard-note">API keys and other secrets are read from environment '
    "variables at process start and are never displayed here, even partially.</p>",
    unsafe_allow_html=True,
)
