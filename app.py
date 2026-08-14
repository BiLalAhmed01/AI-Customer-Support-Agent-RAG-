"""Streamlit chat interface for the RAG customer support agent."""

import html
from pathlib import Path

import streamlit as st

from src.config import settings
from src.llm import prepare_answer_stream
from src.retrieval import collection_count, collection_is_empty

st.set_page_config(
    page_title="Support AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)


def _inject_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


_inject_css()

_MODEL_BY_PROVIDER = {
    "anthropic": settings.anthropic_model,
    "openai": settings.openai_model,
    "groq": settings.groq_model,
}
_active_model = _MODEL_BY_PROVIDER.get(settings.llm_provider, "unknown model")


def _escape(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def render_message(role: str, text: str, sources: list[str] | None = None, typing: bool = False) -> str:
    body = (
        '<div class="typing-dots"><span></span><span></span><span></span></div>'
        if typing
        else _escape(text)
    )

    sources_html = ""
    if sources:
        pills = "".join(f'<span class="source-pill">📄 {html.escape(s)}</span>' for s in sources)
        sources_html = f'<div class="sources-row">{pills}</div>'

    if role == "user":
        return f'<div class="msg-row user"><div class="msg-bubble">{body}</div></div>'

    return (
        f'<div class="msg-row assistant"><div class="msg-inner">'
        f'<div class="avatar">✨</div>'
        f'<div class="msg-bubble">{body}{sources_html}</div>'
        f'</div></div>'
    )


# ---------- Hero header ----------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-badge"><span class="dot"></span> SYSTEM ONLINE · {settings.llm_provider.upper()}</div>
        <h1 class="hero-title">Support AI</h1>
        <p class="hero-sub">Answers are grounded in your documentation only — no guessing, no hallucination.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if collection_is_empty():
    st.warning(
        "The knowledge base is empty. Run `python -m src.ingest` after adding "
        "PDF/text files to `data/docs/` before asking questions."
    )

# ---------- Render existing conversation ----------
for message in st.session_state.messages:
    st.markdown(
        render_message(message["role"], message["content"], message.get("sources")),
        unsafe_allow_html=True,
    )

# ---------- Handle new input ----------
if user_input := st.chat_input("Ask a question about our product..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(render_message("user", user_input), unsafe_allow_html=True)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    placeholder = st.empty()
    placeholder.markdown(render_message("assistant", "", typing=True), unsafe_allow_html=True)

    try:
        sources, grounded, token_stream = prepare_answer_stream(user_input, history=history)
        full_text = ""
        for token in token_stream:
            full_text += token
            placeholder.markdown(
                render_message("assistant", full_text, sources=sources),
                unsafe_allow_html=True,
            )
    except Exception as exc:
        full_text = f"Something went wrong while generating a response: {exc}"
        sources = []
        placeholder.markdown(render_message("assistant", full_text), unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_text, "sources": sources}
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Console")

    st.markdown(
        f"""
        <div class="status-card">
            <div class="label">Model</div>
            <div class="value">{html.escape(_active_model)}</div>
        </div>
        <div class="status-card">
            <div class="label">Knowledge base</div>
            <div class="value"><span class="accent">{collection_count()}</span> chunks indexed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "This assistant answers strictly from the documents you've ingested "
        "into the vector store. If it can't find relevant information, it "
        "says so instead of guessing."
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
