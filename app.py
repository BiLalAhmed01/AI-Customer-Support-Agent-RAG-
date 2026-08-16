"""Streamlit chat interface for Orchis — AI Customer Support Intelligence.

Presentation layer only. All chat/retrieval/generation logic lives in
src/llm.py and src/retrieval.py and is called here unmodified.
"""

import html
import json
import time
from pathlib import Path

import streamlit as st

from src.branding import FAVICON_DIR, brand_hero, brand_mark, inject_favicon_metadata, inject_theme_css
from src.config import settings
from src.llm import prepare_answer_stream
from src.retrieval import collection_count, collection_is_empty
from src.ui import render_header, render_sidebar

st.set_page_config(
    page_title="Orchis — AI Customer Support Intelligence",
    page_icon=str(FAVICON_DIR / "favicon-32.png"),
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_theme_css()
inject_favicon_metadata("chat")

DOCS_DIR = Path(__file__).parent / "data" / "docs"

# (suggested question, backing file, heading that must actually appear in
# it). A question is only offered if both the file and that section really
# exist in the current knowledge base — never a guess about what Orchis can
# answer. Ordered by priority; get_suggested_questions() takes the first
# N whose grounding checks out, so removing/renaming a doc's sections
# silently drops the suggestions that depended on them.
_CANDIDATE_QUESTIONS = [
    ("What products do you offer?", "company_overview.md", "## Our Products"),
    ("How can I place an order?", "ordering_payments.md", "## How to Place an Order"),
    ("What payment methods do you accept?", "ordering_payments.md", "## Payment Methods"),
    ("How long does shipping take?", "shipping_delivery.md", "## Shipping Options"),
    ("What is your return policy?", "returns_refunds.md", "## Return Policy"),
    ("How do I track my order?", "shipping_delivery.md", "## Order Tracking"),
    ("Can I cancel my order?", "returns_refunds.md", "## Order Cancellations"),
    ("What's your warranty policy?", "account_warranty.md", "## Product Warranty"),
    ("How do I contact support?", "support_contact.md", "## Contacting Support"),
]


def get_suggested_questions(limit: int = 5) -> list[str]:
    suggestions = []
    for question, filename, heading in _CANDIDATE_QUESTIONS:
        doc_path = DOCS_DIR / filename
        if not doc_path.exists():
            continue
        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if heading in content:
            suggestions.append(question)
        if len(suggestions) >= limit:
            break
    return suggestions


def _escape(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def _is_timeout_error(exc: Exception) -> bool:
    """Distinguish a timeout from other failures so the user gets an
    honest, specific message instead of a generic connection error."""
    try:
        import openai
        if isinstance(exc, openai.APITimeoutError):
            return True
    except ImportError:
        pass
    name = type(exc).__name__.lower()
    return "timeout" in name or "timeout" in str(exc).lower()


def render_message(role: str, text: str, error: bool = False) -> str:
    """Just the bubble — sources/trace/actions are rendered separately
    (and only once, after generation finishes) so they aren't re-sent on
    every incremental streamed-token update."""
    if error:
        body = f'<span class="error-label">Connection issue</span>{_escape(text)}'
    else:
        body = _escape(text)

    bubble_class = "msg-bubble error" if error else "msg-bubble"

    if role == "user":
        return f'<div class="msg-row user"><div class="msg-bubble">{body}</div></div>'

    return (
        f'<div class="msg-row assistant"><div class="msg-inner">'
        f'<div class="avatar">{brand_mark(26)}</div>'
        f'<div class="{bubble_class}">{body}</div>'
        f'</div></div>'
    )


def render_processing_state(label: str) -> str:
    """A real, honest processing indicator — only ever shown while the
    pipeline stage it names is actually running (see the two call sites
    below), never as decoration split across an already-finished answer."""
    return (
        '<div class="msg-row assistant"><div class="msg-inner">'
        f'<div class="avatar">{brand_mark(26)}</div>'
        f'<div class="msg-bubble processing">'
        f'<span class="processing-icon">✦</span> {html.escape(label)}'
        '<span class="typing-dots"><span></span><span></span><span></span></span>'
        '</div></div></div>'
    )


def render_sources_toggle(sources: list[str], candidates: list[dict] | None) -> str:
    """Compact "Sources · N" pill that expands to the real retrieved
    passages. Falls back to a plain filename list if per-chunk debug data
    isn't available for some reason — never fabricates a section, page,
    or score that wasn't actually returned by retrieval."""
    if not sources:
        return ""
    kept = [c for c in (candidates or []) if c.get("kept")]
    if kept:
        items = "".join(
            f'<div class="source-item"><span class="doc-icon-tiny">📄</span>'
            f'{html.escape(c["source"])}'
            f'<span class="source-score">match {c["rerank_score"]:.1f}</span></div>'
            for c in kept
        )
        count = len(kept)  # matches the passages actually listed below
    else:
        items = "".join(
            f'<div class="source-item"><span class="doc-icon-tiny">📄</span>{html.escape(s)}</div>'
            for s in sources
        )
        count = len(sources)
    # The accordion-body/-inner wrappers are what let style.css animate
    # this open/closed with a CSS grid-rows transition instead of the
    # browser's instant display:none <-> block snap — see the "Accordion
    # motion" section of style.css for why that needs this exact
    # structure (an always-present grid track to transition, clipped by
    # a separate overflow:hidden layer).
    return (
        f'<details class="sources-toggle"><summary>Sources &middot; {count}</summary>'
        f'<div class="accordion-body"><div class="accordion-body-inner">'
        f'<div class="sources-list">{items}</div>'
        f'</div></div></details>'
    )


def render_rag_trace(used_retrieval: bool, candidates: list[dict] | None) -> str:
    """Collapsed-by-default "how this was found" panel. Every value in it
    is read from the real RetrievalDebug the pipeline already produced —
    if a step didn't genuinely run (e.g. a greeting skipped search
    entirely), it's shown as skipped rather than faked as completed."""
    steps = [
        ("Understanding request", True),
        ("Knowledge search", used_retrieval),
        ("Relevant sources", used_retrieval and bool(candidates) and any(c.get("kept") for c in candidates)),
        ("Grounded response", True),
    ]
    step_html = ""
    for i, (label, active) in enumerate(steps):
        cls = "rag-step" if active else "rag-step skipped"
        step_html += f'<div class="{cls}"><span class="rag-dot"></span>{html.escape(label)}</div>'
        if i < len(steps) - 1:
            step_html += '<div class="rag-arrow">↓</div>'

    kept = [c for c in (candidates or []) if c.get("kept")]
    if not used_retrieval:
        detail = (
            '<div class="rag-chunk"><div class="rag-chunk-preview">'
            "This was a general reply — no knowledge search was needed for it."
            "</div></div>"
        )
    elif not kept:
        detail = (
            '<div class="rag-chunk"><div class="rag-chunk-preview">'
            "No knowledge-base passages cleared the relevance bar for this question."
            "</div></div>"
        )
    else:
        detail = "".join(
            f'<div class="rag-chunk">'
            f'<div class="rag-chunk-head"><span>{html.escape(c["source"])}</span>'
            f'<span class="rag-chunk-score">relevance {c["rerank_score"]:.2f}</span></div>'
            f'<div class="rag-chunk-preview">{html.escape(c["preview"])}&hellip;</div>'
            f'</div>'
            for c in kept
        )

    return (
        '<details class="rag-trace"><summary>How Orchis found this answer</summary>'
        f'<div class="accordion-body"><div class="accordion-body-inner">'
        f'<div class="rag-pipeline">{step_html}{detail}</div>'
        f'</div></div></details>'
    )


def render_copy_action(text: str) -> str:
    js_text = json.dumps(text)
    return (
        '<div class="msg-actions" style="margin-left: 40px;">'
        f'<button title="Copy response" onclick="navigator.clipboard.writeText({js_text})">⧉ Copy</button>'
        '</div>'
    )


def render_feedback_row(index: int, message: dict) -> None:
    """Regenerate re-runs the real pipeline for the user turn that
    produced this reply (via the same pending_input mechanism the
    suggestion chips use) — genuine, not a fake reload. Helpful/Not
    helpful are local session-state acknowledgment only; nothing is
    submitted anywhere, and the UI never claims otherwise."""
    cols = st.columns([1.5, 1, 1.4, 6], gap="small")
    with cols[0]:
        if st.button("↻ Regenerate", key=f"regen_{index}", help="Ask Orchis to answer again"):
            prior_user = next(
                (m["content"] for m in reversed(st.session_state.messages[:index]) if m["role"] == "user"),
                None,
            )
            if prior_user:
                st.session_state.messages = st.session_state.messages[:index]
                st.session_state.pending_input = prior_user
                st.rerun()
    with cols[1]:
        # Selected state is the real kind="primary" attribute (a filled
        # pill), not a swapped-in checkmark glyph — the label stays "👍"
        # either way, so the two states read as "same control, toggled"
        # rather than as two different-looking buttons.
        up_active = message.get("feedback") == "up"
        if st.button("👍", key=f"up_{index}", help="Helpful", type="primary" if up_active else "secondary"):
            message["feedback"] = None if up_active else "up"
            st.rerun()
    with cols[2]:
        down_active = message.get("feedback") == "down"
        if st.button("👎", key=f"down_{index}", help="Not helpful", type="primary" if down_active else "secondary"):
            message["feedback"] = None if down_active else "down"
            st.rerun()


def render_debug_expander(debug_info, key: str) -> None:
    """Verbose dev-only diagnostics — gated behind DEBUG_MODE and rendered
    nowhere near the normal chat flow. Never shown unless explicitly
    enabled via .env; end users never see this."""
    with st.expander("Agent debug (verbose, dev-only)", expanded=False):
        st.markdown(
            f"**Intent:** `{debug_info.intent}` · "
            f"**Retrieval used:** {debug_info.used_retrieval}"
        )
        st.markdown(f"**Raw query:** `{debug_info.raw_query}`")
        st.markdown(f"**Normalized query:** `{debug_info.normalized_query}`")
        st.markdown(
            f"**Search query:** `{debug_info.search_query}`"
            + (" _(expanded from conversation history)_" if debug_info.query_expanded_from_history else "")
        )
        st.markdown(
            f"**Candidates fetched:** {debug_info.candidates_fetched} · "
            f"**Kept after rerank:** {debug_info.candidates_kept}"
        )
        if debug_info.candidates:
            st.dataframe(debug_info.candidates, use_container_width=True, hide_index=True, key=key)


# The very first call into src/retrieval.py on a cold process loads the
# local embedding model synchronously (a real, ~15-20s one-time cost —
# see get_embeddings() in src/embeddings.py) before anything below can
# know whether the knowledge base is even populated. Streamlit streams
# markup to the browser as the script executes rather than only at the
# end, so rendering this spinner *before* that first call means the user
# sees "Starting Orchis..." immediately instead of a blank tab — the
# model load itself isn't made any faster, but the perceived wait is a
# spinner instead of nothing.
if "_kb_warm" not in st.session_state:
    with st.spinner("Starting Orchis..."):
        kb_empty = collection_is_empty()
    st.session_state._kb_warm = True
else:
    kb_empty = collection_is_empty()

render_sidebar(active="chat", kb_empty=kb_empty)

render_header("Orchis", "AI Customer Support Intelligence")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# Resolve a starter-prompt click BEFORE deciding whether to show the empty
# state — otherwise the empty state (still true at the top of this same
# script run) renders once more alongside the message it's about to submit.
incoming_input = st.session_state.pending_input
st.session_state.pending_input = None

# ---------- Knowledge-base status ----------
if kb_empty:
    st.markdown(
        """
        <div class="kb-status-card warn">
            <span class="kb-dot"></span>
            Knowledge base is empty — run <code>python -m src.ingest</code> after
            adding documents to <code>data/docs/</code> before asking questions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Empty conversation state ----------
if not st.session_state.messages and not incoming_input and not kb_empty:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="orchis-hero">{brand_hero(72)}</div>
            <div class="empty-brand">Orchis</div>
            <div class="empty-tagline">AI Customer Support Intelligence</div>
            <h2>How can I help you today?</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = get_suggested_questions()
    if suggestions:
        st.markdown('<div class="suggestion-row">', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, prompt in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(prompt, key=f"starter_{i}", use_container_width=True):
                    st.session_state.pending_input = prompt
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Render existing conversation ----------
for i, message in enumerate(st.session_state.messages):
    st.markdown(render_message(message["role"], message["content"], message.get("error")), unsafe_allow_html=True)
    if message["role"] == "assistant" and not message.get("error"):
        candidates = message.get("debug_candidates")
        st.markdown(
            render_sources_toggle(message.get("sources") or [], candidates)
            + render_rag_trace(message.get("used_retrieval", False), candidates),
            unsafe_allow_html=True,
        )
        st.markdown(render_copy_action(message["content"]), unsafe_allow_html=True)
        render_feedback_row(i, message)
        if settings.debug_mode and message.get("raw_debug") is not None:
            render_debug_expander(message["raw_debug"], key=f"debug_{i}")

# ---------- Handle new input ----------
user_input = st.chat_input(
    "Ask a question...", disabled=kb_empty
) or incoming_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(render_message("user", user_input), unsafe_allow_html=True)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    placeholder = st.empty()
    placeholder.markdown(render_processing_state("Understanding your question"), unsafe_allow_html=True)

    debug_info = None
    is_error = False
    sources: list[str] = []
    full_text = ""

    try:
        sources, grounded, token_stream, debug_info = prepare_answer_stream(
            user_input, history=history
        )
        placeholder.markdown(render_processing_state("Preparing your response"), unsafe_allow_html=True)
        try:
            # Throttled to ~20 redraws/sec instead of one per token. Groq's
            # llama-3.3-70b routinely streams well past that rate, and
            # every placeholder.markdown() call is a full websocket
            # round-trip + DOM patch — pushing one for every single token
            # was doing several times more render work than the eye can
            # actually perceive, and was a real source of the reported
            # streaming jank. The remaining text always gets flushed once
            # after the loop, so no token is ever dropped from the view.
            _MIN_REDRAW_INTERVAL = 0.05
            last_redraw = 0.0
            for token in token_stream:
                full_text += token
                now = time.monotonic()
                if now - last_redraw >= _MIN_REDRAW_INTERVAL:
                    # Sources/trace/actions are deliberately withheld until
                    # streaming finishes — text appears progressively first,
                    # then the rest lands once complete.
                    placeholder.markdown(
                        render_message("assistant", full_text), unsafe_allow_html=True
                    )
                    last_redraw = now
        except Exception:
            if full_text.strip():
                # The connection dropped partway through, but real content
                # already reached the user — keep it rather than discarding
                # a partial answer behind a generic error screen.
                full_text += (
                    "\n\n_Response interrupted — the connection was lost "
                    "partway through. You can ask again for the rest._"
                )
            else:
                raise  # nothing usable was generated; let the outer handler report it
    except Exception as exc:
        is_error = True
        if _is_timeout_error(exc):
            full_text = "That took too long to respond and was stopped. Please try again."
        else:
            # A clean, generic message for end users — the raw provider
            # exception (e.g. a 401 with the auth error body) is an
            # implementation detail, not something to surface in the chat.
            # Still visible in debug mode, where developers need it.
            full_text = "I couldn't reach the language model backend right now. Please try again in a moment."
            if settings.debug_mode:
                full_text += f"\n\n_Debug detail: {exc}_"
        sources = []
        placeholder.markdown(render_message("assistant", full_text, error=True), unsafe_allow_html=True)

    if not is_error and not full_text.strip():
        full_text = "Orchis didn't return a response for that. Could you try rephrasing your question?"

    used_retrieval = bool(debug_info and debug_info.used_retrieval)
    candidates = debug_info.candidates if debug_info else None

    if not is_error:
        placeholder.markdown(render_message("assistant", full_text), unsafe_allow_html=True)
        st.markdown(
            render_sources_toggle(sources, candidates) + render_rag_trace(used_retrieval, candidates),
            unsafe_allow_html=True,
        )
        st.markdown(render_copy_action(full_text), unsafe_allow_html=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "sources": sources,
            "error": is_error,
            "used_retrieval": used_retrieval,
            "debug_candidates": candidates,
            "raw_debug": debug_info if settings.debug_mode else None,
            "feedback": None,
        }
    )
    st.rerun()
