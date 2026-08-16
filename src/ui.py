"""Shared application chrome — sidebar, header, and the theme toggle.

Kept in one place so the chat page and the four secondary pages
(Dashboard, Conversations, Knowledge Base, Analytics, Settings) render an
identical, consistent shell instead of each reimplementing it slightly
differently. Presentation only — nothing here touches retrieval,
generation, or ingestion.
"""

import streamlit as st

from src.branding import brand_mark, get_theme, set_theme
from src.config import settings
from src.retrieval import collection_count, collection_is_empty

_MODEL_BY_PROVIDER = {
    "anthropic": settings.anthropic_model,
    "openai": settings.openai_model,
    "groq": settings.groq_model,
}

# (nav key, sidebar label, emoji, page path). "chat" has no page path of
# its own here — it's reached via the "New Chat" action, not a plain link,
# since clicking it should also reset the conversation.
NAV_ITEMS: list[tuple[str, str, str, str]] = [
    ("dashboard", "Dashboard", "◧", "pages/2_Dashboard.py"),
    ("conversations", "Conversations", "🗂", "pages/3_Conversations.py"),
    ("knowledge_base", "Knowledge Base", "📚", "pages/1_Knowledge_Base.py"),
    ("analytics", "Analytics", "◔", "pages/4_Analytics.py"),
    ("settings", "Settings", "⚙", "pages/5_Settings.py"),
]


def active_model() -> str:
    return _MODEL_BY_PROVIDER.get(settings.llm_provider, "unknown model")


def _apply_theme_toggle(key: str) -> None:
    """on_change callback — fires only for the specific toggle instance
    the user just clicked, with its fresh post-click value already in
    session_state[key]."""
    set_theme("dark" if st.session_state[key] else "light")


def render_theme_toggle(key: str = "theme_toggle") -> None:
    """Sun/moon-flanked sliding switch. Built on st.toggle rather than a
    hand-rolled control — it already ships a proper animated thumb (a real
    transform-based slide, not a swapped icon), so restyling its track colors
    to the brand palette gets the "smooth sliding switch" the redesign
    calls for without fighting an undocumented third-party DOM to fake one.

    Two independent instances of this exist (header + sidebar, both
    driving the same shared theme), which rules out the seemingly obvious
    "compare this widget's value to the current theme and set_theme() if
    they differ" approach: a st.toggle's `value=` argument is a one-time-
    only default (Streamlit ignores it forever once the key exists in
    session_state), so the *other*, un-clicked instance keeps whatever
    boolean it last had — and on the very next rerun, comparing ITS stale
    value against the just-changed theme looks like another edit, silently
    flipping the theme straight back. Explicitly re-syncing
    session_state[key] from the real theme on every render (instead of
    trusting the widget's own memory) keeps both instances visually
    correct, and driving the actual write through on_change means
    set_theme() only ever runs for the instance the user actually
    touched — an st.toggle's on_change already triggers Streamlit's
    normal rerun, so no manual st.rerun() is needed either."""
    current = get_theme()
    st.session_state[key] = (current == "dark")
    cols = st.columns([1, 2.6, 1], gap="small", vertical_alignment="center")
    with cols[0]:
        st.markdown('<span class="theme-icon sun">☀</span>', unsafe_allow_html=True)
    with cols[1]:
        st.toggle(
            "Dark mode",
            key=key,
            label_visibility="collapsed",
            on_change=_apply_theme_toggle,
            args=(key,),
        )
    with cols[2]:
        st.markdown('<span class="theme-icon moon">☾</span>', unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "", show_status: bool = True) -> None:
    """Left: page title (+ optional subtitle). Right: AI status + theme
    toggle. Rendered as a real Streamlit column row (not raw HTML) so the
    toggle control can live inside it."""
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        st.markdown(
            f"""
            <div class="page-header">
                <div class="page-header-title">{title}</div>
                {f'<div class="page-header-subtitle">{subtitle}</div>' if subtitle else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="page-header-right">', unsafe_allow_html=True)
        cols = st.columns([2, 1], vertical_alignment="center")
        if show_status:
            with cols[0]:
                st.markdown(
                    '<div class="status-pill" style="margin-left:auto;">'
                    '<span class="dot"></span> AI ONLINE</div>',
                    unsafe_allow_html=True,
                )
        with cols[1]:
            render_theme_toggle()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-header-rule"></div>', unsafe_allow_html=True)


def render_sidebar(active: str, kb_empty: bool | None = None) -> None:
    """`active` is one of the NAV_ITEMS keys, or "chat" for the main page.

    `kb_empty`: pass the already-computed value when the caller needs it
    anyway (app.py does, to decide whether to disable chat input) so this
    doesn't run a second collection_is_empty()/collection_count() round
    trip to the vector store on every single rerun just to render the
    "Indexed · N chunks" status line. Pages that don't otherwise need the
    value can omit it and this computes it once, itself.
    """
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="orchis-mark">{brand_mark(34)}</div>
                <div class="brand-text">
                    <div class="brand-name">Orchis</div>
                    <div class="brand-tagline">AI Customer Support</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Duplicated (not just in the header): the chat page lives inside
        # Streamlit's chat-scroll container, which auto-scrolls to the
        # composer on load — anything placed only in the in-content header
        # can end up scrolled out of view before the user ever touches it.
        # The sidebar is a separate, independently-visible region, so the
        # toggle here is the one guaranteed to always be reachable.
        render_theme_toggle(key="theme_toggle_sidebar")

        st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

        if st.button("＋  New Chat", key="nav_new_chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.pending_input = None
            st.switch_page("app.py")

        st.markdown('<div class="nav-group">', unsafe_allow_html=True)
        for key, label, icon, path in NAV_ITEMS:
            if key == active:
                st.markdown(
                    f'<div class="nav-link active">{icon}&nbsp;&nbsp;{label}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.page_link(path, label=f"{icon}  {label}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Knowledge base</div>', unsafe_allow_html=True)
        if kb_empty is None:
            kb_empty = collection_is_empty()
        if kb_empty:
            st.markdown(
                '<div class="kb-mini-status warn"><span class="kb-mini-dot"></span> Not connected</div>',
                unsafe_allow_html=True,
            )
        else:
            count = collection_count()
            st.markdown(
                f'<div class="kb-mini-status"><span class="kb-mini-dot"></span> '
                f'Indexed &middot; {count} chunk{"s" if count != 1 else ""}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sidebar-footer">
                <div class="sidebar-footer-status"><span class="dot"></span> AI SYSTEM ONLINE</div>
                <div class="sidebar-footer-model">{active_model()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
