"""Shared application chrome — sidebar, header, and the theme toggle.

Kept in one place so the chat page and the four secondary pages
(Dashboard, Conversations, Knowledge Base, Analytics, Settings) render an
identical, consistent shell instead of each reimplementing it slightly
differently. Presentation only — nothing here touches retrieval,
generation, or ingestion.
"""

import html

import streamlit as st

from src.branding import brand_mark, get_theme, set_theme
from src.config import settings
from src.retrieval import collection_count, collection_is_empty

_MODEL_BY_PROVIDER = {
    "anthropic": settings.anthropic_model,
    "openai": settings.openai_model,
    "groq": settings.groq_model,
}

# (nav key, sidebar label, icon, page path). "chat" has no page path of
# its own here — it's reached via the "New Chat" action, not a plain link,
# since clicking it should also reset the conversation.
#
# Icons are deliberately all plain geometric/symbol glyphs, not emoji —
# 🗂 and 📚 (the two icons this replaced) are full pictographic emoji that
# render in flat color on every modern platform, which clashed hard next
# to ◧/◔/⚙'s monochrome, theme-colored glyphs. One consistent icon
# language reads as considered; a mix of colorful and monochrome icons in
# the same list reads as unfinished regardless of what else is right.
NAV_ITEMS: list[tuple[str, str, str, str]] = [
    ("dashboard", "Dashboard", "◧", "pages/2_Dashboard.py"),
    ("conversations", "Conversations", "▤", "pages/3_Conversations.py"),
    ("knowledge_base", "Knowledge Base", "▦", "pages/1_Knowledge_Base.py"),
    ("analytics", "Analytics", "◔", "pages/4_Analytics.py"),
    ("settings", "Settings", "⚙", "pages/5_Settings.py"),
]


def active_model() -> str:
    """HTML-escaped: every caller drops this straight into an
    unsafe_allow_html block, and the value comes from an environment
    variable (ANTHROPIC_MODEL / OPENAI_MODEL / GROQ_MODEL), not from a
    fixed list — a typo'd or hostile .env shouldn't be able to inject
    markup into every page's sidebar footer."""
    return html.escape(_MODEL_BY_PROVIDER.get(settings.llm_provider, "unknown model"))


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
    toggle control can live inside it.

    `title`/`subtitle` are escaped, so callers pass plain text (`&`, not
    `&amp;`) and never markup."""
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        safe_title = html.escape(title)
        safe_subtitle = html.escape(subtitle)
        st.markdown(
            f"""
            <div class="page-header">
                <div class="page-header-title">{safe_title}</div>
                {f'<div class="page-header-subtitle">{safe_subtitle}</div>' if subtitle else ''}
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

        # A plain ASCII "+" (not the fullwidth "＋") — the fullwidth form is
        # a CJK-width glyph sized for monospace/CJK text and renders
        # noticeably oversized and misaligned next to Inter's proportional
        # Latin glyphs.
        if st.button("+ New Chat", key="nav_new_chat", width="stretch", type="primary"):
            st.session_state.messages = []
            st.session_state.pending_input = None
            st.switch_page("app.py")

        for key, label, icon, path in NAV_ITEMS:
            if key == active:
                # A disabled st.button, not a raw st.markdown div — Streamlit
                # reserves a fixed ~26px layout slot for markdown-content
                # containers regardless of what CSS padding later renders
                # into them, so the div version visually overflowed ~14px
                # into the next nav item's space (confirmed by inspecting
                # the live DOM: the div's own ancestor flex wrapper measured
                # 26px tall while the padded pill inside it rendered at
                # 40px). st.button participates in Streamlit's normal
                # widget sizing instead, which page_link already proved
                # reserves the right amount of space.
                st.button(
                    f"{icon}  {label}",
                    key=f"nav_active_{key}",
                    disabled=True,
                    width="stretch",
                )
            else:
                st.page_link(path, label=f"{icon}  {label}")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Knowledge Base</div>', unsafe_allow_html=True)
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
