"""Orchis brand assets and theme tokens — presentation layer only.

Wraps the mark image and design tokens in assets/ so app.py and
pages/*.py can drop them into markup without duplicating file-reading,
base64, or CSS-injection boilerplate. `brand_mark()`/`brand_hero()` both
render the same source mascot artwork (assets/orchis-mark.png) at
different sizes — see their docstrings for why each caches its own
size-specific re-encode instead of just scaling one embed with CSS.
"""

import base64
import json
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ASSETS_DIR = Path(__file__).parent.parent / "assets"
FAVICON_DIR = ASSETS_DIR / "favicon"

# ============================================================
# Theme tokens
#
# Built from 4 fixed ORCHIS brand colors (primary blue / AI cyan / deep
# navy / light blue, exposed verbatim as --color-* below) by deriving
# backgrounds, surfaces, borders, and text from them per theme — not just
# swapping the raw values in as text/background colors directly. Card
# treatment is flat (white/elevated-navy fill + a real hairline border),
# not neumorphic — the shadow-* tokens below are now single soft drop
# shadows rather than the old dual-tone embossed pair, but keep their
# original names so style.css didn't need to change shape, only values.
# Every color-bearing rule in style.css reads these via var(--name), so
# switching themes only ever means swapping which dict gets rendered into
# the injected :root block below.
# ============================================================

# The 4 fixed ORCHIS brand colors, identical in both themes — exposed as
# their own variables (not just baked into the derived tokens below) so
# anything that specifically wants the raw brand color can reach it (e.g.
# the ambient background glow, which blends both directly).
_BRAND_COLORS: dict[str, str] = {
    "color-primary-blue": "#2563EB",
    "color-ai-cyan": "#06B6D4",
    "color-deep-navy": "#0F172A",
    "color-light-blue": "#E0F2FE",
}

LIGHT_THEME: dict[str, str] = {
    **_BRAND_COLORS,

    "bg": "#F8FAFC",
    "surface": "#FFFFFF",
    "surface-tint": "#E0F2FE",       # light-blue wash — hover states, active nav
    "surface-forest": "#FFFFFF",     # sidebar panel — white, distinct from the
                                      # off-white page bg via border only
    "border": "#E2E8F0",
    "border-strong": "#CBD5E1",

    "text-primary": "#0F172A",
    "text-secondary": "#64748B",
    "text-tertiary": "#64748B",

    "accent": "#2563EB",             # primary blue — CTAs, links, selected states
    "accent-strong": "#1D4ED8",      # hover — slightly darker blue
    "accent-on-accent": "#FFFFFF",   # text color on blue-filled buttons
    "accent-tint": "#E0F2FE",
    "accent-tint-strong": "#D5EAF8",
    "ai-accent": "#06B6D4",          # decorative only (glow rings, washes) —
                                      # fails AA/non-text contrast as a light-mode
                                      # foreground, see ai-accent-text below
    "ai-accent-text": "#0E7490",     # 4.67:1+ on light surfaces — AI status/
                                      # processing dots and icons that need to
                                      # actually read against a light background
    "ai-surface": "#E0F2FE",         # AI message bubble fill (distinct from
                                      # plain white cards elsewhere)

    "status-online": "#10B981",
    "status-online-text": "#047857", # 5.24:1 on bg — #059669 (3.60:1) fails AA
    "status-warn": "#B45309",
    "status-error": "#DC2626",
    "status-online-bg": "rgba(16, 185, 129, 0.10)",
    "status-warn-bg": "rgba(180, 83, 9, 0.10)",
    "status-error-bg": "rgba(220, 38, 38, 0.10)",
    "status-online-ring": "rgba(16, 185, 129, 0.16)",

    "shadow-md": "0 8px 24px rgba(15, 23, 42, 0.10)",
    "focus-ring": "rgba(37, 99, 235, 0.35)",

    # Flat-card tokens. A "card" is a solid white fill on the off-white
    # page background, separated by a real hairline border plus a single
    # soft, low-opacity shadow for a touch of lift — the restrained,
    # premium-SaaS treatment the brief calls for, not an embossed
    # neumorphic surface. shadow-pressed (inset) is reserved for controls
    # that should read as "carved in" (the chat input, a toggle's track,
    # a button's :active state).
    "surface-neu": "#FFFFFF",
    "neu-light": "rgba(255, 255, 255, 0.6)",
    "neu-dark": "rgba(15, 23, 42, 0.08)",
    "neu-border": "#E2E8F0",
    "shadow-raised": "0 1px 2px rgba(15, 23, 42, 0.04), 0 2px 6px rgba(15, 23, 42, 0.06)",
    "shadow-raised-sm": "0 1px 2px rgba(15, 23, 42, 0.05)",
    "shadow-raised-lg": "0 6px 16px rgba(15, 23, 42, 0.10)",
    "shadow-pressed": "inset 0 1px 2px rgba(15, 23, 42, 0.06)",
}

DARK_THEME: dict[str, str] = {
    **_BRAND_COLORS,

    "bg": "#0B1120",
    "surface": "#0F172A",
    "surface-tint": "rgba(37, 99, 235, 0.12)",   # blue wash — hover/active
    "surface-forest": "#0F172A",                 # sidebar panel
    "border": "#263449",
    "border-strong": "#33455F",

    "text-primary": "#F8FAFC",
    "text-secondary": "#94A3B8",
    "text-tertiary": "#94A3B8",

    "accent": "#2563EB",              # primary blue stays consistent across themes
    "accent-strong": "#60A5FA",       # lighter blue — text-safe on dark surfaces
    "accent-on-accent": "#FFFFFF",
    "accent-tint": "rgba(37, 99, 235, 0.16)",
    "accent-tint-strong": "rgba(37, 99, 235, 0.26)",
    "ai-accent": "#06B6D4",
    "ai-accent-text": "#06B6D4",      # raw cyan already clears AA (6.7-7.8:1) on dark surfaces
    "ai-surface": "#172033",          # elevated surface — AI bubble + source/RAG cards

    "status-online": "#10B981",
    "status-online-text": "#34D399",
    "status-warn": "#E3A857",
    "status-error": "#F87171",
    "status-online-bg": "rgba(16, 185, 129, 0.16)",
    "status-warn-bg": "rgba(227, 168, 87, 0.14)",
    "status-error-bg": "rgba(248, 113, 113, 0.14)",
    "status-online-ring": "rgba(16, 185, 129, 0.24)",

    "shadow-md": "0 8px 28px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(37, 99, 235, 0.08)",
    "focus-ring": "rgba(6, 182, 212, 0.40)",

    # Elevated surface (#172033) stands in for the old neumorphic
    # surface-neu — cards read as a distinct, lighter panel against the
    # deep navy base via fill + border, not an embossed shadow pair.
    "surface-neu": "#172033",
    "neu-light": "rgba(148, 163, 184, 0.10)",
    "neu-dark": "rgba(0, 0, 0, 0.5)",
    "neu-border": "#263449",
    "shadow-raised": "0 1px 2px rgba(0, 0, 0, 0.3), 0 2px 8px rgba(0, 0, 0, 0.35)",
    "shadow-raised-sm": "0 1px 2px rgba(0, 0, 0, 0.3)",
    "shadow-raised-lg": "0 8px 20px rgba(0, 0, 0, 0.4)",
    "shadow-pressed": "inset 0 1px 2px rgba(0, 0, 0, 0.4)",
}


def get_theme() -> str:
    """Session-state-backed (so the choice survives navigating between
    pages via st.page_link, which loads a fresh URL with no query string
    and would otherwise silently drop back to light) with the ?theme=
    query param as a fallback for a first load / full reload / new tab,
    where no session state exists yet."""
    if "theme" in st.session_state:
        theme = st.session_state.theme
    else:
        theme = st.query_params.get("theme", "light")
    theme = theme if theme in ("light", "dark") else "light"
    st.session_state.theme = theme
    return theme


def set_theme(theme: str) -> None:
    st.session_state.theme = theme
    st.query_params["theme"] = theme


@lru_cache(maxsize=1)
def _static_css() -> str:
    """The stylesheet body never changes at runtime (the dev file-watcher
    is deliberately disabled — see .streamlit/config.toml), but Streamlit
    reruns this whole script on every single interaction. Without this
    cache, every click/keystroke/message reread and reparsed the ~30KB
    style.css from disk before Streamlit could paint anything — a real,
    compounding cost since it sat in the hot path of every rerun, not
    just app startup."""
    return (ASSETS_DIR / "style.css").read_text(encoding="utf-8")


def inject_theme_css() -> str:
    """Renders the current theme's color tokens as a :root block, then the
    static stylesheet (which only ever references those tokens via var()).
    Returns the active theme name so callers don't need a second
    get_theme() call.

    The <style> tag itself still has to be re-emitted every rerun — skip
    it and Streamlit's reconciliation removes it from the DOM along with
    everything else this script didn't re-render that pass — but the
    tokens dict lookup and the (now-cached) CSS text are cheap, so the
    per-rerun cost is just building one string and handing it to
    st.markdown, not disk I/O."""
    theme = get_theme()
    tokens = DARK_THEME if theme == "dark" else LIGHT_THEME
    root_vars = "".join(f"--{name}:{value};" for name, value in tokens.items())
    st.markdown(
        f"<style>:root {{ {root_vars} }}\n{_static_css()}</style>",
        unsafe_allow_html=True,
    )
    return theme

_MARK_SOURCE = ASSETS_DIR / "orchis-mark.png"


@lru_cache(maxsize=None)
def _mark_data_uri(size: int) -> str:
    """Re-encodes the 1024px master mark at the exact display size, once
    per distinct size ever requested (there are only three call sites —
    26px avatar, 34px sidebar mark, 72px hero — so this cache never grows
    past a handful of entries).

    This matters for more than tidiness: the master PNG is ~500KB, and the
    chat avatar re-renders it on every message. Inlining the full-res file
    as a base64 data URI at 26px would ship ~660KB of base64 text per
    message bubble — a real, compounding page-weight cost as a
    conversation grows. Re-encoding at the actual display size keeps each
    inlined copy in the low single-digit KB.
    """
    from io import BytesIO

    from PIL import Image

    with Image.open(_MARK_SOURCE) as img:
        resized = img.convert("RGBA").resize((size, size), Image.LANCZOS)
        buf = BytesIO()
        resized.save(buf, format="PNG", optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mark_img(size: int, css_class: str) -> str:
    uri = _mark_data_uri(size)
    return (
        f'<img class="{css_class}" src="{uri}" width="{size}" height="{size}" '
        f'alt="Orchis" loading="lazy"/>'
    )


def brand_mark(size: int = 34) -> str:
    """Compact mark for the header, sidebar logo, and chat avatars."""
    return _mark_img(size, "orchis-mark-img")


def brand_hero(size: int = 72) -> str:
    """Larger rendering of the same mark for the empty-chat hero moment."""
    return _mark_img(size, "orchis-hero-img")


# Click handler for the per-message copy buttons app.py renders. It has to
# be a real listener rather than an inline onclick= because Streamlit's
# markdown renderer hands raw HTML to React, which drops string-valued
# event-handler attributes instead of attaching them (see
# app.py's render_copy_action docstring).
#
# Delegated from the document, so it covers every copy button that exists
# now or gets re-rendered later without re-running any JS per message —
# and injected as a <script> element in the *parent* document rather than
# left running inside the component iframe, so the handler keeps working
# after Streamlit unmounts that iframe on a later rerun (a listener whose
# function object lives in a torn-down iframe realm does not).
_COPY_HANDLER_JS = """
(function () {
  if (window.__orchisCopyBound) { return; }
  window.__orchisCopyBound = true;
  document.addEventListener('click', function (event) {
    var button = event.target.closest('[data-orchis-copy]');
    if (!button) { return; }
    var text = button.getAttribute('data-orchis-copy');
    var original = button.innerHTML;
    var confirm = function () {
      button.innerHTML = '\\u2713 Copied';
      button.setAttribute('data-copied', '1');
      setTimeout(function () {
        button.innerHTML = original;
        button.removeAttribute('data-copied');
      }, 1500);
    };
    var fallback = function () {
      var area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      try { document.execCommand('copy'); confirm(); } catch (err) { /* clipboard unavailable */ }
      document.body.removeChild(area);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(confirm, fallback);
    } else {
      fallback();
    }
  });
})();
"""


def install_copy_handler() -> None:
    """Inject the copy-button click handler once per session."""
    if st.session_state.get("_copy_handler_installed"):
        return
    st.session_state._copy_handler_installed = True

    components.html(
        f"""
        <script>
        (function() {{
            try {{
                var doc = window.parent.document;
                if (doc.getElementById('orchis-copy-handler')) {{ return; }}
                var el = doc.createElement('script');
                el.id = 'orchis-copy-handler';
                el.textContent = {json.dumps(_COPY_HANDLER_JS)};
                doc.head.appendChild(el);
            }} catch (e) {{ /* cross-origin embed — copy button stays inert */ }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


@lru_cache(maxsize=None)
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def inject_favicon_metadata(page_id: str) -> None:
    """Give the browser tab, mobile home-screen, and any PWA install
    prompt the full mark icon set (16/32/48/180/192/512 + .ico) instead of
    the single low-res image st.set_page_config's page_icon allows.

    `page_id` (e.g. "chat", "dashboard" — one per caller in app.py /
    pages/*.py) guards against the real cost here: Streamlit reruns the
    whole script on *every* interaction, not just on navigating to a new
    page, and this function used to remount a real hidden <iframe> and
    rerun DOM-mutation JS in it on every single one of those reruns —
    clicking thumbs-up on a message was, incidentally, also re-injecting
    the favicon. The <head> <link> tags this writes persist in the
    browser across reruns on their own (they're outside Streamlit's
    managed React tree, so its reconciliation never touches them), so
    once they're set for the current page there's nothing to redo until
    the user actually navigates somewhere else.

    Streamlit's components.html renders in a same-origin iframe, so the
    injected script can reach the real page's <head> via window.parent —
    this is the standard, if unofficial, way to add <link> tags Streamlit
    itself doesn't expose an API for. Zero-height so it takes no layout
    space; nothing here talks to the backend.
    """
    if st.session_state.get("_favicon_page") == page_id:
        return
    st.session_state._favicon_page = page_id

    icons = {
        16: _b64(FAVICON_DIR / "favicon-16.png"),
        32: _b64(FAVICON_DIR / "favicon-32.png"),
        48: _b64(FAVICON_DIR / "favicon-48.png"),
        180: _b64(FAVICON_DIR / "favicon-180.png"),
        192: _b64(FAVICON_DIR / "favicon-192.png"),
        512: _b64(FAVICON_DIR / "favicon-512.png"),
    }
    ico_b64 = _b64(FAVICON_DIR / "favicon.ico")

    links = [
        ('shortcut icon', 'image/x-icon', f"data:image/x-icon;base64,{ico_b64}", None),
        ('icon', 'image/png', f"data:image/png;base64,{icons[16]}", '16x16'),
        ('icon', 'image/png', f"data:image/png;base64,{icons[32]}", '32x32'),
        ('icon', 'image/png', f"data:image/png;base64,{icons[48]}", '48x48'),
        ('apple-touch-icon', 'image/png', f"data:image/png;base64,{icons[180]}", '180x180'),
        ('icon', 'image/png', f"data:image/png;base64,{icons[192]}", '192x192'),
        ('icon', 'image/png', f"data:image/png;base64,{icons[512]}", '512x512'),
    ]

    link_js = "\n".join(
        f'''
        (function() {{
            var l = doc.createElement('link');
            l.rel = {rel!r};
            l.type = {type_!r};
            {"l.sizes = " + repr(sizes) + ";" if sizes else ""}
            l.href = {href!r};
            doc.head.appendChild(l);
        }})();'''
        for rel, type_, href, sizes in links
    )

    components.html(
        f"""
        <script>
        (function() {{
            try {{
                var doc = window.parent.document;
                doc.querySelectorAll(
                    "link[rel*='icon']"
                ).forEach(function(el) {{ el.parentNode.removeChild(el); }});
                {link_js}
            }} catch (e) {{ /* cross-origin embed — favicon stays as set by page_icon */ }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )
