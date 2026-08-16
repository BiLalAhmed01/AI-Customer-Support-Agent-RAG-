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
# Built from 5 fixed brand colors (forest/lilac/eggplant/mint/emerald,
# exposed verbatim as --color-* below) by generating a real tint/shade
# scale off them for backgrounds, surfaces, borders, and text — not just
# swapping the 5 raw values in as text/background colors directly. Every
# pairing that can carry text was checked against WCAG AA (4.5:1 normal
# text, 3:1 large text / non-text UI) with a relative-luminance contrast
# calculation, not eyeballed — a few consequences of that:
#   - Raw --lilac (#B979E4) fails AA as small text on both theme's base
#     backgrounds (3.83:1 dark, would be worse on light), so it's reserved
#     for large text/icons/borders/focus-rings; `accent-strong` is a
#     separate, darker/brighter text-safe variant for links and labels.
#   - Raw --emerald (#07B45C) fails AA as text on light backgrounds
#     (2.57:1) and only marginally clears non-text/dot thresholds
#     elsewhere — it's a decorative/dot/icon color only, never a text
#     color; status copy itself renders in text-primary/secondary.
# Every color-bearing rule in style.css reads these via var(--name), so
# switching themes only ever means swapping which dict gets rendered into
# the injected :root block below.
# ============================================================

# The 5 fixed brand colors, identical in both themes — exposed as their
# own variables (not just baked into the derived tokens below) so
# anything that specifically wants the raw brand color can reach it.
_BRAND_COLORS: dict[str, str] = {
    "color-forest": "#1D684A",
    "color-lilac": "#B979E4",
    "color-eggplant": "#3D325B",
    "color-mint": "#5EC780",
    "color-emerald": "#07B45C",
}

LIGHT_THEME: dict[str, str] = {
    **_BRAND_COLORS,

    # Soft lilac wash instead of pure white, per brief.
    "bg": "#FAF7FE",
    "surface": "#FDFBFF",
    "surface-tint": "#F3EAFB",       # lilac wash — hover states, active nav
    "surface-forest": "#E9F5EC",     # pale forest/mint tint — sidebar's distinct
                                      # identity vs. the lilac-toned main pane,
                                      # the light-mode echo of dark mode's forest
                                      # sidebar (see DARK_THEME's surface-forest)
    "border": "#E4D9F0",
    "border-strong": "#D9C7EE",

    "text-primary": "#241B36",
    "text-secondary": "#5C5270",
    "text-tertiary": "#756B87",      # 4.70:1 on bg — verified, not eyeballed

    "accent": "#B979E4",             # raw lilac — bg/border/icon use only
    "accent-strong": "#8A3FBD",      # 5.61:1 on bg — text-safe lilac (links, labels)
    "accent-on-accent": "#241B36",   # text color for lilac-filled buttons (5.29:1)
    "accent-tint": "#F3EAFB",
    "accent-tint-strong": "#E8D2F7",

    "status-online": "#07B45C",      # dot/icon only — see module docstring
    "status-online-text": "#0A7A3E", # 5.12:1 on bg — for status copy, not the dot
    "status-warn": "#96600F",        # 4.98:1 on bg — safe as text too
    "status-error": "#C4483B",       # 4.57:1 on bg — safe as text too
    "status-online-bg": "rgba(7, 180, 92, 0.10)",
    "status-warn-bg": "rgba(150, 96, 15, 0.10)",
    "status-error-bg": "rgba(196, 72, 59, 0.10)",
    "status-online-ring": "rgba(7, 180, 92, 0.16)",

    "shadow-md": "0 8px 24px rgba(61, 50, 91, 0.10)",
    "focus-ring": "rgba(185, 121, 228, 0.35)",

    # Neumorphism tokens. The defining trait (and the thing that makes it
    # look wrong if got wrong) is that a "card" is the *same* color as the
    # surface it sits on — depth comes entirely from a matched pair of
    # soft shadows (a highlight up-left, a shadow down-right), never from
    # a different fill color or a hard border. surface-neu is therefore
    # set equal to bg, not a step away from it like the old surface/
    # surface-tint tokens. shadow-raised is the resting "embossed" state;
    # shadow-pressed (inset, same two colors) is for anything that should
    # read as pushed in — the chat input, a toggle's track, a button's
    # :active state — so press interactions look like something real
    # physically depressing rather than just a color swap. A very
    # low-opacity neu-border is included despite pure neumorphism
    # avoiding borders entirely: at this text scale, shadow-only edges
    # were too faint to reliably tell where one card ends and the next
    # begins, which is the classic, well-documented usability failure
    # mode of the style — a hairline assist keeps it accessible without
    # visually reading as a "bordered card."
    "surface-neu": "#FAF7FE",
    "neu-light": "rgba(255, 255, 255, 0.9)",
    "neu-dark": "rgba(163, 150, 195, 0.55)",
    "neu-border": "rgba(61, 50, 91, 0.05)",
    "shadow-raised": "6px 6px 16px rgba(163, 150, 195, 0.5), -6px -6px 16px rgba(255, 255, 255, 0.9)",
    "shadow-raised-sm": "3px 3px 8px rgba(163, 150, 195, 0.45), -3px -3px 8px rgba(255, 255, 255, 0.85)",
    # A deeper version of shadow-raised (bigger offset/blur, same two
    # colors) for hover states on cards — lifting a raised shape further
    # off the surface, not swapping its color, is what "hover" should
    # mean in a system where color never carried the depth cue.
    "shadow-raised-lg": "10px 10px 24px rgba(163, 150, 195, 0.55), -8px -8px 20px rgba(255, 255, 255, 0.95)",
    "shadow-pressed": "inset 4px 4px 10px rgba(163, 150, 195, 0.45), inset -4px -4px 10px rgba(255, 255, 255, 0.8)",
}

DARK_THEME: dict[str, str] = {
    **_BRAND_COLORS,

    # Eggplant as the deepest/base surface (per brief), with forest used
    # sparingly as a distinct, darker contrast panel — e.g. the sidebar
    # reads as a different, cooler surface than the main chat pane
    # instead of just a lighter/darker shade of the same hue.
    "bg": "#3D325B",
    "surface": "#4A3D6E",            # one step up — cards, message bubbles
    "surface-tint": "rgba(185, 121, 228, 0.10)",  # hover/active wash
    "surface-forest": "#12271D",     # sidebar contrast panel (forest, darkened)
    "border": "rgba(185, 121, 228, 0.16)",
    "border-strong": "rgba(185, 121, 228, 0.28)",

    "text-primary": "#F4EFFB",
    "text-secondary": "#C9BEDE",
    "text-tertiary": "#ABA0C6",       # 4.75:1 on bg — verified, not eyeballed

    "accent": "#B979E4",              # raw lilac — bg/border/icon use only
    "accent-strong": "#D2A6F0",        # 5.79:1 on bg — text-safe lilac (links, labels)
    "accent-on-accent": "#241B36",    # text color for lilac-filled buttons (5.29:1)
    "accent-tint": "rgba(185, 121, 228, 0.16)",
    "accent-tint-strong": "rgba(185, 121, 228, 0.26)",

    "status-online": "#07B45C",       # dot/icon only — see module docstring
    "status-online-text": "#5EC780",  # 8.51:1 on bg — for status copy, not the dot
    "status-warn": "#E3A857",         # 5.54:1 on bg
    "status-error": "#EA7A6C",        # 4.15:1 on bg (large/UI text weight)
    "status-online-bg": "rgba(7, 180, 92, 0.16)",
    "status-warn-bg": "rgba(227, 168, 87, 0.14)",
    "status-error-bg": "rgba(234, 122, 108, 0.14)",
    "status-online-ring": "rgba(7, 180, 92, 0.24)",

    "shadow-md": "0 8px 28px rgba(10, 6, 18, 0.45), 0 0 0 1px rgba(185, 121, 228, 0.08)",
    "focus-ring": "rgba(185, 121, 228, 0.40)",

    # See LIGHT_THEME's neumorphism comment for the general approach.
    # Dark mode's shadow pair is a lighter lilac lift off the eggplant
    # base (not white — a literal white highlight would read as a
    # light-mode leak) paired with a near-black shadow, rather than the
    # light theme's white-highlight/muted-lavender-shadow pair.
    "surface-neu": "#3D325B",
    "neu-light": "rgba(150, 125, 195, 0.22)",
    "neu-dark": "rgba(15, 10, 25, 0.55)",
    "neu-border": "rgba(185, 121, 228, 0.08)",
    "shadow-raised": "6px 6px 16px rgba(15, 10, 25, 0.55), -6px -6px 16px rgba(150, 125, 195, 0.18)",
    "shadow-raised-sm": "3px 3px 8px rgba(15, 10, 25, 0.5), -3px -3px 8px rgba(150, 125, 195, 0.15)",
    "shadow-raised-lg": "10px 10px 24px rgba(15, 10, 25, 0.6), -8px -8px 20px rgba(150, 125, 195, 0.22)",
    "shadow-pressed": "inset 4px 4px 10px rgba(15, 10, 25, 0.5), inset -4px -4px 10px rgba(150, 125, 195, 0.15)",
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
