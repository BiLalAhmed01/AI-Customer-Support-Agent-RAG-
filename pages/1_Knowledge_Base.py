"""Orchis — Knowledge Base management page.

Shows the ACTUAL indexed state of the knowledge base: every number here is
read live from the persisted vector store or the filesystem, never
fabricated. Ingestion itself is unchanged — this page is a UI on top of
the existing src/ingest.py pipeline, calling the same ingest() function
the CLI uses.
"""

import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

from src.auth import require_access
from src.branding import FAVICON_DIR, inject_favicon_metadata, inject_theme_css
from src.config import settings
from src.ingest import SUPPORTED_EXTENSIONS, delete_document, ingest
from src.ratelimit import check_upload_rate_limit
from src.retrieval import collection_count, list_indexed_sources
from src.ui import render_header, render_sidebar

st.set_page_config(
    page_title="Orchis — Knowledge Base",
    page_icon=str(FAVICON_DIR / "favicon-32.png"),
    layout="centered",
)

inject_theme_css()
inject_favicon_metadata("knowledge_base")
require_access()

DOCS_DIR = Path(settings.docs_dir)


# Upload guard rails. `type=` on st.file_uploader is a browser-side accept
# filter and a client-side check only — it does not stop a crafted POST to
# Streamlit's upload endpoint from carrying any filename and any bytes, so
# the real validation has to happen here, before anything is written to
# disk. settings.max_upload_mb lives in src/config.py alongside every
# other tunable; it's also mirrored in .streamlit/config.toml's
# server.maxUploadSize (see that setting's comment for why both exist).
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024


def safe_destination(raw_name: str) -> Path:
    """Resolve an uploaded file's client-supplied name to a path that is
    provably inside DOCS_DIR, or raise.

    The name in an upload is attacker-controlled: "../../app.py" or
    "C:/Windows/Temp/x.md" both arrive intact if a client chooses to send
    them, and `DOCS_DIR / raw_name` happily resolves either one outside
    the docs directory (verified: `Path("data/docs") / "../../evil.md"`
    resolves to the repo root). Overwriting app.py that way is remote code
    execution on the next rerun, since Streamlit re-executes the script
    file on every interaction.

    Three checks, because each covers a different escape: taking the
    basename (after normalizing "\\" to "/", so a Windows-style
    "..\\..\\app.py" is split the same way on any platform) removes
    traversal segments; the extension allowlist stops a .py/.exe landing
    in a directory the app itself reads; and re-resolving the final path
    and requiring its parent to still be DOCS_DIR is the backstop that
    fails closed if either of the first two ever misses a form.
    """
    name = Path(raw_name.replace("\\", "/")).name
    if not name or name in {".", ".."} or name.startswith("."):
        raise ValueError(f"Rejected unsafe filename: {raw_name!r}")

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix or name}' — allowed: .pdf, .txt, .md")

    docs_root = DOCS_DIR.resolve()
    dest = (docs_root / name).resolve()
    if dest.parent != docs_root:
        raise ValueError(f"Rejected unsafe filename: {raw_name!r}")
    return dest


@lru_cache(maxsize=64)
def _probe_readability_cached(path_str: str, mtime_ns: int, size: int) -> tuple[bool, str]:
    """The actual probe, cached on (path, mtime, size) — get_documents()
    calls this on every single page rerun for every not-yet-indexed file
    (nav clicks, button presses, anything that triggers a Streamlit rerun
    while a doc is still "processing"), and it fully parses the file
    (a real PDF/text load, not a cheap check) each time. Keying on mtime+
    size rather than just the path means a re-uploaded/edited file with
    the same name still gets re-probed, while an unchanged file — the
    common case while sitting in "processing" — reuses the prior result
    instead of re-parsing on every rerun."""
    path = Path(path_str)
    try:
        if path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            PyPDFLoader(str(path)).load()
        else:
            from langchain_community.document_loaders import TextLoader
            TextLoader(str(path), encoding="utf-8").load()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _probe_readability(path: Path) -> tuple[bool, str]:
    """Read-only check using the exact same loader classes src/ingest.py
    uses for real ingestion — if this fails, ingestion would fail on this
    file too, so the signal is genuine, not guessed."""
    stat = path.stat()
    return _probe_readability_cached(str(path), stat.st_mtime_ns, stat.st_size)


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def get_documents() -> list[dict]:
    """Union of files actually on disk and sources actually present in the
    vector store — nothing here is invented. Each entry's status is
    derived from real, checkable facts:
      - chunks > 0            -> Indexed (it's genuinely queryable)
      - on disk, 0 chunks     -> Processing (not yet run through ingest)
        (readability probe failed) -> Failed, with the real error
      - in the store but the source file is gone from disk -> still
        Indexed (the chunks are real and still queryable), noted as such
    """
    indexed = list_indexed_sources()  # {filename: chunk_count}
    disk_files = []
    if DOCS_DIR.exists():
        disk_files = sorted(
            p.name for p in DOCS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    # Every name here came from either disk_files or indexed, so a name
    # with 0 chunks is necessarily on disk — the two branches below are
    # exhaustive.
    all_names = sorted(set(disk_files) | set(indexed.keys()))
    docs = []
    for name in all_names:
        chunk_count = indexed.get(name, 0)
        file_path = DOCS_DIR / name
        on_disk = file_path.exists()

        if chunk_count > 0:
            status, error_detail = "indexed", ""
            missing_note = not on_disk
        else:
            ok, error_detail = _probe_readability(file_path)
            status = "processing" if ok else "failed"
            missing_note = False

        size_str, modified_str = None, None
        if on_disk:
            try:
                stat = file_path.stat()
                size_str = _format_size(stat.st_size)
                from datetime import datetime
                modified_str = datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %Y")
            except OSError:
                pass

        docs.append({
            "name": name,
            "file_type": Path(name).suffix.lstrip(".").upper() or "—",
            "chunks": chunk_count,
            "status": status,
            "error_detail": error_detail,
            "missing_from_disk": missing_note,
            "size": size_str,
            "modified": modified_str,
        })
    return docs


def render_status_badge(status: str) -> str:
    if status == "indexed":
        return '<span class="doc-status indexed">● Indexed</span>'
    if status == "processing":
        return '<span class="doc-status processing">◌ Processing</span>'
    return '<span class="doc-status failed">! Failed</span>'


def render_doc_card(doc: dict) -> str:
    meta_parts = [f"{doc['file_type']} document"]
    if doc["size"]:
        meta_parts.append(doc["size"])
    if doc["modified"]:
        meta_parts.append(f"modified {doc['modified']}")
    if doc["missing_from_disk"]:
        meta_parts.append("source file no longer on disk")
    meta = " · ".join(meta_parts)

    chunks_label = f"{doc['chunks']} chunk{'s' if doc['chunks'] != 1 else ''}"

    return (
        f'<div class="doc-card">'
        f'<div class="doc-main">'
        f'<div class="doc-icon">{html.escape(doc["file_type"][:3] or "DOC")}</div>'
        f'<div class="doc-info">'
        f'<div class="doc-name">{html.escape(doc["name"])}</div>'
        f'<div class="doc-meta">{html.escape(meta)}</div>'
        f'</div></div>'
        f'<div class="doc-side">'
        f'<span class="doc-chunks">{chunks_label}</span>'
        f'{render_status_badge(doc["status"])}'
        f'</div></div>'
    )


# ---------- Header ----------
render_sidebar(active="knowledge_base")
render_header("Knowledge Base", "Documents indexed for retrieval")

upload_result = st.session_state.pop("kb_upload_result", None)
if upload_result:
    level, text = upload_result
    (st.success if level == "success" else st.error)(text)

documents = get_documents()
total_docs = len(documents)
total_indexed = sum(1 for d in documents if d["status"] == "indexed")
total_chunks = collection_count()

# ---------- Overview stats (real numbers only) ----------
st.markdown(
    f"""
    <div class="kb-stat-row">
        <div class="kb-stat">
            <div class="kb-stat-label">Documents</div>
            <div class="kb-stat-value">{total_docs}</div>
        </div>
        <div class="kb-stat">
            <div class="kb-stat-label">Indexed</div>
            <div class="kb-stat-value">{total_indexed}</div>
        </div>
        <div class="kb-stat">
            <div class="kb-stat-label">Total Chunks</div>
            <div class="kb-stat-value">{total_chunks}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Document list / empty state ----------
if not documents:
    st.markdown(
        """
        <div class="kb-empty">
            <div class="kb-empty-icon">O</div>
            <h3>No documents yet</h3>
            <p>Add a document below to start building Orchis's knowledge base.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="section-heading">Documents</div>', unsafe_allow_html=True)
    for doc in documents:
        st.markdown(render_doc_card(doc), unsafe_allow_html=True)
        if doc["status"] == "failed":
            st.markdown(
                f'<div class="doc-error-detail">{html.escape(doc["error_detail"])}</div>',
                unsafe_allow_html=True,
            )

        # Two-click confirm (arm, then confirm) rather than a single
        # button, since this is a destructive action with no undo —
        # removes the document's chunks from the vector store and its
        # file from disk (see src/ingest.py's delete_document()).
        confirm_key = f"kb_confirm_delete_{doc['name']}"
        if st.session_state.get(confirm_key):
            st.warning(f"Remove **{doc['name']}** from the knowledge base? This can't be undone.")
            cols = st.columns([1, 1, 4], gap="small")
            with cols[0]:
                if st.button("Confirm remove", key=f"kb_do_delete_{doc['name']}", type="primary"):
                    removed = delete_document(doc["name"])
                    st.session_state.pop(confirm_key, None)
                    st.session_state.kb_upload_result = (
                        "success" if removed else "error",
                        f"Removed {doc['name']}." if removed
                        else f"Nothing to remove for {doc['name']}.",
                    )
                    st.rerun()
            with cols[1]:
                if st.button("Cancel", key=f"kb_cancel_delete_{doc['name']}"):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
        else:
            if st.button("Remove", key=f"kb_remove_{doc['name']}"):
                st.session_state[confirm_key] = True
                st.rerun()

# ---------- Add Knowledge Source ----------
st.markdown('<div class="section-heading">Add Knowledge Source</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-panel">', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
st.markdown(
    '<div class="upload-hint">Supported formats: PDF, TXT, MD. Files are added to the '
    "existing knowledge base — nothing already indexed is removed.</div>",
    unsafe_allow_html=True,
)

if uploaded_files:
    if st.button(f"Index {len(uploaded_files)} document(s)", type="primary"):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        saved_names = []
        rejected: list[str] = []
        for f in uploaded_files:
            data = f.getvalue()
            try:
                if len(data) > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        f"{_format_size(len(data))} exceeds the "
                        f"{_format_size(MAX_UPLOAD_BYTES)} per-file limit"
                    )
                # Cumulative-per-session cap, on top of the per-file limit
                # above — bounds repeated uploads filling the disk over
                # time, not just a single oversized file.
                rate_ok, rate_message = check_upload_rate_limit(len(data))
                if not rate_ok:
                    raise ValueError(rate_message)
                dest = safe_destination(f.name)
            except ValueError as exc:
                rejected.append(f"{f.name}: {exc}")
                continue
            dest.write_bytes(data)
            saved_names.append(dest.name)

        if saved_names:
            with st.spinner("Indexing document(s)... this runs the real ingestion pipeline and may take a moment."):
                try:
                    # Restricted to just the files saved above — without this,
                    # every already-indexed document in DOCS_DIR would be
                    # re-embedded and re-stored too, duplicating its chunks in
                    # the collection on every upload.
                    chunk_count = ingest(str(DOCS_DIR), reset=False, only_filenames=set(saved_names))
                    level = "success"
                    message = (
                        f"Indexed {', '.join(saved_names)} — {chunk_count} chunk(s) added "
                        "to the knowledge base."
                    )
                except Exception as exc:
                    level, message = "error", f"Indexing failed: {exc}"
        else:
            level, message = "error", "Nothing was indexed."

        if rejected:
            message += "  Rejected: " + "; ".join(rejected)

        # Stashed rather than rendered here: the st.rerun() below (needed
        # so the document list above reflects the new chunks) throws away
        # everything this script run has already emitted, so an st.success/
        # st.error written at this point would flash and vanish before the
        # user could read it. The next run renders it from session state
        # and clears it.
        st.session_state.kb_upload_result = (level, message)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
