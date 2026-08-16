"""Orchis — Knowledge Base management page.

Shows the ACTUAL indexed state of the knowledge base: every number here is
read live from the persisted vector store or the filesystem, never
fabricated. Ingestion itself is unchanged — this page is a UI on top of
the existing src/ingest.py pipeline, calling the same ingest() function
the CLI uses.
"""

import html
from pathlib import Path

import streamlit as st

from src.branding import FAVICON_DIR, inject_favicon_metadata, inject_theme_css
from src.config import settings
from src.ingest import SUPPORTED_EXTENSIONS, ingest
from src.retrieval import collection_count, list_indexed_sources
from src.ui import render_header, render_sidebar

st.set_page_config(
    page_title="Orchis — Knowledge Base",
    page_icon=str(FAVICON_DIR / "favicon-32.png"),
    layout="centered",
)

inject_theme_css()
inject_favicon_metadata("knowledge_base")

DOCS_DIR = Path(settings.docs_dir)


def _probe_readability(path: Path) -> tuple[bool, str]:
    """Read-only check using the exact same loader classes src/ingest.py
    uses for real ingestion — if this fails, ingestion would fail on this
    file too, so the signal is genuine, not guessed."""
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

    all_names = sorted(set(disk_files) | set(indexed.keys()))
    docs = []
    for name in all_names:
        chunk_count = indexed.get(name, 0)
        file_path = DOCS_DIR / name
        on_disk = file_path.exists()

        if chunk_count > 0:
            status, error_detail = "indexed", ""
            missing_note = not on_disk
        elif on_disk:
            ok, error_detail = _probe_readability(file_path)
            status = "processing" if ok else "failed"
            missing_note = False
        else:
            status, error_detail, missing_note = "indexed", "", False  # unreachable in practice

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
        for f in uploaded_files:
            dest = DOCS_DIR / f.name
            dest.write_bytes(f.getvalue())
            saved_names.append(f.name)

        with st.spinner("Indexing document(s)... this runs the real ingestion pipeline and may take a moment."):
            try:
                # Restricted to just the files saved above — without this,
                # every already-indexed document in DOCS_DIR would be
                # re-embedded and re-stored too, duplicating its chunks in
                # the collection on every upload.
                chunk_count = ingest(str(DOCS_DIR), reset=False, only_filenames=set(saved_names))
                st.success(
                    f"Indexed {', '.join(saved_names)} — {chunk_count} chunk(s) added to the knowledge base."
                )
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
