"""Ingest PDF/text documents from data/docs into the Chroma vector store.

Usage:
    python -m src.ingest
    python -m src.ingest --docs-dir ./data/docs --reset
"""

import argparse
import sys
import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.embeddings import get_embeddings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Streamlit runs every user's session in the same process (multiple
# threads), and pages/1_Knowledge_Base.py calls ingest() directly from a
# request handler — so two uploads landing at the same time could
# otherwise interleave their delete-then-add sequence below (e.g. session
# A's delete running between session B's delete and add), leaving the
# collection in a state neither caller intended. This serializes the
# actual read-modify-write against the vector store; it does not change
# what a single call does, only guarantees calls don't overlap.
_INGEST_LOCK = threading.Lock()


def load_documents(docs_dir: Path, only_filenames: set[str] | None = None):
    documents = []
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if only_filenames is not None:
        files = [p for p in files if p.name in only_filenames]

    if not files:
        print(f"No supported files (.pdf, .txt, .md) found in {docs_dir}")
        return documents

    for path in files:
        print(f"Loading {path}...")
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = path.name
            documents.extend(docs)
        except Exception as exc:
            print(f"  Failed to load {path}: {exc}", file=sys.stderr)

    return documents


def ingest(docs_dir: str, reset: bool = False, only_filenames: set[str] | None = None) -> int:
    """Load, chunk, embed, and store documents from `docs_dir`.

    By default (`only_filenames=None`) this processes every supported file
    in the directory — the right behavior for the CLI, and for `--reset`
    rebuilds. Pass `only_filenames` to restrict ingestion to specific
    files instead: without it, re-running this against a directory that
    already has indexed files (e.g. after adding one new upload) would
    re-embed and re-store every existing file too, duplicating their
    chunks in the collection on every call.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Docs directory does not exist: {docs_path}", file=sys.stderr)
        return 0

    raw_documents = load_documents(docs_path, only_filenames=only_filenames)
    if not raw_documents:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_documents)
    print(f"Split {len(raw_documents)} document(s) into {len(chunks)} chunk(s).")

    embeddings = get_embeddings()

    # Everything from here on is a read-modify-write against the shared
    # persisted collection — locked so two concurrent ingest() calls (e.g.
    # two uploads landing at the same time in the same Streamlit process)
    # can't interleave their delete-then-add sequence. Document loading/
    # chunking above is local, CPU-only work and deliberately stays
    # outside the lock so it isn't serialized for no reason.
    with _INGEST_LOCK:
        if reset:
            print("Resetting existing collection...")
            Chroma(
                collection_name=settings.collection_name,
                embedding_function=embeddings,
                persist_directory=settings.chroma_persist_dir,
            ).delete_collection()

        vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

        # Re-ingesting a filename that's already indexed (re-uploading a
        # corrected/updated document through the Knowledge Base page is
        # the common case) must replace its old chunks, not add alongside
        # them — otherwise the store ends up with both the stale and the
        # current text simultaneously retrievable, and Orchis can answer
        # from either one depending on which chunk scores higher.
        # only_filenames is exactly the set of sources this call is about
        # to (re-)write, so clearing their existing chunks first is safe
        # and specific: it never touches any other document's chunks.
        if only_filenames:
            vectorstore.delete(where={"source": {"$in": sorted(only_filenames)}})

        vectorstore.add_documents(chunks)

    print(
        f"Ingested {len(chunks)} chunks into Chroma collection "
        f"'{settings.collection_name}' at {settings.chroma_persist_dir}"
    )
    return len(chunks)


def delete_document(filename: str, docs_dir: str | None = None) -> bool:
    """Remove a single document from the knowledge base: its chunks from
    the vector store, and its file from disk if still present.

    `filename` is treated as untrusted input the same way
    pages/1_Knowledge_Base.py's safe_destination() treats an upload's
    name — basename-only, no traversal segments — since this is reachable
    from a UI action driven by data the Knowledge Base page reads back
    from the vector store's own metadata, not a value this function
    should ever trust blindly. Returns True if anything was actually
    removed (vectors or file), False if the name was invalid or nothing
    matched.
    """
    docs_root = Path(docs_dir or settings.docs_dir).resolve()
    name = Path(filename.replace("\\", "/")).name
    if not name or name in {".", ".."} or name.startswith("."):
        return False

    embeddings = get_embeddings()
    removed = False

    with _INGEST_LOCK:
        vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        existing = vectorstore._collection.get(where={"source": name}, include=[])
        if existing["ids"]:
            vectorstore.delete(where={"source": name})
            removed = True

    file_path = (docs_root / name).resolve()
    if file_path.parent == docs_root and file_path.exists():
        file_path.unlink()
        removed = True

    return removed


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--docs-dir", default=settings.docs_dir, help="Directory of source docs")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection first")
    args = parser.parse_args()

    ingest(args.docs_dir, reset=args.reset)


if __name__ == "__main__":
    main()
