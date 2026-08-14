"""Ingest PDF/text documents from data/docs into the Chroma vector store.

Usage:
    python -m src.ingest
    python -m src.ingest --docs-dir ./data/docs --reset
"""

import argparse
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.embeddings import get_embeddings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(docs_dir: Path):
    documents = []
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

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


def ingest(docs_dir: str, reset: bool = False) -> int:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Docs directory does not exist: {docs_path}", file=sys.stderr)
        return 0

    raw_documents = load_documents(docs_path)
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
    vectorstore.add_documents(chunks)

    print(
        f"Ingested {len(chunks)} chunks into Chroma collection "
        f"'{settings.collection_name}' at {settings.chroma_persist_dir}"
    )
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--docs-dir", default=settings.docs_dir, help="Directory of source docs")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection first")
    args = parser.parse_args()

    ingest(args.docs_dir, reset=args.reset)


if __name__ == "__main__":
    main()
