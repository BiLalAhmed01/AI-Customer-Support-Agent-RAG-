"""Retrieval function: fetch the most relevant chunks for a query,
filtered by a relevance threshold so irrelevant chunks are dropped
rather than passed to the LLM as false context.
"""

from dataclasses import dataclass
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.embeddings import get_embeddings


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float  # lower = more similar (Chroma uses L2/cosine distance)


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """Cached Chroma connection — avoids re-opening the persisted DB and
    reloading the embedding model on every single query."""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def collection_is_empty() -> bool:
    return collection_count() == 0


def collection_count() -> int:
    return _get_vectorstore()._collection.count()


def retrieve(query: str, k: int | None = None) -> list[RetrievedChunk]:
    """Return relevant chunks for `query`, or an empty list if nothing
    in the vector store clears the relevance threshold.

    Chroma's default `similarity_search_with_score` returns a *distance*
    (lower is better). We keep only chunks below RELEVANCE_SCORE_THRESHOLD.
    """
    if collection_is_empty():
        return []

    vectorstore = _get_vectorstore()
    k = k or settings.retrieval_k

    results: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(
        query, k=k
    )

    relevant = [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            score=score,
        )
        for doc, score in results
        if score <= settings.relevance_score_threshold
    ]
    return relevant


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt."""
    if not chunks:
        return ""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {chunk.source})\n{chunk.content}")
    return "\n\n".join(blocks)
