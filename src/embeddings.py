"""Shared embedding function used by both ingestion and retrieval."""

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local sentence-transformers embedding model — no API key required.

    Using the same embedding function at ingest time and query time is
    required: Chroma similarity search only works if vectors were produced
    by the same model.

    Cached (loaded once per process) — reloading transformer weights from
    disk on every call was the single largest source of query latency.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
