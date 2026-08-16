"""Central configuration loaded from environment variables (.env)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Groq: free-tier option, OpenAI-compatible API (no billing required)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "support_docs")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Final number of chunks handed to the LLM as context.
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))

    # How many candidates to pull from the vector store (by raw embedding
    # distance) before reranking. Wider than retrieval_k on purpose — this
    # is what lets a semantically-correct but lexically-different chunk
    # survive long enough for the reranker to recognize it, even if the
    # bi-encoder alone would have ranked it outside the top retrieval_k.
    fetch_k: int = int(os.getenv("FETCH_K", "12"))

    # Cross-encoder relevance gate (raw ms-marco-MiniLM-L-6-v2 logit scale,
    # roughly -11..+11). This — not the legacy distance threshold below —
    # is what decides whether a chunk is actually relevant.
    rerank_score_threshold: float = float(
        os.getenv("RERANK_SCORE_THRESHOLD", "-8.0")
    )

    # Legacy raw-L2-distance cutoff from the bi-encoder-only pipeline. No
    # longer used to gate results (see rerank_score_threshold) — kept only
    # so debug mode can show the old score alongside the new one.
    relevance_score_threshold: float = float(
        os.getenv("RELEVANCE_SCORE_THRESHOLD", "1.5")
    )

    debug_mode: bool = os.getenv("DEBUG_MODE", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    docs_dir: str = os.getenv("DOCS_DIR", "./data/docs")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "75"))


settings = Settings()
