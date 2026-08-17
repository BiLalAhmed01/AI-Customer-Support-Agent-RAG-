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

    # Kept modest on purpose: shorter completions finish faster and support
    # answers rarely need more than this to be complete.
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "600"))

    # Hard ceiling on a single LLM request. Without this, a stalled
    # connection to the provider hangs indefinitely instead of failing
    # predictably — the UI has no way to recover from a request that
    # never resolves either way.
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

    # Per-file cap on Knowledge Base uploads. Mirrored by
    # .streamlit/config.toml's server.maxUploadSize, which bounds what the
    # server accepts at all — this is enforced again here since that
    # setting isn't read from Python.
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))

    # Most recent messages (user + assistant combined) sent to the LLM as
    # conversation history, on top of the current question. Unbounded
    # history was a real audit finding: token cost per turn grew linearly
    # with conversation length, and a long enough session eventually
    # exceeds the model's context window outright. 16 (~8 exchanges) keeps
    # follow-up questions coherent without that growth — chosen as a
    # reasonable default, not measured against real conversation data.
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "16"))

    # Shared-passcode gate (src/auth.py). Empty (the default) disables it
    # entirely — this app shipped with zero authentication anywhere, so a
    # non-empty default here would silently lock out every existing
    # deployment the moment this code lands. Opt in by setting
    # APP_ACCESS_CODE before deploying anywhere the knowledge-base upload
    # page or the chat itself shouldn't be open to the public internet.
    access_code: str = os.getenv("APP_ACCESS_CODE", "")

    # Chat messages allowed per session per rolling window (src/ratelimit.py).
    # 0 disables the limit.
    chat_rate_limit_count: int = int(os.getenv("CHAT_RATE_LIMIT_COUNT", "20"))
    chat_rate_limit_window_seconds: int = int(
        os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60")
    )

    # Cumulative Knowledge Base upload volume allowed per session, on top
    # of the existing per-file max_upload_mb cap. 0 disables the limit.
    upload_rate_limit_mb: int = int(os.getenv("UPLOAD_RATE_LIMIT_MB", "50"))
    upload_rate_limit_window_seconds: int = int(
        os.getenv("UPLOAD_RATE_LIMIT_WINDOW_SECONDS", "3600")
    )

    # Most recent messages rendered on the Chat page by default. The full
    # transcript is always kept in session state (Regenerate, Dashboard's
    # counts, and the Conversations page still see everything) — this only
    # bounds how much HTML app.py builds and re-renders on every rerun,
    # which otherwise grows without limit as a conversation gets long.
    # Older messages are still reachable via an in-page "Show earlier
    # messages" control, not deleted. 0 disables the window (render
    # everything, matching the app's original behavior).
    chat_display_window: int = int(os.getenv("CHAT_DISPLAY_WINDOW", "60"))


settings = Settings()
