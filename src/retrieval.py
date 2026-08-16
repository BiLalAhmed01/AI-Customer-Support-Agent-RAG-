"""Retrieval pipeline: normalize the query, optionally fold in conversation
context for follow-ups, pull a broad candidate pool by embedding distance,
rerank that pool with a cross-encoder for actual relevance, dedupe, and
return the top-k.

Why two stages (embedding distance, then cross-encoder rerank) instead of
one: a bi-encoder embeds the query and each chunk independently and compares
vectors, which is fast but blind to relevance when wording differs — "How
can I order?" and "purchase products through our online store" share almost
no surface vocabulary, so raw embedding distance ranks them as barely
related even though they answer the same question. A cross-encoder scores
the query and passage *together*, which captures that kind of match far more
reliably. It's too slow to run against the whole corpus, so it only reranks
the shortlist the bi-encoder already narrowed down.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.embeddings import get_embeddings
from src.reranker import score_pairs

# Queries this short, or starting/containing these words, are likely
# referring back to the previous turn ("what about international?", "and
# returns?", "how much does that cost?") rather than standing alone.
_REFERENTIAL_STARTERS = (
    "what about",
    "how about",
    "and what",
    "and how",
    "and is",
    "and does",
    "what if",
    "is it",
    "does it",
    "can it",
)
_REFERENTIAL_WORDS = {"it", "that", "this", "those", "these", "them", "there", "they"}
_SHORT_QUERY_WORD_LIMIT = 4

# Exact-match only, deliberately — "thanks" skips retrieval, but "thanks,
# and what about shipping?" must NOT (it's a real question riding on a
# pleasantry). A substring check would wrongly skip that second case.
_SMALL_TALK_PHRASES = {
    "hi", "hello", "hey", "hiya", "yo", "howdy",
    "hi there", "hey there", "hello there", "howdy there",
    "good morning", "good afternoon", "good evening", "good day", "greetings",
    "thanks", "thank you", "thanks a lot", "thx", "ty", "appreciate it", "much appreciated",
    "ok", "okay", "k", "kk", "alright", "sure", "got it", "sounds good",
    "cool", "great", "nice", "perfect", "awesome",
    "bye", "goodbye", "see you", "later", "cya", "take care",
    "yes", "no", "yep", "yeah", "nope", "nah",
}


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float  # raw vector distance (lower = more similar) — legacy metric
    rerank_score: float = 0.0  # cross-encoder relevance (higher = more relevant)


@dataclass
class RetrievalDebug:
    """Diagnostics for development mode only — never shown to end users."""

    raw_query: str
    normalized_query: str
    search_query: str
    query_expanded_from_history: bool
    candidates_fetched: int
    candidates_kept: int
    candidates: list[dict] = field(default_factory=list)
    # Filled in by llm.py after intent classification — kept here rather
    # than a separate object so the whole turn's diagnostics live in one
    # place for the dev-only debug panel.
    intent: str = ""
    used_retrieval: bool = True


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


def list_indexed_sources() -> dict[str, int]:
    """Real per-document chunk counts, read directly from the persisted
    vector store — {} if nothing is indexed yet. Read-only: this does not
    change ingestion or storage behavior, it only exposes what's already
    there, the same way collection_count() does."""
    if collection_is_empty():
        return {}
    data = _get_vectorstore()._collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        source = meta.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def normalize_query(raw_query: str) -> str:
    """Deterministic cleanup — no model call. Trims stray whitespace and
    collapses repeated whitespace/punctuation that add noise without adding
    meaning (e.g. "how   can i order???")."""
    text = raw_query.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([?!.]){2,}", r"\1", text)
    return text


def is_small_talk(normalized_query: str) -> bool:
    """True for pure greetings/acknowledgments/farewells ("hi", "thanks",
    "okay") that carry no information need — retrieval would only ever
    return nothing for these, so it's wasted embedding + rerank compute.

    Exact match against a fixed phrase set, not a substring/keyword check:
    "thanks, what about shipping?" must still retrieve normally. This is
    deliberately conservative — when in doubt, it falls through to real
    retrieval rather than risk silently dropping a real question.
    """
    text = re.sub(r"[!?.,]+$", "", normalized_query.strip().lower())
    return text in _SMALL_TALK_PHRASES


def build_search_query(normalized_query: str, history: list[dict] | None) -> tuple[str, bool]:
    """Fold in the previous user turn when the current message looks like a
    follow-up, so retrieval isn't done blind to conversation context.

    Deterministic heuristic, not an LLM call: a query is treated as a
    follow-up if it's short or contains a referential word/opener. This
    covers the common "what about X?" / "and Y?" pattern without the cost
    or latency of a query-rewriting model call on every message.

    Returns (search_query, was_expanded).
    """
    if not history:
        return normalized_query, False

    lowered = normalized_query.lower()
    words = lowered.split()
    looks_referential = (
        len(words) <= _SHORT_QUERY_WORD_LIMIT
        or lowered.startswith(_REFERENTIAL_STARTERS)
        or any(w in _REFERENTIAL_WORDS for w in words)
    )
    if not looks_referential:
        return normalized_query, False

    last_user_turn = next(
        (turn["content"] for turn in reversed(history) if turn["role"] == "user"),
        None,
    )
    if not last_user_turn:
        return normalized_query, False

    combined = f"{last_user_turn.strip()} {normalized_query}"
    return combined, True


def retrieve(
    query: str,
    history: list[dict] | None = None,
    k: int | None = None,
    debug: RetrievalDebug | None = None,
) -> list[RetrievedChunk]:
    """Return relevant chunks for `query`, or an empty list if nothing
    clears the relevance bar.

    Pass a `RetrievalDebug()` instance via `debug` to have it populated
    in-place with diagnostics (only meant to be constructed when
    settings.debug_mode is True — see llm.py).
    """
    k = k or settings.retrieval_k

    normalized = normalize_query(query)
    search_query, was_expanded = build_search_query(normalized, history)

    if debug is not None:
        debug.raw_query = query
        debug.normalized_query = normalized
        debug.search_query = search_query
        debug.query_expanded_from_history = was_expanded

    if collection_is_empty():
        if debug is not None:
            debug.candidates_fetched = 0
            debug.candidates_kept = 0
        return []

    vectorstore = _get_vectorstore()
    fetch_k = max(settings.fetch_k, k)

    candidates: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(
        search_query, k=fetch_k
    )

    if debug is not None:
        debug.candidates_fetched = len(candidates)

    if not candidates:
        if debug is not None:
            debug.candidates_kept = 0
        return []

    rerank_scores = score_pairs(search_query, [doc.page_content for doc, _ in candidates])

    scored = [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            score=distance,
            rerank_score=rerank_score,
        )
        for (doc, distance), rerank_score in zip(candidates, rerank_scores)
    ]

    # Relevance gate: the cross-encoder score decides, not the raw distance.
    relevant = [c for c in scored if c.rerank_score >= settings.rerank_score_threshold]
    relevant.sort(key=lambda c: c.rerank_score, reverse=True)

    # Dedupe exact-duplicate content (can happen with overlapping chunks or
    # a document ingested more than once) — keep the first (highest-scoring)
    # occurrence.
    seen_content: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in relevant:
        if chunk.content in seen_content:
            continue
        seen_content.add(chunk.content)
        deduped.append(chunk)

    kept = deduped[:k]

    if debug is not None:
        debug.candidates_kept = len(kept)
        debug.candidates = [
            {
                "source": c.source,
                "rerank_score": round(c.rerank_score, 3),
                "vector_distance": round(c.score, 3),
                "passed_legacy_threshold": c.score <= settings.relevance_score_threshold,
                "kept": c in kept,
                "preview": c.content[:80].replace("\n", " "),
            }
            for c in scored
        ]

    return kept


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt,
    ordered by relevance (most relevant first)."""
    if not chunks:
        return ""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {chunk.source})\n{chunk.content}")
    return "\n\n".join(blocks)
