"""Cross-encoder reranking.

A bi-encoder (the embedding model in embeddings.py) scores query and
passage independently, then compares vectors — fast, but it misses
relevance when the wording differs (e.g. "How can I order?" vs "purchase
products through our online store" share little surface vocabulary).

A cross-encoder scores the query and passage *together* in one forward
pass, so it captures that kind of semantic match far more reliably. It's
a small local model (no API call, no added latency budget beyond a few
tens of milliseconds for a handful of candidates), which is why this is
the fix for wording mismatches rather than an LLM-based query rewrite.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Loaded once per process — same caching rationale as get_embeddings()."""
    return CrossEncoder(RERANKER_MODEL)


def score_pairs(query: str, passages: list[str]) -> list[float]:
    """Score each passage's relevance to `query`. Higher = more relevant.

    Raw cross-encoder logits, not a bounded [0,1] score — typically range
    roughly -11 (irrelevant) to +11 (highly relevant) for this model.
    """
    if not passages:
        return []
    reranker = get_reranker()
    pairs = [(query, passage) for passage in passages]
    return [float(s) for s in reranker.predict(pairs)]
