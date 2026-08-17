"""Characterization tests for src/retrieval.py.

normalize_query / is_small_talk / build_search_query are pure and tested
directly. retrieve()'s threshold/dedupe/sort logic is tested against a
faked vectorstore + reranker (monkeypatched) so this suite never opens a
real Chroma DB or downloads a model — it locks in the *filtering logic*,
not the real embedding/reranking quality.
"""

from dataclasses import dataclass

import src.retrieval as retrieval
from src.retrieval import build_search_query, is_small_talk, normalize_query


def test_normalize_query_collapses_whitespace_and_punctuation():
    assert normalize_query("how   can i order???") == "how can i order?"
    assert normalize_query("  hi there  ") == "hi there"


def test_is_small_talk_exact_matches():
    assert is_small_talk("hi") is True
    assert is_small_talk("thanks") is True
    assert is_small_talk("Thanks!") is True  # normalize_query lowercases upstream in practice
    assert is_small_talk("ok.") is True


def test_is_small_talk_does_not_swallow_a_real_question():
    # A real question riding on a pleasantry must NOT be treated as small talk.
    assert is_small_talk("thanks, what about shipping?") is False
    assert is_small_talk("how much does this cost") is False


def test_build_search_query_no_history_returns_unchanged():
    query, expanded = build_search_query("what about international?", history=None)
    assert query == "what about international?"
    assert expanded is False


def test_build_search_query_expands_short_referential_query():
    history = [
        {"role": "user", "content": "Do you ship to Canada?"},
        {"role": "assistant", "content": "Yes, we ship to Canada."},
    ]
    query, expanded = build_search_query("what about the US?", history)
    assert expanded is True
    assert query.startswith("Do you ship to Canada?")
    assert "what about the US?" in query


def test_build_search_query_leaves_standalone_question_alone():
    history = [
        {"role": "user", "content": "Do you ship to Canada?"},
        {"role": "assistant", "content": "Yes, we ship to Canada."},
    ]
    # Long, self-contained question with no referential words/openers —
    # must NOT be expanded with unrelated prior context.
    query, expanded = build_search_query(
        "What is your standard warranty period for electronics?", history
    )
    assert expanded is False
    assert query == "What is your standard warranty period for electronics?"


@dataclass
class _FakeDoc:
    page_content: str
    metadata: dict


class _FakeVectorstore:
    def __init__(self, candidates):
        self._candidates = candidates

    def similarity_search_with_score(self, query, k):
        return self._candidates[:k]


def _patch_pipeline(monkeypatch, candidates, rerank_scores):
    monkeypatch.setattr(retrieval, "collection_is_empty", lambda: False)
    monkeypatch.setattr(retrieval, "_get_vectorstore", lambda: _FakeVectorstore(candidates))
    monkeypatch.setattr(retrieval, "score_pairs", lambda query, passages: rerank_scores)


def test_retrieve_filters_below_threshold(monkeypatch):
    candidates = [
        (_FakeDoc("relevant passage", {"source": "a.md"}), 0.5),
        (_FakeDoc("irrelevant passage", {"source": "b.md"}), 0.6),
    ]
    # Default threshold is -8.0 (settings.rerank_score_threshold) — one
    # chunk clears it, one doesn't.
    _patch_pipeline(monkeypatch, candidates, rerank_scores=[5.0, -9.0])

    result = retrieval.retrieve("some query")

    assert len(result) == 1
    assert result[0].source == "a.md"


def test_retrieve_sorts_by_rerank_score_descending(monkeypatch):
    candidates = [
        (_FakeDoc("lower relevance", {"source": "a.md"}), 0.5),
        (_FakeDoc("higher relevance", {"source": "b.md"}), 0.5),
    ]
    _patch_pipeline(monkeypatch, candidates, rerank_scores=[1.0, 5.0])

    result = retrieval.retrieve("some query")

    assert [c.source for c in result] == ["b.md", "a.md"]


def test_retrieve_dedupes_identical_content(monkeypatch):
    candidates = [
        (_FakeDoc("duplicate text", {"source": "a.md"}), 0.5),
        (_FakeDoc("duplicate text", {"source": "b.md"}), 0.5),
    ]
    _patch_pipeline(monkeypatch, candidates, rerank_scores=[5.0, 4.0])

    result = retrieval.retrieve("some query")

    assert len(result) == 1
    assert result[0].source == "a.md"  # higher-scoring occurrence kept


def test_retrieve_respects_k_limit(monkeypatch):
    candidates = [
        (_FakeDoc(f"passage {i}", {"source": f"{i}.md"}), 0.5) for i in range(5)
    ]
    _patch_pipeline(monkeypatch, candidates, rerank_scores=[5.0] * 5)

    result = retrieval.retrieve("some query", k=2)

    assert len(result) == 2


def test_retrieve_returns_empty_list_when_collection_empty(monkeypatch):
    monkeypatch.setattr(retrieval, "collection_is_empty", lambda: True)

    result = retrieval.retrieve("some query")

    assert result == []
