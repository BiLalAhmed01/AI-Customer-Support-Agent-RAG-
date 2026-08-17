"""Characterization tests for src/ingest.py's delete_document() — the
vector store is faked (no real Chroma DB touched) but the filesystem
deletion is real, against a pytest tmp_path.
"""

import src.ingest as ingest_module
from src.ingest import delete_document


class _FakeCollection:
    def __init__(self, sources):
        self._sources = sources

    def get(self, where=None, include=None):
        source = where.get("source") if where else None
        ids = [f"id-{i}" for i, s in enumerate(self._sources) if s == source]
        return {"ids": ids}


class _FakeVectorstore:
    def __init__(self, sources):
        self._collection = _FakeCollection(sources)
        self.deleted_where = None

    def delete(self, where=None):
        self.deleted_where = where


def _patch_chroma(monkeypatch, sources):
    fake = _FakeVectorstore(sources)
    monkeypatch.setattr(ingest_module, "Chroma", lambda **kwargs: fake)
    monkeypatch.setattr(ingest_module, "get_embeddings", lambda: object())
    return fake


def test_delete_document_removes_vectors_and_file(monkeypatch, tmp_path):
    fake = _patch_chroma(monkeypatch, sources=["keep.md", "remove.md"])
    target = tmp_path / "remove.md"
    target.write_text("content")

    removed = delete_document("remove.md", docs_dir=str(tmp_path))

    assert removed is True
    assert fake.deleted_where == {"source": "remove.md"}
    assert not target.exists()


def test_delete_document_removes_vectors_only_when_file_absent(monkeypatch, tmp_path):
    fake = _patch_chroma(monkeypatch, sources=["remove.md"])

    removed = delete_document("remove.md", docs_dir=str(tmp_path))

    assert removed is True
    assert fake.deleted_where == {"source": "remove.md"}


def test_delete_document_removes_file_only_when_not_indexed(monkeypatch, tmp_path):
    fake = _patch_chroma(monkeypatch, sources=[])
    target = tmp_path / "orphan.md"
    target.write_text("content")

    removed = delete_document("orphan.md", docs_dir=str(tmp_path))

    assert removed is True
    assert fake.deleted_where is None  # nothing indexed, delete() never called
    assert not target.exists()


def test_delete_document_returns_false_when_nothing_to_remove(monkeypatch, tmp_path):
    _patch_chroma(monkeypatch, sources=[])

    removed = delete_document("nonexistent.md", docs_dir=str(tmp_path))

    assert removed is False


def test_delete_document_rejects_path_traversal(monkeypatch, tmp_path):
    fake = _patch_chroma(monkeypatch, sources=["evil.md"])

    removed = delete_document("../../evil.md", docs_dir=str(tmp_path))

    # "../../evil.md" normalizes to basename "evil.md" — a real, safe
    # name, not a rejected one. This test locks in that normalization
    # behavior (matching safe_destination()'s approach in the Knowledge
    # Base page) rather than asserting rejection of the raw string.
    assert fake.deleted_where == {"source": "evil.md"}


def test_delete_document_rejects_dot_names(monkeypatch, tmp_path):
    _patch_chroma(monkeypatch, sources=["x"])

    assert delete_document("..", docs_dir=str(tmp_path)) is False
    assert delete_document(".", docs_dir=str(tmp_path)) is False
    assert delete_document("", docs_dir=str(tmp_path)) is False
