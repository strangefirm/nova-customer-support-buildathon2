"""FAISS as the embedding store: persistence, incremental updates, search quality and fallbacks (offline)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

import kb.vector_index as vi
from config import get_settings
from kb.bm25 import tokenize
from kb.store import KnowledgeBase


def _vec(text, dim=64):
    v = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / (n if n else 1)


class Counter:
    """A fake embedding function that records exactly which texts it was asked to embed."""

    def __init__(self, dim=64):
        self.calls, self.dim = [], dim

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.stack([_vec(t, self.dim) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    @property
    def embedded(self):
        return [t for c in self.calls for t in c]


DOCS = ["refund within thirty days", "reset your password by email", "standard shipping takes five days", "blender warranty twelve months"]


def build(texts=DOCS, emb=None, model="m", backend="faiss", cache=None):
    emb = emb or Counter()
    return vi.VectorIndex.build(texts, emb, model, backend, cache), emb


# ---------- the index is the embedding store ----------
def test_faiss_is_used_and_exact_for_small_corpora(tmp_path):
    idx, _ = build(cache=tmp_path)
    assert idx.backend == "faiss" and idx.kind == "flat"


def test_search_returns_the_right_chunk(tmp_path):
    idx, _ = build(cache=tmp_path)
    s = idx.scores(_vec("password email"), k=2)
    assert s.shape == (4,) and int(np.argmax(s)) == 1


def test_first_build_embeds_everything_and_persists_the_index(tmp_path):
    idx, emb = build(cache=tmp_path)
    assert sorted(emb.embedded) == sorted(DOCS) and idx.stats == {"embedded": 4, "reused": 0, "rewritten": True}
    assert (tmp_path / vi.INDEX_FILE).exists()


def test_restart_makes_zero_embedding_calls(tmp_path):
    build(cache=tmp_path)
    idx, emb = build(cache=tmp_path)  # a "new process": nothing in memory except the index file
    assert emb.embedded == [] and idx.stats == {"embedded": 0, "reused": 4, "rewritten": False}
    assert int(np.argmax(idx.scores(_vec("shipping days"), k=2))) == 2  # and it still searches correctly


def test_only_changed_chunks_are_re_embedded(tmp_path):
    build(cache=tmp_path)
    edited = DOCS[:3] + ["blender warranty eighteen months"]
    idx, emb = build(edited, cache=tmp_path)
    assert emb.embedded == ["blender warranty eighteen months"]
    assert idx.stats["reused"] == 3 and idx.stats["rewritten"]
    # unchanged vectors were reconstructed from the old index and still rank correctly
    assert int(np.argmax(idx.scores(_vec("refund days"), k=2))) == 0
    _, emb2 = build(edited, cache=tmp_path)
    assert emb2.embedded == []  # and the rewritten index is now the stored one


def test_removed_chunks_do_not_linger(tmp_path):
    build(cache=tmp_path)
    idx, emb = build(DOCS[:2], cache=tmp_path)
    assert emb.embedded == [] and idx.n == 2 and idx._index.ntotal == 2


def test_changing_the_model_never_reuses_old_vectors(tmp_path):
    build(cache=tmp_path, model="small")
    idx, emb = build(cache=tmp_path, model="large")
    assert sorted(emb.embedded) == sorted(DOCS) and idx.stats["reused"] == 0


def test_dimension_change_rebuilds_cleanly(tmp_path):
    build(cache=tmp_path, emb=Counter(dim=32))
    edited = DOCS[:3] + ["blender warranty eighteen months"]
    idx, emb = build(edited, cache=tmp_path, emb=Counter(dim=64))  # embedding size changed under the same identity
    assert idx.stats["reused"] == 0 and idx.backend == "faiss" and len(emb.embedded) == 4  # nothing old is reusable
    assert int(np.argmax(idx.scores(_vec("password email", 64), k=2))) == 1


def test_query_with_wrong_dimension_is_a_clear_error(tmp_path):
    idx, _ = build(cache=tmp_path, emb=Counter(dim=32))
    with pytest.raises(ValueError, match="dimensions"):
        idx.scores(_vec("refund", 64), k=2)


def test_provider_change_never_reuses_vectors(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS", "on")
    monkeypatch.setattr("kb.store.Embedder", HashEmbedder)
    HashEmbedder.seen = []
    KnowledgeBase(get_settings())
    first = len(HashEmbedder.seen)
    monkeypatch.setenv("LLM_BASE_URL", "https://other-provider.example/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    HashEmbedder.seen = []
    KnowledgeBase(get_settings())
    assert len(HashEmbedder.seen) == first  # a different provider URL re-embeds everything


def test_duplicate_chunks_are_embedded_once_and_both_scored(tmp_path):
    idx, emb = build(["same text", "other words", "same text"], cache=tmp_path)
    assert emb.embedded.count("same text") == 1
    s = idx.scores(_vec("same text"), k=3)
    assert s[0] == s[2] > s[1]


def test_corrupt_index_file_is_rebuilt_not_fatal(tmp_path):
    build(cache=tmp_path)
    (tmp_path / vi.INDEX_FILE).write_bytes(b"garbage")
    idx, emb = build(cache=tmp_path)
    assert idx.backend == "faiss" and sorted(emb.embedded) == sorted(DOCS)


def test_index_write_is_atomic_no_temp_file_left(tmp_path):
    build(cache=tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [vi.INDEX_FILE]


def test_k_larger_than_corpus_is_clamped(tmp_path):
    idx, _ = build(cache=tmp_path)
    ids, _ = idx.top_k(_vec("refund"), 100)
    assert len(ids) == 4


def test_hnsw_used_for_large_corpora_and_finds_the_nearest(tmp_path):
    rng = np.random.default_rng(2)
    n = vi.HNSW_MIN_VECTORS + 100
    texts = [f"chunk-{i}" for i in range(n)]
    table = rng.normal(size=(n, 48)).astype(np.float32)
    table /= np.linalg.norm(table, axis=1, keepdims=True)
    lookup = dict(zip(texts, table))
    idx = vi.VectorIndex.build(texts, lambda ts: np.stack([lookup[t] for t in ts]), "m", "faiss", tmp_path)
    assert idx.kind == "hnsw"
    hits = sum(int(np.argmax(idx.scores(table[i] + 0.01 * rng.normal(size=48).astype(np.float32), k=1)) == i) for i in range(0, 2000, 100))
    assert hits >= 19  # approximate search, near-perfect recall on exact-neighbour queries
    again = vi.VectorIndex.build(texts, lambda ts: (_ for _ in ()).throw(AssertionError("must not embed")), "m", "faiss", tmp_path)
    assert again.stats["embedded"] == 0 and again.kind == "hnsw"  # HNSW index also reloads from disk


# ---------- fallbacks ----------
def test_numpy_backend_searches_without_persisting(tmp_path):
    idx, emb = build(backend="numpy", cache=tmp_path)
    assert idx.backend == "numpy" and not list(tmp_path.iterdir())
    assert int(np.argmax(idx.scores(_vec("password email"), k=2))) == 1


def test_falls_back_to_numpy_when_faiss_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(vi, "faiss", None)
    idx, _ = build(cache=tmp_path)
    assert idx.backend == "numpy" and int(np.argmax(idx.scores(_vec("shipping days"), k=2))) == 2


def test_falls_back_to_numpy_when_faiss_errors(monkeypatch, tmp_path):
    class Broken:
        METRIC_INNER_PRODUCT = 0

        def IndexFlatIP(self, *_):
            raise RuntimeError("boom")

    monkeypatch.setattr(vi, "faiss", Broken())
    idx, _ = build(cache=tmp_path)
    assert idx.backend == "numpy" and int(np.argmax(idx.scores(_vec("password email"), k=2))) == 1


# ---------- hybrid retrieval end to end ----------
class HashEmbedder:
    """Deterministic bag-of-words embedder; counts the texts it embeds across instances."""

    seen: list = []

    def __init__(self, *_a, **_k):
        pass

    def embed(self, texts):
        HashEmbedder.seen.extend(texts)
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 512] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def hybrid_kb(monkeypatch):
    HashEmbedder.seen = []
    monkeypatch.setenv("EMBEDDINGS", "on")
    monkeypatch.setattr("kb.store.Embedder", HashEmbedder)
    return KnowledgeBase(get_settings())


def test_hybrid_mode_uses_faiss(hybrid_kb):
    assert hybrid_kb.mode == "hybrid" and hybrid_kb.index.backend == "faiss" and "FAISS" in hybrid_kb.retrieval_label


@pytest.mark.parametrize("query,stem", [
    ("Where is my order? It was supposed to arrive yesterday.", "shipping_policy"),
    ("I want to return the shoes I bought last week. How do I get a refund?", "refund_policy"),
    ("I forgot my password and can't log in.", "account_and_login"),
    ("The blender I received is broken. Can I get a replacement?", "product_faq"),
    ("You charged my card three times!", "billing_and_payments"),
])
def test_hybrid_retrieval_finds_right_document(hybrid_kb, query, stem):
    hits = hybrid_kb.search(query)
    assert hits and Path(hits[0].source).stem == stem


def test_hybrid_still_reports_no_match_for_out_of_scope(hybrid_kb):
    assert hybrid_kb.search("Do you sell gift cards for a company of 500 employees with custom branding?") == []
    assert hybrid_kb.search("what is the capital of France") == []


def test_knowledge_base_restart_does_not_re_embed_documents(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS", "on")
    monkeypatch.setattr("kb.store.Embedder", HashEmbedder)
    HashEmbedder.seen = []
    first = KnowledgeBase(get_settings())
    assert len(HashEmbedder.seen) == len(first.chunks) > 0
    HashEmbedder.seen = []
    again = KnowledgeBase(get_settings())  # simulated restart
    assert HashEmbedder.seen == [] and again.index.stats["reused"] > 0
    again.search("I forgot my password")
    assert HashEmbedder.seen == ["I forgot my password"]  # only the query is embedded


def test_numpy_backend_setting_is_respected(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS", "on")
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setattr("kb.store.Embedder", HashEmbedder)
    kb = KnowledgeBase(get_settings())
    assert kb.index.backend == "numpy" and kb.search("I forgot my password")[0].source.startswith("account_and_login")


def test_embedding_failure_degrades_to_keyword(monkeypatch):
    class Down:
        def __init__(self, *_a, **_k):
            pass

        def embed(self, _):
            raise ConnectionError("provider unreachable")

    monkeypatch.setenv("EMBEDDINGS", "on")
    monkeypatch.setattr("kb.store.Embedder", Down)
    kb = KnowledgeBase(get_settings())
    assert kb.mode == "keyword" and kb.index is None and kb.retrieval_label.startswith("Keyword")
    assert kb.search("I forgot my password and can't log in.")


def test_missing_faiss_is_reported_not_silent(monkeypatch, tmp_path):
    monkeypatch.setattr(vi, "faiss", None)
    idx, _ = build(cache=tmp_path)
    assert idx.backend == "numpy" and "not installed" in idx.note


def test_label_and_note_tell_the_user_when_faiss_is_unavailable(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS", "on")
    monkeypatch.setattr("kb.store.Embedder", HashEmbedder)
    monkeypatch.setattr(vi, "faiss", None)
    kb = KnowledgeBase(get_settings())
    assert kb.retrieval_label == "Hybrid · numpy (no FAISS)" and "faiss-cpu" in kb.index.note


def test_explicit_numpy_backend_has_no_warning(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS", "on")
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    monkeypatch.setattr("kb.store.Embedder", HashEmbedder)
    kb = KnowledgeBase(get_settings())
    assert kb.retrieval_label == "Hybrid · numpy" and kb.index.note == ""
