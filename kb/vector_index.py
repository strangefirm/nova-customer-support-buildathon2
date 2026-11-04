"""FAISS vector index that IS the embedding store.

Every chunk's vector is stored in the index under an ID derived from a hash of (model, chunk text)
(IndexIDMap2 around the real index). So the index file alone knows which vectors belong to which
content: no separate cache, database table or JSON file.

  * restart:           the index file is loaded; if the chunk set is unchanged, ZERO embedding calls are made
  * a document edited: only the new/changed chunks are embedded; unchanged vectors are reconstructed
                       straight out of the old index, then the index is rewritten atomically
  * model changed:     IDs include the model identity (provider URL + model name), so old vectors are never reused
                       for a different model
  * small corpora:     IndexFlatIP (exact search); >= HNSW_MIN_VECTORS: IndexHNSWFlat (approximate, scales to millions)
  * vectors are unit-normalised, so inner product == cosine similarity
  * no faiss / faiss error / VECTOR_BACKEND=numpy: search falls back to numpy over in-memory vectors
    (that path is not persisted, so embeddings are recomputed at startup)
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import faiss
except ImportError:  # faiss-cpu not installed
    faiss = None

log = logging.getLogger(__name__)
HNSW_MIN_VECTORS = 2000
INDEX_FILE = "kb_vectors.faiss"


def content_id(model: str, text: str) -> int:
    """Stable positive int64 for (model, text); this is the vector's ID inside FAISS."""
    return int.from_bytes(hashlib.sha256(f"{model}\n{text}".encode()).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


class VectorIndex:
    def __init__(self):
        self.backend = "numpy"
        self.kind = "numpy"
        self.note = ""  # why FAISS is not in use, if it was requested but unavailable
        self.n = 0  # number of chunks (including duplicates)
        self.stats = {"embedded": 0, "reused": 0, "rewritten": False}
        self._index = None
        self._vectors: np.ndarray | None = None  # only kept for the numpy path
        self._ids: np.ndarray | None = None
        self._pos: dict[int, list[int]] = {}  # content id -> chunk positions

    # ---------- construction ----------
    @classmethod
    def build(cls, texts: list[str], embed_fn: Callable[[list[str]], np.ndarray], model: str,
              backend: str = "faiss", cache_dir: Path | None = None) -> "VectorIndex":
        self = cls()
        self.n = len(texts)
        ids = [content_id(model, t) for t in texts]
        for pos, i in enumerate(ids):
            self._pos.setdefault(i, []).append(pos)
        first_text = {i: texts[p[0]] for i, p in self._pos.items()}
        want = list(first_text)  # unique ids, stable order

        use_faiss = backend == "faiss" and faiss is not None and bool(want)
        if backend == "faiss" and faiss is None:
            self.note = "faiss-cpu is not installed in this Python environment (pip install faiss-cpu)"
            log.warning("FAISS requested but %s", self.note)
        path = Path(cache_dir) / INDEX_FILE if cache_dir else None
        known: dict[int, np.ndarray] = {}
        loaded = None
        if use_faiss and path and path.exists():
            loaded, known = self._load(path)

        missing = [i for i in want if i not in known]
        new_vecs = embed_fn([first_text[i] for i in missing]) if missing else np.zeros((0, 0), dtype=np.float32)
        fresh = dict(zip(missing, new_vecs))
        if missing and known and next(iter(known.values())).shape[0] != new_vecs.shape[1]:
            # embedding dimension changed: nothing stored is reusable, so embed only the chunks we have not just embedded
            rest = [i for i in want if i not in fresh]
            fresh.update(zip(rest, embed_fn([first_text[i] for i in rest])))
            known, loaded, missing = {}, None, want
        self.stats.update(embedded=len(missing), reused=len(want) - len(missing))

        if use_faiss:
            try:
                if loaded is not None and not missing and set(known) == set(want):
                    self._index = loaded  # identical chunk set: use the stored index as is
                else:
                    mat = np.stack([known.get(i, fresh.get(i)) for i in want]).astype(np.float32)
                    self._index = self._make(mat, np.array(want, dtype=np.int64))
                    self.stats["rewritten"] = True
                    if path:
                        self._save(path)
                self.backend, self.kind = "faiss", "hnsw" if len(want) >= HNSW_MIN_VECTORS else "flat"
                return self
            except Exception as exc:  # noqa: BLE001 - never let the index take retrieval down
                self.note = f"FAISS failed ({type(exc).__name__}); using numpy"
                log.warning("FAISS unavailable (%s); using numpy search.", type(exc).__name__)
                self._index = None
        if want:
            self._ids = np.array(want, dtype=np.int64)
            self._vectors = np.stack([known.get(i, fresh.get(i)) for i in want]).astype(np.float32)
        return self

    @staticmethod
    def _make(mat: np.ndarray, ids: np.ndarray):
        dim = mat.shape[1]
        if len(ids) >= HNSW_MIN_VECTORS:
            base = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = 80
            base.hnsw.efSearch = 64
        else:
            base = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(np.ascontiguousarray(mat), ids)
        return index

    @staticmethod
    def _load(path: Path):
        """Read the stored index and pull every vector back out of it, keyed by content id."""
        try:
            index = faiss.read_index(str(path))
            if index.ntotal == 0:
                return None, {}
            stored_ids = faiss.vector_to_array(index.id_map)
            vectors = index.index.reconstruct_n(0, index.ntotal)
            return index, {int(i): v for i, v in zip(stored_ids, vectors)}
        except Exception as exc:  # noqa: BLE001 - a corrupt or incompatible file is simply rebuilt
            log.warning("Stored FAISS index unreadable (%s); rebuilding.", type(exc).__name__)
            return None, {}

    def _save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            faiss.write_index(self._index, str(tmp))
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written index
        except Exception as exc:  # noqa: BLE001 - persistence is an optimisation only
            log.warning("FAISS index not saved (%s).", type(exc).__name__)

    # ---------- search ----------
    def top_k(self, query: np.ndarray, k: int) -> tuple[list[int], np.ndarray]:
        """(content ids, cosine scores) of the k nearest unique vectors, best first."""
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        dim = self._index.d if self._index is not None else (self._vectors.shape[1] if self._vectors is not None else q.shape[1])
        if q.shape[1] != dim:
            raise ValueError(f"query has {q.shape[1]} dimensions but the index has {dim}; the embedding model changed")
        if self._index is not None:
            k = max(1, min(k, self._index.ntotal))
            scores, ids = self._index.search(q, k)
            keep = ids[0] >= 0
            return [int(i) for i in ids[0][keep]], scores[0][keep]
        if self._vectors is None:
            return [], np.zeros(0, dtype=np.float32)
        sims = self._vectors @ q[0]
        order = np.argsort(-sims)[: max(1, min(k, len(sims)))]
        return [int(self._ids[o]) for o in order], sims[order]

    def scores(self, query: np.ndarray, k: int) -> np.ndarray:
        """Full-length (one per chunk) cosine array: top-k from the index, zero elsewhere, so it can be fused with BM25."""
        out = np.zeros(self.n, dtype=np.float32)
        ids, sims = self.top_k(query, k)
        for cid, sim in zip(ids, sims):
            for pos in self._pos.get(cid, ()):
                out[pos] = sim
        return out
