"""OpenAI-compatible embeddings client. Stateless: persistence is the FAISS index's job (kb/vector_index.py).

Any failure raises, and the caller degrades to keyword search.
"""
from __future__ import annotations

import numpy as np

BATCH = 96
_QUERY_MEMO_MAX = 256


class Embedder:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self._api_key, self._base_url = api_key, base_url
        self._client = None
        self._memo: dict[str, np.ndarray] = {}  # in-memory only: repeated identical queries cost nothing

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url or None, timeout=30, max_retries=2)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) unit-normalised float32 matrix."""
        if len(texts) == 1 and texts[0] in self._memo:
            return self._memo[texts[0]][None, :]
        rows: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            resp = self._ensure_client().embeddings.create(model=self.model, input=texts[i : i + BATCH])
            rows.extend(item.embedding for item in resp.data)
        mat = np.array(rows, dtype=np.float32)
        mat = mat / np.where((n := np.linalg.norm(mat, axis=1, keepdims=True)) == 0, 1, n)
        if len(texts) == 1:
            if len(self._memo) >= _QUERY_MEMO_MAX:
                self._memo.pop(next(iter(self._memo)))
            self._memo[texts[0]] = mat[0]
        return mat
