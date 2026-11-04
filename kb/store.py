"""Knowledge base: heading-aware chunking + hybrid (BM25 + embeddings) retrieval."""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import Settings, get_settings
from kb.bm25 import BM25
from kb.embeddings import Embedder
from kb.vector_index import VectorIndex
from models import KBHit

log = logging.getLogger(__name__)
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    text: str


def chunk_markdown(source: str, text: str) -> list[Chunk]:
    """Split on headings; every chunk keeps the document title for context."""
    title, heading, buf, chunks = "", "", [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            label = f"{title} - {heading}" if heading and heading != title else (title or heading)
            chunks.append(Chunk(source, heading or title, f"{label}\n{body}"))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            if len(m.group(1)) == 1:
                title = heading = m.group(2).strip()
            else:
                heading = m.group(2).strip()
        else:
            buf.append(line)
    flush()
    return chunks


def pdf_to_markdown(path: Path) -> str:
    """Extract text from a PDF, turning large-font lines into markdown headings.

    Headings are recognised by font size (>= 13pt => heading, >= 18pt => title), which is how our
    generated KB PDFs are styled (tools/build_kb_pdfs.py). Body text is kept as plain lines.
    """
    from pypdf import PdfReader

    segments: list[tuple[str, float]] = []

    def visit(text, cm, tm, font_dict, font_size):
        if text and text.strip():
            size = float(font_size or 0) * abs(float(tm[3])) if tm and tm[3] else float(font_size or 0)
            segments.append((text.strip(), size))

    for page in PdfReader(str(path)).pages:
        page.extract_text(visitor_text=visit)

    lines: list[str] = []
    pending_marker = ""
    for text, size in segments:
        if size >= 13:
            lines.append(("# " if size >= 18 else "## ") + text)
        elif re.fullmatch(r"[•\-*]|\d{1,2}\.?", text):  # list marker emitted as its own text run
            pending_marker = f"{text.rstrip('.')}. " if text[0].isdigit() else "- "
        else:
            lines.append(pending_marker + text)
            pending_marker = ""
    return "\n".join(lines)


def _read_document(path: Path) -> str:
    return pdf_to_markdown(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")


class KnowledgeBase:
    def __init__(self, settings: Settings | None = None):
        s = self.settings = settings or get_settings()
        self.chunks: list[Chunk] = []
        kb_dir = Path(s.kb_dir)
        for path in sorted(p for p in kb_dir.iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"}):
            try:
                self.chunks.extend(chunk_markdown(path.name, _read_document(path)))
            except Exception as exc:  # noqa: BLE001 - one unreadable file must not take the desk down
                log.warning("Skipping unreadable knowledge file %s (%s)", path.name, type(exc).__name__)
        self.bm25 = BM25([c.text for c in self.chunks])
        self.index: VectorIndex | None = None
        self.embedder: Embedder | None = None
        self.mode = "keyword"
        if s.embeddings_enabled and self.chunks:
            try:
                self.embedder = Embedder(s.api_key, s.base_url, s.embedding_model)
                self.index = VectorIndex.build(
                    [c.text for c in self.chunks], self.embedder.embed, f"{s.base_url}|{s.embedding_model}", s.vector_backend, s.cache_dir
                )
                self.mode = "hybrid"
            except Exception as exc:  # noqa: BLE001 - any provider failure must degrade, not crash
                log.warning("Embeddings unavailable (%s); using BM25 keyword search only.", type(exc).__name__)
                self.embedder = self.index = None

    @property
    def retrieval_label(self) -> str:
        if self.index is None:
            return "Keyword retrieval (BM25)"
        if self.index.backend == "faiss":
            return f"Hybrid · FAISS {self.index.kind}"
        return "Hybrid · numpy (no FAISS)" if self.index.note else "Hybrid · numpy"

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def search(self, query: str, k: int = 3, min_score: float | None = None) -> list[KBHit]:
        threshold = self.settings.kb_min_score if min_score is None else min_score
        if not self.chunks or not query.strip():
            return []
        kw = np.array(self.bm25.scores(query))
        scores = kw
        if self.index is not None and self.embedder is not None:
            try:
                q = self.embedder.embed([query])[0]
                cos = np.clip(self.index.scores(q, k=max(k * 6, 20)), 0, 1)
                # Cosine for unrelated text sits around 0.1-0.25 with current embedding models, so
                # rescale before blending so it does not mask the keyword signal.
                cos = np.clip((cos - 0.25) / 0.45, 0, 1)
                scores = 0.4 * kw + 0.6 * cos
            except Exception as exc:  # noqa: BLE001
                log.warning("Query embedding failed (%s); keyword only.", type(exc).__name__)
        order = np.argsort(-scores)[:k]
        return [
            KBHit(source=self.chunks[i].source, heading=self.chunks[i].heading, passage=self.chunks[i].text, score=round(float(scores[i]), 3))
            for i in order
            if scores[i] >= threshold
        ]


_lock = threading.Lock()
_cache: dict[tuple, KnowledgeBase] = {}


def get_kb(settings: Settings | None = None) -> KnowledgeBase:
    """Process-wide cached KnowledgeBase, rebuilt when documents or retrieval config change."""
    s = settings or get_settings()
    files = sorted(Path(s.kb_dir).glob("*.*"))
    sig = (str(s.kb_dir), s.embeddings_enabled, s.embedding_model, s.vector_backend, tuple((f.name, f.stat().st_mtime_ns) for f in files))
    with _lock:
        if sig not in _cache:
            _cache.clear()
            _cache[sig] = KnowledgeBase(s)
        return _cache[sig]
