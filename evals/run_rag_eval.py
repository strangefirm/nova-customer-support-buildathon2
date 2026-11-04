"""Retrieval quality evaluation against the labelled gold set (evals/rag_gold.yaml).

  python -m evals.run_rag_eval                # keyword (BM25) retrieval, offline and free
  python -m evals.run_rag_eval --embeddings   # hybrid BM25 + embeddings + FAISS (calls your embedding provider)
  python -m evals.run_rag_eval --k 3 --out evals/rag_retrieval_report.md
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.rag_metrics import abstention, aggregate, key, score_retrieval  # noqa: E402

GOLD = Path(__file__).parent / "rag_gold.yaml"


def load_gold() -> list[dict]:
    return yaml.safe_load(GOLD.read_text(encoding="utf-8"))


def evaluate(kb, gold: list[dict], k: int = 3) -> dict:
    answerable = [g for g in gold if g["relevant"]]
    unanswerable = [g for g in gold if not g["relevant"]]
    scores, per_question = [], []
    for g in answerable:
        retrieved = [key(h.source, h.heading) for h in kb.search(g["question"], k=k)]
        sc = score_retrieval(retrieved, g["relevant"], k)
        scores.append(sc)
        per_question.append((g, sc))
    leaks = []  # unanswerable questions that still retrieved something: (question, [(passage, score)])
    unanswerable_retrieved = []
    for g in unanswerable:
        hits = kb.search(g["question"], k=k)
        unanswerable_retrieved.append([key(h.source, h.heading) for h in hits])
        if hits:
            leaks.append((g, [(key(h.source, h.heading), h.score) for h in hits]))
    return {"k": k, "retrieval": aggregate(scores), "abstention": abstention(unanswerable_retrieved),
            "per_question": per_question, "leaks": leaks}


def render(result: dict, mode: str) -> str:
    r, a, k = result["retrieval"], result["abstention"], result["k"]
    lines = [
        "# Retrieval evaluation", "",
        f"- Mode: **{mode}**  ·  k = {k}  ·  {r['n']} answerable questions, {a['n']} unanswerable",
        "- Labels are hand-written section-level judgements (`evals/rag_gold.yaml`); a small, author-labelled set, so treat the numbers as a regression gauge, not a benchmark.", "",
        "| Metric | Value |", "|---|---|",
        f"| Precision@1 (top passage is relevant) | {r['p1']:.1%} |", f"| Precision@{k} | {r['precision']:.1%} |", f"| Recall@{k} | {r['recall']:.1%} |",
        f"| Hit rate@{k} (any relevant section retrieved) | {r['hit']:.1%} |", f"| MRR | {r['mrr']:.3f} |", f"| nDCG@{k} | {r['ndcg']:.3f} |",
        f"| Abstention on unanswerable questions | {a['abstained']}/{a['n']} ({a['rate']:.0%}) |", "",
    ]
    misses = [(g, s) for g, s in result["per_question"] if s.recall < 1.0 or s.mrr < 1.0]
    if misses:
        lines += ["## Questions where a relevant section was missed or ranked below the top", "", "| Id | Question | Relevant | Retrieved |", "|---|---|---|---|"]
        for g, s in misses:
            lines.append(f"| {g['id']} | {g['question'][:60]} | {'; '.join(s.relevant)} | {'; '.join(s.retrieved) or '(nothing)'} |")
    if result["leaks"]:
        lines += ["", "## Unanswerable questions that still retrieved passages (false support)", "",
                  "| Id | Question | Retrieved (score) |", "|---|---|---|"]
        for g, hits in result["leaks"]:
            lines.append(f"| {g['id']} | {g['question'][:60]} | {'; '.join(f'{p} ({sc:.2f})' for p, sc in hits)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true", help="use hybrid retrieval (needs an embedding provider)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).parent / "rag_retrieval_report.md"))
    args = ap.parse_args()

    if not args.embeddings:
        os.environ["EMBEDDINGS"] = "off"
    else:
        os.environ["EMBEDDINGS"] = "on"
    os.environ.setdefault("CACHE_DIR", str(Path(tempfile.mkdtemp(prefix="ragcache_"))) if not args.embeddings else str(ROOT / ".cache"))
    os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="ragdb_")) / "x.db")

    from config import get_settings
    from kb.store import KnowledgeBase

    kb = KnowledgeBase(get_settings())
    mode = kb.retrieval_label
    if args.embeddings and kb.index is None:
        print("Embeddings are not available (check OPENAI_API_KEY / provider); aborting rather than reporting keyword numbers as hybrid.")
        return 2
    report = render(evaluate(kb, load_gold(), args.k), mode)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
