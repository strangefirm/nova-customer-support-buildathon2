"""Pure metric functions for RAG evaluation (no I/O, no model calls), so they are unit-testable.

Retrieval is scored per question against the labelled relevant sections; the unit is the retrieved
passage's (document, section). "k" is the number of passages the Resolution agent actually receives.

  precision@k  relevant retrieved / retrieved            (retrieved can be < k: weak matches are dropped; with one relevant
               section and k=3 it cannot exceed 1/3, so read it together with precision@1)
  precision@1  1 if the top-ranked passage is relevant
  recall@k     relevant retrieved / relevant total
  hit@k        1 if at least one relevant section was retrieved
  MRR          1 / rank of the first relevant section (0 if none)
  nDCG@k       rank-aware gain with binary relevance
  abstention   for questions the KB cannot answer: retrieved nothing
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def key(source: str, heading: str) -> str:
    """Canonical id of a passage: '<file stem>#<section heading>'."""
    stem = source.rsplit(".", 1)[0]
    return f"{stem}#{heading.strip()}"


@dataclass
class RetrievalScore:
    precision: float
    p1: float
    recall: float
    hit: float
    mrr: float
    ndcg: float
    retrieved: list[str] = field(default_factory=list)
    relevant: list[str] = field(default_factory=list)


def score_retrieval(retrieved: list[str], relevant: list[str], k: int = 3) -> RetrievalScore:
    """Score one answerable question. `retrieved` is ranked best first."""
    top = retrieved[:k]
    rel = set(relevant)
    hits = [r for r in top if r in rel]
    precision = len(hits) / len(top) if top else 0.0
    recall = len(set(hits)) / len(rel) if rel else 0.0
    first = next((i for i, r in enumerate(top, 1) if r in rel), None)
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(top, 1) if r in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return RetrievalScore(precision, 1.0 if top and top[0] in rel else 0.0, recall, 1.0 if hits else 0.0, 1.0 / first if first else 0.0,
                          dcg / ideal if ideal else 0.0, top, sorted(rel))


def aggregate(scores: list[RetrievalScore]) -> dict:
    n = len(scores)
    if not n:
        return {"n": 0, "precision": 0.0, "p1": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0, "ndcg": 0.0}
    mean = lambda f: round(sum(f(s) for s in scores) / n, 4)  # noqa: E731
    return {"n": n, "precision": mean(lambda s: s.precision), "p1": mean(lambda s: s.p1), "recall": mean(lambda s: s.recall),
            "hit": mean(lambda s: s.hit), "mrr": mean(lambda s: s.mrr), "ndcg": mean(lambda s: s.ndcg)}


def abstention(retrieved_per_unanswerable: list[list[str]]) -> dict:
    """Of the questions the KB cannot answer, how many correctly retrieved nothing?"""
    n = len(retrieved_per_unanswerable)
    correct = sum(1 for r in retrieved_per_unanswerable if not r)
    return {"n": n, "abstained": correct, "rate": round(correct / n, 4) if n else 0.0}


# ---------- generation / hallucination ----------
@dataclass
class Claim:
    text: str
    supported: bool
    cited: list[int] = field(default_factory=list)


def hallucination_stats(replies: list[list[Claim]]) -> dict:
    """claim-level rate = unsupported claims / all claims; response-level = replies with >= 1 unsupported claim / replies."""
    claims = [c for r in replies for c in r]
    unsupported = [c for c in claims if not c.supported]
    with_claims = [r for r in replies if r]
    return {
        "replies": len(replies),
        "claims": len(claims),
        "unsupported_claims": len(unsupported),
        "claim_hallucination_rate": round(len(unsupported) / len(claims), 4) if claims else 0.0,
        "response_hallucination_rate": round(sum(1 for r in with_claims if any(not c.supported for c in r)) / len(with_claims), 4)
        if with_claims else 0.0,
    }
