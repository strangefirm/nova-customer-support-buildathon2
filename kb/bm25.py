"""Dependency-free BM25 with a coverage-normalised score so a threshold can mean "no match"."""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from had has have how i if in into is it its
    me my of on or our please so that the their them there they this to up us was we were what when where
    which who why will with would you your am been being did get got just not no than then very want need
    would should about after again all also any because before both each few more most other out over own
    same some such too until while yes hi hello thanks thank nobody
    novacart""".split()  # the brand name is in every document title, so it cannot discriminate between them
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ing", "ed", "es", "ly", "s", "e")

# Query-side expansion (applied to stems) so everyday phrasing reaches policy vocabulary.
EXPANSIONS = {
    "arriv": ["deliver", "ship", "track"],
    "late": ["delay", "deliver", "track"],
    "order": ["track"],
    "where": ["track"],
    "return": ["refund"],
    "broken": ["defective", "damaged", "replace"],
    "replac": ["defective", "replace"],
    "forgot": ["reset", "password"],
    "login": ["password", "reset"],
    "charg": ["billing", "payment"],
    "bank": ["dispute", "payment"],
}


def stem(token: str) -> str:
    if token.isdigit() or len(token) <= 3:
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.avgdl = (sum(len(t) for t in self.tokens) / self.n) if self.n else 0.0
        self.tf = [Counter(t) for t in self.tokens]
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))
        self.df = df

    def _idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def scores(self, query: str) -> list[float]:
        """Per-document score in roughly [0, 1]: BM25 divided by its maximum attainable value.

        Query terms absent from the whole corpus still count in the denominator, so a query that
        is mostly off-topic scores low everywhere (which is what lets us detect "no match").
        """
        terms = set(tokenize(query))
        for t in list(terms):
            terms.update(stem(x) for x in EXPANSIONS.get(t, ()))
        if not terms or not self.n:
            return [0.0] * self.n
        ceiling = sum(self._idf(t) for t in terms) * (self.k1 + 1)
        out = []
        for i in range(self.n):
            dl = len(self.tokens[i]) or 1
            s = 0.0
            for t in terms:
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                s += self._idf(t) * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s / ceiling if ceiling else 0.0)
        return out
