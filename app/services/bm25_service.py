"""BM25 scorer — exact port of the original experiment script.

idf = log(1 + (N - f + 0.5) / (f + 0.5))
score(d, q) = sum over query terms of
    idf(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

Query term frequency is not used (only term presence), matching the original.
"""

import math

import numpy as np


class BM25:
    """BM25 over pre-tokenized chunks."""

    def __init__(
        self, chunks: list[dict], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1, self.b = k1, b
        self.n = len(chunks)
        self.docs = [c["tokens"] for c in chunks]
        self.lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.lens) / max(1, self.n)
        df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {
            t: math.log(1 + (self.n - f + 0.5) / (f + 0.5))
            for t, f in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """Score every chunk for the query tokens (one score per chunk)."""
        qtf: dict[str, int] = {}
        for t in query_tokens:
            qtf[t] = qtf.get(t, 0) + 1
        out = np.zeros(self.n)
        for t, _ in qtf.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, d in enumerate(self.docs):
                f = d.count(t)
                if f:
                    denom = f + self.k1 * (
                        1 - self.b + self.b * self.lens[i] / self.avgdl
                    )
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out
