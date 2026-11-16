"""TF-IDF scorer — exact port of the original experiment script.

Term weight = (1 + log(tf)) * idf, with idf = log((N + 1) / (df + 1)) + 1.
Document rows are L2-normalized; the query vector is L2-normalized, so the
score is cosine similarity. The vocabulary order is deterministic
(first-appearance order) instead of the original's set-iteration order;
scores are identical up to floating-point summation order.
"""

import math

import numpy as np


class TFIDF:
    """TF-IDF with log term frequency and cosine similarity."""

    def __init__(self, chunks: list[dict]) -> None:
        self.n = len(chunks)
        vocab: dict[str, int] = {}
        df: dict[str, int] = {}
        for c in chunks:
            for t in set(c["tokens"]):
                df[t] = df.get(t, 0) + 1
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab
        self.idf = np.array(
            [math.log((self.n + 1) / (df[t] + 1)) + 1 for t in vocab]
        )
        mat = np.zeros((self.n, len(vocab)))
        for i, c in enumerate(chunks):
            tf: dict[str, int] = {}
            for t in c["tokens"]:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                mat[i, vocab[t]] = (1 + math.log(f)) * self.idf[vocab[t]]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.mat = mat / np.where(norms == 0, 1, norms)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """Score every chunk for the query tokens (one score per chunk)."""
        q = np.zeros(len(self.vocab))
        for t in query_tokens:
            if t in self.vocab:
                q[self.vocab[t]] += 1
        n = np.linalg.norm(q)
        if n == 0:
            return np.zeros(self.n)
        return self.mat @ (q / n)
