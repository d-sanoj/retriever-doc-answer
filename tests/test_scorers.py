"""Unit tests for the BM25 and TF-IDF scorers."""

import math

import numpy as np

from app.services.bm25_service import BM25
from app.services.retrieval_service import RetrievalService
from app.services.text_service import tokenize
from app.services.tfidf_service import TFIDF


def _chunks():
    docs = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are animals",
    ]
    return [
        {"slug": f"doc{i}", "text": d, "tokens": tokenize(d)}
        for i, d in enumerate(docs)
    ]


def test_bm25_idf_formula():
    chunks = _chunks()
    bm25 = BM25(chunks)
    # "cat" appears in 1 of 3 docs; "sat" in 2 of 3.
    assert bm25.idf["cat"] == math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    assert bm25.idf["sat"] == math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    # "the" is a stopword: not in the vocabulary.
    assert "the" not in bm25.idf


def test_bm25_ranks_matching_doc_first():
    chunks = _chunks()
    scores = BM25(chunks).scores(tokenize("cat sat"))
    assert scores[0] > scores[1] > scores[2]
    # Unknown query terms score zero everywhere.
    assert np.all(BM25(chunks).scores(tokenize("zebra")) == 0)


def test_bm25_ignores_query_term_frequency():
    chunks = _chunks()
    once = BM25(chunks).scores(tokenize("cat"))
    twice = BM25(chunks).scores(tokenize("cat cat"))
    np.testing.assert_array_equal(once, twice)


def test_tfidf_cosine_and_unknown_terms():
    chunks = _chunks()
    tfidf = TFIDF(chunks)
    scores = tfidf.scores(tokenize("cat mat"))
    assert scores[0] > scores[1]
    assert scores[0] <= 1.0 + 1e-12  # cosine similarity is bounded by 1
    assert np.all(tfidf.scores(tokenize("zebra")) == 0)


def test_tfidf_rows_are_l2_normalized():
    chunks = _chunks()
    tfidf = TFIDF(chunks)
    norms = np.linalg.norm(tfidf.mat, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(chunks)))


def test_index_chunk_counts_match_original(repository):
    """BM25 @200w over the bundled corpus must have 244 chunks."""
    posts = repository.load_posts()
    assert len(posts) == 14
    chunks = RetrievalService.build_index(posts, 200)
    assert len(chunks) == 244
    assert len(RetrievalService.build_index(posts, 100)) == 483
    assert len(RetrievalService.build_index(posts, 400)) == 123
    assert len(RetrievalService.build_index(posts, 0)) == 14
