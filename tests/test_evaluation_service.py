"""Unit tests for the evaluation service."""

from app.services.evaluation_service import is_gold_chunk


def test_is_gold_chunk_requires_gold_post_and_terms(evaluation_service):
    questions = evaluation_service.repository.load_questions()
    q1 = questions[0]  # gold: your-agents-code-looks-nothing-like-yours
    assert q1.answer_terms == ["convention", "checker", "exit code"]

    gold_chunk = {
        "slug": q1.gold,
        "text": "A convention checker fails with a non-zero exit code.",
        "tokens": [],
    }
    assert is_gold_chunk(gold_chunk, q1) is True
    assert is_gold_chunk(gold_chunk, q1, min_terms=1) is True

    one_term = {
        "slug": q1.gold,
        "text": "A convention is a rule.",
        "tokens": [],
    }
    assert is_gold_chunk(one_term, q1) is False  # only 1 of 2 terms
    assert is_gold_chunk(one_term, q1, min_terms=1) is True

    wrong_post = {
        "slug": "how-a-model-learns",
        "text": gold_chunk["text"],
        "tokens": [],
    }
    assert is_gold_chunk(wrong_post, q1) is False


def test_evaluate_bm25_200w_headline(evaluation_service):
    result = evaluation_service.evaluate(retriever="bm25", chunk_size=200)
    assert result["chunks"] == 244
    assert result["recall@1"] == 0.85
    assert result["recall@5"] == 0.95
    assert result["missed_at_5"] == 1
    assert result["never_found"] == 0
    assert len(result["gold_ranks"]) == 20


def test_evaluate_chunk_level_200w_headline(evaluation_service):
    result = evaluation_service.evaluate(
        retriever="bm25", chunk_size=200, grading="chunk"
    )
    assert result["recall@1"] == 0.55
    assert result["recall@5"] == 0.70
    assert result["missed_at_5"] == 6
    assert result["never_found"] == 0


def test_evaluate_title_prepend_post_level(evaluation_service):
    result = evaluation_service.evaluate(
        retriever="bm25", chunk_size=200, title_prepend=True
    )
    assert result["recall@1"] == 0.90
    assert result["recall@5"] == 1.0
    assert result["missed_at_5"] == 0


def test_evaluate_loose_min_terms(evaluation_service):
    result = evaluation_service.evaluate(
        retriever="bm25", chunk_size=200, grading="chunk", min_terms=1
    )
    assert result["recall@1"] == 0.80
    assert result["recall@5"] == 0.95


def test_evaluate_tfidf_200w(evaluation_service):
    result = evaluation_service.evaluate(retriever="tfidf", chunk_size=200)
    assert result["recall@1"] == 0.85
    assert result["recall@5"] == 0.95
