"""Replication test: the API must reproduce the reference results.json.

The reference was produced by the original experiment script
(``measure-retrieval.py``). Any math deviation in the port shows up here:
recall values are fractions of 20, so a single flipped rank moves a metric
by 0.05. Headline numbers are asserted with exact equality.
"""

from datetime import date

import pytest

# Headline numbers from the blog post / reference results.json.
HEADLINES = {
    ("bm25@200w", "recall@1"): 0.85,
    ("bm25@200w", "recall@5"): 0.95,
    ("chunk-bm25@200w", "recall@1"): 0.55,
    ("chunk-bm25@200w", "recall@5"): 0.70,
    ("bm25+title@200w", "recall@5"): 1.0,
    ("chunk-bm25@400w", "recall@5"): 0.90,
    ("chunk-loose>=1-bm25@200w", "recall@5"): 0.95,
}


def _assert_close(actual, expected, path=""):
    """Recursive comparison: floats within 1e-9, everything else exact."""
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)), path
        assert actual == pytest.approx(expected, abs=1e-9), path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, path


def test_headline_numbers_exact(evaluation_service, repository):
    result = evaluation_service.run_full_experiment()
    for (experiment, metric), expected in HEADLINES.items():
        actual = result["experiments"][experiment][metric]
        assert actual == expected, (
            f"{experiment} {metric}: {actual} != {expected}"
        )


def test_full_experiments_match_reference(evaluation_service, repository):
    result = evaluation_service.run_full_experiment()
    reference = repository.load_reference_results()
    assert set(result["experiments"]) == set(reference["experiments"])
    for name, expected in reference["experiments"].items():
        _assert_close(
            result["experiments"][name], expected, path=f"experiments.{name}"
        )


def test_corpus_and_questions_match_reference(evaluation_service, repository):
    result = evaluation_service.run_full_experiment()
    reference = repository.load_reference_results()
    assert result["corpus"]["posts"] == reference["corpus"]["posts"] == 14
    assert result["corpus"]["words"] == reference["corpus"]["words"] == 38433
    assert result["corpus"]["files"] == reference["corpus"]["files"]
    assert result["corpus"]["excluded"] == reference["corpus"]["excluded"]
    assert result["corpus"]["generated"] == date.today().isoformat()
    assert result["questions"] == reference["questions"] == 20


def test_per_question_rows_match_reference(evaluation_service, repository):
    result = evaluation_service.run_full_experiment()
    reference = repository.load_reference_results()
    assert result["per_question_200w"] == reference["per_question_200w"]


def test_rewording_ranks_match_reference(evaluation_service, repository):
    result = evaluation_service.run_full_experiment()
    reference = repository.load_reference_results()
    actual_items = result["rewording"]["items"]
    expected_items = reference["rewording"]["items"]
    assert len(actual_items) == len(expected_items)
    for actual, expected in zip(actual_items, expected_items):
        assert actual["id"] == expected["id"]
        assert actual["gold"] == expected["gold"]
        for a_row, e_row in zip(actual["rows"], expected["rows"]):
            assert a_row["label"] == e_row["label"]
            assert a_row["text"] == e_row["text"]
            assert a_row["gold_rank"] == e_row["gold_rank"]
            assert a_row["answer_chunk_rank"] == e_row["answer_chunk_rank"]
            assert [t["slug"] for t in a_row["top5"]] == [
                t["slug"] for t in e_row["top5"]
            ]
            for a_top, e_top in zip(a_row["top5"], e_row["top5"]):
                assert a_top["score"] == pytest.approx(
                    e_top["score"], abs=0.01
                )
                assert a_top["preview"] == e_top["preview"]


def test_top5_dump_matches_reference(evaluation_service, repository):
    result = evaluation_service.run_full_experiment()
    reference = repository.load_reference_results()
    assert len(result["top5_bm25_200w"]) == len(reference["top5_bm25_200w"])
    for actual, expected in zip(
        result["top5_bm25_200w"], reference["top5_bm25_200w"]
    ):
        assert actual["id"] == expected["id"]
        assert actual["q"] == expected["q"]
        assert actual["gold"] == expected["gold"]
        assert actual["gold_rank"] == expected["gold_rank"]
        assert [t["slug"] for t in actual["top5"]] == [
            t["slug"] for t in expected["top5"]
        ]
        for a_top, e_top in zip(actual["top5"], expected["top5"]):
            assert a_top["score"] == pytest.approx(e_top["score"], abs=0.01)
            assert a_top["preview"] == e_top["preview"]
