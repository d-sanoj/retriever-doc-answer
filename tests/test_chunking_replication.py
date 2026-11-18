"""Replication test: the chunking experiment must reproduce the
reference data/chunking/results-chunking.json (2026-11-18 post).

Headline numbers are asserted with exact equality. Recall values are
fractions of 22, so a single flipped rank moves a metric by ~0.045.
"""

from datetime import date

import pytest

# Headline numbers from the blog post / reference results-chunking.json.
HEADLINES = {
    ("chunk-bm25@section", "recall@1"): 0.5,
    ("chunk-bm25@section", "recall@5"): 20 / 22,
    ("chunk-bm25@fixed200", "recall@1"): 0.5454545454545454,
    ("chunk-bm25@fixed200", "recall@5"): 16 / 22,
    ("bm25@section", "recall@5"): 1.0,
    ("bm25@fixed200", "recall@5"): 21 / 22,
    ("chunk-bm25@fixed400", "recall@5"): 20 / 22,
    ("chunk-bm25@fixed200+head", "recall@5"): 17 / 22,
    ("chunk-bm25@para200", "recall@5"): 16 / 22,
}

CHUNK_COUNTS = {
    "fixed200": 263,
    "fixed400": 132,
    "fixed200+head": 256,
    "section": 170,
    "para200": 271,
}

MEDIAN_WORDS = {
    "fixed200": 200,
    "fixed400": 400,
    "fixed200+head": 206,
    "section": 236,
    "para200": 182,
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


def test_headline_numbers_exact(
    chunking_evaluation_service, chunking_repository
):
    result = chunking_evaluation_service.run_chunking_experiment()
    for (experiment, metric), expected in HEADLINES.items():
        actual = result["experiments"][experiment][metric]
        assert actual == expected, (
            f"{experiment} {metric}: {actual} != {expected}"
        )


def test_chunk_counts_and_medians(
    chunking_evaluation_service, chunking_repository
):
    result = chunking_evaluation_service.run_chunking_experiment()
    for strategy, count in CHUNK_COUNTS.items():
        entry = result["experiments"][f"bm25@{strategy}"]
        assert entry["chunks"] == count, strategy
        assert entry["median_chunk_words"] == MEDIAN_WORDS[strategy], strategy
        chunk_entry = result["experiments"][f"chunk-bm25@{strategy}"]
        assert chunk_entry["chunks"] == count, strategy


def test_full_experiments_match_reference(
    chunking_evaluation_service, chunking_repository
):
    result = chunking_evaluation_service.run_chunking_experiment()
    reference = chunking_repository.load_reference_results()
    assert set(result["experiments"]) == set(reference["experiments"])
    for name, expected in reference["experiments"].items():
        _assert_close(
            result["experiments"][name], expected, path=f"experiments.{name}"
        )


def test_corpus_and_questions_match_reference(
    chunking_evaluation_service, chunking_repository
):
    result = chunking_evaluation_service.run_chunking_experiment()
    reference = chunking_repository.load_reference_results()
    assert result["corpus"]["posts"] == reference["corpus"]["posts"] == 15
    assert result["corpus"]["words"] == reference["corpus"]["words"] == 41375
    assert result["corpus"]["files"] == reference["corpus"]["files"]
    assert (
        result["corpus"]["excluded_folders"]
        == reference["corpus"]["excluded_folders"]
    )
    assert result["corpus"]["generated"] == date.today().isoformat()
    assert result["questions"] == reference["questions"] == 22


def test_per_question_rows_match_reference(
    chunking_evaluation_service, chunking_repository
):
    result = chunking_evaluation_service.run_chunking_experiment()
    reference = chunking_repository.load_reference_results()
    assert set(result["per_question"]) == set(reference["per_question"])
    for strategy, expected in reference["per_question"].items():
        assert result["per_question"][strategy] == expected, strategy


def test_rewording_ranks_match_reference(
    chunking_evaluation_service, chunking_repository
):
    result = chunking_evaluation_service.run_chunking_experiment()
    reference = chunking_repository.load_reference_results()
    actual_items = result["rewording"]["items"]
    expected_items = reference["rewording"]["items"]
    assert len(actual_items) == len(expected_items)
    for actual, expected in zip(actual_items, expected_items):
        assert actual["id"] == expected["id"]
        assert actual["gold"] == expected["gold"]
        for a_row, e_row in zip(actual["rows"], expected["rows"]):
            assert a_row["label"] == e_row["label"]
            assert a_row["text"] == e_row["text"]
            assert a_row["fixed200"] == e_row["fixed200"]
            assert a_row["section"] == e_row["section"]


def test_top5_dump_matches_reference(
    chunking_evaluation_service, chunking_repository
):
    result = chunking_evaluation_service.run_chunking_experiment()
    reference = chunking_repository.load_reference_results()
    assert len(result["top5_bm25_section"]) == len(
        reference["top5_bm25_section"]
    )
    for actual, expected in zip(
        result["top5_bm25_section"], reference["top5_bm25_section"]
    ):
        assert actual["id"] == expected["id"]
        assert actual["q"] == expected["q"]
        assert actual["gold"] == expected["gold"]
        assert [t["slug"] for t in actual["top5"]] == [
            t["slug"] for t in expected["top5"]
        ]
        for a_top, e_top in zip(actual["top5"], expected["top5"]):
            assert a_top["score"] == pytest.approx(e_top["score"], abs=0.01)
            assert a_top["preview"] == e_top["preview"]


def test_nov16_snapshot_untouched(
    evaluation_service, repository, chunking_repository
):
    """The chunking snapshot must not change the 2026-11-16 experiment."""
    result = evaluation_service.run_full_experiment()
    reference = repository.load_reference_results()
    assert result["corpus"]["posts"] == 14
    assert result["questions"] == 20
    assert (
        result["experiments"]["chunk-bm25@200w"]["recall@5"]
        == reference["experiments"]["chunk-bm25@200w"]["recall@5"]
        == 0.70
    )
