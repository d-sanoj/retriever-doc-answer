"""End-to-end API tests (TestClient) for every endpoint."""



def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"


def test_openapi_lists_all_routes(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for path in (
        "/health",
        "/api/v1/corpus",
        "/api/v1/search",
        "/api/v1/evaluate",
        "/api/v1/experiments/run",
        "/api/v1/experiments/results",
        "/api/v1/experiments/rewording",
    ):
        assert path in paths


def test_corpus_stats(client):
    response = client.get("/api/v1/corpus")
    assert response.status_code == 200
    body = response.json()
    assert body["posts"] == 14
    assert body["words"] == 38433
    assert body["files"] == sorted(body["files"])
    assert "retriever-finds-the-document-not-the-answer" not in body["files"]
    assert len(body["posts_detail"]) == 14
    assert sum(p["words"] for p in body["posts_detail"]) == 38433


def test_search_bm25(client):
    response = client.post(
        "/api/v1/search",
        json={
            "query": "How do I stop my coding agent "
            "from ignoring style rules?",
            "retriever": "bm25",
            "chunk_size": 200,
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_total"] == 244
    assert len(body["results"]) == 5
    assert body["results"][0]["slug"] == (
        "your-agents-code-looks-nothing-like-yours"
    )
    assert body["results"][0]["rank"] == 1
    assert body["results"][0]["score"] > 0
    assert body["results"][0]["preview"]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_tfidf_and_defaults(client):
    response = client.post(
        "/api/v1/search",
        json={"query": "tokenizer non-English cost"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["retriever"] == "bm25"  # default
    assert body["chunk_size"] == 200
    assert body["results"][0]["slug"] == "inside-gpt-tokenizer-vocabulary"

    response = client.post(
        "/api/v1/search",
        json={"query": "tokenizer non-English cost", "retriever": "tfidf"},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["slug"] == (
        "inside-gpt-tokenizer-vocabulary"
    )


def test_search_validation(client):
    assert client.post("/api/v1/search", json={"query": ""}).status_code == 422
    assert (
        client.post(
            "/api/v1/search",
            json={"query": "x", "retriever": "embedding"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/search", json={"query": "x", "chunk_size": -1}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/search", json={"query": "x", "top_k": 0}
        ).status_code
        == 422
    )


def test_evaluate_post_level(client):
    response = client.post(
        "/api/v1/evaluate",
        json={"retriever": "bm25", "chunk_size": 200, "grading": "post"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] == 244
    assert body["recall@1"] == 0.85
    assert body["recall@5"] == 0.95
    assert body["missed_at_5"] == 1
    assert body["never_found"] == 0
    assert len(body["gold_ranks"]) == 20


def test_evaluate_chunk_level_and_whole_post(client):
    response = client.post(
        "/api/v1/evaluate",
        json={"retriever": "bm25", "chunk_size": 400, "grading": "chunk"},
    )
    assert response.status_code == 200
    assert response.json()["recall@5"] == 0.90

    response = client.post(
        "/api/v1/evaluate",
        json={"retriever": "bm25", "chunk_size": 0, "grading": "post"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] == 14
    assert body["recall@5"] == 1.0


def test_reference_results(client):
    response = client.get("/api/v1/experiments/results")
    assert response.status_code == 200
    body = response.json()
    assert body["corpus"]["posts"] == 14
    assert body["corpus"]["words"] == 38433
    assert body["questions"] == 20
    assert body["experiments"]["bm25@200w"]["recall@5"] == 0.95
    assert len(body["per_question_200w"]) == 20
    assert len(body["top5_bm25_200w"]) == 20


def test_rewording_all_and_subset(client):
    response = client.post("/api/v1/experiments/rewording", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["experiment"] == "rewording"
    assert body["retriever"] == "bm25@200w"
    assert len(body["items"]) == 6
    assert [row["label"] for row in body["items"][0]["rows"]] == [
        "original",
        "variant1",
        "variant2",
    ]

    response = client.post(
        "/api/v1/experiments/rewording", json={"question_ids": [3]}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == 3
    # Q3 variant2 ("Can an AI coding helper remember what we did yesterday?")
    # ranks the gold post 6th and the answer chunk 17th — the known
    # rewording failure: it matches one query word ("remember", once).
    assert body["items"][0]["rows"][2]["gold_rank"] == 6
    assert body["items"][0]["rows"][2]["answer_chunk_rank"] == 17


def test_experiments_run_headlines(client):
    response = client.post("/api/v1/experiments/run")
    assert response.status_code == 200
    body = response.json()
    assert body["corpus"]["posts"] == 14
    assert body["questions"] == 20
    assert body["experiments"]["bm25@200w"]["recall@5"] == 0.95
    assert body["experiments"]["chunk-bm25@200w"]["recall@5"] == 0.70
    assert len(body["per_question_200w"]) == 20
    assert len(body["rewording"]["items"]) == 6
    assert len(body["top5_bm25_200w"]) == 20
