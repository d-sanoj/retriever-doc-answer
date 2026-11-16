# retriever-doc-answer

BM25 / TF-IDF retrieval measurement as an API service.

This is the retrieval experiment from the sanoj.dev post
["Your retriever finds the right document. It doesn't always find the answer."](https://sanoj.dev/blog/retriever-finds-the-document-not-the-answer/)
rebuilt as a FastAPI service. The original was a single CLI script; this repo
keeps the scoring math identical and exposes it over HTTP: search the corpus,
evaluate a retriever configuration, or re-run the whole experiment suite.

The headline finding: on this 14-post blog corpus, BM25 at 200-word chunks
finds the right *document* 95% of the time (recall@5) but the chunk that
actually contains the *answer* only 70% of the time. Post-level and
chunk-level grading disagree — and that gap is the post.

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/d-sanoj/retriever-doc-answer.git
cd retriever-doc-answer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app
```

The API listens on `http://127.0.0.1:8000`. Interactive docs at
`http://127.0.0.1:8000/docs`.

## Endpoints

Health check:

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok", "version": "0.1.0"}
```

Corpus stats — post count, word count, file list:

```bash
curl http://127.0.0.1:8000/api/v1/corpus
```

Search — ranked chunks for a query (`bm25` or `tfidf`; `chunk_size` 0 means
one chunk per post; `title_prepend` prepends the post title to chunk tokens):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "How do I stop my coding agent from ignoring style rules?",
       "retriever": "bm25", "chunk_size": 200, "top_k": 5}'
```

Evaluate — recall@1/3/5/10, MRR, median gold rank, gold ranks, missed@5 for
one configuration. `grading` is `post` (chunk correct if from the gold post)
or `chunk` (chunk correct if from the gold post *and* it contains at least
`min_terms` answer terms):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"retriever": "bm25", "chunk_size": 200, "grading": "post"}'

curl -X POST http://127.0.0.1:8000/api/v1/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"retriever": "bm25", "chunk_size": 200, "grading": "chunk",
       "min_terms": 2}'
```

Run the full experiment — every chunk size, both retrievers, title prepend,
chunk-level grading, the loose (>=1 term) reading, rewording, and the top-5
dump. Returns the `results.json`-shaped object:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments/run
```

Reference results — the `results.json` produced by the original script:

```bash
curl http://127.0.0.1:8000/api/v1/experiments/results
```

Rewording — how the same question, phrased three ways, ranks the gold post
and the answer chunk. Omit `question_ids` for all six rewording items:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments/rewording \
  -H 'Content-Type: application/json' \
  -d '{"question_ids": [3]}'
```

## Replicating the experiment

The corpus is bundled in `data/corpus/` (the 14 sanoj.dev blog posts, frontmatter
stripped at load time) with `data/questions.json` (20 reader-style questions
with gold posts and answer terms) and `data/results.json` (the reference
output of the original script). No network access and no external corpus are
needed at runtime.

To reproduce the headline numbers, run the full experiment and compare
against the reference:

```bash
python -m app &
curl -s -X POST http://127.0.0.1:8000/api/v1/experiments/run \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print({k: d['experiments'][k] for k in ('bm25@200w','chunk-bm25@200w')})"
```

Expected headline numbers (exact):

| Experiment | recall@1 | recall@5 |
| --- | --- | --- |
| bm25@200w (post-level) | 0.85 | 0.95 |
| chunk-bm25@200w (strict, >=2 terms) | 0.55 | 0.70 |
| bm25+title@200w (post-level) | 0.90 | 1.00 |
| chunk-bm25@400w (strict) | 0.70 | 0.90 |
| chunk-loose>=1-bm25@200w | 0.80 | 0.95 |

The test suite checks all of these against the reference:

```bash
pip install -r requirements-dev.txt
python -m pytest
python -m ruff check app tests
```

`tests/test_replication.py` runs the full suite through the service layer and
compares every experiment, every per-question rank, every rewording rank, and
the top-5 dump against `data/results.json`.

## How the measurement works

- **Corpus:** 14 posts, 38,433 words. Each post is cleaned (frontmatter
  stripped, code-fence/image/`:::` lines dropped, table pipes to spaces,
  links to anchor text) and split into sliding windows: size `N` words,
  overlap `N // 5`, step `N - overlap`. A trailing chunk under 20 words is
  dropped. Size 0 means one chunk per post.
- **BM25:** k1 = 1.5, b = 0.75, idf = log(1 + (N - f + 0.5) / (f + 0.5)).
  Query term frequency is not used — only term presence.
- **TF-IDF:** (1 + log tf) * idf, idf = log((N + 1) / (df + 1)) + 1,
  L2-normalized rows, cosine similarity against the normalized query.
- **Tokenization:** lowercase `[a-z0-9]+`, minus a fixed stopword list. No
  stemming, no embeddings.
- **Grading:** post-level (chunk correct if from the gold post) vs
  chunk-level (gold post *and* at least 2 answer terms appear in the chunk
  text, substring-matched). Recall@k is the fraction of the 20 questions
  with a correct chunk in the top k.

## Project structure

```text
retriever-doc-answer/
  app/
    __init__.py            # package version
    __main__.py            # `python -m app` -> uvicorn
    main.py                # FastAPI app factory, lifespan, router wiring
    core/
      config.py            # pydantic-settings (RETRIEVER_* env overrides)
      logging.py           # logging setup
    models/                # Pydantic request/response schemas (M)
      corpus.py  search.py  evaluation.py
    repositories/          # data access (M in MVC terms: the data layer)
      corpus_repository.py # loads corpus .md files, questions, results.json
    services/              # business logic (C in MVC terms: the model layer)
      text_service.py      # tokenize, clean_markdown, chunk_words
      bm25_service.py      # BM25 scorer
      tfidf_service.py     # TF-IDF scorer
      retrieval_service.py # index building, ranking, search
      evaluation_service.py# grading, metrics, full experiment suite
    api/v1/                # controllers / routers (V in MVC terms: the view)
      deps.py              # shared dependencies (app-state singletons)
      health.py  corpus.py  search.py  evaluation.py  experiments.py
  data/
    corpus/                # 14 bundled blog posts (.md)
    questions.json         # 20 eval questions + rewording variants
    results.json           # reference output of the original script
  tests/
    test_text_service.py  test_scorers.py  test_evaluation_service.py
    test_api.py           # TestClient end-to-end, every endpoint
    test_replication.py   # API output vs reference results.json
```

## Architecture

MVC, with the layers kept separate:

- **Models** (`app/models/`) are Pydantic schemas — the shapes that cross the
  HTTP boundary, nothing else.
- **Repositories** (`app/repositories/`) own data access: reading the bundled
  `.md` files and JSON, cached in-process. They know nothing about scoring.
- **Services** (`app/services/`) own the business logic: text processing, the
  two scorers, index building, ranking, grading, and the experiment suite.
  They depend on the repository, never on HTTP.
- **Controllers** (`app/api/v1/`) are thin FastAPI routers: validate the
  request body, call one service method, return the response model. They
  hold no scoring or grading logic.
- **Composition** happens in `app/main.py`: the lifespan handler builds the
  repository and both services once and stores them on `app.state`; routers
  receive them through `app/api/v1/deps.py`.

Configuration lives in `app/core/config.py` (pydantic-settings, `RETRIEVER_`
prefix): corpus directory, question/results paths, BM25 k1/b, defaults, host
and port. Defaults point at the bundled `data/` directory.

## License

MIT — see `LICENSE`.
