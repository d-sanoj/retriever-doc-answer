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

The follow-up experiment (sanoj.dev post ["I chunked my blog by section.
Retrieval jumped from 73% to 91%."](https://sanoj.dev/blog/chunk-by-section-not-by-word-count/))
measured the fix the first post recommended: chunk by structure, not by
word count. On the 15-post corpus, one chunk per markdown section (heading
carried, long sections split at sentence boundaries) lifts strict
answer-chunk recall@5 from 73% (fixed 200-word windows) to 91% — tying the
best fixed window (400 words) with far fewer, more focused chunks. See
[Chunking experiment](#chunking-experiment) below.

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
one chunk per post; `title_prepend` prepends the post title to chunk tokens;
`strategy` picks a chunking strategy instead of fixed windows):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "How do I stop my coding agent from ignoring style rules?",
       "retriever": "bm25", "chunk_size": 200, "top_k": 5}'

curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "How do I stop my coding agent from ignoring style rules?",
       "retriever": "bm25", "strategy": "section", "top_k": 5}'
```

Evaluate — recall@1/3/5/10, MRR, median gold rank, gold ranks, missed@5 for
one configuration. `grading` is `post` (chunk correct if from the gold post)
or `chunk` (chunk correct if from the gold post *and* it contains at least
`min_terms` answer terms). `strategy` selects a chunking strategy; when set,
`chunk_size` and `title_prepend` are ignored:

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

## Chunking experiment

The 2026-11-18 follow-up compares five chunking strategies (BM25 only) on
a 15-post corpus (the 14 posts above plus the jokeGPT post) with 22
questions — the 20 from the first experiment plus 2 for the jokeGPT post.
Strategies: `fixed200` (200-word windows, 40 overlap), `fixed400`
(400-word windows, 80 overlap), `fixed200+head` (200-word windows carrying
their section heading), `section` (one chunk per markdown section, heading
carried, long sections split at sentence boundaries, max ~400 words), and
`para200` (whole paragraphs accumulated to ~200 words, heading carried).

Run the comparison — every strategy, post- and chunk-level grading,
per-question ranks, the fixed200-vs-section rewording comparison, and the
top-5 dump for the section run. Returns the
`results-chunking.json`-shaped object:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments/chunking
```

Reference results — the `results-chunking.json` produced by the original
script:

```bash
curl http://127.0.0.1:8000/api/v1/experiments/chunking/results
```

Headline numbers (strict chunk-level recall@5, exact):

| Strategy | Chunks | Median words | Post recall@5 | Chunk recall@5 |
| --- | --- | --- | --- | --- |
| fixed200 | 263 | 200 | 0.95 | 0.73 |
| fixed400 | 132 | 400 | 1.00 | 0.91 |
| fixed200+head | 256 | 206 | 1.00 | 0.77 |
| section | 170 | 236 | 1.00 | 0.91 |
| para200 | 271 | 182 | 1.00 | 0.73 |

Section chunking ties fixed400 on answer-chunk recall (0.91) with a median
chunk of 236 words instead of 400 — and it fixes 5 of the 6 questions
fixed200 misses. The snapshot lives in `data/chunking/` (corpus,
questions, reference results) and is kept separate from the 2026-11-16
snapshot in `data/`, which stays frozen: its 14 posts, 20 questions, and
headline numbers are unchanged by this experiment.

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

`tests/test_chunking_replication.py` does the same for the chunking
experiment against `data/chunking/results-chunking.json` (15 posts, 22
questions, all five strategies).

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
- **Chunking strategies:** `fixed200` / `fixed400` slide fixed windows
  (overlap `size // 5`); `fixed200+head` windows on a 160-word step with
  each chunk carrying the heading of its first word's section;
  `section` is one chunk per markdown section (heading carried as
  `heading — body`, long sections split at sentence boundaries, max ~400
  words); `para200` accumulates whole sentences to ~200 words per chunk,
  heading carried, never splitting mid-sentence. A trailing fixed window
  under 20 words is dropped.

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
      text_service.py      # tokenize, clean_markdown, chunk_words,
                           #   split_sections, split_sentences
      bm25_service.py      # BM25 scorer
      tfidf_service.py     # TF-IDF scorer
      chunking_service.py  # five chunking strategies (fixed200, fixed400,
                           #   fixed200+head, section, para200)
      retrieval_service.py # index building, ranking, search
      evaluation_service.py# grading, metrics, full experiment suite,
                           #   chunking comparison
    api/v1/                # controllers / routers (V in MVC terms: the view)
      deps.py              # shared dependencies (app-state singletons)
      health.py  corpus.py  search.py  evaluation.py  experiments.py
  data/
    corpus/                # 14 bundled blog posts (.md)
    questions.json         # 20 eval questions + rewording variants
    results.json           # reference output of the original script
    chunking/              # 2026-11-18 snapshot (frozen separately)
      corpus/              # 15 bundled blog posts (.md)
      questions.json       # 22 eval questions + rewording variants
      results-chunking.json# reference output of the chunking script
  tests/
    test_text_service.py  test_scorers.py  test_evaluation_service.py
    test_api.py           # TestClient end-to-end, every endpoint
    test_replication.py   # API output vs reference results.json
    test_chunking_service.py  # chunking strategies, unit level
    test_chunking_replication.py  # chunking output vs reference
                           #   results-chunking.json
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
