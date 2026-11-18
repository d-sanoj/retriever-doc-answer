"""Application settings.

All paths default to the bundled ``data/`` directory, so the service is
self-contained: it never reads the sine-portfolio checkout at runtime.
Every setting can be overridden with an environment variable prefixed
``RETRIEVER_`` (e.g. ``RETRIEVER_CORPUS_DIR=/path/to/posts``).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the retrieval service."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVER_")

    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    questions_file: Path = BASE_DIR / "data" / "questions.json"
    results_file: Path = BASE_DIR / "data" / "results.json"

    # Chunking experiment snapshot (2026-11-18 post): the 15-post corpus,
    # 22 questions, and its reference results. Kept separate from the
    # 2026-11-16 snapshot above, which stays frozen.
    chunking_corpus_dir: Path = BASE_DIR / "data" / "chunking" / "corpus"
    chunking_questions_file: Path = (
        BASE_DIR / "data" / "chunking" / "questions.json"
    )
    chunking_results_file: Path = (
        BASE_DIR / "data" / "chunking" / "results-chunking.json"
    )

    # Scoring defaults, identical to the original experiment script.
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    default_chunk_size: int = 200
    min_chunk_words: int = 20

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
