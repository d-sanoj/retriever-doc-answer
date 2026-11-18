"""FastAPI application factory.

The app is built by :func:`create_app`, which wires the v1 routers and
creates the service singletons (repository, retrieval, evaluation) once at
startup so the corpus is loaded and indexes are cached across requests.
Run it with ``python -m app`` or ``uvicorn app.main:app``.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.v1 import router as v1_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.repositories.corpus_repository import CorpusRepository
from app.services.evaluation_service import EvaluationService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load the corpus once. Shutdown: nothing to clean up."""
    configure_logging()
    repository = CorpusRepository()
    posts = repository.load_posts()
    questions = repository.load_questions()
    app.state.repository = repository
    app.state.retrieval_service = RetrievalService(repository)
    app.state.evaluation_service = EvaluationService(
        repository, app.state.retrieval_service
    )
    logger.info(
        "corpus loaded: %d posts, %d words, %d questions",
        len(posts),
        sum(p["words"] for p in posts.values()),
        len(questions),
    )
    # Chunking snapshot (2026-11-18 post): 15 posts, 22 questions.
    # Separate repository so the 2026-11-16 experiment stays frozen.
    chunking_repository = CorpusRepository(
        corpus_dir=settings.chunking_corpus_dir,
        questions_file=settings.chunking_questions_file,
        results_file=settings.chunking_results_file,
    )
    chunking_retrieval = RetrievalService(chunking_repository)
    app.state.chunking_evaluation_service = EvaluationService(
        chunking_repository, chunking_retrieval
    )
    chunking_posts = chunking_repository.load_posts()
    logger.info(
        "chunking corpus loaded: %d posts, %d words, %d questions",
        len(chunking_posts),
        sum(p["words"] for p in chunking_posts.values()),
        len(chunking_repository.load_questions()),
    )
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="retriever-doc-answer",
        description=(
            "BM25 / TF-IDF retrieval measurement as an API service. "
            "Reproduces the sanoj.dev blog experiment "
            "'Your retriever finds the right document. "
            "It doesn't always find the answer.' "
            "Also runs the follow-up chunking comparison from "
            "'I chunked my blog by section. Retrieval jumped from 73% "
            "to 91%.'"
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(v1_router)
    return app


app = create_app()
