"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.repositories.corpus_repository import CorpusRepository
from app.services.evaluation_service import EvaluationService
from app.services.retrieval_service import RetrievalService


@pytest.fixture(scope="session")
def repository() -> CorpusRepository:
    """Corpus repository over the bundled data directory."""
    return CorpusRepository()


@pytest.fixture(scope="session")
def retrieval_service(repository: CorpusRepository) -> RetrievalService:
    """Retrieval service wired to the bundled corpus."""
    return RetrievalService(repository)


@pytest.fixture(scope="session")
def evaluation_service(
    repository: CorpusRepository, retrieval_service: RetrievalService
) -> EvaluationService:
    """Evaluation service wired to the bundled corpus."""
    return EvaluationService(repository, retrieval_service)


@pytest.fixture(scope="session")
def client() -> TestClient:
    """TestClient with the lifespan (corpus load) executed."""
    with TestClient(create_app()) as test_client:
        yield test_client
