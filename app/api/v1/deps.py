"""Shared FastAPI dependencies.

Services are singletons held on ``app.state`` (created once in the
lifespan handler), so the corpus is loaded and indexes are built once and
reused across requests.
"""

from fastapi import Request

from app.repositories.corpus_repository import CorpusRepository
from app.services.evaluation_service import EvaluationService
from app.services.retrieval_service import RetrievalService


def get_repository(request: Request) -> CorpusRepository:
    """Return the app-wide corpus repository."""
    return request.app.state.repository


def get_retrieval_service(request: Request) -> RetrievalService:
    """Return the app-wide retrieval service."""
    return request.app.state.retrieval_service


def get_evaluation_service(request: Request) -> EvaluationService:
    """Return the app-wide evaluation service."""
    return request.app.state.evaluation_service
