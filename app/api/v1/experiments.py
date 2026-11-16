"""Experiment endpoints: full suite, reference results, rewording."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import get_evaluation_service, get_repository
from app.models.evaluation import (
    ExperimentRunResponse,
    RewordingRequest,
)
from app.repositories.corpus_repository import CorpusRepository
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/run", response_model=ExperimentRunResponse)
def run_experiment(
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    """Run the full experiment suite and return the results object.

    Same shape as ``data/results.json``: corpus block, per-configuration
    metrics, per-question ranks, rewording results, and the top-5 dump.
    Takes a few seconds (BM25 scoring is a pure-Python loop).
    """
    return service.run_full_experiment()


@router.get("/results")
def get_reference_results(
    repository: CorpusRepository = Depends(get_repository),
) -> dict:
    """The reference ``results.json`` from the original experiment script."""
    return repository.load_reference_results()


@router.post("/rewording")
def run_rewording(
    request: RewordingRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    """Rewording experiment for the given question ids (or all)."""
    return service.run_rewording(
        question_ids=request.question_ids,
        chunk_size=request.chunk_size,
        retriever=request.retriever,
    )
