"""Experiment endpoints: full suite, reference results, rewording,
chunking comparison."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import (
    get_chunking_evaluation_service,
    get_evaluation_service,
    get_repository,
)
from app.models.evaluation import (
    ChunkingExperimentResponse,
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


@router.post("/chunking", response_model=ChunkingExperimentResponse)
def run_chunking_experiment(
    service: EvaluationService = Depends(get_chunking_evaluation_service),
) -> dict:
    """Run the chunking comparison (2026-11-18 post) and return the
    results object.

    Same shape as ``data/chunking/results-chunking.json``: corpus block,
    per-strategy metrics (post- and chunk-level), per-question ranks,
    the fixed200-vs-section rewording comparison, and the top-5 dump for
    the section run. Runs on the 15-post chunking snapshot, not the
    2026-11-16 corpus. Takes a few seconds (BM25 scoring is a
    pure-Python loop).
    """
    return service.run_chunking_experiment()


@router.get("/chunking/results")
def get_chunking_reference_results(
    service: EvaluationService = Depends(get_chunking_evaluation_service),
) -> dict:
    """The reference ``results-chunking.json`` from the chunking script."""
    return service.repository.load_reference_results()
