"""Evaluation endpoints."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import get_evaluation_service
from app.models.evaluation import EvaluateRequest, EvaluationResult
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.post("", response_model=EvaluationResult)
def evaluate(
    request: EvaluateRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationResult:
    """Evaluate one retriever configuration against the 20 questions."""
    result = service.evaluate(
        retriever=request.retriever,
        chunk_size=request.chunk_size,
        grading=request.grading,
        min_terms=request.min_terms,
        title_prepend=request.title_prepend,
    )
    return EvaluationResult(**result)
