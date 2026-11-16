"""Health endpoint."""

from fastapi import APIRouter

from app import __version__
from app.models.evaluation import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. No dependencies, no side effects."""
    return HealthResponse(status="ok", version=__version__)
