"""Search endpoints."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import get_retrieval_service
from app.models.search import RankedChunk, SearchRequest, SearchResponse
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    request: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """Rank corpus chunks for a query with BM25 or TF-IDF."""
    results, chunks_total = service.search(
        query=request.query,
        retriever=request.retriever,
        chunk_size=request.chunk_size,
        top_k=request.top_k,
        title_prepend=request.title_prepend,
    )
    return SearchResponse(
        query=request.query,
        retriever=request.retriever,
        chunk_size=request.chunk_size,
        title_prepend=request.title_prepend,
        chunks_total=chunks_total,
        results=[RankedChunk(**item) for item in results],
    )
