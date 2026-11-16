"""Pydantic models: search resources."""

from typing import Literal

from pydantic import BaseModel, Field

RetrieverName = Literal["bm25", "tfidf"]


class SearchRequest(BaseModel):
    """Search request body for ``POST /api/v1/search``."""

    query: str = Field(min_length=1, description="Natural-language query.")
    retriever: RetrieverName = Field(
        default="bm25", description="Scoring model: bm25 or tfidf."
    )
    chunk_size: int = Field(
        default=200,
        ge=0,
        description="Chunk size in words; 0 means one chunk per post.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Chunks to return.")
    title_prepend: bool = Field(
        default=False,
        description=(
            "Prepend the post title (slug words) to each chunk's tokens."
        ),
    )


class RankedChunk(BaseModel):
    """One ranked chunk from a search."""

    rank: int = Field(description="1-based rank in the result list.")
    slug: str = Field(description="Post slug the chunk comes from.")
    score: float = Field(description="Retriever score (higher is better).")
    preview: str = Field(description="First 140 characters of the chunk text.")


class SearchResponse(BaseModel):
    """Search response: the top-k ranked chunks for the query."""

    query: str
    retriever: RetrieverName
    chunk_size: int
    title_prepend: bool
    chunks_total: int = Field(description="Chunks in the index.")
    results: list[RankedChunk]
