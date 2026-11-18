"""Pydantic models: evaluation and experiment resources."""

from typing import Literal

from pydantic import BaseModel, Field

from app.models.search import ChunkingStrategy, RetrieverName

GradingName = Literal["post", "chunk"]


class EvaluateRequest(BaseModel):
    """Evaluation request body for ``POST /api/v1/evaluate``."""

    retriever: RetrieverName = Field(default="bm25")
    chunk_size: int = Field(
        default=200, ge=0, description="Chunk size in words; 0 = whole post."
    )
    grading: GradingName = Field(
        default="post",
        description="post: chunk correct if from the gold post; "
        "chunk: chunk correct if from the gold post AND it contains "
        "at least min_terms answer terms.",
    )
    min_terms: int = Field(
        default=2,
        ge=1,
        description="Answer terms required for chunk-level grading.",
    )
    title_prepend: bool = Field(default=False)
    strategy: ChunkingStrategy | None = Field(
        default=None,
        description=(
            "Chunking strategy instead of fixed windows; chunk_size "
            "and title_prepend are then ignored."
        ),
    )


class EvaluationResult(BaseModel):
    """Recall / MRR metrics for one retriever configuration.

    Serialized with the original experiment's key names (``recall@5`` etc.)
    so API output matches ``results.json`` exactly.
    """

    model_config = {"populate_by_name": True}

    chunks: int = Field(description="Chunks in the index.")
    recall_at_1: float = Field(alias="recall@1")
    recall_at_3: float = Field(alias="recall@3")
    recall_at_5: float = Field(alias="recall@5")
    recall_at_10: float = Field(alias="recall@10")
    mrr: float
    median_gold_rank: float | None = Field(
        description="Median rank of the gold chunk; null if never found."
    )
    gold_ranks: list[int | None] = Field(
        description="Gold rank per question, in question order."
    )
    missed_at_5: int = Field(
        description="Questions with no gold chunk in top 5."
    )
    never_found: int = Field(
        description="Questions with no gold chunk at all."
    )


class ExperimentRunResponse(BaseModel):
    """Full experiment suite result, shaped like ``data/results.json``."""

    corpus: dict
    questions: int
    experiments: dict[str, dict]
    per_question_200w: list[dict]
    rewording: dict
    top5_bm25_200w: list[dict]


class ChunkingExperimentResponse(BaseModel):
    """Chunking comparison result, shaped like
    ``data/chunking/results-chunking.json``."""

    corpus: dict
    questions: int
    experiments: dict[str, dict]
    per_question: dict[str, list[dict]]
    rewording: dict
    top5_bm25_section: list[dict]


class RewordingRequest(BaseModel):
    """Rewording experiment request body."""

    question_ids: list[int] | None = Field(
        default=None,
        description="Question ids to reword; null means all rewording items.",
    )
    chunk_size: int = Field(default=200, ge=0)
    retriever: RetrieverName = Field(default="bm25")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
