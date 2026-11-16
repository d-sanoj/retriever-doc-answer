"""Pydantic models: corpus resources."""

from pydantic import BaseModel, Field


class PostInfo(BaseModel):
    """One corpus post: its slug and word count after markdown cleaning."""

    slug: str
    words: int


class CorpusStats(BaseModel):
    """Corpus summary returned by ``GET /api/v1/corpus``."""

    posts: int = Field(description="Number of posts in the corpus.")
    words: int = Field(description="Total words across all posts.")
    files: list[str] = Field(description="Post slugs, sorted.")
    posts_detail: list[PostInfo] = Field(description="Per-post word counts.")


class Question(BaseModel):
    """One eval question, as stored in ``data/questions.json``."""

    id: int
    q: str
    gold: str
    answer_terms: list[str] = Field(default_factory=list)


class Rewording(BaseModel):
    """Rewording variants for one question."""

    id: int
    variants: list[str]
