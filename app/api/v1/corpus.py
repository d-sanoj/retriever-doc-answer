"""Corpus endpoints."""

from fastapi import APIRouter, Depends

from app.api.v1.deps import get_repository
from app.models.corpus import CorpusStats, PostInfo
from app.repositories.corpus_repository import CorpusRepository

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("", response_model=CorpusStats)
def get_corpus_stats(
    repository: CorpusRepository = Depends(get_repository),
) -> CorpusStats:
    """Corpus stats: post count, word count, and the file list."""
    posts = repository.load_posts()
    return CorpusStats(
        posts=len(posts),
        words=sum(p["words"] for p in posts.values()),
        files=sorted(posts),
        posts_detail=[
            PostInfo(slug=slug, words=posts[slug]["words"])
            for slug in sorted(posts)
        ],
    )
