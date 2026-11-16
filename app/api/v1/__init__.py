"""API v1 router: aggregates all resource routers."""

from fastapi import APIRouter

from app.api.v1 import corpus, evaluation, experiments, health, search

router = APIRouter()
router.include_router(health.router)
router.include_router(corpus.router, prefix="/api/v1")
router.include_router(search.router, prefix="/api/v1")
router.include_router(evaluation.router, prefix="/api/v1")
router.include_router(experiments.router, prefix="/api/v1")
