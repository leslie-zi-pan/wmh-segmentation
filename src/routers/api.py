from fastapi import APIRouter
from src.routers import segment

router = APIRouter()

router.include_router(segment.router, tags=["White Matter Hyperintensity Segmenter"])