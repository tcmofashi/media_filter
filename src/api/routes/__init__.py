from fastapi import APIRouter

from src.api.routes.batch import router as batch_router
from src.api.routes.export import router as export_router
from src.api.routes.label import router as label_router
from src.api.routes.media import router as media_router
from src.api.routes.score import router as score_router
from src.api.routes.pipeline import router as pipeline_router
from src.api.routes.status import router as status_router

api_router = APIRouter()
api_router.include_router(batch_router)
api_router.include_router(export_router)
api_router.include_router(label_router)
api_router.include_router(media_router)
api_router.include_router(pipeline_router)
api_router.include_router(score_router)
api_router.include_router(status_router)
