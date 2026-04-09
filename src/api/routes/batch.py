"""Batch scoring endpoint for existing local media paths."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.routes.media import get_media_type, validate_path
from src.config import settings
from src.logger import get_logger
from src.models import FrozenClipEngine, create_engine_from_settings

logger = get_logger(__name__)

router = APIRouter(prefix="/batch-score", tags=["batch-scoring"])


class BatchScoreItem(BaseModel):
    """One media path to score."""

    path: str = Field(..., description="Absolute or allowed local media path")
    is_video: bool | None = Field(default=None, description="Optional type override")


class BatchScoreRequest(BaseModel):
    """Batch scoring request."""

    items: list[BatchScoreItem] = Field(
        ...,
        min_length=1,
        max_length=settings.max_batch_size,
        description="Media paths to score",
    )


def get_engine() -> FrozenClipEngine:
    """Get the shared Frozen CLIP engine."""
    return create_engine_from_settings()


def ensure_engine_loaded(engine: FrozenClipEngine) -> FrozenClipEngine:
    """Load the configured checkpoint on first use."""
    if not engine.is_loaded:
        engine.load_model(settings.model_checkpoint)
    return engine


@router.post("")
async def score_batch(
    request: BatchScoreRequest,
    engine: FrozenClipEngine = Depends(get_engine),
):
    """Score a batch of existing local media files."""
    if len(request.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {settings.max_batch_size}. Got {len(request.items)}.",
        )

    engine = ensure_engine_loaded(engine)

    results: list[dict] = []
    errors: list[dict] = []

    for item in request.items:
        try:
            media_path = validate_path(item.path)
            if not media_path.is_file():
                raise HTTPException(status_code=400, detail=f"Path is not a file: {item.path}")

            result = await asyncio.to_thread(engine.score, media_path, item.is_video)
            result["name"] = Path(item.path).name
            result["media_type"] = get_media_type(media_path)
            result["success"] = True
            results.append(result)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Batch scoring failed for %s: %s", item.path, exc)
            error_message = exc.detail if isinstance(exc, HTTPException) else str(exc)
            error_payload = {
                "path": item.path,
                "success": False,
                "error": error_message,
            }
            results.append(error_payload)
            errors.append(error_payload)

    return {
        "results": results,
        "errors": errors,
        "total": len(request.items),
        "success_count": len(request.items) - len(errors),
        "failed_count": len(errors),
    }
