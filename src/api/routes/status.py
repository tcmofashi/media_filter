"""Status endpoint for the Frozen CLIP baseline."""

from typing import Any

from fastapi import APIRouter

from src.config import settings
from src.models import create_engine_from_settings, get_gpu_memory_usage
from src.storage.database import db

router = APIRouter(tags=["status"])


@router.get("/status")
async def get_status() -> dict[str, Any]:
    """Return model, GPU, and label status."""
    engine = create_engine_from_settings()

    model_status: dict[str, Any] = {
        "loaded": engine.is_loaded,
        "checkpoint": settings.model_checkpoint,
        "device": str(engine.device),
        "sample_frames": settings.sample_frames,
        "clip_model": settings.clip_model_name,
    }
    if engine.is_loaded:
        model_status["details"] = engine.get_model_info()

    try:
        label_count = await db.get_label_count()
    except Exception:
        label_count = 0

    return {
        "model": model_status,
        "gpu": get_gpu_memory_usage(),
        "labels": {"total": label_count},
    }
