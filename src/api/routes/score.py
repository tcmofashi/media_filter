"""Single-file scoring endpoint for the Frozen CLIP baseline."""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.logger import get_logger
from src.models import FrozenClipEngine, create_engine_from_settings
from src.storage.database import db

logger = get_logger(__name__)

router = APIRouter(prefix="/score", tags=["scoring"])

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
UPLOAD_DIR = Path("data/uploads")


def get_upload_dir() -> Path:
    """Ensure the upload directory exists."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def get_engine() -> FrozenClipEngine:
    """Get the shared Frozen CLIP engine."""
    return create_engine_from_settings()


def detect_file_type(filename: str) -> tuple[str, bool]:
    """Detect media type by extension."""
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return "image", False
    if ext in ALLOWED_VIDEO_EXTENSIONS:
        return "video", True
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
    )


async def save_upload_file(file: UploadFile) -> Path:
    """Persist an uploaded file under data/uploads."""
    upload_dir = get_upload_dir()
    original_ext = Path(file.filename or "").suffix.lower() or ".bin"
    file_path = upload_dir / f"{uuid.uuid4()}{original_ext}"
    content = await file.read()
    file_path.write_bytes(content)
    return file_path


def ensure_engine_loaded(engine: FrozenClipEngine) -> FrozenClipEngine:
    """Load the configured checkpoint on first use."""
    if not engine.is_loaded:
        from src.config import settings

        engine.load_model(settings.model_checkpoint)
    return engine


@router.post("")
async def score_media(
    file: UploadFile = File(..., description="Media file to score"),
    is_video: Optional[bool] = Form(
        default=None,
        description="Optional explicit media type override for videos.",
    ),
    engine: FrozenClipEngine = Depends(get_engine),
):
    """Upload and score a single media file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_type, detected_is_video = detect_file_type(file.filename)
    media_path = await save_upload_file(file)
    engine = ensure_engine_loaded(engine)

    try:
        result = await asyncio.to_thread(
            engine.score,
            media_path,
            detected_is_video if is_video is None else is_video,
        )
    except Exception as exc:
        logger.error("Failed to score %s: %s", media_path, exc)
        raise

    try:
        await db.add_media(str(media_path), file_type)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to index uploaded media %s: %s", media_path, exc)

    return result
