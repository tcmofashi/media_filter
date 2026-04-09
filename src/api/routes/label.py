from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path

from src.storage.database import db
from src.config import settings

# Image and video extensions for type detection
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
router = APIRouter(prefix="/label", tags=["labels"])


class LabelCreate(BaseModel):
    media_path: str = Field(..., description="Path to the media file")
    score: float = Field(..., ge=0, le=9, description="Score from 0 to 9")


class LabelResponse(BaseModel):
    id: int
    media_path: str
    score: float
    created_at: str


@router.post("", status_code=201, response_model=LabelResponse)
async def create_label(label: LabelCreate):
    """Create a new label (simplified - single dimension)"""
    # Validate media path
    media_path = Path(label.media_path).resolve()
    
    # Check if path is in allowed directories
    allowed = False
    for allowed_base in settings.allowed_paths:
        allowed_base_path = Path(allowed_base).resolve()
        try:
            media_path.relative_to(allowed_base_path)
            allowed = True
            break
        except ValueError:
            continue
    
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Path not in allowed directories"
        )
    
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    
    # Determine file type
    ext = media_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        file_type = "image"
    elif ext in VIDEO_EXTENSIONS:
        file_type = "video"
    else:
        file_type = "unknown"
    
    # Check if media exists in DB, if not add it
    media = await db.get_media(str(media_path))
    if not media:
        await db.add_media(str(media_path), file_type)
        media = await db.get_media(str(media_path))
    
    if not media:
        raise HTTPException(status_code=500, detail="Failed to add media to database")

    await db.add_label(
        media_id=media["id"],
        score=label.score,
    )

    labels = await db.get_labels(limit=1)
    created_label = labels[0]

    return LabelResponse(
        id=created_label["id"],
        media_path=created_label["media_path"],
        score=created_label["score"],
        created_at=created_label["created_at"],
    )


@router.get("s", response_model=List[LabelResponse])
async def list_labels(limit: int = Query(100, ge=1, le=50000)):
    """List all labels (simplified - no dimension filter)"""
    labels = await db.get_labels(limit=limit)

    return [
        LabelResponse(
            id=label["id"],
            media_path=label["media_path"],
            score=label["score"],
            created_at=label["created_at"],
        )
        for label in labels
    ]


@router.delete("/{label_id}")
async def delete_label(label_id: int):
    """Delete a label"""
    label = await db.get_label(label_id)
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")

    await db.delete_label(label_id)
    return {"message": "Label deleted"}
