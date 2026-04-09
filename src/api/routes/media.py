"""Media file handling APIs for folder scanning, thumbnails, and streaming with async task support."""

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path

import mimetypes

from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.responses import StreamingResponse, Response

from pydantic import BaseModel

from src.config import settings
from src.services.task_queue import (
    TaskType,
    TaskStatus,
)
from src.logger import get_logger
from src.storage.database import db

logger = get_logger(__name__)

# Supported media extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

router = APIRouter(prefix="/media", tags=["media"])


class PreloadRequest(BaseModel):
    """Request model for batch preload."""

    paths: list[str]
    types: list[str] = ["thumbnail"]  # can include "transcode"


class PreloadResponse(BaseModel):
    """Response model for batch preload."""

    submitted: int  # newly submitted tasks
    cached: int  # already cached (skipped)
    total: int  # total files processed


class InferenceMediaItem(BaseModel):
    """Media item loaded from stored inference results."""

    name: str
    path: str
    type: str
    inference_score: float
    content_hash: str
    inference_updated_at: str


class InferenceMediaResponse(BaseModel):
    """Response model for inferred media listing."""

    source: str
    sort: str
    total: int
    root_path: str | None = None
    files: list[InferenceMediaItem]


def validate_path(path: str) -> Path:
    """Validate that the path is within allowed directories."""
    resolved_path = Path(path).resolve()

    allowed = False
    for allowed_base in settings.allowed_paths:
        allowed_base_path = Path(allowed_base).resolve()
        try:
            resolved_path.relative_to(allowed_base_path)
            allowed = True
            break
        except ValueError:
            continue

    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Path not in allowed directories: {settings.allowed_paths}",
        )

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {path}")

    return resolved_path


def get_media_type(file_path: Path) -> str:
    """Determine media type from file extension."""
    ext = file_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def get_file_info(file_path: Path) -> dict:
    """Get file metadata."""
    stat = file_path.stat()
    mime_type, _ = mimetypes.guess_type(str(file_path))

    return {
        "name": file_path.name,
        "path": str(file_path),
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "type": get_media_type(file_path),
        "mime_type": mime_type or "application/octet-stream",
    }


def scan_directory(path: Path, recursive: bool = False) -> list[dict]:
    """Scan directory for media files."""
    results = []

    try:
        if recursive:
            for item in path.rglob("*"):
                if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
                    results.append(get_file_info(item))
        else:
            for item in path.iterdir():
                if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
                    results.append(get_file_info(item))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    # Sort by modified time, newest first
    results.sort(key=lambda x: x["modified"], reverse=True)

    return results


def get_inference_media_info(file_path: Path, row: dict) -> dict:
    """Build lightweight media metadata for inferred media listing."""
    return {
        "name": file_path.name,
        "path": str(file_path),
        "type": get_media_type(file_path),
        "inference_score": float(row["score"]),
        "content_hash": row["content_hash"],
        "inference_updated_at": row["updated_at"],
    }


@router.get("/scan")
async def scan_folder(path: str = Query(...), recursive: bool = False):
    """
    Scan a server-side directory for media files.

    Args:
        path: Directory path to scan
        recursive: Whether to scan subdirectories recursively

    Returns:
        List of media files with metadata
    """
    resolved_path = validate_path(path)

    if not resolved_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    return {
        "path": str(resolved_path),
        "recursive": recursive,
        "files": scan_directory(resolved_path, recursive),
    }


@router.get("/inference", response_model=InferenceMediaResponse)
async def list_inferred_media(
    root_path: str | None = Query(None),
    limit: int = Query(50000, ge=1, le=50000),
):
    """
    List completed media inference results sorted by score descending.

    Args:
        root_path: Optional directory prefix to filter results.
        limit: Maximum number of results to return.

    Returns:
        Media files loaded from the inference cache/results table.
    """
    resolved_root: Path | None = None
    db_root: str | None = None
    if root_path:
        resolved_root = validate_path(root_path)
        db_root = str(resolved_root)

    rows = await db.list_top_media_inference(
        limit=limit,
        root_path=db_root,
    )

    files: list[dict] = []
    for row in rows:
        media_path = Path(row["media_path"]).resolve()
        if not media_path.exists() or not media_path.is_file():
            continue
        files.append(get_inference_media_info(media_path, row))

    return InferenceMediaResponse(
        source="media_inference",
        sort="score_desc",
        total=len(files),
        root_path=str(resolved_root) if resolved_root else None,
        files=files,
    )


@router.get("/thumbnail")
async def get_thumbnail(path: str = Query(...)):
    """
    Get thumbnail for media file.

    For images: Returns scaled version (min 1000x1000)
    For videos: Extracts first frame using FFmpeg

    If thumbnail is being generated, returns task status instead.
    """
    media_path = validate_path(path)

    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if media_path.stat().st_size > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    media_type = get_media_type(media_path)
    if media_type == "unknown":
        raise HTTPException(status_code=400, detail="Not a supported media file")

    # Check if we have a cached thumbnail
    from src.services.task_queue import task_queue

    cache_path, content_hash = task_queue.get_cached_path(
        str(media_path), TaskType.THUMBNAIL
    )

    if cache_path and cache_path.exists():
        # Verify cache is still valid (source not newer)
        if cache_path.stat().st_mtime >= media_path.stat().st_mtime:
            # Return cached thumbnail
            def iter_file():
                with open(cache_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        yield chunk

            return StreamingResponse(
                iter_file(),
                media_type="image/jpeg",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(cache_path.stat().st_size),
                    "X-Cache-Status": "hit",
                },
            )

    # No cache - create and submit async task
    task = task_queue.create_task(str(media_path), TaskType.THUMBNAIL)
    await task_queue.submit(task)

    # Return task status (processing)
    return {
        "task_id": task.id,
        "status": task.status,
        "message": "Thumbnail generation started",
        "poll_url": f"/media/task/{task.id}",
    }
class ScreenshotsResponse(BaseModel):
    """Response model for screenshots."""
    status: str  # "cached" | "processing" | "error"
    screenshot_url: str | None = None  # URL to the stitched screenshot image
    content_hash: str | None = None
    task_id: str | None = None
    message: str | None = None


@router.get("/screenshots", response_model=ScreenshotsResponse)
async def get_screenshots(path: str = Query(...)):
    """
    Get stitched screenshots image for video file.

    For videos: Returns URL to the 5x5 stitched screenshot image.
    If screenshots are being generated, returns task status instead.
    """
    media_path = validate_path(path)

    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if media_path.stat().st_size > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    media_type = get_media_type(media_path)
    if media_type != "video":
        raise HTTPException(status_code=400, detail="Screenshots only available for video files")

    from src.services.task_queue import task_queue

    # Check if we have cached screenshots
    cache_path, content_hash = task_queue.get_cached_path(
        str(media_path), TaskType.SCREENSHOTS
    )

    if cache_path and cache_path.exists() and cache_path.is_file():
        # Return URL to cached stitched screenshot image
        return ScreenshotsResponse(
            status="cached",
            screenshot_url=f"/api/media/screenshot/{content_hash}",
            content_hash=content_hash,
        )

    # Check if task is already in queue or processing
    task_id = f"{TaskType.SCREENSHOTS}_{media_path}"
    existing_task = task_queue.get_task(task_id)
    
    if existing_task and existing_task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING):
        # Task already exists and is being processed
        return ScreenshotsResponse(
            status="processing",
            task_id=existing_task.id,
            message="Screenshots generation in progress",
        )

    # No cache and no existing task - create and submit async task
    task = task_queue.create_task(str(media_path), TaskType.SCREENSHOTS)
    await task_queue.submit(task)

    return ScreenshotsResponse(
        status="processing",
        task_id=task.id,
        message="Screenshots generation started",
    )


@router.get("/screenshot/{content_hash}")
async def get_screenshot(content_hash: str):
    """
    Get the stitched screenshot image by content hash.
    """
    screenshot_path = Path(settings.screenshots_cache_dir) / f"{content_hash}.jpg"

    if not screenshot_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")

    def iter_file():
        with open(screenshot_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type="image/jpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(screenshot_path.stat().st_size),
            "X-Cache-Status": "hit",
        },
    )

@router.post("/preload", response_model=PreloadResponse)
async def preload_media(request: PreloadRequest):
    """
    Batch submit media processing tasks.

    For each path and type combination:
    - Check if cache exists
    - If not cached, create and submit task asynchronously

    Returns counts of submitted and cached tasks.
    """
    from src.services.task_queue import task_queue

    submitted = 0
    cached = 0

    # Validate and process each path
    for path in request.paths:
        try:
            media_path = validate_path(path)
        except HTTPException:
            # Skip invalid paths - let task processing handle errors
            continue

        if not media_path.is_file():
            continue

        # Process each requested type
        for type_str in request.types:
            try:
                task_type = TaskType(type_str)
            except ValueError:
                # Skip invalid types
                continue

            # Check if already cached
            cache_path, _ = task_queue.get_cached_path(str(media_path), task_type)
            if cache_path and cache_path.exists():
                if cache_path.stat().st_mtime >= media_path.stat().st_mtime:
                    cached += 1
                    continue

            # Create and submit task
            task = task_queue.create_task(str(media_path), task_type)
            await task_queue.submit(task)
            submitted += 1

    return PreloadResponse(submitted=submitted, cached=cached, total=submitted + cached)


@router.get("/task/{task_id:path}")
async def get_task_status(task_id: str):
    """Get status of a media processing task."""
    from urllib.parse import unquote
    from src.services.task_queue import task_queue, TaskType

    # Decode the task_id (may be URL-encoded by frontend)
    decoded_task_id = unquote(task_id)

    # Try to find task in memory
    task = task_queue.get_task(decoded_task_id)

    if task:
        # Task exists in memory - return its status
        response = {
            "task_id": task.id,
            "media_path": task.media_path,
            "task_type": str(task.task_type),
            "status": str(task.status),
            "progress": task.progress,
            "created_at": task.created_at.isoformat(),
        }
        if task.started_at:
            response["started_at"] = task.started_at.isoformat()
        if task.completed_at:
            response["completed_at"] = task.completed_at.isoformat()
        if task.result_path:
            response["result_path"] = task.result_path
        if task.error:
            response["error"] = task.error
        return response

    # Task not in memory - check if cached result exists
    # Parse task_id format: "{task_type}_{media_path}"
    if "_" not in decoded_task_id:
        raise HTTPException(status_code=404, detail="Task not found")

    task_type_str, media_path = decoded_task_id.split("_", 1)
    try:
        task_type = TaskType(task_type_str)
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid task type")

    # Check if cached result exists
    cache_path, content_hash = task_queue.get_cached_path(media_path, task_type)

    if cache_path and cache_path.exists():
        # Cache exists - return completed status
        return {
            "task_id": decoded_task_id,
            "media_path": media_path,
            "task_type": task_type_str,
            "status": "completed",
            "progress": 100,
            "result_path": str(cache_path),
        }

    # No task and no cache
    raise HTTPException(status_code=404, detail="Task not found")

@router.get("/thumbnail/wait")
async def wait_for_thumbnail(path: str = Query(...), timeout: float = Query(30.0)):
    """
    Wait for thumbnail to be ready,    Returns the thumbnail when ready, or timeout.
    """
    media_path = validate_path(path)

    from src.services.task_queue import task_queue, TaskType

    # Check cache first
    cache_path, _ = task_queue.get_cached_path(str(media_path), TaskType.THUMBNAIL)

    if cache_path and cache_path.exists():
        if cache_path.stat().st_mtime >= media_path.stat().st_mtime:

            def iter_file():
                with open(cache_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        yield chunk

            return StreamingResponse(
                iter_file(),
                media_type="image/jpeg",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(cache_path.stat().st_size),
                    "X-Cache-Status": "hit",
                },
            )

    # Create and submit task if not already processing
    task = task_queue.create_task(str(media_path), TaskType.THUMBNAIL)
    if task.status == TaskStatus.PENDING:
        await task_queue.submit(task)
    # Wait for completion
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if task.status == TaskStatus.COMPLETED:
            # Return the completed thumbnail
            if task.result_path:
                result_path = Path(task.result_path)
                if result_path.exists():

                    def iter_file():
                        with open(result_path, "rb") as f:
                            while chunk := f.read(1024 * 1024):
                                yield chunk

                    return StreamingResponse(
                        iter_file(),
                        media_type="image/jpeg",
                        headers={
                            "Accept-Ranges": "bytes",
                            "Content-Length": str(result_path.stat().st_size),
                            "X-Cache-Status": "generated",
                        },
                    )
        elif task.status == TaskStatus.FAILED:
            raise HTTPException(
                status_code=500,
                detail=f"Thumbnail generation failed: {task.error}",
            )

        await asyncio.sleep(0.5)

    raise HTTPException(status_code=408, detail="Timeout waiting for thumbnail")


@router.get("/stream")
async def stream_media(
    path: str = Query(...),
    range: str = Header(None),
    format: str = Query("original", pattern="^(original|webm|mp4)$"),
):
    """
    Stream media file with range request support.

    Supports:
    - Range requests for video seeking
    - Transcoding to webm/mp4 for browser compatibility (async)
    """
    media_path = validate_path(path)

    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if media_path.stat().st_size > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    media_type = get_media_type(media_path)
    if media_type == "unknown":
        raise HTTPException(status_code=400, detail="Not a supported media file")

    file_size = media_path.stat().st_size
    file_range = None

    # Parse range header
    if range:
        try:
            range_str = range.replace("bytes=", "")
            start_str, end_str = range_str.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            file_range = (start, end)
        except (ValueError, AttributeError):
            raise HTTPException(status_code=400, detail="Invalid range header")

    # For videos with format conversion
    if media_type == "video" and format != "original":
        from src.services.task_queue import task_queue, TaskType

        # Check if we have a cached transcode
        cache_path, _ = task_queue.get_cached_path(str(media_path), TaskType.TRANSCODE)

        if cache_path and cache_path.exists():
            # Verify cache is still valid
            if cache_path.stat().st_mtime >= media_path.stat().st_mtime:
                media_path = cache_path
                file_size = media_path.stat().st_size
            else:
                # Cache outdated, create new task
                cache_path = None

        if not cache_path or not cache_path.exists():
            # Create and submit transcode task
            task = task_queue.create_task(str(media_path), TaskType.TRANSCODE)
            await task_queue.submit(task)

            # Return task status (video will be transcoded in background)
            return {
                "task_id": task.id,
                "status": task.status,
                "message": "Video transcode started. Poll /media/task/{task_id} for status.",
                "poll_url": f"/media/task/{task.id}",
                "original_format": media_path.suffix.lower(),
                "target_format": format,
            }

    # Determine content type
    mime_type, _ = mimetypes.guess_type(str(media_path))

    # For videos, use range requests
    if media_type == "video" and file_range:
        start, end = file_range
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_file_range(range_start: int, range_end: int):
            with open(media_path, "rb") as f:
                f.seek(range_start)
                remaining = range_end - range_start + 1
                while remaining > 0:
                    chunk_size = min(remaining, 1024 * 1024)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
                    remaining -= chunk_size

        return StreamingResponse(
            iter_file_range(start, end),
            media_type=mime_type or "video/mp4",
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )
    else:
        # For images or full video files
        def iter_file_full():
            with open(media_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(
            iter_file_full(),
            media_type=mime_type or "application/octet-stream",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )


@router.get("/stream/wait")
async def wait_for_stream(
    path: str = Query(...),
    format: str = Query("webm", pattern="^(webm|mp4)$"),
    timeout: float = Query(300.0),
):
    """Wait for video transcode to complete, then stream."""
    media_path = validate_path(path)

    from src.services.task_queue import task_queue, TaskType

    # Check cache first
    cache_path, _ = task_queue.get_cached_path(str(media_path), TaskType.TRANSCODE)

    if cache_path and cache_path.exists():
        if cache_path.stat().st_mtime >= media_path.stat().st_mtime:
            # Stream the cached file
            file_size = cache_path.stat().st_size
            mime_type, _ = mimetypes.guess_type(str(cache_path))

            def iter_file():
                with open(cache_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        yield chunk

            return StreamingResponse(
                iter_file(),
                media_type=mime_type or "video/webm",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(file_size),
                    "X-Cache-Status": "hit",
                },
            )

    # Create and submit task if not already processing
    task = task_queue.create_task(str(media_path), TaskType.TRANSCODE)
    if task.status == TaskStatus.PENDING:
        await task_queue.submit(task)
    # Wait for completion
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if task.status == TaskStatus.COMPLETED:
            if task.result_path:
                result_path = Path(task.result_path)
                if result_path.exists():
                    file_size = result_path.stat().st_size
                    mime_type, _ = mimetypes.guess_type(str(result_path))

                    def iter_file():
                        with open(result_path, "rb") as f:
                            while chunk := f.read(1024 * 1024):
                                yield chunk

                    return StreamingResponse(
                        iter_file(),
                        media_type=mime_type or "video/webm",
                        headers={
                            "Accept-Ranges": "bytes",
                            "Content-Length": str(file_size),
                            "X-Cache-Status": "generated",
                        },
                    )
        elif task.status == TaskStatus.FAILED:
            raise HTTPException(
                status_code=500,
                detail=f"Transcode failed: {task.error}",
            )

        await asyncio.sleep(1.0)

    raise HTTPException(status_code=408, detail="Timeout waiting for transcode")


@router.get("/stream/live")
async def live_stream(
    path: str = Query(...),
    start_time: float = Query(0.0, description="Start time in seconds"),
    duration: float = Query(30.0, description="Duration to transcode in seconds"),
):
    """
    Stream transcoded video segment in real-time.

    This endpoint enables real-time transcoding from a specific start time,
    allowing users to seek to any position without waiting for the full video.

    Args:
        path: Path to media file
        start_time: Start time in seconds (default: 0)
        duration: Duration to transcode in seconds (default: 30)
    """
    media_path = validate_path(path)

    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if media_path.stat().st_size > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    media_type = get_media_type(media_path)
    if media_type != "video":
        raise HTTPException(status_code=400, detail="Live stream only available for video files")

    async def generate_stream():
        """
        Generate transcoded stream using ffmpeg.
        Uses -ss to seek to start time, and -t to limit duration for faster seeking.
        """
        # FFmpeg command for live transcoding
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_time),  # Seek to start time
            "-i", str(media_path),
            "-t", str(duration),  # Limit duration
            "-c:v", "libvpx-vp9",
            "-c:a", "libopus",
            "-threads", "2",  # Use 2 threads for faster start
            "-speed", "4",  # Fast encoding for live stream
            "-crf", "30",  # Quality
            "-b:v", "0",  # Use CRF mode
            "-vf", "scale=-2:480",  # Max 480p for web
            "-row-mt", "1",
            "-f", "webm",  # Output format
            "-",  # Output to stdout
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream():
            try:
                while True:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(65536),
                        timeout=0.1
                    )
                    if not chunk:
                        break
                    yield chunk
            except asyncio.TimeoutError:
                pass
            finally:
                # Ensure process is cleaned up
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        process.kill()

        return StreamingResponse(
            read_stream(),
            media_type="video/webm",
            headers={
                "X-Stream-Type": "live-transcode",
                "X-Start-Time": str(start_time),
                "Cache-Control": "no-cache",
            },
        )

    return await generate_stream()


# HLS Streaming endpoints
@router.get("/hls/playlist")
async def get_hls_playlist(path: str = Query(...)):
    """
    Generate HLS playlist (m3u8) for video file.
    
    This endpoint generates an HLS manifest that allows browsers to
    stream video with proper seeking support.
    """
    media_path = validate_path(path)
    
    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    if media_path.stat().st_size > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    media_type = get_media_type(media_path)
    if media_type != "video":
        raise HTTPException(status_code=400, detail="HLS only available for video files")
    
    from src.services.task_queue import task_queue
    
    # Compute content hash for cache key
    content_hash = task_queue._compute_content_hash(media_path)
    
    # Get video duration using ffprobe
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path)
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
    except Exception:
        duration = 60.0  # Default duration if probe fails
    
    # Generate HLS playlist with 10-second segments
    segment_duration = 10
    num_segments = int(duration / segment_duration) + 1
    
    playlist_lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{segment_duration}",
        "#EXT-X-PLAYLIST-TYPE:VOD",
    ]
    
    for i in range(num_segments):
        start_time = i * segment_duration
        seg_duration = min(segment_duration, duration - start_time)
        if seg_duration > 0:
            playlist_lines.append(f"#EXTINF:{seg_duration:.1f},")
            playlist_lines.append(f"/api/media/hls/segment/{content_hash}/{i}.ts?path={path}")
    
    playlist_lines.append("#EXT-X-ENDLIST")
    
    playlist_content = "\n".join(playlist_lines)
    
    return Response(
        content=playlist_content,
        media_type="application/vnd.apple.mpegurl",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.get("/hls/segment/{content_hash}/{segment}.ts")
async def get_hls_segment(
    content_hash: str,
    segment: str,
    path: str = Query(...)
):
    """
    Generate a single HLS TS segment.
    
    Segments are cached to disk for faster subsequent access.
    """
    media_path = validate_path(path)
    
    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Parse segment number from filename (e.g., "0.ts" -> 0)
    try:
        segment_num = int(segment.replace(".ts", ""))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid segment number")
    
    # Check for cached segment
    cache_dir = Path(settings.transcode_cache_dir) / "hls" / content_hash
    cache_path = cache_dir / f"{segment_num}.ts"
    
    if cache_path.exists():
        # Return cached segment
        def iter_cached():
            with open(cache_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        
        return StreamingResponse(
            iter_cached(),
            media_type="video/mp2t",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    # Generate segment using ffmpeg
    segment_duration = 10
    start_time = segment_num * segment_duration
    
    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command to generate TS segment
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),  # Fast input seek
        "-i", str(media_path),
        "-t", str(segment_duration),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ac", "2",
        "-ar", "44100",  # Standard audio sample rate
        "-af", "aresample=async=1:first_pts=0",  # Sync audio timestamps after seek
        "-vf", "scale=-2:720",
        "-avoid_negative_ts", "make_zero",  # Fix timestamp issues
        "-f", "mpegts",
        str(cache_path)  # Output to file for caching
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"FFmpeg segment error: {result.stderr}")
            raise HTTPException(status_code=500, detail="Failed to generate segment")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Segment generation timeout")
    
    # Return the generated segment
    def iter_segment():
        with open(cache_path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk
    
    return StreamingResponse(
        iter_segment(),
        media_type="video/mp2t",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
        }
    )

# Folder scanning for media files
@router.get("/scan")
async def folder_scan(path: str = Query(...), recursive: bool = False):
    """
    Scan a server-side directory for media files.

    Args:
        path: Directory path to scan
        recursive: Whether to scan subdirectories recursively

    Returns:
        List of media files with metadata
    """
    resolved_path = validate_path(path)

    if not resolved_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    return {
        "path": str(resolved_path),
        "recursive": recursive,
        "files": scan_directory(resolved_path, recursive),
    }
