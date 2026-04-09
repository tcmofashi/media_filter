"""Async media processing task queue for thumbnails and video transcoding."""

import asyncio
from concurrent.futures import ProcessPoolExecutor
import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from PIL import Image

from src.config import settings
from src.logger import get_logger
from src.models.video_sampler import VideoSampler

logger = get_logger(__name__)


# Module-level functions for ProcessPoolExecutor (must be picklable)
def _run_ffmpeg_thumbnail(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    """Run FFmpeg command for thumbnail generation. Returns (returncode, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stderr


def _create_video_grid(source_str: str, output_str: str, min_width: int, min_height: int, grid_size: int = 5) -> str:
    """Create a grid thumbnail from video frames. Returns output path string."""
    from pathlib import Path
    from PIL import Image
    from src.models.video_sampler import VideoSampler
    
    source = Path(source_str)
    output = Path(output_str)
    frame_width = min_width // grid_size
    frame_height = min_height // grid_size
    
    # Sample 25 frames from video
    sampler = VideoSampler(num_frames=25, output_format="pil")
    frames = sampler.sample(source)

    # Create grid image
    grid_image = Image.new("RGB", (min_width, min_height))

    for idx, frame in enumerate(frames):
        row = idx // grid_size
        col = idx % grid_size

        # Resize frame to fit grid cell (maintain aspect ratio, pad if needed)
        frame_resized = frame.resize(
            (frame_width, frame_height), Image.Resampling.LANCZOS  # pyright: ignore[reportArgumentType]
        )

        # Paste into grid
        x_offset = col * frame_width
        y_offset = row * frame_height
        grid_image.paste(frame_resized, (x_offset, y_offset))  # type: ignore[arg-type]
    # Save as JPEG
    grid_image.save(output, "JPEG", quality=95)
    return str(output)


def _create_screenshots_grid(source_str: str, output_str: str, grid_width: int = 1000, grid_height: int = 1000) -> str:
    """Create a 5x5 stitched screenshots image for video. Returns output path string."""
    from pathlib import Path
    from PIL import Image
    from src.models.video_sampler import VideoSampler
    
    source = Path(source_str)
    output = Path(output_str)
    grid_size = 5
    frame_width = grid_width // grid_size
    frame_height = grid_height // grid_size
    
    # Sample 25 frames from video
    sampler = VideoSampler(num_frames=25, output_format="pil")
    frames = sampler.sample(source)

    # Create grid image
    grid_image = Image.new("RGB", (grid_width, grid_height))

    for idx, frame in enumerate(frames):
        row = idx // grid_size
        col = idx % grid_size

        # Resize frame to fit grid cell
        frame_resized = frame.resize(
            (frame_width, frame_height), Image.Resampling.LANCZOS  # pyright: ignore[reportArgumentType]
        )
        # Paste into grid
        x_offset = col * frame_width
        y_offset = row * frame_height
        grid_image.paste(frame_resized, (x_offset, y_offset))  # type: ignore[arg-type]
    # Save as JPEG
    grid_image.save(output, "JPEG", quality=90)
    return str(output)


class StrEnum(str, Enum):
    """String enum compatible with Python 3.10+."""

    def __str__(self) -> str:
        return str(self.value)


class TaskType(StrEnum):
    """Type of media processing task."""

    THUMBNAIL = "thumbnail"
    SCREENSHOTS = "screenshots"
    TRANSCODE = "transcode"
class TaskStatus(StrEnum):
    """Status of a processing task."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MediaTask:
    """Represents a media processing task."""

    id: str
    media_path: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0  # 0-100
    error: Optional[str] = None
    result_path: Optional[str] = None

    def __post_init__(self):
        """Ensure timestamps are timezone-aware."""
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=None)


class MediaTaskQueue:
    """Async task queue for media processing."""

    def __init__(self):
        self._tasks: dict[str, MediaTask] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()
        # Process pool for CPU-intensive tasks (thumbnails, screenshots)
        self._process_pool: ProcessPoolExecutor | None = None
        self._tasks: dict[str, MediaTask] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()

    def _compute_content_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content (first 64KB for efficiency)."""
        file_size = file_path.stat().st_size
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            chunk = f.read(65536)
            if chunk:
                hasher.update(f"{file_size}:".encode() + chunk)
        return hasher.hexdigest()

    def _get_cache_path(
        self, media_path: Path, task_type: TaskType, content_hash: str
    ) -> Path:
        """Get cache file path based on content hash."""
        if task_type == TaskType.THUMBNAIL:
            cache_dir = Path(settings.thumbnail_cache_dir)
            return cache_dir / f"{content_hash}.jpg"
        elif task_type == TaskType.SCREENSHOTS:
            # For screenshots, return a single stitched image file
            cache_dir = Path(settings.screenshots_cache_dir)
            return cache_dir / f"{content_hash}.jpg"
        else:  # TRANSCODE
            cache_dir = Path(settings.transcode_cache_dir)
            # Determine output format based on original file
            return cache_dir / f"{content_hash}.webm"
    def create_task(self, media_path: str, task_type: TaskType) -> MediaTask:
        """Create a new processing task."""
        task_id = f"{task_type}_{media_path}"

        # Check if task already exists
        if task_id in self._tasks:
            return self._tasks[task_id]

        task = MediaTask(
            id=task_id,
            media_path=media_path,
            task_type=task_type,
        )
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[MediaTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_cached_path(
        self, media_path: str, task_type: TaskType
    ) -> tuple[Optional[Path], Optional[str]]:
        """Check if cached result exists for given media path.

        Returns:
            (cache_path, content_hash) if cache exists and valid, (None, None) otherwise
        """
        path = Path(media_path)
        if not path.exists():
            return None, None

        try:
            content_hash = self._compute_content_hash(path)
            cache_path = self._get_cache_path(path, task_type, content_hash)

            # Check if cache exists and is newer than source
            if cache_path.exists():
                # For screenshots, check if it's a file (stitched image)
                if task_type == TaskType.SCREENSHOTS:
                    if cache_path.is_file() and cache_path.stat().st_mtime >= path.stat().st_mtime:
                        return cache_path, content_hash
                elif cache_path.stat().st_mtime >= path.stat().st_mtime:
                    return cache_path, content_hash
        except Exception as e:
            logger.error(f"Error checking cache for {media_path}: {e}")

        return None, None

    async def start(self):
        """Start the task queue processor."""
        if self._running:
            return

        self._running = True
        
        # Create process pool for CPU-intensive tasks
        self._process_pool = ProcessPoolExecutor(max_workers=settings.process_pool_workers)
        logger.info(f"Process pool created with {settings.process_pool_workers} workers")
        
        # Start worker tasks
        for i in range(settings.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(
            f"Media task queue started with {settings.max_concurrent_tasks} async workers"
        )
        """Start the task queue processor."""
        if self._running:
            return

        self._running = True
        # Start worker tasks
        for i in range(settings.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(
            f"Media task queue started with {settings.max_concurrent_tasks} workers"
        )

    async def stop(self):
        """Stop the task queue processor."""
        self._running = False
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        
        # Shutdown process pool
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
            logger.info("Process pool shutdown complete")
        
        logger.info("Media task queue stopped")
        """Stop the task queue processor."""
        self._running = False
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        logger.info("Media task queue stopped")

    async def submit(self, task: MediaTask) -> str:
        """Submit a task to the queue."""
        async with self._lock:
            # Only submit if task is PENDING (not already processing or completed)
            if task.status != TaskStatus.PENDING:
                logger.info(f"Task {task.id} already {task.status}, skipping submit")
                return task.id
            
            if task.id not in self._tasks:
                self._tasks[task.id] = task
            await self._queue.put(task)
            logger.info(f"Submitted task {task.id}")
        
        return task.id

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks from queue."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(), timeout=settings.task_poll_interval
                )
                await self._process_task(task)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _process_task(self, task: MediaTask):
        """Process a single task."""
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()

        try:
            media_path = Path(task.media_path)

            # Compute content hash
            content_hash = self._compute_content_hash(media_path)
            cache_path = self._get_cache_path(media_path, task.task_type, content_hash)

            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if task.task_type == TaskType.THUMBNAIL:
                await self._generate_thumbnail(media_path, cache_path, task)
            elif task.task_type == TaskType.SCREENSHOTS:
                await self._generate_screenshots(media_path, cache_path, task)
            else:  # TRANSCODE
                await self._transcode_video(media_path, cache_path, task)
            task.result_path = str(cache_path)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100

            logger.info(f"Task {task.id} completed: {cache_path}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error(f"Task {task.id} failed: {e}")

    def _is_video_file(self, source: Path) -> bool:
        """Check if file is a video based on extension."""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
        return source.suffix.lower() in video_extensions

    async def _generate_thumbnail(self, source: Path, output: Path, task: MediaTask):
        """Generate thumbnail using FFmpeg for images, or 5x5 grid for videos."""
        if self._is_video_file(source):
            await self._generate_video_grid_thumbnail(source, output, task)
        else:
            await self._generate_image_thumbnail(source, output, task)

    async def _generate_image_thumbnail(self, source: Path, output: Path, task: MediaTask):
        """Generate thumbnail for image using FFmpeg (runs in process pool)."""
        min_width, min_height = settings.thumbnail_min_size

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vframes",
            "1",
            "-vf",
            f"scale={min_width}:{min_height}:force_original_aspect_ratio=decrease,pad={min_width}:{min_height}:(ow-iw)/2:(oh-ih)/2",
            "-q:v",
            str(settings.thumbnail_quality),
            str(output),
        ]

        loop = asyncio.get_event_loop()
        returncode, stderr = await loop.run_in_executor(
            self._process_pool,
            _run_ffmpeg_thumbnail,
            cmd,
            30,  # timeout
        )

        if returncode != 0:
            raise RuntimeError(f"FFmpeg thumbnail failed: {stderr}")
    async def _generate_video_grid_thumbnail(self, source: Path, output: Path, task: MediaTask):
        """Generate 5x5 grid thumbnail for video using 25 equally spaced frames (runs in process pool)."""
        min_width, min_height = settings.thumbnail_min_size
        
        # Run grid creation in process pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._process_pool,
            _create_video_grid,
            str(source),
            str(output),
            min_width,
            min_height,
            5,  # grid_size
        )

        logger.info(f"Generated 5x5 grid thumbnail for video: {source}")

    async def _generate_screenshots(self, source: Path, output: Path, task: MediaTask):
        """Generate a single 5x5 stitched screenshots image for video display (runs in process pool)."""
        # Run screenshots creation in process pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._process_pool,
            _create_screenshots_grid,
            str(source),
            str(output),
            1000,  # grid_width
            1000,  # grid_height
        )

        logger.info(f"Generated 5x5 stitched screenshots for video: {source}")

    async def _transcode_video(self, source: Path, output: Path, task: MediaTask):
        """Transcode video to webm format with optimized settings."""
        # Optimized encoding settings for better quality and smooth playback
        # -threads 0: use all available threads
        # -speed 2: balanced encoding speed (better quality than speed 4)
        # -crf 28: good quality (lower = better, range 0-63)
        # -vf scale: limit resolution to 720p for faster decoding
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libvpx-vp9",
            "-c:a",
            "libopus",
            "-threads",
            "0",
            "-speed",
            "2",  # Balanced speed for better quality
            "-crf",
            "28",  # Better quality (was 35)
            "-b:v",
            "0",  # Use CRF mode
            "-vf",
            "scale=-2:720",  # Limit to 720p for smoother playback
            "-row-mt",
            "1",  # Multi-threaded row encoding
            str(output),
        ]

        logger.info(f"Starting transcode: {source} -> {output}")

        # Run in thread pool to avoid blocking (longer timeout for large files)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=3600),
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg transcode failed: {result.stderr}")


# Global task queue instance
task_queue = MediaTaskQueue()
