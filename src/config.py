"""Configuration management for the Frozen CLIP baseline."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="MF_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Frozen CLIP model settings
    model_checkpoint: str = "checkpoints/checkpoint_best.pt"
    model_device: str = "auto"
    clip_model_name: str = "openai/clip-vit-large-patch14"

    # Media sampling settings
    max_video_duration: int = 300
    sample_frames: int = 12
    video_segment_duration: int = 60
    video_max_segments: int = 2
    video_sample_fps: float = 1.0

    # API settings
    max_batch_size: int = 50
    request_timeout: int = 300

    # Storage settings
    database_path: str = "data/media_filter.db"

    # Media file handling settings
    allowed_paths: list[str] = ["/media", "/home", "/data"]
    max_file_size: int = 5 * 1024 * 1024 * 1024

    # Cache settings
    cache_base_dir: str = "data/cache"
    thumbnail_cache_dir: str = "data/cache/thumbnails"
    screenshots_cache_dir: str = "data/cache/screenshots"
    transcode_cache_dir: str = "data/cache/transcodes"

    # Thumbnail settings
    thumbnail_min_size: tuple[int, int] = (1000, 1000)
    thumbnail_quality: int = 2

    # Task queue settings
    max_concurrent_tasks: int = 8
    task_poll_interval: float = 0.5
    process_pool_workers: int = 12

    @property
    def cache_paths(self) -> dict[str, Path]:
        """Return all cache directory paths."""
        return {
            "base": Path(self.cache_base_dir),
            "thumbnails": Path(self.thumbnail_cache_dir),
            "screenshots": Path(self.screenshots_cache_dir),
            "transcodes": Path(self.transcode_cache_dir),
        }

    def ensure_cache_dirs(self) -> None:
        """Ensure all cache directories exist."""
        for path in self.cache_paths.values():
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_cache_dirs()
