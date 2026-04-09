"""Frozen CLIP inference engine for media quality scoring."""

from pathlib import Path
from typing import Dict, Any, Optional, Union, cast

import torch
from PIL import Image

from src.config import settings
from src.logger import get_logger
from src.models.video_sampler import VideoSampler
from src.training.frozen_clip_encoder import FrozenCLIPEncoder
from src.training.temporal_attention import TemporalAttention
from src.training.score_head import ScoreHead

logger = get_logger(__name__)
_ENGINE: Optional["FrozenClipEngine"] = None
_ENGINE_KEY: Optional[tuple[str, str, int]] = None


class FrozenClipEngineError(Exception):
    """Base exception for FrozenClipEngine errors."""

    pass


class ModelNotLoadedError(FrozenClipEngineError):
    """Exception raised when model is not loaded."""

    pass


class FileNotFoundError(FrozenClipEngineError):
    """Exception raised when media file is not found."""

    pass


class FrozenClipEngine:
    """Inference engine using frozen CLIP encoder with temporal attention.

    This engine loads a trained model consisting of:
    1. Frozen CLIPEncoder (openai/clip-vit-large-patch14) - 768-dim features
    2. TemporalAttention - aggregates video frame features
    3. ScoreHead - MLP predicting quality score in the configured 0-9 range

    Attributes:
        clip_encoder: Frozen CLIP vision encoder
        temporal_attention: Temporal attention module for video aggregation
        score_head: MLP score prediction head
        video_sampler: Video frame sampler
        device: Device for inference (cuda/cpu)

    Example:
        >>> engine = FrozenClipEngine(device="cuda")
        >>> engine.load_model("model_checkpoint.pt")
        >>>
        >>> # Score single image
        >>> score = engine.score_image("/path/to/image.jpg")
        >>> print(f"Image score: {score:.2f}")
        >>>
        >>> # Score video
        >>> score = engine.score_video("/path/to/video.mp4")
        >>> print(f"Video score: {score:.2f}")
    """

    # Supported media formats
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def __init__(
        self,
        device: str = "auto",
        num_frames: int = 12,
        temporal_heads: int = 8,
        temporal_dim: int = 256,
        score_hidden_dims: tuple[int, ...] = (256, 64),
    ):
        """Initialize the frozen CLIP inference engine.

        Args:
            device: Device for inference. "auto" uses CUDA if available.
            num_frames: Number of frames to sample from videos.
            temporal_heads: Number of attention heads in temporal attention.
            temporal_dim: Projection dimension for temporal attention.
            score_hidden_dims: Hidden dimensions for score head MLP.
        """
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Configuration
        self.num_frames = num_frames
        self.temporal_heads = temporal_heads
        self.temporal_dim = temporal_dim
        self.score_hidden_dims = score_hidden_dims
        self.long_video_strategy = "compress"
        self.min_long_frames: Optional[int] = None
        self.max_long_frames: Optional[int] = None
        self.score_range: tuple[float, float] = (0.0, 9.0)
        self.model_max_frames: Optional[int] = None

        # Components (lazy loaded)
        self._clip_encoder: Optional[FrozenCLIPEncoder] = None
        self._temporal_attention: Optional[TemporalAttention] = None
        self._score_head: Optional[ScoreHead] = None
        self._video_sampler: Optional[VideoSampler] = None

        # State
        self._is_loaded = False

        logger.info(
            f"FrozenClipEngine initialized on {self.device}, num_frames={num_frames}"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._is_loaded:
            raise ModelNotLoadedError(
                "Model not loaded. Call load_model() before inference."
            )

    def load_model(self, checkpoint_path: Union[str, Path]) -> "FrozenClipEngine":
        """Load trained model weights from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint file (.pt or .pth).

        Returns:
            self for method chaining.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            ModelNotLoadedError: If checkpoint format is invalid.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading model from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        config = checkpoint.get("config", {})

        # Align inference modules with the training-time architecture.
        self.num_frames = int(config.get("num_frames", self.num_frames))
        temporal_heads = int(config.get("num_heads", self.temporal_heads))
        temporal_dim = int(config.get("temporal_dim", self.temporal_dim))
        hidden_dims = tuple(config.get("hidden_dims", self.score_hidden_dims))
        max_frames = int(config.get("max_frames", 32))
        score_range = tuple(config.get("score_range", (0.0, 9.0)))
        self.long_video_strategy = str(
            config.get("long_video_strategy", self.long_video_strategy)
        )
        self.min_long_frames = config.get("min_long_frames", self.min_long_frames)
        self.max_long_frames = config.get("max_long_frames", self.max_long_frames)
        self.score_range = score_range
        self.model_max_frames = max_frames

        clip_model_name = str(config.get("clip_model_name", FrozenCLIPEncoder.MODEL_NAME))

        # Initialize CLIP encoder using the training-time base checkpoint.
        self._clip_encoder = FrozenCLIPEncoder(
            device=str(self.device),
            model_name=clip_model_name,
        )
        feature_dim = self._clip_encoder.get_feature_dim()

        # Initialize temporal attention
        self._temporal_attention = TemporalAttention(
            feature_dim=feature_dim,
            num_heads=temporal_heads,
            temporal_dim=temporal_dim,
            max_frames=max_frames,
        )

        # Initialize score head
        self._score_head = ScoreHead(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            score_range=score_range,
        )

        clip_state = checkpoint.get("clip_encoder_trainable_state")
        if clip_state:
            incompatible = self._clip_encoder.load_trainable_state_dict(clip_state)
            logger.info(
                "Loaded clip_encoder trainable state (%d tensors, missing=%d, unexpected=%d)",
                len(clip_state),
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )

        # Load weights from checkpoint
        if "temporal_attention" in checkpoint:
            self._temporal_attention.load_state_dict(checkpoint["temporal_attention"])
            logger.info("Loaded temporal_attention weights")
        elif "model_state_dict" in checkpoint:
            # Try loading from unified model state
            state_dict = checkpoint["model_state_dict"]
            temporal_state = {
                k.replace("temporal_attention.", ""): v
                for k, v in state_dict.items()
                if k.startswith("temporal_attention.")
            }
            if temporal_state:
                self._temporal_attention.load_state_dict(temporal_state)
                logger.info("Loaded temporal_attention from model_state_dict")

        if "score_head" in checkpoint:
            self._score_head.load_state_dict(checkpoint["score_head"])
            logger.info("Loaded score_head weights")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            head_state = {
                k.replace("score_head.", ""): v
                for k, v in state_dict.items()
                if k.startswith("score_head.")
            }
            if head_state:
                self._score_head.load_state_dict(head_state)
                logger.info("Loaded score_head from model_state_dict")

        # Move modules to device and set to eval mode
        self._temporal_attention = self._temporal_attention.to(self.device)
        self._score_head = self._score_head.to(self.device)

        self._temporal_attention.eval()
        self._score_head.eval()

        # Initialize video sampler using the training-time sampling config.
        self._video_sampler = VideoSampler(
            num_frames=self.num_frames,
            output_format="pil",
            long_video_strategy=self.long_video_strategy,
            min_long_frames=self.min_long_frames,
            max_long_frames=self.max_long_frames,
        )

        self._is_loaded = True
        logger.info(
            "Model loaded successfully with sampler config: "
            f"num_frames={self.num_frames}, "
            f"long_video_strategy={self.long_video_strategy}, "
            f"min_long_frames={self.min_long_frames}, "
            f"max_long_frames={self.max_long_frames}"
        )

        return self

    def _is_video(self, path: Path) -> bool:
        """Check if file is a video based on extension."""
        return path.suffix.lower() in self.VIDEO_EXTENSIONS

    def _is_image(self, path: Path) -> bool:
        """Check if file is an image based on extension."""
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    @torch.no_grad()
    def score_image(self, image_path: Union[str, Path]) -> float:
        """Score a single image.

        Args:
            image_path: Path to image file.

        Returns:
            Quality score in range [0, 9].

        Raises:
            ModelNotLoadedError: If model is not loaded.
            FileNotFoundError: If image file doesn't exist.
            FrozenClipEngineError: If image format is not supported.
        """
        self._ensure_loaded()

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not self._is_image(image_path):
            raise FrozenClipEngineError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported: {self.IMAGE_EXTENSIONS}"
            )

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Type narrowing - these are guaranteed non-None after _ensure_loaded()
        assert self._clip_encoder is not None
        assert self._temporal_attention is not None
        assert self._score_head is not None

        # Extract CLIP features
        features = self._clip_encoder.extract_features(image)

        # For single image, temporal attention returns input unchanged
        aggregated = self._temporal_attention(features.unsqueeze(0))
        aggregated = aggregated.squeeze(0)

        # Predict score
        score = self._score_head.predict(aggregated)

        return float(score.item())

    @torch.no_grad()
    def score_video(self, video_path: Union[str, Path]) -> float:
        """Score a video file.

        Samples frames from the video, extracts CLIP features,
        applies temporal attention for aggregation, and predicts
        a quality score.

        Args:
            video_path: Path to video file.

        Returns:
            Quality score in range [0, 9].

        Raises:
            ModelNotLoadedError: If model is not loaded.
            FileNotFoundError: If video file doesn't exist.
            VideoLoadError: If video cannot be loaded.
            FrozenClipEngineError: If video format is not supported.
        """
        self._ensure_loaded()

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not self._is_video(video_path):
            raise FrozenClipEngineError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported: {self.VIDEO_EXTENSIONS}"
            )

        # Sample frames from video
        assert self._video_sampler is not None
        frames = self._video_sampler.sample(video_path)

        if not frames:
            raise FrozenClipEngineError(f"No frames extracted from video: {video_path}")

        # Type narrowing - these are guaranteed non-None after _ensure_loaded()
        assert self._clip_encoder is not None
        assert self._temporal_attention is not None
        assert self._score_head is not None

        # Extract CLIP features for all frames (cast needed for type checker)
        features = self._clip_encoder.extract_features(cast(list[Image.Image], frames))

        # Apply temporal attention
        features = features.unsqueeze(0)
        attended = self._temporal_attention(features)

        # Aggregate temporal features (mean pooling)
        aggregated = self._temporal_attention.aggregate_temporal(
            attended, method="mean"
        )

        # Predict score
        score = self._score_head.predict(aggregated)

        return float(score.item())

    def score(
        self,
        media_path: Union[str, Path],
        is_video: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Score media file (auto-detect image or video).

        Args:
            media_path: Path to media file.
            is_video: Override auto-detection. None for auto-detect.

        Returns:
            Dict with:
                - score: Quality score in [0, 9]
                - media_path: Path to media file
                - media_type: "image" or "video"
        """
        media_path = Path(media_path)

        # Auto-detect media type
        if is_video is None:
            is_video = self._is_video(media_path)

        if is_video:
            score = self.score_video(media_path)
            media_type = "video"
        else:
            score = self.score_image(media_path)
            media_type = "image"

        return {
            "score": score,
            "media_path": str(media_path),
            "media_type": media_type,
        }

    def score_batch(
        self,
        media_paths: list[Union[str, Path]],
    ) -> list[Dict[str, Any]]:
        """Score multiple media files.

        Args:
            media_paths: List of paths to media files.

        Returns:
            List of result dicts, each with score, media_path, media_type.
            Failed items have score=None and error message.
        """
        self._ensure_loaded()

        paths = [Path(path) for path in media_paths]
        results: list[Dict[str, Any]] = [None] * len(paths)  # type: ignore[list-item]

        image_indices: list[int] = []
        image_inputs: list[Image.Image] = []
        image_paths: list[Path] = []
        video_indices: list[int] = []
        video_paths: list[Path] = []

        for index, media_path in enumerate(paths):
            try:
                if not media_path.exists():
                    raise FileNotFoundError(f"Media file not found: {media_path}")
                if not self._is_image(media_path) and not self._is_video(media_path):
                    raise FrozenClipEngineError(
                        f"Unsupported media format: {media_path.suffix}. "
                        f"Supported: {self.IMAGE_EXTENSIONS | self.VIDEO_EXTENSIONS}"
                    )

                if self._is_video(media_path):
                    video_indices.append(index)
                    video_paths.append(media_path)
                else:
                    image_inputs.append(Image.open(media_path).convert("RGB"))
                    image_indices.append(index)
                    image_paths.append(media_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"Failed to prepare {media_path}: {exc}")
                results[index] = {
                    "score": None,
                    "media_path": str(media_path),
                    "media_type": "unknown",
                    "error": str(exc),
                }

        if image_inputs:
            try:
                image_features = self._clip_encoder.extract_features(image_inputs)
                aggregated = self._temporal_attention(image_features.unsqueeze(1))
                aggregated = self._temporal_attention.aggregate_temporal(aggregated)
                image_scores = self._score_head.predict(aggregated)
                scores = image_scores.view(-1).tolist()
                for idx, media_path, score in zip(
                    image_indices,
                    image_paths,
                    scores,
                ):
                    results[idx] = {
                        "score": float(score),
                        "media_path": str(media_path),
                        "media_type": "image",
                    }
            except Exception as exc:  # pylint: disable=broad-except
                for idx, media_path in zip(image_indices, image_paths):
                    logger.error(f"Failed to score {media_path}: {exc}")
                    results[idx] = {
                        "score": None,
                        "media_path": str(media_path),
                        "media_type": "image",
                        "error": str(exc),
                    }

        for index, media_path in zip(video_indices, video_paths):
            if results[index] is not None:
                continue
            try:
                score = self.score_video(media_path)
                results[index] = {
                    "score": score,
                    "media_path": str(media_path),
                    "media_type": "video",
                }
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"Failed to score {media_path}: {exc}")
                results[index] = {
                    "score": None,
                    "media_path": str(media_path),
                    "media_type": "video",
                    "error": str(exc),
                }

        return [
            entry
            if entry is not None
            else {
                "score": None,
                "media_path": str(path),
                "media_type": "unknown",
                "error": "unknown scoring failure",
            }
            for path, entry in zip(paths, results)
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dict with model configuration and parameter counts.
        """
        self._ensure_loaded()

        assert self._temporal_attention is not None
        assert self._score_head is not None

        temporal_params = sum(p.numel() for p in self._temporal_attention.parameters())
        score_params = sum(p.numel() for p in self._score_head.parameters())
        assert self._clip_encoder is not None

        return {
            "device": str(self.device),
            "clip_model": self._clip_encoder.model_name,
            "clip_feature_dim": self._clip_encoder.get_feature_dim(),
            "temporal_attention_params": temporal_params,
            "score_head_params": score_params,
            "total_trainable_params": temporal_params + score_params,
            "num_frames": self.num_frames,
            "long_video_strategy": self.long_video_strategy,
            "min_long_frames": self.min_long_frames,
            "max_long_frames": self.max_long_frames,
            "model_max_frames": self.model_max_frames,
            "score_range": self.score_range,
            "is_loaded": self._is_loaded,
        }

    def __repr__(self) -> str:
        """String representation of the engine."""
        status = "loaded" if self._is_loaded else "not loaded"
        return f"FrozenClipEngine(device={self.device}, status={status})"


def get_gpu_memory_usage() -> list[dict[str, float | int]]:
    """Return per-device GPU memory stats without forcing model load."""
    if not torch.cuda.is_available():
        return []

    usage: list[dict[str, float | int]] = []
    for gpu_id in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
        allocated_bytes = torch.cuda.memory_allocated(gpu_id)
        reserved_bytes = torch.cuda.memory_reserved(gpu_id)
        usage.append(
            {
                "id": gpu_id,
                "allocated_gb": round(allocated_bytes / (1024**3), 3),
                "reserved_gb": round(reserved_bytes / (1024**3), 3),
                "free_gb": round(free_bytes / (1024**3), 3),
                "total_gb": round(total_bytes / (1024**3), 3),
            }
        )
    return usage


def create_engine_from_settings() -> FrozenClipEngine:
    """Return a cached Frozen CLIP engine configured from application settings."""
    global _ENGINE, _ENGINE_KEY

    engine_key = (
        settings.model_checkpoint,
        settings.model_device,
        settings.sample_frames,
    )
    if _ENGINE is None or _ENGINE_KEY != engine_key:
        _ENGINE = FrozenClipEngine(
            device=settings.model_device,
            num_frames=settings.sample_frames,
        )
        _ENGINE_KEY = engine_key
    return _ENGINE


def create_engine(
    checkpoint_path: Union[str, Path],
    device: str = "auto",
    **kwargs,
) -> FrozenClipEngine:
    """Convenience function to create and load a FrozenClipEngine.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device for inference.
        **kwargs: Additional arguments for FrozenClipEngine.

    Returns:
        Loaded FrozenClipEngine instance.

    Example:
        >>> engine = create_engine("checkpoints/best_model.pt")
        >>> score = engine.score_image("photo.jpg")
    """
    engine = FrozenClipEngine(device=device, **kwargs)
    engine.load_model(checkpoint_path)
    return engine


if __name__ == "__main__":
    import sys

    # Quick test
    if len(sys.argv) < 2:
        print("Usage: python frozen_clip_engine.py <checkpoint_path> [media_path]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    engine = FrozenClipEngine()
    engine.load_model(checkpoint_path)

    print("\nModel Info:")
    info = engine.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    if len(sys.argv) >= 3:
        media_path = sys.argv[2]
        result = engine.score(media_path)
        print("\nScoring result:")
        print(f"  Path: {result['media_path']}")
        print(f"  Type: {result['media_type']}")
        print(f"  Score: {result['score']:.2f}")
