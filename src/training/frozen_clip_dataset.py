"""Unified dataset for frozen CLIP encoder training.

Loads both images and videos with frame sampling support.
Images return single frame, videos return multiple sampled frames.
"""

import json
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any, Literal

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.video_sampler import VideoSampler
from src.logger import get_logger

logger = get_logger(__name__)


class FrozenClipDataset(Dataset):
    """Unified dataset for frozen CLIP encoder training.

    Loads labeled media samples from JSON Lines file and provides
    frames and scores for training.

    Images are returned as single-frame lists.
    Videos are sampled using VideoSampler with configurable frame count.

    Attributes:
        IMAGE_EXTENSIONS: Supported image file extensions
        VIDEO_EXTENSIONS: Supported video file extensions
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def __init__(
        self,
        labels_path: str,
        num_frames: int = 8,
        validate_files: bool = True,
        score_range: Optional[Tuple[float, float]] = None,
        normalize_scores: bool = False,
        long_video_strategy: Literal["compress", "expand"] = "compress",
        min_long_frames: Optional[int] = None,
        max_long_frames: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            labels_path: Path to JSON Lines file with labels.
                        Each line: {"media_path": str, "score": float}
            num_frames: Number of frames to sample from videos (default: 8).
            validate_files: Whether to validate file existence during init.
            score_range: Expected score range for normalization (min, max).
                        If None, infer from loaded labels.
            normalize_scores: Whether to return normalized scores in __getitem__.
            long_video_strategy: Sampling strategy for videos longer than 5 minutes.
            min_long_frames: Minimum frames to retain for long videos.
            max_long_frames: Maximum frames to retain for long videos.
        """
        self.labels_path = Path(labels_path)
        self.num_frames = num_frames
        self.score_range = score_range
        self.normalize_scores = normalize_scores
        self.long_video_strategy = long_video_strategy
        self.min_long_frames = min_long_frames
        self.max_long_frames = max_long_frames

        # Initialize video sampler with specified frame count
        self._video_sampler = VideoSampler(
            num_frames=num_frames,
            output_format="pil",
            long_video_strategy=long_video_strategy,
            min_long_frames=min_long_frames,
            max_long_frames=max_long_frames,
        )

        # Load samples
        self._samples: List[dict] = []
        self._load_samples(validate_files)

        if self.score_range is None:
            self.score_range = self._infer_score_range()

        logger.info(
            f"FrozenClipDataset initialized: {len(self._samples)} samples, "
            f"num_frames={num_frames}, long_video_strategy={long_video_strategy}, "
            f"min_long_frames={min_long_frames}, max_long_frames={max_long_frames}, "
            f"score_range={score_range}"
        )

    def _load_samples(self, validate_files: bool) -> None:
        """Load samples from JSON Lines file.

        Args:
            validate_files: Whether to check file existence.
        """
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    continue

                media_path = item.get("media_path")
                if not media_path:
                    logger.warning(f"Missing media_path at line {line_num}")
                    continue

                score = item.get("score")
                if score is None:
                    logger.warning(f"Missing score at line {line_num}")
                    continue

                # Validate file existence if requested
                if validate_files:
                    path = Path(media_path)
                    if not path.exists():
                        logger.debug(f"Media file not found, skipping: {media_path}")
                        continue

                self._samples.append(
                    {
                        "media_path": media_path,
                        "score": float(score),
                    }
                )

        if not self._samples:
            raise ValueError(f"No valid samples loaded from {self.labels_path}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[Union[Image.Image, np.ndarray]], float]:
        """Get frames and score by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (frames, score) where:
            - frames: List of frames (PIL Images or numpy arrays)
            - score: Raw score or normalized score depending on normalize_scores

        Raises:
            IndexError: If index out of range.
            FileNotFoundError: If media file not found.
            VideoLoadError: If video cannot be loaded.
        """
        if idx < 0 or idx >= len(self._samples):
            raise IndexError(f"Index {idx} out of range [0, {len(self._samples)})")

        sample = self._samples[idx]
        media_path = Path(sample["media_path"])

        is_video = self._is_video(media_path)

        frames = self._load_frames(media_path, is_video)

        if self.normalize_scores:
            score = self._normalize_score(sample["score"])
        else:
            score = float(sample["score"])

        return frames, score

    def _infer_score_range(self) -> Tuple[float, float]:
        """Infer score range from loaded samples."""
        if not self._samples:
            return (0.0, 1.0)

        scores = [float(sample["score"]) for sample in self._samples]
        return (float(min(scores)), float(max(scores)))

    def _is_video(self, path: Path) -> bool:
        """Check if file is a video based on extension.

        Args:
            path: File path.

        Returns:
            True if video extension, False otherwise.
        """
        return path.suffix.lower() in self.VIDEO_EXTENSIONS

    def _is_image(self, path: Path) -> bool:
        """Check if file is an image based on extension.

        Args:
            path: File path.

        Returns:
            True if image extension, False otherwise.
        """
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _load_frames(
        self, media_path: Path, is_video: bool
    ) -> List[Union[Image.Image, np.ndarray]]:
        """Load frames from media file.

        Args:
            media_path: Path to media file.
            is_video: Whether the file is a video.

        Returns:
            List of frames (PIL Images or numpy arrays depending on sampler config).

        Raises:
            FileNotFoundError: If file not found.
            ValueError: If unsupported format.
            VideoLoadError: If video cannot be loaded.
        """
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        if is_video:
            try:
                return self._video_sampler.sample(media_path)
            except Exception as exc:
                logger.warning(
                    "Failed to decode video %s, using blank-frame fallback: %s",
                    media_path,
                    exc,
                )
                return self._create_blank_frames(self.num_frames)
        elif self._is_image(media_path):
            try:
                image = Image.open(media_path).convert("RGB")
            except Exception as exc:
                logger.warning(
                    "Failed to load image %s, using blank-frame fallback: %s",
                    media_path,
                    exc,
                )
                return self._create_blank_frames(1)
            return [image]
        else:
            raise ValueError(
                f"Unsupported media format: {media_path.suffix}. "
                f"Image: {self.IMAGE_EXTENSIONS}, Video: {self.VIDEO_EXTENSIONS}"
            )

    def _create_blank_frames(self, frame_count: int) -> List[Image.Image]:
        """Create blank RGB frames as a last-resort media fallback."""
        frame_count = max(1, int(frame_count))
        return [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(frame_count)]

    def _normalize_score(self, score: float) -> float:
        """Normalize score to [0, 1] range.

        Args:
            score: Raw score value.

        Returns:
            Normalized score in [0, 1].
        """
        min_score, max_score = self.score_range
        normalized = (score - min_score) / (max_score - min_score)
        # Clamp to [0, 1] in case of outliers
        return max(0.0, min(1.0, normalized))

    def get_sample_info(self, idx: int) -> dict:
        """Get sample information without loading frames.

        Args:
            idx: Sample index.

        Returns:
            Dict with media_path, score, and is_video.
        """
        if idx < 0 or idx >= len(self._samples):
            raise IndexError(f"Index {idx} out of range")

        sample = self._samples[idx]
        media_path = Path(sample["media_path"])

        return {
            "media_path": str(media_path),
            "score": sample["score"],
            "normalized_score": self._normalize_score(sample["score"]),
            "is_video": self._is_video(media_path),
        }

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dict with total count, image/video counts, score statistics.
        """
        if not self._samples:
            return {
                "total": 0,
                "images": 0,
                "videos": 0,
                "score_min": 0.0,
                "score_max": 0.0,
                "score_mean": 0.0,
            }

        scores = [s["score"] for s in self._samples]
        image_count = sum(
            1 for s in self._samples if not self._is_video(Path(s["media_path"]))
        )
        video_count = len(self._samples) - image_count

        return {
            "total": len(self._samples),
            "images": image_count,
            "videos": video_count,
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }


def create_train_val_split(
    dataset: FrozenClipDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split dataset indices into training and validation sets.

    Args:
        dataset: FrozenClipDataset instance.
        val_ratio: Proportion of data for validation (0-1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices).
    """
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    logger.info(f"Split dataset: {len(train_indices)} train, {len(val_indices)} val")

    return train_indices, val_indices


def collate_fn(
    batch: List[Tuple[List[Union[Image.Image, np.ndarray]], float]],
) -> Dict[str, Any]:
    """Collate function for batching FrozenClipDataset samples.

    Handles variable-length frame lists by keeping them as nested lists.
    Each sample's frames remain separate for later CLIP processing.

    Args:
        batch: List of (frames, score) tuples from FrozenClipDataset.
               frames: List of PIL Images or numpy arrays (variable length)
               score: Normalized score in [0, 1]

    Returns:
        Dict with:
            - frames: List of frame lists (batch_size samples, each with variable frames)
            - scores: Tensor of shape (batch_size,) with normalized scores
            - num_frames: List of frame counts per sample

    Example:
        >>> dataset = FrozenClipDataset("labels.json", num_frames=8)
        >>> loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        >>> batch = next(iter(loader))
        >>> batch["frames"]  # List of 4 frame lists
        >>> batch["scores"]  # Tensor shape (4,)
        >>> batch["num_frames"]  # [1, 8, 8, 1] for 2 images + 2 videos
    """
    frames_list: List[List[Union[Image.Image, np.ndarray]]] = []
    scores_list: List[float] = []
    num_frames_list: List[int] = []

    for frames, score in batch:
        frames_list.append(frames)
        scores_list.append(score)
        num_frames_list.append(len(frames))

    return {
        "frames": frames_list,
        "scores": torch.tensor(scores_list, dtype=torch.float32),
        "num_frames": num_frames_list,
    }


def collate_fn_padded(
    batch: List[Tuple[List[Union[Image.Image, np.ndarray]], float]],
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Collate function with padding for fixed-size tensor output.

    Pads frame lists to the same length for models that require fixed-size input.
    Note: This is less memory-efficient than collate_fn for CLIP-based training.

    Args:
        batch: List of (frames, score) tuples from FrozenClipDataset.
        max_frames: Maximum number of frames to pad to. If None, uses max in batch.

    Returns:
        Dict with:
            - frames: Tensor of shape (batch_size, max_frames, C, H, W) or nested list
            - scores: Tensor of shape (batch_size,) with normalized scores
            - masks: Tensor of shape (batch_size, max_frames) indicating valid frames

    Example:
        >>> loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_padded)
        >>> batch = next(iter(loader))
        >>> batch["masks"]  # [[1, 0, ...], [1, 1, ...], ...]
    """
    frames_list: List[List[Union[Image.Image, np.ndarray]]] = []
    scores_list: List[float] = []
    num_frames_list: List[int] = []

    raw_max_len = max((len(frames) for frames, _ in batch), default=0)
    max_len = raw_max_len if max_frames is None else min(max_frames, raw_max_len)

    for frames, score in batch:
        trimmed_frames = frames[:max_len] if max_len else []
        frames_list.append(trimmed_frames)
        scores_list.append(score)
        num_frames_list.append(len(trimmed_frames))

    # Create masks
    batch_size = len(batch)
    masks = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, num in enumerate(num_frames_list):
        masks[i, :num] = 1.0

    return {
        "frames": frames_list,  # Keep as list for CLIP processor compatibility
        "scores": torch.tensor(scores_list, dtype=torch.float32),
        "num_frames": num_frames_list,
        "masks": masks,
        "max_frames": max_len,
    }


def create_dataloader(
    dataset: FrozenClipDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    use_padding: bool = False,
    max_frames: Optional[int] = None,
) -> DataLoader:
    """Create DataLoader for FrozenClipDataset with appropriate collate function.

    Args:
        dataset: FrozenClipDataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data at each epoch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether to copy tensors to CUDA pinned memory.
        drop_last: Whether to drop the last incomplete batch.
        use_padding: If True, use collate_fn_padded with frame masks.
        max_frames: Maximum frames for padding (only used if use_padding=True).

    Returns:
        DataLoader configured for FrozenClipDataset.

    Example:
        >>> dataset = FrozenClipDataset("labels.json", num_frames=8)
        >>> loader = create_dataloader(dataset, batch_size=16, shuffle=True)
        >>> for batch in loader:
        ...     frames = batch["frames"]  # List of frame lists
        ...     scores = batch["scores"]  # Tensor (batch_size,)
    """
    collate = collate_fn_padded if use_padding else collate_fn

    if use_padding:

        def collate_with_max(batch):
            return collate_fn_padded(batch, max_frames=max_frames)

        collate = collate_with_max

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
    )


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Test with labels.json
    labels_path = "labels.json"
    if len(sys.argv) > 1:
        labels_path = sys.argv[1]

    try:
        dataset = FrozenClipDataset(
            labels_path=labels_path,
            num_frames=8,
            validate_files=False,  # Skip validation for demo
        )

        # Print stats
        stats = dataset.get_stats()
        print("\nDataset Statistics:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Images: {stats['images']}")
        print(f"  Videos: {stats['videos']}")
        print(f"  Score range: [{stats['score_min']:.1f}, {stats['score_max']:.1f}]")
        print(f"  Score mean: {stats['score_mean']:.2f} ± {stats['score_std']:.2f}")

        # Test getitem
        if len(dataset) > 0:
            print("\nTesting __getitem__(0):")
            frames, score = dataset[0]
            print(f"  Frames: {len(frames)} frame(s)")
            print(f"  First frame size: {frames[0].size}")
            print(f"  Score: {score:.3f}")

            # Sample info
            info = dataset.get_sample_info(0)
            print(f"  Media path: {info['media_path']}")
            print(f"  Is video: {info['is_video']}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
