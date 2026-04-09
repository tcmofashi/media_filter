"""Video sampling module for efficient frame extraction."""

import math
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
from PIL import Image

from decord import VideoReader, cpu

from src.config import settings
from src.logger import get_logger


logger = get_logger(__name__)


class VideoLoadError(Exception):
    """Exception raised when video loading fails."""

    pass


class VideoSampler:
    """Efficient video frame sampler using decord.

    Supports two sampling strategies:
    - Short videos (<5min): Uniform sampling
    - Long videos (>5min): Smart keyframe sampling to compress to 5min equivalent
    """

    SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def __init__(
        self,
        num_frames: int | None = None,
        max_duration: int | None = None,
        output_format: Literal["pil", "numpy"] = "pil",
        long_video_strategy: Literal["compress", "expand"] = "compress",
        min_long_frames: int | None = None,
        max_long_frames: int | None = None,
    ):
        """Initialize VideoSampler.

        Args:
            num_frames: Number of frames to sample. Defaults to config.sample_frames (12).
            max_duration: Maximum video duration in seconds. Defaults to config.max_video_duration (300).
            output_format: Output format for frames. Either "pil" (PIL Images) or "numpy" (numpy arrays).
            long_video_strategy: Strategy for videos longer than ``max_duration``.
                ``compress`` keeps roughly the old 5-minute-equivalent density.
                ``expand`` increases sampled frames with duration until ``max_long_frames``.
            min_long_frames: Minimum frames to keep for long videos. Defaults to ``num_frames``.
            max_long_frames: Maximum frames for long videos. Defaults to ``num_frames * 4``.
        """
        self.num_frames = num_frames or settings.sample_frames
        self.max_duration = max_duration or settings.max_video_duration
        self.output_format = output_format
        self.long_video_strategy = long_video_strategy
        self.min_long_frames = (
            max(1, int(min_long_frames))
            if min_long_frames is not None
            else self.num_frames
        )
        default_max_long_frames = max(self.min_long_frames, self.num_frames * 4)
        self.max_long_frames = (
            max(self.min_long_frames, int(max_long_frames))
            if max_long_frames is not None
            else default_max_long_frames
        )

    def sample(
        self, video_path: Union[str, Path]
    ) -> List[Union[Image.Image, np.ndarray]]:
        """Sample frames from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of sampled frames as PIL Images or numpy arrays.

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video format is not supported.
            VideoLoadError: If video cannot be loaded.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        except Exception as e:
            raise VideoLoadError(f"Failed to load video {video_path}: {e}") from e

        duration = len(vr) / vr.get_avg_fps()

        if duration <= self.max_duration:
            frames = self._uniform_sample(vr)
        else:
            frames = self._keyframe_sample(vr, duration)

        return frames

    def _uniform_sample(self, vr: VideoReader) -> List[Union[Image.Image, np.ndarray]]:
        """Sample frames uniformly from a short video.

        Args:
            vr: Decord VideoReader instance.

        Returns:
            List of sampled frames.
        """
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = self._get_frames(vr, indices)

        return self._convert_frames(frames)

    def _keyframe_sample(
        self, vr: VideoReader, duration: float
    ) -> List[Union[Image.Image, np.ndarray]]:
        """Sample keyframes from a long video, compressing to 5min equivalent.

        For long videos, we calculate how many frames would represent a 5min video
        at the same frame density, then sample keyframes uniformly across the video.

        Args:
            vr: Decord VideoReader instance.
            duration: Video duration in seconds.

        Returns:
            List of sampled keyframes.
        """
        target_frames = self._compute_long_video_target_frames(duration)

        # Sample uniformly across video
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

        # Get keyframes (decord returns keyframes efficiently)
        frames = self._get_frames(vr, indices)

        return self._convert_frames(frames)

    def _compute_long_video_target_frames(self, duration: float) -> int:
        """Determine how many frames to preserve for long videos."""
        if self.long_video_strategy == "compress":
            target_frames = int(self.num_frames * (self.max_duration / duration) * 2)
            target_frames = max(target_frames, 4)
        else:
            density_scaled_frames = math.ceil(
                self.num_frames * (duration / self.max_duration)
            )
            target_frames = max(density_scaled_frames, self.min_long_frames)

        return max(
            self.min_long_frames,
            min(int(target_frames), self.max_long_frames),
        )

    def _get_frames(self, vr: VideoReader, indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """Load frames with a per-frame fallback when batched decoding fails."""
        indices = np.asarray(indices, dtype=int)

        try:
            return vr.get_batch(indices).asnumpy()
        except Exception as batch_error:
            logger.warning(
                "decord batched decode failed for %d frames; falling back to per-frame decode: %s",
                len(indices),
                batch_error,
            )

        recovered_frames = []
        for idx in indices:
            try:
                recovered_frames.append(vr[int(idx)].asnumpy())
            except Exception as frame_error:
                logger.warning("decord failed to decode frame %d: %s", int(idx), frame_error)

        if not recovered_frames:
            raise VideoLoadError("Failed to decode any requested frame from video.")

        while len(recovered_frames) < len(indices):
            recovered_frames.append(recovered_frames[-1].copy())

        return np.stack(recovered_frames, axis=0)

    def _convert_frames(
        self, frames: np.ndarray
    ) -> List[Union[Image.Image, np.ndarray]]:
        """Convert frames to the desired output format.

        Args:
            frames: Numpy array of frames (N, H, W, C).

        Returns:
            List of frames in the desired format.
        """
        if self.output_format == "pil":
            return [Image.fromarray(frame) for frame in frames]
        else:
            return [frame for frame in frames]

    @classmethod
    def get_video_info(cls, video_path: Union[str, Path]) -> dict:
        """Get video metadata without loading frames.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with video metadata (fps, frame_count, duration, width, height).
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        except Exception as e:
            raise VideoLoadError(f"Failed to load video {video_path}: {e}") from e

        return {
            "fps": vr.get_avg_fps(),
            "frame_count": len(vr),
            "duration": len(vr) / vr.get_avg_fps(),
            "width": vr[0].shape[1],
            "height": vr[0].shape[0],
        }

    def compute_motion_scores(
        self,
        video_path: Union[str, Path],
        segment_duration: float = 60.0,
        sample_fps: float = 1.0,
    ) -> List[dict]:
        """Compute motion scores for video segments.

        Uses frame differencing to calculate how much motion/change
        occurs in each segment of the video.

        Args:
            video_path: Path to the video file.
            segment_duration: Duration of each segment in seconds (default: 60s = 1min).
            sample_fps: Frames per second to sample for motion analysis (default: 1fps).

        Returns:
            List of dicts with segment info: {"start_time", "end_time", "motion_score", "frame_indices"}
        """
        video_path = Path(video_path)
        vr = VideoReader(str(video_path), ctx=cpu(0))

        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps

        # Calculate frame indices to sample at target fps
        sample_interval = int(fps / sample_fps)  # e.g., 30fps / 1fps = 30
        sample_indices = list(range(0, total_frames, sample_interval))

        if len(sample_indices) < 2:
            # Too short for motion analysis
            return [
                {
                    "start_time": 0.0,
                    "end_time": duration,
                    "motion_score": 0.0,
                    "frame_indices": sample_indices,
                }
            ]

        # Load all sample frames
        frames = self._get_frames(vr, sample_indices)

        # Convert to grayscale and compute frame differences
        gray_frames = np.array([self._to_gray(f) for f in frames])
        diffs = np.abs(np.diff(gray_frames.astype(np.float32), axis=0))

        # Sum of differences per frame pair (motion score)
        frame_motion = np.sum(diffs, axis=(1, 2))

        # Group by segments
        segment_duration_frames = int(
            segment_duration * sample_fps
        )  # frames per segment
        segments = []

        for seg_idx in range(
            int(np.ceil(len(sample_indices) / segment_duration_frames))
        ):
            start_frame = seg_idx * segment_duration_frames
            end_frame = min(start_frame + segment_duration_frames, len(sample_indices))

            # Motion scores for this segment (frame differences)
            seg_motion_indices = range(start_frame, min(end_frame, len(frame_motion)))
            if len(seg_motion_indices) > 0:
                seg_motion = np.mean([frame_motion[i] for i in seg_motion_indices])
            else:
                seg_motion = 0.0

            segments.append(
                {
                    "start_time": start_frame / sample_fps,
                    "end_time": end_frame / sample_fps,
                    "motion_score": float(seg_motion),
                    "frame_indices": sample_indices[start_frame:end_frame],
                }
            )

        return segments

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB frame to grayscale using luminosity method.

        Args:
            frame: RGB frame array (H, W, C).

        Returns:
            Grayscale array (H, W).
        """
        return (
            0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        ).astype(np.float32)

    def sample_active_segments(
        self,
        video_path: Union[str, Path],
        max_segments: int = 2,
        segment_duration: float = 60.0,
        frames_per_segment: int = 12,
        sample_fps: float = 1.0,
    ) -> List[Union[Image.Image, np.ndarray]]:
        """Sample frames from the most active segments of a video.

        This method:
        1. Computes motion scores for each 1-minute segment
        2. Selects the top N segments with highest motion
        3. Samples frames uniformly from selected segments at 1fps

        Args:
            video_path: Path to the video file.
            max_segments: Maximum number of segments to sample (default: 2).
            segment_duration: Duration of each segment in seconds (default: 60s).
            frames_per_segment: Number of frames to sample per segment (default: 12).
            sample_fps: Sampling rate for motion analysis (default: 1fps).

        Returns:
            List of sampled frames from the most active segments.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        except Exception as e:
            raise VideoLoadError(f"Failed to load video {video_path}: {e}") from e

        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps

        # If video is shorter than segment_duration, just do uniform sampling
        if duration <= segment_duration:
            return self._uniform_sample(vr)

        # Compute motion scores for all segments
        segments = self.compute_motion_scores(video_path, segment_duration, sample_fps)

        # Sort by motion score (descending) and select top N
        sorted_segments = sorted(
            segments, key=lambda x: x["motion_score"], reverse=True
        )
        top_segments = sorted_segments[:max_segments]

        # Sort selected segments by time (for temporal consistency)
        top_segments = sorted(top_segments, key=lambda x: x["start_time"])

        # Sample frames from each selected segment
        all_frames = []
        for seg in top_segments:
            # Get frame indices within this segment (clamp to video bounds)
            seg_start_frame = max(0, int(seg["start_time"] * fps))
            seg_end_frame = min(total_frames, int(seg["end_time"] * fps))
            seg_total_frames = seg_end_frame - seg_start_frame

            if seg_total_frames <= 0:
                continue

            if seg_total_frames <= frames_per_segment:
                # Take all frames
                indices = list(range(seg_start_frame, seg_end_frame))
            else:
                # Uniform sample within segment
                indices = np.linspace(
                    seg_start_frame, seg_end_frame - 1, frames_per_segment, dtype=int
                ).tolist()

            if indices:
                frames = self._get_frames(vr, indices)
                all_frames.extend(self._convert_frames(frames))

        return all_frames
