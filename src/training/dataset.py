"""Compatibility dataset structures for training regression tests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ScoreRegressionSample:
    """Dataclass representing one scored media sample."""

    media_path: Path
    score: float
    dimension: str = "general"
    is_video: bool = False
    media_id: int | None = None


class LRUCache:
    """Minimal least-recently-used cache used by regression tests."""

    def __init__(self, max_size: int = 128) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self._cache: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str):
        if key not in self._cache:
            return None

        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def put(self, key: str, value: object) -> None:
        if key in self._cache:
            self._cache.pop(key)

        self._cache[key] = value

        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
