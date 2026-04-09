"""Unit tests for regression training components."""

import pytest
import torch

from src.training.config import (
    TrainingConfig,
    LoRAConfig,
    ModelConfig,
    Stage1Config,
    Stage2Config,
    LossConfig,
    HardwareConfig,
    PathsConfig,
)
from src.training.loss import CombinedRegressionRankingLoss
from src.training.model import MLPRegressionHead
from src.training.dataset import ScoreRegressionSample, LRUCache


class TestCombinedRegressionRankingLoss:
    """Tests for CombinedRegressionRankingLoss class."""

    def test_perfect_predictions(self):
        """When predictions == targets, loss should be 0."""
        loss_fn = CombinedRegressionRankingLoss()
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])

        loss = loss_fn(predictions, targets)

        assert loss.item() == 0.0

    def test_ranking_correct_order(self):
        """When ordering is correct, ranking loss should be 0."""
        loss_fn = CombinedRegressionRankingLoss(lambda_rank=0.3, margin=0.5)
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])

        _, regression_loss, ranking_loss = loss_fn(
            predictions, targets, return_components=True
        )

        assert ranking_loss.item() == 0.0

    def test_ranking_wrong_order(self):
        """When ordering is wrong, ranking loss should be > 0."""
        loss_fn = CombinedRegressionRankingLoss(lambda_rank=0.3, margin=0.5)
        # Predictions in wrong order compared to targets
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # Targets in reversed order
        targets = torch.tensor([4.0, 3.0, 2.0, 1.0])

        _, regression_loss, ranking_loss = loss_fn(
            predictions, targets, return_components=True
        )

        assert ranking_loss.item() > 0.0

    def test_loss_components(self):
        """Verify return_components returns 3 values."""
        loss_fn = CombinedRegressionRankingLoss()
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.5, 2.5, 3.5, 4.5])

        result = loss_fn(predictions, targets, return_components=True)

        assert isinstance(result, tuple)
        assert len(result) == 3
        total_loss, reg_loss, rank_loss = result
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(reg_loss, torch.Tensor)
        assert isinstance(rank_loss, torch.Tensor)


class TestMLPRegressionHead:
    """Tests for MLPRegressionHead class."""

    def test_forward_shape(self):
        """Input (batch, 3584) -> Output (batch, 1)."""
        model = MLPRegressionHead(hidden_dim=3584, layer_sizes=[512, 128])
        batch_size = 4
        x = torch.randn(batch_size, 3584)

        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_output_range(self):
        """After sigmoid, output should be in [0, 1]."""
        model = MLPRegressionHead(hidden_dim=3584, layer_sizes=[512, 128])
        x = torch.randn(4, 3584)

        logits = model(x)
        output = torch.sigmoid(logits)

        assert output.min() >= 0.0
        assert output.max() <= 1.0


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Verify default config values are correct."""
        config = TrainingConfig()

        assert config.model.hidden_dim == 3584
        assert config.stage1.epochs == 2
        assert config.stage1.lr == 1e-3
        assert config.stage2.epochs == 3
        assert config.stage2.lr == 5e-5
        assert config.loss.lambda_ == 0.3
        assert config.hardware.batch_size == 2
        assert config.hardware.fp16 is True

    def test_nested_config(self):
        """Verify nested configs (model, stage1, stage2, etc.)."""
        config = TrainingConfig()

        # Model config
        assert isinstance(config.model, ModelConfig)
        assert config.model.mlp_layers == [512, 128]

        # Stage configs
        assert isinstance(config.stage1, Stage1Config)
        assert isinstance(config.stage2, Stage2Config)

        # LoRA config
        assert isinstance(config.lora, LoRAConfig)
        assert config.lora.r == 8
        assert config.lora.lora_alpha == 16
        assert config.lora.target_modules == ["q_proj", "v_proj"]

        # Loss config
        assert isinstance(config.loss, LossConfig)
        assert config.loss.smooth_l1_beta == 1.0

        # Hardware config
        assert isinstance(config.hardware, HardwareConfig)
        assert config.hardware.gradient_accumulation == 8

        # Paths config
        assert isinstance(config.paths, PathsConfig)


class TestScoreRegressionDataset:
    """Tests for ScoreRegressionSample and LRUCache."""

    def test_sample_dataclass(self):
        """Test ScoreRegressionSample creation."""
        from pathlib import Path

        sample = ScoreRegressionSample(
            media_path=Path("/path/to/video.mp4"),
            score=8.5,
            dimension="aesthetic",
            is_video=True,
            media_id=123,
        )

        assert sample.media_path == Path("/path/to/video.mp4")
        assert sample.score == 8.5
        assert sample.dimension == "aesthetic"
        assert sample.is_video is True
        assert sample.media_id == 123

    def test_lru_cache(self):
        """Test LRUCache put/get/eviction."""
        cache = LRUCache(max_size=3)

        # Test put and get
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        # Test update existing key
        cache.put("key1", "value1_updated")
        assert cache.get("key1") == "value1_updated"

        # Test eviction when full
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key2 (first in)

        assert cache.get("key1") == "value1_updated"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
        assert cache.get("key2") is None  # Evicted

        # Test get non-existent key
        assert cache.get("nonexistent") is None
