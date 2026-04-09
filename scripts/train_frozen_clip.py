#!/usr/bin/env python
"""
Training script for Frozen CLIP + Temporal Attention + Score Head.

Architecture:
    - FrozenCLIPEncoder: Configurable CLIP vision encoder
    - TemporalAttention: Multi-head attention for frame aggregation
    - ScoreHead: MLP for score prediction in the configured 0-9 range

Usage:
    # Single GPU
    python scripts/train_frozen_clip.py --config configs/train_frozen_clip.yaml

    # Multi-GPU with DeepSpeed
    torchrun --nproc_per_node=4 scripts/train_frozen_clip.py \
        --deepspeed_config configs/ds_frozen_clip.json \
        --batch_size 16

    # DeepSpeed with offload
    torchrun --nproc_per_node=2 scripts/train_frozen_clip.py \
        --deepspeed_config configs/ds_frozen_clip_v100_fp16.json \
        --batch_size 8
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger
from src.training.frozen_clip_encoder import FrozenCLIPEncoder
from src.training.temporal_attention import TemporalAttention
from src.training.score_head import ScoreHead
from src.training.frozen_clip_dataset import (
    FrozenClipDataset,
    collate_fn_padded,
    create_train_val_split,
)
from src.training.score_loss import create_score_loss

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STABLE_BEST_DIR = PROJECT_ROOT / "checkpoints" / "frozen_clip"
STABLE_BEST_PATH = STABLE_BEST_DIR / "checkpoint_best.pt"
STABLE_BEST_META_PATH = STABLE_BEST_DIR / "checkpoint_best.meta.json"


# ============================================================================
# Model Definition
# ============================================================================


class FrozenClipScoreModel(nn.Module):
    """Complete model: CLIP + Temporal Attention + Score Head.

    Architecture:
        1. FrozenCLIPEncoder: Extract visual features from frames
        2. TemporalAttention: Aggregate frame features with attention
        3. ScoreHead: MLP to predict score in the configured score range

    Total trainable parameters: temporal/score heads plus optional CLIP tail.

    Args:
        feature_dim: CLIP feature dimension (default: 768)
        num_heads: Number of attention heads (default: 8)
        temporal_dim: Temporal projection dimension (default: 256)
        hidden_dims: Hidden dimensions for ScoreHead (default: (256, 64))
        dropout: Dropout rate (default: 0.1)
        max_frames: Maximum frames for positional encoding (default: 32)
    """

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        num_heads: int = 8,
        temporal_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (256, 64),
        dropout: float = 0.1,
        max_frames: int = 32,
        score_range: Tuple[float, float] = (0.0, 9.0),
        clip_batch_size: int = 64,
        clip_model_name: str = FrozenCLIPEncoder.MODEL_NAME,
        unfreeze_last_n_vision_layers: int = 0,
    ):
        super().__init__()

        # CLIP encoder
        self.clip_encoder = FrozenCLIPEncoder(
            device="cpu",  # Will be moved later
            model_name=clip_model_name,
            unfreeze_last_n_vision_layers=unfreeze_last_n_vision_layers,
        )
        clip_feature_dim = self.clip_encoder.get_feature_dim()
        if feature_dim is not None and feature_dim != clip_feature_dim:
            raise ValueError(
                f"feature_dim={feature_dim} does not match CLIP output dim {clip_feature_dim}"
            )
        feature_dim = clip_feature_dim

        # Trainable components
        self.temporal_attention = TemporalAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            temporal_dim=temporal_dim,
            max_frames=max_frames,
            dropout=dropout,
        )

        self.score_head = ScoreHead(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            score_range=score_range,
        )

        # Store config
        self.clip_model_name = clip_model_name
        self.unfreeze_last_n_vision_layers = unfreeze_last_n_vision_layers
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.temporal_dim = temporal_dim
        self.hidden_dims = hidden_dims
        self.clip_batch_size = clip_batch_size

        self._log_parameters()

    def _extract_clip_features(self, frames_list: List[List]) -> Tuple[List[torch.Tensor], int]:
        """Extract CLIP features in larger cross-sample chunks for better GPU utilization."""
        frame_counts = [len(frames) for frames in frames_list]
        flat_frames = [frame for frames in frames_list for frame in frames]

        if not flat_frames:
            return [], 0

        chunk_size = len(flat_frames) if self.clip_batch_size <= 0 else self.clip_batch_size
        flat_features = []

        for start in range(0, len(flat_frames), chunk_size):
            frame_chunk = flat_frames[start : start + chunk_size]
            flat_features.append(self.clip_encoder.extract_features(frame_chunk))

        concatenated = torch.cat(flat_features, dim=0)

        all_features = []
        offset = 0
        for frame_count in frame_counts:
            all_features.append(concatenated[offset : offset + frame_count])
            offset += frame_count

        return all_features, max(frame_counts, default=0)

    def _log_parameters(self):
        """Log parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        logger.info(
            f"Model parameters - Total: {total:,}, Trainable: {trainable:,}, Frozen: {frozen:,}"
        )

    def forward(
        self,
        frames_list: List[List],
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            frames_list: List of frame lists, each sample has variable frames.
                        Each frame is a PIL Image or numpy array.
            masks: Optional attention mask of shape (batch_size, num_frames).

        Returns:
            Predicted scores of shape (batch_size, 1) in the configured range.
        """
        batch_size = len(frames_list)

        # Extract CLIP features across the whole batch instead of one sample at a time.
        all_features, max_frames = self._extract_clip_features(frames_list)

        device = next(self.temporal_attention.parameters()).device
        dtype = all_features[0].dtype if all_features else torch.float32

        # Pad and stack features
        padded_features = torch.zeros(
            batch_size,
            max_frames,
            self.feature_dim,
            device=device,
            dtype=dtype,
        )
        if masks is None or masks.shape[1] != max_frames:
            aligned_masks = torch.zeros(batch_size, max_frames, device=device)
        else:
            aligned_masks = masks.to(device)

        for i, features in enumerate(all_features):
            n = len(features)
            padded_features[i, :n] = features
            aligned_masks[i, :n] = 1.0

        # Apply temporal attention
        temporal_out = self.temporal_attention(
            padded_features, aligned_masks
        )  # (B, T, D)

        # Aggregate temporal dimension
        # Use mask-aware mean pooling
        mask_expanded = aligned_masks.unsqueeze(-1)
        aggregated = (temporal_out * mask_expanded).sum(dim=1) / mask_expanded.sum(
            dim=1
        ).clamp(min=1e-9)
        # (B, D)

        # Predict score
        scores = self.score_head(aggregated)  # (B, 1)

        return scores

    def predict_scores(
        self,
        frames_list: List[List],
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference method without gradient computation.

        Args:
            frames_list: List of frame lists.
            masks: Optional attention mask.

        Returns:
            Predicted scores of shape (batch_size, 1).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(frames_list, masks)


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    labels_path: str = "labels.json"
    val_ratio: float = 0.2
    num_frames: int = 8
    score_range: Optional[Tuple[float, float]] = None
    long_video_strategy: str = "expand"
    min_long_frames: Optional[int] = None
    max_long_frames: Optional[int] = 32

    # Training
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    seed: int = 42
    precision: str = "fp32"
    clip_batch_size: int = 64

    # Model
    clip_model_name: str = FrozenCLIPEncoder.MODEL_NAME
    unfreeze_last_n_vision_layers: int = 0
    feature_dim: Optional[int] = None
    num_heads: int = 8
    temporal_dim: int = 256
    hidden_dims: Tuple[int, ...] = (256, 64)
    dropout: float = 0.1
    max_frames: int = 32

    # Loss
    loss_type: str = "mse"  # "mse" or "smooth_l1"
    lambda_l1: float = 0.0

    # Output
    output_dir: str = "checkpoints/frozen_clip"
    save_every: int = 1  # Save every N epochs

    # DeepSpeed
    deepspeed_config: Optional[str] = None
    local_rank: int = -1

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ============================================================================
# Trainer
# ============================================================================


class FrozenClipTrainer:
    """Trainer for FrozenClipScoreModel."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main_process = self.local_rank == 0

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None
        self.val_sampler = None
        self.best_val_loss = float("inf")

        # DeepSpeed
        self.deepspeed_engine = None
        self.use_deepspeed = config.deepspeed_config is not None
        self.use_amp = (
            self.device.type == "cuda"
            and not self.use_deepspeed
            and config.precision != "fp32"
        )
        self.amp_dtype = (
            torch.float16 if config.precision == "fp16" else torch.bfloat16
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        if self.is_main_process:
            logger.info(
                f"Training config: {json.dumps(config.to_dict(), indent=2, default=str)}"
            )
            logger.info(f"World size: {self.world_size}, Local rank: {self.local_rank}")
            logger.info(
                "Precision setup: precision=%s, autocast=%s, clip_batch_size=%d",
                self.config.precision,
                self.use_amp,
                self.config.clip_batch_size,
            )

    def setup(self):
        """Setup all components."""
        if self.is_main_process:
            logger.info("Setting up trainer...")

        self._set_seed()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_distributed()

        if self.is_main_process:
            logger.info("Trainer setup complete.")

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np

        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True

    def _autocast_context(self):
        """Return autocast context when mixed precision is enabled."""
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _setup_model(self):
        """Initialize the model."""
        self.model = FrozenClipScoreModel(
            clip_model_name=self.config.clip_model_name,
            unfreeze_last_n_vision_layers=self.config.unfreeze_last_n_vision_layers,
            feature_dim=self.config.feature_dim,
            num_heads=self.config.num_heads,
            temporal_dim=self.config.temporal_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            max_frames=self.config.max_frames,
            score_range=self.config.score_range,
            clip_batch_size=self.config.clip_batch_size,
        )
        self.config.feature_dim = self.model.feature_dim

        # Move the entire model so trainable heads and frozen CLIP stay aligned.
        self.model = self.model.to(self.device)
        self.model.clip_encoder.device = self.device

        if self.is_main_process:
            logger.info("Model initialized")

    def _setup_loss(self):
        """Initialize loss function."""
        self.loss_fn = create_score_loss(
            loss_type=self.config.loss_type,
            lambda_l1=self.config.lambda_l1,
            score_range=self.config.score_range,
        )

        if self.is_main_process:
            logger.info(
                f"Loss function: {self.config.loss_type}, lambda_l1={self.config.lambda_l1}"
            )

    def _setup_data(self):
        """Load and prepare datasets."""
        if self.is_main_process:
            logger.info(f"Loading data from {self.config.labels_path}")

        # Create dataset
        full_dataset = FrozenClipDataset(
            labels_path=self.config.labels_path,
            num_frames=self.config.num_frames,
            validate_files=True,
            score_range=self.config.score_range,
            normalize_scores=False,
            long_video_strategy=self.config.long_video_strategy,
            min_long_frames=self.config.min_long_frames,
            max_long_frames=self.config.max_long_frames,
        )

        if self.config.score_range is None:
            self.config.score_range = full_dataset.score_range
            if self.is_main_process:
                logger.info(f"Inferred score range from data: {self.config.score_range}")

        # Split into train/val
        train_indices, val_indices = create_train_val_split(
            full_dataset,
            val_ratio=self.config.val_ratio,
            seed=self.config.seed,
        )

        # Create subset datasets
        from torch.utils.data import Subset

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # Create samplers for distributed training
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
            )
            self.val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False,
            )
        else:
            self.train_sampler = None
            self.val_sampler = None

        # Create dataloaders
        def _collate_fn(batch):
            return collate_fn_padded(batch, max_frames=self.config.max_frames)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(
                self.config.persistent_workers if self.config.num_workers > 0 else False
            ),
            prefetch_factor=(
                self.config.prefetch_factor if self.config.num_workers > 0 else None
            ),
            collate_fn=_collate_fn,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(
                self.config.persistent_workers if self.config.num_workers > 0 else False
            ),
            prefetch_factor=(
                self.config.prefetch_factor if self.config.num_workers > 0 else None
            ),
            collate_fn=_collate_fn,
            drop_last=False,
        )

        if self.is_main_process:
            logger.info(
                f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, "
                f"batch_size={self.config.batch_size}, "
                f"effective_batch_size={self.config.batch_size * self.world_size * self.config.gradient_accumulation_steps}"
            )

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler with warmup
        steps_per_epoch = math.ceil(
            len(self.train_loader) / max(1, self.config.gradient_accumulation_steps)
        )
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if self.is_main_process:
            logger.info(
                f"Optimizer: AdamW, lr={self.config.learning_rate}, "
                f"warmup_steps={warmup_steps}, total_steps={total_steps}, "
                f"grad_accum={self.config.gradient_accumulation_steps}"
            )

    def _setup_distributed(self):
        """Setup distributed training (DDP or DeepSpeed)."""
        if self.use_deepspeed:
            self._setup_deepspeed()
        elif self.world_size > 1:
            self._setup_ddp()

    def _setup_deepspeed(self):
        """Initialize DeepSpeed engine."""
        import deepspeed

        # Load DeepSpeed config
        with open(self.config.deepspeed_config) as f:
            ds_config = json.load(f)

        # Set auto values
        ds_config["train_batch_size"] = (
            self.world_size
            * self.config.batch_size
            * self.config.gradient_accumulation_steps
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.config.batch_size
        ds_config["gradient_accumulation_steps"] = (
            self.config.gradient_accumulation_steps
        )

        if self.is_main_process:
            logger.info(f"DeepSpeed config: {json.dumps(ds_config, indent=2)}")

        # Initialize DeepSpeed
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            config=ds_config,
        )

        self.deepspeed_engine = self.model
        if self.is_main_process:
            logger.info("DeepSpeed engine initialized")

    def _setup_ddp(self):
        """Initialize DistributedDataParallel."""
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            # Some ranks can receive image-only batches where TemporalAttention
            # short-circuits on a single frame, so attention params are unused
            # for that step. DDP must tolerate those sparse branches.
            find_unused_parameters=True,
        )

        if self.is_main_process:
            logger.info("DDP initialized")

    def train(self):
        """Run training loop."""
        if self.is_main_process:
            logger.info(f"Starting training for {self.config.epochs} epochs")

        global_step = 0
        total_steps = (
            math.ceil(len(self.train_loader) / max(1, self.config.gradient_accumulation_steps))
            * self.config.epochs
        )

        for epoch in range(self.config.epochs):
            # Set epoch for distributed sampler
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # Training epoch
            train_loss, global_step = self._train_epoch(epoch, global_step, total_steps)

            # Validation
            val_loss = self._validate_epoch(epoch)

            if self.is_main_process:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            # Save checkpoint
            if self.is_main_process and (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch, val_loss)

        if self.is_main_process:
            self._sync_stable_best_checkpoint()
            logger.info("Training complete!")

    def _refresh_stable_best_checkpoint(
        self,
        run_best_path: Path,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Refresh the stable best-checkpoint alias and metadata."""
        run_best_path = run_best_path.resolve()
        STABLE_BEST_DIR.mkdir(parents=True, exist_ok=True)

        current_target = None
        if STABLE_BEST_PATH.exists() or STABLE_BEST_PATH.is_symlink():
            try:
                current_target = STABLE_BEST_PATH.resolve()
            except OSError:
                current_target = None

        if current_target != run_best_path:
            if STABLE_BEST_PATH.exists() or STABLE_BEST_PATH.is_symlink():
                STABLE_BEST_PATH.unlink()
            if STABLE_BEST_PATH != run_best_path:
                STABLE_BEST_PATH.symlink_to(run_best_path)

        if checkpoint is None:
            checkpoint = torch.load(run_best_path, map_location="cpu")

        metadata = {
            "stable_path": str(STABLE_BEST_PATH),
            "source_path": str(run_best_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "best_val_loss": checkpoint.get(
                "best_val_loss", checkpoint.get("val_loss")
            ),
            "output_dir": self.config.output_dir,
        }
        STABLE_BEST_META_PATH.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )
        logger.info(
            "Stable best checkpoint refreshed: %s -> %s",
            STABLE_BEST_PATH,
            run_best_path,
        )

    def _sync_stable_best_checkpoint(self) -> None:
        """Sync the stable best-checkpoint alias to this run's best checkpoint."""
        run_best_path = Path(self.config.output_dir) / "checkpoint_best.pt"
        if not run_best_path.exists():
            logger.warning(
                "Run best checkpoint not found, skip stable alias refresh: %s",
                run_best_path,
            )
            return
        self._refresh_stable_best_checkpoint(run_best_path)

    def _train_epoch(
        self, epoch: int, global_step: int, total_steps: int
    ) -> Tuple[float, int]:
        """Run one training epoch."""
        if self.use_deepspeed:
            self.deepspeed_engine.train()
        else:
            self.model.train()

        total_loss = 0.0
        num_batches = 0
        accum_steps = max(1, self.config.gradient_accumulation_steps)

        if not self.use_deepspeed:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            # Get batch data
            frames_list = batch["frames"]
            scores = batch["scores"].to(self.device)
            masks = batch.get("masks")
            if masks is not None:
                masks = masks.to(self.device)

            with self._autocast_context():
                # Forward pass
                if self.use_deepspeed:
                    predictions = self.deepspeed_engine(frames_list, masks)
                    predictions = predictions.squeeze(-1)  # (B, 1) -> (B,)
                else:
                    predictions = self.model(frames_list, masks)
                    predictions = predictions.squeeze(-1)

                # Get trainable parameters for L1 regularization
                if self.use_deepspeed:
                    model_params = [
                        p for p in self.deepspeed_engine.parameters() if p.requires_grad
                    ]
                else:
                    model_params = [p for p in self.model.parameters() if p.requires_grad]

                # Compute loss
                loss = self.loss_fn(predictions, scores, model_params)

            loss_value = loss.item()

            # Backward pass
            if self.use_deepspeed:
                self.deepspeed_engine.backward(loss)
                self.deepspeed_engine.step()
                if self.deepspeed_engine.is_gradient_accumulation_boundary():
                    global_step += 1
            else:
                scaled_loss = loss / accum_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (
                    (batch_idx + 1) % accum_steps == 0
                    or (batch_idx + 1) == len(self.train_loader)
                )
                if should_step:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)

                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.max_grad_norm,
                        )

                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            total_loss += loss_value
            num_batches += 1

            # Logging
            if self.is_main_process and (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                progress = (global_step / total_steps * 100) if total_steps else 0.0
                lr = (
                    self.scheduler.get_last_lr()[0]
                    if not self.use_deepspeed
                    else self.config.learning_rate
                )
                logger.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, "
                    f"LR: {lr:.2e}, Progress: {progress:.1f}%"
                )

        return total_loss / num_batches, global_step

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        """Run validation."""
        if self.use_deepspeed:
            self.deepspeed_engine.eval()
        else:
            self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        for batch in self.val_loader:
            frames_list = batch["frames"]
            scores = batch["scores"].to(self.device)
            masks = batch.get("masks")
            if masks is not None:
                masks = masks.to(self.device)

            with self._autocast_context():
                # Forward pass
                if self.use_deepspeed:
                    predictions = self.deepspeed_engine(frames_list, masks)
                    predictions = predictions.squeeze(-1)
                else:
                    predictions = self.model(frames_list, masks)
                    predictions = predictions.squeeze(-1)

                # Compute loss (without regularization)
                loss = self.loss_fn(predictions, scores, None)
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

        avg_loss = total_loss / num_batches

        # Compute additional metrics
        if self.is_main_process and num_batches > 0:
            import numpy as np
            from scipy.stats import pearsonr

            predictions_arr = np.array(all_predictions)
            targets_arr = np.array(all_targets)

            mae = np.mean(np.abs(predictions_arr - targets_arr))
            mse = np.mean((predictions_arr - targets_arr) ** 2)
            rmse = np.sqrt(mse)

            try:
                pearson_corr, _ = pearsonr(predictions_arr, targets_arr)
            except Exception:
                pearson_corr = 0.0

            logger.info(
                f"Validation - Loss: {avg_loss:.4f}, MAE: {mae:.4f}, "
                f"RMSE: {rmse:.4f}, Pearson: {pearson_corr:.4f}, "
                f"PredRange: [{predictions_arr.min():.2f}, {predictions_arr.max():.2f}]"
            )

        return avg_loss

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        # Get underlying model for saving
        if self.use_deepspeed:
            model_to_save = self.deepspeed_engine.module
        elif isinstance(self.model, DDP):
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
            "temporal_attention": model_to_save.temporal_attention.state_dict(),
            "score_head": model_to_save.score_head.state_dict(),
        }
        clip_trainable_state = model_to_save.clip_encoder.get_trainable_state_dict()
        if clip_trainable_state:
            checkpoint["clip_encoder_trainable_state"] = clip_trainable_state

        # Save latest
        checkpoint_path = output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save epoch checkpoint
        epoch_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, epoch_path)

        # Save best
        if is_best:
            best_path = output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            self._refresh_stable_best_checkpoint(best_path, checkpoint)
            logger.info(f"New best model saved! Val loss: {val_loss:.4f}")

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get underlying model
        if self.use_deepspeed:
            model_to_load = self.deepspeed_engine.module
        elif isinstance(self.model, DDP):
            model_to_load = self.model.module
        else:
            model_to_load = self.model

        # Load state dicts
        clip_state = checkpoint.get("clip_encoder_trainable_state")
        if clip_state:
            incompatible = model_to_load.clip_encoder.load_trainable_state_dict(clip_state)
            logger.info(
                "Loaded clip_encoder trainable state (%d tensors, missing=%d, unexpected=%d)",
                len(clip_state),
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        model_to_load.temporal_attention.load_state_dict(
            checkpoint["temporal_attention"]
        )
        model_to_load.score_head.load_state_dict(checkpoint["score_head"])

        self.best_val_loss = checkpoint.get(
            "best_val_loss", checkpoint.get("val_loss", self.best_val_loss)
        )

        logger.info(
            "Checkpoint loaded from %s (epoch=%s, val_loss=%s, best_val_loss=%s)",
            checkpoint_path,
            checkpoint.get("epoch"),
            checkpoint.get("val_loss"),
            self.best_val_loss,
        )

        return checkpoint.get("epoch", 0)


# ============================================================================
# Main
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Frozen CLIP + Temporal Attention + Score Head"
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Data
    parser.add_argument(
        "--labels_path",
        type=str,
        default="labels.json",
        help="Path to labels JSONL file",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample from videos",
    )
    parser.add_argument(
        "--long_video_strategy",
        type=str,
        default="expand",
        choices=["compress", "expand"],
        help="Sampling strategy for videos longer than 5 minutes",
    )
    parser.add_argument(
        "--min_long_frames",
        type=int,
        default=None,
        help="Minimum frames to keep for long videos; default follows num_frames",
    )
    parser.add_argument(
        "--max_long_frames",
        type=int,
        default=32,
        help="Maximum frames to keep for long videos",
    )
    parser.add_argument(
        "--score_min",
        type=float,
        default=None,
        help="Optional minimum score. If omitted, infer from labels.",
    )
    parser.add_argument(
        "--score_max",
        type=float,
        default=None,
        help="Optional maximum score. If omitted, infer from labels.",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precision mode for non-DeepSpeed training",
    )
    parser.add_argument(
        "--clip_batch_size",
        type=int,
        default=64,
        help="Frames per CLIP forward chunk; <=0 means process the whole batch at once",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default=FrozenCLIPEncoder.MODEL_NAME,
        help="Hugging Face CLIP model name",
    )
    parser.add_argument(
        "--unfreeze_last_n_vision_layers",
        type=int,
        default=0,
        help="Number of final CLIP vision transformer layers to unfreeze",
    )

    # Model
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--temporal_dim", type=int, default=256, help="Temporal projection dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=32,
        help="Maximum frames for positional encoding",
    )

    # Loss
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "smooth_l1"],
        help="Loss type",
    )
    parser.add_argument(
        "--lambda_l1", type=float, default=0.0, help="L1 regularization weight"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/frozen_clip",
        help="Output directory",
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="Save checkpoint every N epochs"
    )

    # DeepSpeed
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )

    # Data loading
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor when num_workers > 0",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        default=True,
        help="Keep DataLoader workers alive between epochs",
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_false",
        dest="persistent_workers",
        help="Disable persistent DataLoader workers",
    )

    # Resume
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize distributed training
    if args.deepspeed_config:
        import deepspeed

        deepspeed.init_distributed()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")

    if (args.score_min is None) != (args.score_max is None):
        raise ValueError("score_min and score_max must be provided together")

    score_range = None
    if args.score_min is not None and args.score_max is not None:
        score_range = (args.score_min, args.score_max)

    # Create config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(
            labels_path=args.labels_path,
            val_ratio=args.val_ratio,
            num_frames=args.num_frames,
            long_video_strategy=args.long_video_strategy,
            min_long_frames=args.min_long_frames,
            max_long_frames=args.max_long_frames,
            score_range=score_range,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            precision=args.precision,
            clip_batch_size=args.clip_batch_size,
            clip_model_name=args.clip_model_name,
            unfreeze_last_n_vision_layers=args.unfreeze_last_n_vision_layers,
            num_heads=args.num_heads,
            temporal_dim=args.temporal_dim,
            dropout=args.dropout,
            max_frames=args.max_frames,
            loss_type=args.loss_type,
            lambda_l1=args.lambda_l1,
            output_dir=args.output_dir,
            save_every=args.save_every,
            deepspeed_config=args.deepspeed_config,
            local_rank=args.local_rank,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

    # Create trainer
    trainer = FrozenClipTrainer(config)
    trainer.setup()

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
