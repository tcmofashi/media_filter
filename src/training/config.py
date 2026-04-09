"""Public-compatible training configuration dataclasses.

These lightweight configurations are used by training-related unit tests to
provide a stable API surface after the cleanup of experimental training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class ModelConfig:
    """Model architecture settings."""

    hidden_dim: int = 3584
    mlp_layers: list[int] = field(default_factory=lambda: [512, 128])
    dropout: float = 0.1
    use_bias: bool = True
    score_range_min: float = 0.0
    score_range_max: float = 1.0


@dataclass(slots=True)
class Stage1Config:
    """Stage-1 training options."""

    epochs: int = 2
    lr: float = 1e-3
    batch_size: int = 2
    warmup_steps: int = 0


@dataclass(slots=True)
class Stage2Config:
    """Stage-2 training options."""

    epochs: int = 3
    lr: float = 5e-5
    freeze_encoder: bool = True


@dataclass(slots=True)
class LoRAConfig:
    """LoRA hyper-parameters for compatibility."""

    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(slots=True)
class LossConfig:
    """Loss config."""

    lambda_: float = 0.3
    smooth_l1_beta: float = 1.0
    regression_weight: float = 1.0
    ranking_weight: float = 0.0


@dataclass(slots=True)
class HardwareConfig:
    """Hardware/runtime settings."""

    batch_size: int = 2
    fp16: bool = True
    gradient_accumulation: int = 8
    num_workers: int = 4
    seed: int = 42


@dataclass(slots=True)
class PathsConfig:
    """Filesystem paths for optional training artifacts."""

    dataset_path: Optional[Path] = None
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    ckpt_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None


@dataclass(slots=True)
class TrainingConfig:
    """Aggregated configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
