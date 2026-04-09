"""Temporal attention module for video frame aggregation.

This module implements a lightweight temporal attention mechanism to aggregate
features from multiple video frames while supporting single images seamlessly.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logger import get_logger

logger = get_logger(__name__)


class TemporalAttention(nn.Module):
    """Lightweight temporal attention for aggregating frame features.

    This module applies multi-head attention across the temporal dimension
    to capture temporal relationships in video features. For single images,
    it returns the input features unchanged (identity operation).

    Architecture:
        - Multi-head self-attention across temporal dimension
        - Learnable temporal positional encoding
        - Optional layer normalization

    Parameters:
        - feature_dim: 768 (from CLIP ViT-L/14)
        - num_heads: 8
        - temporal_dim: 256 (projection dimension)
        - Total trainable params: ~800K

    Example:
        >>> attention = TemporalAttention(feature_dim=768)
        >>>
        >>> # Single image (batch_size, 1, 768)
        >>> img_feat = torch.randn(4, 1, 768)
        >>> out = attention(img_feat)  # (4, 1, 768)
        >>>
        >>> # Video frames (batch_size, num_frames, 768)
        >>> video_feat = torch.randn(4, 12, 768)
        >>> out = attention(video_feat)  # (4, 12, 768)
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        temporal_dim: int = 256,
        max_frames: int = 32,
        dropout: float = 0.1,
        use_norm: bool = True,
    ):
        """Initialize temporal attention module.

        Args:
            feature_dim: Input feature dimension (768 for CLIP ViT-L/14).
            num_heads: Number of attention heads.
            temporal_dim: Dimension for temporal projection.
            max_frames: Maximum number of frames for positional encoding.
            dropout: Dropout probability.
            use_norm: Whether to use layer normalization.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.temporal_dim = temporal_dim
        self.max_frames = max_frames

        assert temporal_dim % num_heads == 0, (
            "temporal_dim must be divisible by num_heads"
        )
        self.head_dim = temporal_dim // num_heads

        self.query_proj = nn.Linear(feature_dim, temporal_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, temporal_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, temporal_dim, bias=False)
        self.output_proj = nn.Linear(temporal_dim, feature_dim, bias=False)

        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, feature_dim) * 0.02)

        self.norm = nn.LayerNorm(feature_dim) if use_norm else None
        self.dropout = nn.Dropout(dropout)

        self._log_parameters()

    def _log_parameters(self) -> None:
        """Log total trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"TemporalAttention initialized with {total_params:,} trainable parameters"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply temporal attention to frame features.

        Args:
            x: Input tensor of shape (batch_size, num_frames, feature_dim).
               For single images, num_frames=1.
            mask: Optional attention mask of shape (batch_size, num_frames).
                  1 for valid frames, 0 for padding.

        Returns:
            Output tensor of shape (batch_size, num_frames, feature_dim).
            For single images (num_frames=1), returns input unchanged.
        """
        batch_size, num_frames, _ = x.shape

        if num_frames == 1:
            return x

        pos_encoding = self.temporal_pos[:, :num_frames, :]
        x = x + pos_encoding

        if self.norm is not None:
            x = self.norm(x)

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = Q.view(batch_size, num_frames, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.view(batch_size, num_frames, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        V = V.view(batch_size, num_frames, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask.expand_as(scores) == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_frames, self.temporal_dim)

        output = self.output_proj(attn_output)

        output = output + x

        return output

    def aggregate_temporal(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        method: str = "mean",
    ) -> torch.Tensor:
        """Aggregate temporal features to a single vector.

        Args:
            x: Input tensor of shape (batch_size, num_frames, feature_dim).
            mask: Optional mask of shape (batch_size, num_frames).
            method: Aggregation method - "mean", "max", "attention".

        Returns:
            Aggregated tensor of shape (batch_size, feature_dim).
        """
        if mask is None:
            if method == "mean":
                return x.mean(dim=1)
            elif method == "max":
                return x.max(dim=1)[0]
            else:
                return x.mean(dim=1)

        mask_expanded = mask.unsqueeze(-1).float()

        if method == "mean":
            return (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(
                min=1e-9
            )
        elif method == "max":
            x_masked = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            return x_masked.max(dim=1)[0]
        else:
            return (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(
                min=1e-9
            )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
