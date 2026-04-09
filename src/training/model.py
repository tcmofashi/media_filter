"""Compatibility model adapters for regression testing and public API."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class MLPRegressionHead(nn.Module):
    """Simple MLP head producing one score value per sample.

    This module is intentionally compact and independent from experimental code.
    It is used by tests to validate baseline model shape and range behavior.
    """

    def __init__(
        self,
        hidden_dim: int = 3584,
        layer_sizes: Sequence[int] | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        if layer_sizes is None:
            layer_sizes = [512, 128]

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError(f"layer_sizes must be positive, got {layer_sizes}")

        self.hidden_dim = hidden_dim
        self.layer_sizes = list(layer_sizes)
        self.dropout = dropout

        act = nn.ReLU if activation.lower() == "relu" else nn.GELU

        layers = []
        in_features = hidden_dim
        for out_features in self.layer_sizes:
            layers.append(nn.Linear(in_features, out_features, bias=use_bias))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1, bias=use_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Input must be 2D (batch, feature), got shape {tuple(x.shape)}")
        return self.net(x)
