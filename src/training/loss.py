"""Compatibility losses for public regression baseline tests."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedRegressionRankingLoss(nn.Module):
    """Combines regression loss with ranking pairwise hinge loss.

    The ranking term is zero when predictions preserve target order with sufficient
    margin, and increases when pairwise ordering is violated.
    """

    def __init__(self, lambda_rank: float = 0.4, margin: float = 0.5) -> None:
        super().__init__()
        self.lambda_rank = lambda_rank
        self.lambda_reg = 1.0 - lambda_rank
        self.margin = margin

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        regression_loss = F.mse_loss(predictions, targets, reduction="mean")
        ranking_loss = self._ranking_loss(predictions, targets)
        total_loss = self.lambda_reg * regression_loss + self.lambda_rank * ranking_loss

        if return_components:
            return total_loss, regression_loss, ranking_loss
        return total_loss

    def _ranking_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.numel() < 2:
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)

        # Compare all ordered pairs and penalize inverse predictions.
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)

        # Keep non-zero target differences only.
        valid_mask = target_diff != 0
        if not torch.any(valid_mask):
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)

        target_sign = torch.where(target_diff > 0, 1.0, -1.0)
        violated_margin = self.margin - target_sign * pred_diff
        active_margin = F.relu(violated_margin[valid_mask])
        return active_margin.mean()
