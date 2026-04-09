"""Distribution-aware loss functions to prevent mean collapse."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionAwareLoss(nn.Module):
    """Combined loss to prevent mean collapse and encourage diverse predictions.

    Components:
    1. Weighted MSE: Higher weight for rare label ranges
    2. Variance matching: Penalize when prediction variance differs from target variance
    3. Range penalty: Encourage predictions to span a wider range
    """

    def __init__(
        self,
        label_range: tuple = (0.0, 1.0),
        num_bins: int = 10,
        variance_weight: float = 0.1,
        range_weight: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.label_range = label_range
        self.num_bins = num_bins
        self.variance_weight = variance_weight
        self.range_weight = range_weight
        self.reduction = reduction

        bin_edges = torch.linspace(label_range[0], label_range[1], num_bins + 1)
        self.register_buffer("bin_edges", bin_edges)

    def _compute_sample_weights(self, labels: torch.Tensor) -> torch.Tensor:
        bin_edges = self.bin_edges.to(labels.device)
        bin_indices = torch.searchsorted(bin_edges, labels) - 1
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)

        bin_counts = torch.bincount(bin_indices, minlength=self.num_bins).float()

        bin_weights = torch.where(
            bin_counts > 0, 1.0 / (bin_counts + 1e-6), torch.zeros_like(bin_counts)
        )
        bin_weights = bin_weights / bin_weights.mean()

        sample_weights = bin_weights[bin_indices]
        return sample_weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute distribution-aware loss.

        Args:
            predictions: Model predictions (batch_size,)
            targets: Ground truth labels (batch_size,)
            return_components: If True, return dict with loss components

        Returns:
            Total loss or dict with loss components
        """
        sample_weights = self._compute_sample_weights(targets)

        weighted_mse = sample_weights * (predictions - targets) ** 2
        if self.reduction == "mean":
            mse_loss = weighted_mse.mean()
        else:
            mse_loss = weighted_mse.sum()

        pred_var = (
            predictions.var(unbiased=False)
            if predictions.numel() > 1
            else torch.tensor(0.0, device=predictions.device)
        )
        target_var = (
            targets.var(unbiased=False)
            if targets.numel() > 1
            else torch.tensor(1.0, device=targets.device)
        )
        variance_loss = (pred_var - target_var).abs()

        pred_range = predictions.max() - predictions.min()
        target_range = (
            targets.max() - targets.min()
            if targets.numel() > 1
            else torch.tensor(1.0, device=targets.device)
        )
        range_loss = F.relu(target_range * 0.5 - pred_range)

        total_loss = (
            mse_loss
            + self.variance_weight * variance_loss
            + self.range_weight * range_loss
        )

        if return_components:
            return {
                "total": total_loss,
                "mse": mse_loss,
                "variance": variance_loss,
                "range": range_loss,
                "pred_var": pred_var,
                "target_var": target_var,
                "pred_range": pred_range,
            }

        return total_loss


class FocalMSELoss(nn.Module):
    """Focal MSE Loss - gives more weight to hard examples (large errors)."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = (predictions - targets).abs()
        weights = (errors + 0.1) ** self.gamma
        weighted_errors = weights * (errors**2)

        if self.reduction == "mean":
            return weighted_errors.mean()
        else:
            return weighted_errors.sum()


class HuberLoss(nn.Module):
    """Huber Loss - more robust to outliers than MSE."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = predictions - targets
        abs_errors = errors.abs()

        quadratic = torch.min(
            abs_errors, torch.tensor(self.delta, device=errors.device)
        )
        linear = abs_errors - quadratic

        loss = 0.5 * quadratic**2 + self.delta * linear

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
