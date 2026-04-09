"""Score regression loss functions for media quality rating.

This module provides loss functions specifically designed for score prediction
tasks, combining MSE/Smooth L1 loss with optional L1 regularization.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreLoss(nn.Module):
    """
    Combined loss for score regression with optional L1 regularization.

    This loss combines base regression loss (MSE or Smooth L1) with optional
    L1 regularization on model parameters. Designed for score prediction
    tasks with values in the configured score range, typically [0, 9].

    The total loss is computed as:
        total_loss = base_loss + lambda_l1 * l1_regularization

    Args:
        loss_type: Type of base loss, either 'mse' or 'smooth_l1'. Default: 'mse'
        lambda_l1: Weight for L1 regularization. Default: 0.0
        smooth_l1_beta: Beta parameter for Smooth L1 Loss. Default: 1.0
        score_range: Expected score range (min, max). Default: (0.0, 9.0)
        reduction: Reduction method, 'mean' or 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = ScoreLoss(loss_type='smooth_l1', lambda_l1=0.01)
        >>> predictions = torch.tensor([3.5, 5.0, 7.5, 9.0])
        >>> targets = torch.tensor([3.0, 5.5, 7.0, 9.5])
        >>> model_params = [p for p in model.parameters() if p.requires_grad]
        >>> loss = loss_fn(predictions, targets, model_params)
    """

    def __init__(
        self,
        loss_type: str = "mse",
        lambda_l1: float = 0.0,
        smooth_l1_beta: float = 1.0,
        score_range: Tuple[float, float] = (0.0, 9.0),
        reduction: str = "mean",
    ) -> None:
        """
        Initialize the score loss.

        Args:
            loss_type: Type of base loss ('mse' or 'smooth_l1').
            lambda_l1: Weight for L1 regularization on model parameters.
            smooth_l1_beta: Beta parameter for Smooth L1 Loss (transition point).
            score_range: Expected score range for normalization purposes.
            reduction: Reduction method for loss computation.
        """
        super().__init__()

        if loss_type not in ("mse", "smooth_l1"):
            raise ValueError(
                f"loss_type must be 'mse' or 'smooth_l1', got '{loss_type}'"
            )

        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

        self.loss_type = loss_type
        self.lambda_l1 = lambda_l1
        self.smooth_l1_beta = smooth_l1_beta
        self.score_range = score_range
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_params: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the combined regression and regularization loss.

        Args:
            predictions: Predicted scores, shape [batch_size] or [batch_size, ...]
            targets: Ground truth scores, shape [batch_size] or [batch_size, ...]
            model_params: Optional list of model parameters for L1 regularization.
                         If None, L1 regularization is skipped.
            return_components: If True, returns a dict with loss components.
                             If False, returns only total_loss. Default: False

        Returns:
            If return_components is False: total_loss (scalar tensor)
            If return_components is True: dict with 'total', 'base_loss', 'l1_reg'
        """
        targets = targets.to(predictions.device)

        # Compute base regression loss
        base_loss = self._compute_base_loss(predictions, targets)

        # Compute L1 regularization if parameters provided
        l1_reg = self._compute_l1_regularization(
            model_params, fallback_device=predictions.device
        )

        # Combine losses
        total_loss = base_loss + self.lambda_l1 * l1_reg

        if return_components:
            return {
                "total": total_loss,
                "base_loss": base_loss,
                "l1_reg": l1_reg,
            }

        return total_loss

    def _compute_base_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the base regression loss (MSE or Smooth L1).

        Args:
            predictions: Predicted scores
            targets: Ground truth scores

        Returns:
            Base regression loss as a scalar tensor
        """
        if self.loss_type == "mse":
            if self.reduction == "mean":
                base_loss = F.mse_loss(predictions, targets, reduction="mean")
            else:
                base_loss = F.mse_loss(predictions, targets, reduction="sum")
        else:  # smooth_l1
            if self.reduction == "mean":
                base_loss = F.smooth_l1_loss(
                    predictions, targets, beta=self.smooth_l1_beta, reduction="mean"
                )
            else:
                base_loss = F.smooth_l1_loss(
                    predictions, targets, beta=self.smooth_l1_beta, reduction="sum"
                )

        return base_loss

    def _compute_l1_regularization(
        self,
        model_params: Optional[torch.Tensor],
        fallback_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute L1 regularization on model parameters.

        Args:
            model_params: List of model parameters (tensors with requires_grad=True)

        Returns:
            L1 regularization term as a scalar tensor
        """
        device = self._get_device(model_params, fallback_device)
        if model_params is None or self.lambda_l1 == 0.0:
            return torch.tensor(0.0, device=device)

        l1_reg = torch.tensor(0.0, device=device)
        for param in model_params:
            if param.requires_grad:
                l1_reg = l1_reg + param.abs().sum()

        return l1_reg

    def _get_device(
        self,
        model_params: Optional[torch.Tensor] = None,
        fallback_device: Optional[torch.device] = None,
    ) -> torch.device:
        """Get a reasonable device for auxiliary tensors."""
        if model_params:
            for param in model_params:
                return param.device

        for param in self.parameters():
            return param.device

        if fallback_device is not None:
            return fallback_device

        return torch.device("cpu")


class WeightedScoreLoss(nn.Module):
    """
    Weighted score loss with customizable component weights.

    This loss allows fine-grained control over different loss components:
    - MSE loss component
    - Smooth L1 loss component
    - L1 regularization component

    The total loss is computed as:
        total_loss = w_mse * mse_loss + w_smooth_l1 * smooth_l1_loss + w_l1 * l1_reg

    This is useful for experimentation and hyperparameter tuning.

    Args:
        weight_mse: Weight for MSE loss component. Default: 0.5
        weight_smooth_l1: Weight for Smooth L1 loss component. Default: 0.5
        weight_l1: Weight for L1 regularization. Default: 0.0
        smooth_l1_beta: Beta parameter for Smooth L1 Loss. Default: 1.0
        score_range: Expected score range (min, max). Default: (1.0, 10.0)
        reduction: Reduction method, 'mean' or 'sum'. Default: 'mean'

    Example:
        >>> loss_fn = WeightedScoreLoss(weight_mse=0.7, weight_smooth_l1=0.3, weight_l1=0.01)
        >>> predictions = torch.tensor([3.5, 5.0, 7.5])
        >>> targets = torch.tensor([3.0, 5.5, 7.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        weight_mse: float = 0.5,
        weight_smooth_l1: float = 0.5,
        weight_l1: float = 0.0,
        smooth_l1_beta: float = 1.0,
        score_range: Tuple[float, float] = (1.0, 10.0),
        reduction: str = "mean",
    ) -> None:
        """
        Initialize the weighted score loss.

        Args:
            weight_mse: Weight for MSE loss component.
            weight_smooth_l1: Weight for Smooth L1 loss component.
            weight_l1: Weight for L1 regularization.
            smooth_l1_beta: Beta parameter for Smooth L1 Loss.
            score_range: Expected score range.
            reduction: Reduction method for loss computation.
        """
        super().__init__()

        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

        self.weight_mse = weight_mse
        self.weight_smooth_l1 = weight_smooth_l1
        self.weight_l1 = weight_l1
        self.smooth_l1_beta = smooth_l1_beta
        self.score_range = score_range
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_params: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the weighted combined loss.

        Args:
            predictions: Predicted scores, shape [batch_size]
            targets: Ground truth scores, shape [batch_size]
            model_params: Optional model parameters for L1 regularization.
            return_components: If True, returns dict with loss components.

        Returns:
            Total loss or dict with loss components
        """
        # Compute MSE loss
        if self.reduction == "mean":
            mse_loss = F.mse_loss(predictions, targets, reduction="mean")
            smooth_l1_loss = F.smooth_l1_loss(
                predictions, targets, beta=self.smooth_l1_beta, reduction="mean"
            )
        else:
            mse_loss = F.mse_loss(predictions, targets, reduction="sum")
            smooth_l1_loss = F.smooth_l1_loss(
                predictions, targets, beta=self.smooth_l1_beta, reduction="sum"
            )

        # Compute L1 regularization
        l1_reg = torch.tensor(0.0, device=predictions.device)
        if model_params is not None and self.weight_l1 > 0:
            for param in model_params:
                if param.requires_grad:
                    l1_reg = l1_reg + param.abs().sum()

        # Combine with weights
        total_loss = (
            self.weight_mse * mse_loss
            + self.weight_smooth_l1 * smooth_l1_loss
            + self.weight_l1 * l1_reg
        )

        if return_components:
            return {
                "total": total_loss,
                "mse": mse_loss,
                "smooth_l1": smooth_l1_loss,
                "l1_reg": l1_reg,
            }

        return total_loss


def create_score_loss(
    loss_type: str = "mse",
    lambda_l1: float = 0.0,
    smooth_l1_beta: float = 1.0,
    score_range: Tuple[float, float] = (1.0, 10.0),
) -> ScoreLoss:
    """
    Factory function to create a ScoreLoss instance with common configurations.

    Args:
        loss_type: Type of base loss ('mse' or 'smooth_l1').
        lambda_l1: Weight for L1 regularization.
        smooth_l1_beta: Beta parameter for Smooth L1 Loss.
        score_range: Expected score range.

    Returns:
        Configured ScoreLoss instance

    Example:
        >>> loss_fn = create_score_loss(loss_type='smooth_l1', lambda_l1=0.01)
    """
    return ScoreLoss(
        loss_type=loss_type,
        lambda_l1=lambda_l1,
        smooth_l1_beta=smooth_l1_beta,
        score_range=score_range,
    )
