"""MLP score head for predicting quality scores from temporal features.

This module implements a lightweight MLP head that maps aggregated temporal
features to a quality score in a configurable range.

Architecture:
    Input (768) → Linear(256) → LayerNorm → GELU → Dropout
               → Linear(64) → LayerNorm → GELU → Dropout
               → Linear(1) → Sigmoid → Scale to configured score range

Total trainable parameters: ~220K (well under 1M requirement)
"""

import torch
import torch.nn as nn

from src.logger import get_logger

logger = get_logger(__name__)


class ScoreHead(nn.Module):
    """MLP head for predicting quality scores from aggregated features.

    This module takes the output from temporal attention (after aggregation)
    and maps it to a single quality score in a configurable range.

    Parameters:
        - input_dim: 768 (from CLIP ViT-L/14 temporal attention output)
        - hidden_dims: [256, 64] (progressive dimension reduction)
        - output_dim: 1 (single score)
        - Total trainable params: ~220K

    Example:
        >>> score_head = ScoreHead(input_dim=768)
        >>> # Input: aggregated temporal features (batch_size, 768)
        >>> features = torch.randn(4, 768)
        >>> scores = score_head(features)  # (4, 1) in configured score range
    """

    DEFAULT_SCORE_MIN = 0.0
    DEFAULT_SCORE_MAX = 9.0

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: tuple[int, ...] = (256, 64),
        dropout: float = 0.1,
        use_bias: bool = False,
        score_range: tuple[float, float] = (DEFAULT_SCORE_MIN, DEFAULT_SCORE_MAX),
    ):
        """Initialize the score head.

        Args:
            input_dim: Input feature dimension (default: 768 for CLIP ViT-L/14).
            hidden_dims: Tuple of hidden layer dimensions (default: (256, 64)).
            dropout: Dropout probability for regularization.
            use_bias: Whether to use bias in linear layers.
            score_range: Output score range as (min_score, max_score).
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.score_range = score_range

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim, bias=use_bias),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Final output layer (single score)
        layers.append(nn.Linear(prev_dim, 1, bias=use_bias))

        self.mlp = nn.Sequential(*layers)

        min_score, max_score = score_range
        if max_score <= min_score:
            raise ValueError(
                f"score_range must satisfy max > min, got {score_range}"
            )

        # Score scaling: sigmoid output [0, 1] -> configured score range.
        self._score_scale = max_score - min_score
        self._score_offset = min_score

        self._log_parameters()

    def _log_parameters(self) -> None:
        """Log total trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"ScoreHead initialized with {total_params:,} trainable parameters")

    def forward(
        self,
        x: torch.Tensor,
        return_logit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to predict quality score.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
               This should be the aggregated output from temporal attention.
            return_logit: If True, also return the raw logit before scaling.

        Returns:
            If return_logit is False:
                Score tensor of shape (batch_size, 1) in configured score range.
            If return_logit is True:
                Tuple of (score, logit) where logit is in range [0, 1].
        """
        # Get raw logit from MLP
        logit = self.mlp(x)

        # Apply sigmoid to get [0, 1] range
        normalized = torch.sigmoid(logit)

        # Scale to configured range.
        score = normalized * self._score_scale + self._score_offset

        if return_logit:
            return score, normalized
        return score

    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Inference method returning score in configured range.

        This is a convenience method that disables gradient computation.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Score tensor of shape (batch_size, 1) in configured score range.
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            output = self.forward(x)
            # When return_logit=False (default), output is always Tensor
            score = output if isinstance(output, torch.Tensor) else output[0]

        if was_training:
            self.train()

        return score


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = ScoreHead(input_dim=768)
    x = torch.randn(4, 768)
    scores = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min().item():.2f}, {scores.max().item():.2f}]")
    print(f"Total parameters: {count_parameters(model):,}")
