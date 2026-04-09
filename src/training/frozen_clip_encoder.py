"""CLIP vision encoder for feature extraction and tail fine-tuning.

This module wraps a Hugging Face CLIP model for extracting visual features from
images and video frames. It supports the original fully frozen setup as well as
partial fine-tuning of the last N vision transformer layers.
"""

import os
from typing import List, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.logger import get_logger

logger = get_logger(__name__)


class FrozenCLIPEncoder(nn.Module):
    """CLIP encoder for visual feature extraction with optional tail unfreezing.

    This encoder loads a pre-trained CLIP model and freezes all parameters by
    default. When requested, it can unfreeze the last N vision transformer
    blocks plus the post-layernorm/projection layers for partial fine-tuning.

    Features:
        - Loads a configurable Hugging Face CLIP checkpoint
        - All parameters frozen by default
        - Optional unfreezing of the last N vision layers
        - Supports both single images and batches of frames
        - Efficient partial-gradient extraction for trainable vision tails

    Example:
        >>> encoder = FrozenCLIPEncoder()
        >>> # Single image
        >>> features = encoder.extract_features(image)  # (1, 768)
        >>> # Multiple frames (video)
        >>> features = encoder.extract_features(frames)  # (num_frames, 768)
    """

    MODEL_NAME = "openai/clip-vit-large-patch14"
    FEATURE_DIM = 768

    def __init__(
        self,
        device: str = "auto",
        model_name: str = MODEL_NAME,
        unfreeze_last_n_vision_layers: int = 0,
    ):
        """Initialize the frozen CLIP encoder.

        Args:
            device: Device to load the model on. "auto" uses CUDA if available,
                    otherwise CPU.
            model_name: Hugging Face model name for the CLIP checkpoint.
            unfreeze_last_n_vision_layers: Number of final vision transformer
                    blocks to unfreeze for training.
        """
        super().__init__()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.unfreeze_last_n_vision_layers = max(0, int(unfreeze_last_n_vision_layers))

        logger.info(f"Loading CLIP model: {self.model_name}")

        # Check if offline mode is enabled via environment variable
        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

        try:
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
        except Exception as e:
            if not local_only:
                logger.warning(f"Failed to download from Hugging Face Hub: {e}")
                logger.info("Attempting to load from local cache only...")
                self.model = CLIPModel.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )
                self.processor = CLIPProcessor.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )
            else:
                raise

        self._freeze_parameters()
        self._unfreeze_vision_tail()

        self.feature_dim = int(self.model.config.projection_dim)
        self.image_size = int(self.model.config.vision_config.image_size)
        self.num_vision_layers = int(self.model.config.vision_config.num_hidden_layers)
        self._trainable_param_names = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }

        self.model = self.model.to(self.device)

        logger.info(
            "CLIP encoder loaded on %s with model=%s, feature_dim=%d, image_size=%d, "
            "trainable_params=%d",
            self.device,
            self.model_name,
            self.feature_dim,
            self.image_size,
            self.count_trainable_parameters(),
        )

    def _freeze_parameters(self) -> None:
        """Freeze all model parameters to prevent gradient computation."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_vision_tail(self) -> None:
        """Unfreeze the last N vision transformer layers plus output heads."""
        if self.unfreeze_last_n_vision_layers <= 0:
            return

        vision_layers = self.model.vision_model.encoder.layers
        total_layers = len(vision_layers)
        unfreeze_count = min(self.unfreeze_last_n_vision_layers, total_layers)
        start_idx = total_layers - unfreeze_count

        for layer in vision_layers[start_idx:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.model.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        for param in self.model.visual_projection.parameters():
            param.requires_grad = True

        logger.info(
            "Unfroze CLIP vision tail: layers %d-%d of %d, plus post_layernorm and visual_projection",
            start_idx,
            total_layers - 1,
            total_layers,
        )

    def _count_parameters(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters in the CLIP model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def has_trainable_parameters(self) -> bool:
        """Whether any CLIP parameters are trainable."""
        return self.count_trainable_parameters() > 0

    def get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Get the trainable CLIP parameter tensors for checkpointing."""
        state_dict = self.model.state_dict()
        return {
            name: state_dict[name].detach().cpu()
            for name in self._trainable_param_names
            if name in state_dict
        }

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """Load a trainable CLIP tail state dict."""
        return self.model.load_state_dict(state_dict, strict=False)

    def _extract_features_impl(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the full CLIP vision stack."""
        outputs = self.model.vision_model(pixel_values=pixel_values)
        image_features = outputs.pooler_output  # (batch_size, hidden_dim)
        image_features = self.model.visual_projection(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _extract_features_with_trainable_tail(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run frozen early layers under no_grad and keep gradients on the tail."""
        vision_model = self.model.vision_model
        encoder_layers = vision_model.encoder.layers
        total_layers = len(encoder_layers)
        trainable_layers = min(self.unfreeze_last_n_vision_layers, total_layers)
        frozen_layers = total_layers - trainable_layers

        with torch.no_grad():
            hidden_states = vision_model.embeddings(pixel_values)
            hidden_states = vision_model.pre_layrnorm(hidden_states)

            for layer in encoder_layers[:frozen_layers]:
                hidden_states = layer(hidden_states, attention_mask=None)

        hidden_states = hidden_states.detach()

        for layer in encoder_layers[frozen_layers:]:
            hidden_states = layer(hidden_states, attention_mask=None)

        pooled_output = hidden_states[:, 0, :]
        pooled_output = vision_model.post_layernorm(pooled_output)
        image_features = self.model.visual_projection(pooled_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def preprocess(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Preprocess images for CLIP model.

        Args:
            images: Single PIL Image or list of PIL Images.

        Returns:
            Preprocessed pixel values tensor of shape (batch, channels, height, width).
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt", padding=True)

        return inputs["pixel_values"].to(self.device)

    def extract_features(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> torch.Tensor:
        """Extract CLIP features from images.

        This is the main interface for feature extraction. It handles both
        single images and batches of frames uniformly.

        Args:
            images: Single PIL Image or list of PIL Images.
                   - For single image: returns (1, 768) tensor
                   - For video frames: returns (num_frames, 768) tensor

        Returns:
            Feature tensor of shape (batch_size, 768) where batch_size is
            the number of input images.

        Example:
            >>> encoder = FrozenCLIPEncoder()
            >>>
            >>> # Single image
            >>> img = Image.open("photo.jpg")
            >>> features = encoder.extract_features(img)
            >>> print(features.shape)  # torch.Size([1, 768])
            >>>
            >>> # Video frames
            >>> frames = [Image.open(f"frame_{i}.jpg") for i in range(12)]
            >>> features = encoder.extract_features(frames)
            >>> print(features.shape)  # torch.Size([12, 768])
        """
        pixel_values = self.preprocess(images)

        if not self.has_trainable_parameters():
            with torch.no_grad():
                return self._extract_features_impl(pixel_values)

        return self._extract_features_with_trainable_tail(pixel_values)

    def forward(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Forward pass - alias for extract_features for nn.Module compatibility.

        Args:
            images: Single PIL Image or list of PIL Images.

        Returns:
            Feature tensor of shape (batch_size, 768).
        """
        return self.extract_features(images)

    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted features.

        Returns:
            Feature dimension for the loaded CLIP checkpoint.
        """
        return self.feature_dim
