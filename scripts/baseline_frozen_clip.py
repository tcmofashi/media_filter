#!/usr/bin/env python3
"""
Baseline evaluation: Frozen CLIP + Simple MLP
Evaluates feasibility of predicting preference scores from CLIP features.
"""

import json
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Set HuggingFace mirror before importing transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor
from PIL import Image
import cv2
from scipy.stats import pearsonr

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_FILE = PROJECT_ROOT / "labels.json"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
BATCH_SIZE = 32
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Video sampling
NUM_FRAMES = 8


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(labels_file: str) -> List[dict]:
    """Load labels from JSONL file."""
    labels = []
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))
    return labels


def is_video(path: str | Path) -> bool:
    """Check if path is a video file."""
    return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


def extract_video_frames(
    video_path: str, num_frames: int = NUM_FRAMES
) -> List[Image.Image]:
    """Extract frames from video."""
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return frames

    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def extract_clip_features(
    clip_model: CLIPModel, processor: CLIPProcessor, media_path: str, device: str
) -> np.ndarray:
    """Extract CLIP features from image or video."""
    path = resolve_media_path(media_path)
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return np.zeros(768)

    features_list = []

    if is_video(path):
        # Extract frames from video
        frames = extract_video_frames(str(path), NUM_FRAMES)
        if not frames:
            print(f"Warning: No frames extracted from {path}")
            return np.zeros(768)  # Return zero features

        # Process each frame
        for frame in frames:
            inputs = processor(images=frame, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                # get_image_features returns BaseModelOutputWithPooling, extract pooler_output
                image_features = image_features.pooler_output

            features_list.append(image_features.cpu().numpy())

        # Average features across frames
        if features_list:
            return np.mean(features_list, axis=0).squeeze()
        else:
            return np.zeros(768)
    else:
        # Process image
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                # get_image_features returns BaseModelOutputWithPooling, extract pooler_output
                image_features = image_features.pooler_output

            return image_features.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return np.zeros(768)


def resolve_media_path(media_path: str) -> Path:
    """Resolve label paths relative to the repository root when needed."""
    path = Path(media_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class SimpleMLP(nn.Module):
    """Simple MLP regression head: 768 -> 256 -> 1"""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train MLP for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_features, batch_targets in train_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = criterion(predictions.squeeze(), batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_pearson(
    model: nn.Module, val_loader: DataLoader, device: str
) -> Tuple[float, List[float], List[float]]:
    """Evaluate model and return Pearson correlation."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            batch_features = batch_features.to(device)
            predictions = model(batch_features).squeeze()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_targets.numpy())

    # Calculate Pearson correlation
    corr, p_value = pearsonr(all_predictions, all_targets)
    return corr, all_predictions, all_targets


def main():
    print("=" * 60)
    print("Baseline Evaluation: Frozen CLIP + Simple MLP")
    print("=" * 60)

    set_seed(RANDOM_SEED)

    # Load labels
    print(f"\nLoading labels from {LABELS_FILE}...")
    labels = load_labels(LABELS_FILE)
    print(f"Total samples: {len(labels)}")

    # Count images and videos
    num_images = sum(1 for l in labels if not is_video(l["media_path"]))
    num_videos = sum(1 for l in labels if is_video(l["media_path"]))
    print(f"Images: {num_images}, Videos: {num_videos}")

    # Load CLIP model
    print(f"\nLoading CLIP model: {CLIP_MODEL_NAME}...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()

    # Freeze CLIP parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    print(f"CLIP model loaded. Device: {DEVICE}")

    # Extract features
    print("\nExtracting CLIP features...")
    features = []
    scores = []
    valid_paths = []

    for i, label in enumerate(labels):
        media_path = label["media_path"]
        score = label["score"]

        path = resolve_media_path(media_path)
        if not path.exists():
            print(f"Warning: File not found: {media_path}")
            continue

        # Extract features
        feature = extract_clip_features(clip_model, processor, str(path), DEVICE)
        features.append(feature)
        scores.append(score)
        valid_paths.append(str(path))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(labels)} samples...")

    features = np.array(features)
    scores = np.array(scores)

    print(f"\nExtracted features for {len(features)} samples")
    print(f"Features shape: {features.shape}")
    print(f"Score range: {scores.min():.1f} - {scores.max():.1f}")

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    # Split into train/val
    n_samples = len(features)
    n_val = int(n_samples * VAL_RATIO)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train = features[train_indices]
    y_train = scores[train_indices]
    X_val = features[val_indices]
    y_val = scores[val_indices]

    print(f"\nTrain/Val split: {n_train}/{n_val}")
    print(f"Train score mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    print(f"Val score mean: {y_val.mean():.2f}, std: {y_val.std():.2f}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize MLP
    print("\nInitializing MLP (768 -> 256 -> 1)...")
    mlp = SimpleMLP(input_dim=768, hidden_dim=256).to(DEVICE)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    best_corr = -1

    for epoch in range(NUM_EPOCHS):
        train_loss = train_mlp(mlp, train_loader, optimizer, criterion, DEVICE)
        val_corr, val_preds, val_targets = evaluate_pearson(mlp, val_loader, DEVICE)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Pearson: {val_corr:.4f}")

        if val_corr > best_corr:
            best_corr = val_corr

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_corr, final_preds, final_targets = evaluate_pearson(mlp, val_loader, DEVICE)

    print(f"Validation Pearson Correlation: {final_corr:.4f}")
    print(f"Target: > 0.60")
    print(f"Status: {'PASS' if final_corr > 0.60 else 'FAIL'}")

    # Additional statistics
    print("\n--- Score Statistics ---")
    print(
        f"Predictions - Mean: {np.mean(final_preds):.2f}, Std: {np.std(final_preds):.2f}"
    )
    print(
        f"Targets     - Mean: {np.mean(final_targets):.2f}, Std: {np.std(final_targets):.2f}"
    )

    # Error analysis
    errors = np.array(final_preds) - np.array(final_targets)
    print(f"Errors - Mean: {np.mean(errors):.2f}, MAE: {np.mean(np.abs(errors)):.2f}")

    # Report summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: Frozen CLIP ViT-L/14 + MLP (768->256->1)")
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Validation Pearson: {final_corr:.4f}")
    print(f"Target achieved: {'Yes' if final_corr > 0.60 else 'No'}")

    return final_corr


if __name__ == "__main__":
    main()
