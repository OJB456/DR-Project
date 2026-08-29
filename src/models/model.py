"""EfficientNet-B0 classifier and checkpoint utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

CLASS_NAMES = ("No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR")
LOGGER = logging.getLogger(__name__)


def build_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Build the requested EfficientNet-B0, falling back offline if weights are unavailable."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    try:
        model = models.efficientnet_b0(weights=weights)
    except Exception as error:
        if not pretrained:
            raise
        LOGGER.warning("Pretrained EfficientNet weights unavailable (%s); using random initialization.", error)
        model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the final spatial feature block for EfficientNet Grad-CAM."""
    if hasattr(model, "features") and len(model.features) > 0:
        convolutional_layers = [module for module in model.features.modules() if isinstance(module, nn.Conv2d)]
        if convolutional_layers:
            return convolutional_layers[-1]
        return model.features[-1]
    raise ValueError("Model does not expose a convolutional 'features' stack for Grad-CAM")


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load either a project checkpoint dictionary or a raw state dict."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.1 compatibility
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint.get("state_dict", checkpoint))) if isinstance(checkpoint, dict) else checkpoint
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        expected = model.classifier[-1].out_features
        raise ValueError(f"Checkpoint architecture is incompatible with the configured {expected}-class model; refusing to make predictions: {error}") from error
    return checkpoint if isinstance(checkpoint, dict) else {"model": state}
