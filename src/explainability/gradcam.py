"""Genuine Grad-CAM using activations and gradients from the live model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from src.models.model import get_gradcam_target_layer


class GradCAM:
    """Compute Grad-CAM for a selected class at a convolutional feature layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or get_gradcam_target_layer(model)

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        activation: list[torch.Tensor] = []
        gradient: list[torch.Tensor] = []

        def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del module, inputs
            activation.append(output)
            output.register_hook(lambda grad: gradient.append(grad))

        handle = self.target_layer.register_forward_hook(capture)
        try:
            with torch.enable_grad():
                tensor = input_tensor.detach().clone().requires_grad_(True)
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted = int(logits.argmax(dim=1).item())
                selected = predicted if target_class is None else int(target_class)
                self.model.zero_grad(set_to_none=True)
                logits[:, selected].sum().backward()
                if not activation or not gradient:
                    raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
                weights = gradient[-1].mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation[-1]).sum(dim=1, keepdim=True))
                cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
                heatmap = cam[0, 0].detach().cpu().numpy()
                heatmap -= heatmap.min()
                heatmap /= heatmap.max() + 1e-8
            return {
                "heatmap": heatmap.astype(np.float32),
                "predicted_class": predicted,
                "target_class": selected,
                "confidence": float(probabilities[0, selected].detach().cpu().item()),
                "probabilities": probabilities[0].detach().cpu().numpy().astype(float),
            }
        finally:
            handle.remove()
            if not was_training:
                self.model.eval()
            else:
                self.model.train()


def generate_gradcam(model: nn.Module, input_tensor: torch.Tensor, target_class: int | None = None) -> dict[str, Any]:
    return GradCAM(model)(input_tensor, target_class)


def overlay_heatmap(image: Image.Image | np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB original, RGB heatmap, and RGB overlay arrays."""
    if isinstance(image, Image.Image):
        original = np.asarray(image.convert("RGB"))
    else:
        original = np.asarray(image)
    original = original.astype(np.uint8)
    height, width = original.shape[:2]
    resized_heatmap = cv2.resize(np.clip(heatmap, 0, 1), (width, height), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(np.uint8(resized_heatmap * 255), cv2.COLORMAP_TURBO)
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # Preserve the original fundus in low-activation regions. This is a display
    # mask over the real CAM, not a fabricated lesion mask or a new prediction.
    alpha_map = alpha * np.clip((resized_heatmap - 0.35) / 0.65, 0.0, 1.0)
    overlay = (original.astype(np.float32) * (1.0 - alpha_map[..., None]) + color_rgb.astype(np.float32) * alpha_map[..., None]).clip(0, 255).astype(np.uint8)
    return original.astype(np.uint8), color_rgb, overlay


def save_gradcam_visualization(image: Image.Image | np.ndarray, heatmap: np.ndarray, output_dir: str | Path) -> dict[str, str]:
    """Save original, heatmap, and overlay PNGs for a report/demo."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    original, color, overlay = overlay_heatmap(image, heatmap)
    paths = {"original": directory / "original.png", "heatmap": directory / "heatmap.png", "overlay": directory / "overlay.png"}
    Image.fromarray(original).save(paths["original"])
    Image.fromarray(color).save(paths["heatmap"])
    Image.fromarray(overlay).save(paths["overlay"])
    return {key: str(value) for key, value in paths.items()}
