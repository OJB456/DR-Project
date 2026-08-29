"""Prototype image quality gate for safe screening triage."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config.settings import QualityConfig


@dataclass
class QualityResult:
    quality_score: float
    quality_status: str
    quality_reasons: list[str]
    blur_variance: float = 0.0
    brightness_mean: float = 0.0
    contrast_std: float = 0.0
    field_of_view_ratio: float = 0.0
    image_integrity: bool = True

    @property
    def accepted(self) -> bool:
        return self.quality_status == "ACCEPT"

    @property
    def reason(self) -> str:
        return "pass" if self.accepted else ", ".join(self.quality_reasons)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_image(image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return np.asarray(opened.convert("RGB"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    return array.astype(np.uint8)


class QualityAssessment:
    """Classical, configurable checks; thresholds are engineering prototypes."""

    def __init__(self, config: QualityConfig | None = None, **overrides: float) -> None:
        values = vars(config or QualityConfig()).copy()
        values.update(overrides)
        self.config = QualityConfig(**values)

    def assess(self, image: Image.Image | np.ndarray | str | Path) -> QualityResult:
        try:
            rgb = _read_image(image)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            return QualityResult(0.0, "REJECT", [f"Image integrity check failed: {error}"], image_integrity=False)
        if rgb.size == 0 or min(rgb.shape[:2]) < 16:
            return QualityResult(0.0, "REJECT", ["Image integrity check failed: image is empty or too small"], image_integrity=False)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        fov_ratio = float(np.mean(np.max(rgb, axis=2) > 8))
        cfg = self.config
        reasons: list[str] = []
        if blur < cfg.blur_threshold:
            reasons.append("Image too blurry")
        if brightness < cfg.brightness_min:
            reasons.append("Insufficient brightness")
        elif brightness > cfg.brightness_max:
            reasons.append("Overexposed illumination")
        if contrast < cfg.contrast_min:
            reasons.append("Poor contrast")
        if fov_ratio < cfg.field_of_view_min:
            reasons.append("Insufficient retinal field of view")
        elif fov_ratio > cfg.field_of_view_max:
            reasons.append("Retinal field boundary is not apparent")
        blur_score = min(100.0, blur / max(cfg.blur_threshold, 1e-6) * 100.0)
        brightness_score = 100.0 if cfg.brightness_min <= brightness <= cfg.brightness_max else 0.0
        contrast_score = min(100.0, contrast / max(cfg.contrast_min, 1e-6) * 100.0)
        fov_score = 100.0 if cfg.field_of_view_min <= fov_ratio <= cfg.field_of_view_max else 0.0
        score = float(round(0.35 * blur_score + 0.25 * brightness_score + 0.20 * contrast_score + 0.20 * fov_score, 1))
        if score < cfg.min_quality_score and not reasons:
            reasons.append("Overall image quality is below the prototype threshold")
        return QualityResult(score, "ACCEPT" if not reasons else "REJECT", reasons, blur, brightness, contrast, fov_ratio)


def assess_quality(image: Image.Image | np.ndarray | str | Path, config: QualityConfig | None = None, **overrides: float) -> QualityResult:
    return QualityAssessment(config, **overrides).assess(image)
