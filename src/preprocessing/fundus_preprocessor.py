"""Conservative preprocessing for fundus photographs."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def crop_black_border(image: np.ndarray, threshold: int = 8) -> np.ndarray:
    """Crop only pixels that are effectively black background."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image array with shape H x W x 3")
    mask = np.max(image, axis=2) > threshold
    coordinates = cv2.findNonZero(mask.astype(np.uint8))
    if coordinates is None:
        return image
    x, y, width, height = cv2.boundingRect(coordinates)
    return image[y : y + height, x : x + width]


class FundusPreprocessor:
    """Crop, resize, and optionally apply mild luminance CLAHE."""

    def __init__(
        self,
        size: int = 224,
        crop_black_border: bool = True,
        black_threshold: int = 8,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> None:
        self.size = int(size)
        self.crop_enabled = bool(crop_black_border)
        self.black_threshold = int(black_threshold)
        self.apply_clahe = bool(apply_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = int(clahe_tile_grid_size)

    def __call__(self, image: Image.Image | np.ndarray | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                array = np.asarray(opened.convert("RGB"))
        elif isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected an RGB PIL image or H x W x 3 array")
        array = np.ascontiguousarray(array.astype(np.uint8))
        if self.crop_enabled:
            array = crop_black_border(array, self.black_threshold)
        if self.apply_clahe:
            lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            clahe = cv2.createCLAHE(self.clahe_clip_limit, (self.clahe_tile_grid_size, self.clahe_tile_grid_size))
            array = cv2.cvtColor(cv2.merge((clahe.apply(lightness), channel_a, channel_b)), cv2.COLOR_LAB2RGB)
        resized = cv2.resize(array, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized, mode="RGB")


def preprocess_fundus(image: Image.Image | np.ndarray | str | Path, size: int = 224, **kwargs: object) -> Image.Image:
    """Functional wrapper used by the UI and compatibility facade."""
    return FundusPreprocessor(size=size, **kwargs)(image)
