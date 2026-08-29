"""Lazy APTOS CSV/image mapping and PyTorch datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from src.config.settings import ProjectConfig
from src.preprocessing.fundus_preprocessor import FundusPreprocessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = ("No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _column(frame: pd.DataFrame, candidates: Iterable[str], fallback: str) -> str:
    return next((name for name in candidates if name in frame.columns), fallback)


def _image_index(image_dir: Path) -> dict[str, Path]:
    """Index only path metadata; image pixels remain lazy-loaded."""
    index: dict[str, Path] = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem, path)
    return index


def resolve_image_path(image_dir: str | Path, image_id: str) -> Path:
    """Resolve an ID against nested image directories and common extensions."""
    directory = Path(image_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    direct = [directory / image_id, *(directory / f"{image_id}{extension}" for extension in IMAGE_EXTENSIONS)]
    for candidate in direct:
        if candidate.is_file():
            return candidate
    indexed = _image_index(directory)
    try:
        return indexed[Path(str(image_id)).stem]
    except KeyError as error:
        raise FileNotFoundError(f"No image found for ID '{image_id}' below {directory}") from error


def read_records(csv_path: str | Path, image_dir: str | Path) -> list[tuple[Path, int]]:
    """Read labels and resolve paths without opening image pixels."""
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"CSV has no rows: {csv_path}")
    id_column = _column(frame, ("id_code", "image", "image_id", "id"), str(frame.columns[0]))
    label_column = _column(frame, ("diagnosis", "label", "level", "grade"), str(frame.columns[-1]))
    index = _image_index(Path(image_dir))
    records: list[tuple[Path, int]] = []
    missing: list[str] = []
    for row in frame[[id_column, label_column]].itertuples(index=False, name=None):
        image_id, label = str(row[0]), int(row[1])
        image_path = index.get(Path(image_id).stem)
        if image_path is None:
            missing.append(image_id)
        else:
            records.append((image_path, label))
    if missing:
        raise FileNotFoundError(f"{len(missing)} labelled images are missing below {image_dir}; first: {missing[:3]}")
    return records


class APTOSDataset(Dataset[tuple[torch.Tensor, int]]):
    """One-image-at-a-time APTOS dataset with deterministic split transforms."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        train: bool = False,
        image_size: int = 224,
        preprocessor: FundusPreprocessor | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.records = read_records(csv_path, image_dir)
        self.train = train
        self.preprocessor = preprocessor or FundusPreprocessor(size=image_size)
        self.transform = transform or self._default_transform(train)

    @staticmethod
    def _default_transform(train: bool) -> transforms.Compose:
        operations = []
        if train:
            operations.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
            ])
        operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.records[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                processed = self.preprocessor(image)
            tensor = self.transform(processed)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise RuntimeError(f"Unable to read APTOS image '{path}': {error}") from error
        return tensor, int(label)


def build_datasets(config: ProjectConfig) -> tuple[APTOSDataset, APTOSDataset, APTOSDataset]:
    """Build train/validation/test datasets using the verified official splits."""
    preprocessor = FundusPreprocessor(size=config.image_size, **vars(config.preprocessing))
    return (
        APTOSDataset(config.train_csv, config.train_images, train=True, preprocessor=preprocessor),
        APTOSDataset(config.val_csv, config.val_images, train=False, preprocessor=preprocessor),
        APTOSDataset(config.test_csv, config.test_images, train=False, preprocessor=preprocessor),
    )
