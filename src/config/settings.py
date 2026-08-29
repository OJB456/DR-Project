"""Project configuration loaded from a small, machine-independent YAML file."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QualityConfig:
    blur_threshold: float = 35.0
    brightness_min: float = 12.0
    brightness_max: float = 245.0
    contrast_min: float = 15.0
    field_of_view_min: float = 0.15
    field_of_view_max: float = 0.98
    min_quality_score: float = 60.0


@dataclass
class PreprocessingConfig:
    crop_black_border: bool = True
    black_threshold: int = 8
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8


@dataclass
class ProjectConfig:
    project_root: Path
    dataset_root: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    train_images: Path
    val_images: Path
    test_images: Path
    image_size: int = 224
    num_classes: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 4
    num_workers: int = 0
    seed: int = 42
    pretrained: bool = True
    checkpoint_path: Path = Path("artifacts/checkpoints/best_model.pt")
    last_checkpoint_path: Path = Path("artifacts/checkpoints/last_model.pt")
    training_history_path: Path = Path("artifacts/training_history.json")
    metrics_path: Path = Path("artifacts/metrics.json")
    confidence_threshold: float = 0.60
    referable_classes: tuple[int, ...] = (2, 3, 4)
    quality: QualityConfig = field(default_factory=QualityConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def resolve(self, value: str | Path) -> Path:
        """Resolve a path relative to the project root unless it is absolute."""
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation used in artifacts."""
        data = asdict(self)
        data.pop("project_root", None)
        for key in ("dataset_root", "train_csv", "val_csv", "test_csv", "train_images", "val_images", "test_images", "checkpoint_path", "last_checkpoint_path", "training_history_path", "metrics_path"):
            data[key] = str(data[key])
        data["referable_classes"] = list(data["referable_classes"])
        return data


def _path(raw: Any, project_root: Path) -> Path:
    value = Path(str(raw))
    return value if value.is_absolute() else project_root / value


def load_config(path: str | Path = "config.yaml", project_root: str | Path | None = None) -> ProjectConfig:
    """Load configuration and resolve all paths without assuming a Windows layout."""
    config_path = Path(path)
    if not config_path.is_absolute():
        root = Path(project_root) if project_root is not None else Path.cwd()
        config_path = root / config_path
    else:
        root = Path(project_root) if project_root is not None else config_path.parent
    root = root.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    raw = raw or {}
    quality = QualityConfig(**(raw.get("quality") or {}))
    preprocessing = PreprocessingConfig(**(raw.get("preprocessing") or {}))
    path_keys = ("dataset_root", "train_csv", "val_csv", "test_csv", "train_images", "val_images", "test_images", "checkpoint_path", "last_checkpoint_path", "training_history_path", "metrics_path")
    paths = {key: _path(raw.get(key, getattr(ProjectConfig, key, key)), root) for key in path_keys}
    return ProjectConfig(
        project_root=root,
        **paths,
        image_size=int(raw.get("image_size", 224)),
        num_classes=int(raw.get("num_classes", 5)),
        batch_size=int(raw.get("batch_size", 16)),
        learning_rate=float(raw.get("learning_rate", 1e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        epochs=int(raw.get("epochs", 15)),
        patience=int(raw.get("patience", 4)),
        num_workers=int(raw.get("num_workers", 0)),
        seed=int(raw.get("seed", 42)),
        pretrained=bool(raw.get("pretrained", True)),
        confidence_threshold=float(raw.get("confidence_threshold", 0.60)),
        referable_classes=tuple(int(item) for item in raw.get("referable_classes", [2, 3, 4])),
        quality=quality,
        preprocessing=preprocessing,
    )
