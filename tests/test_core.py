from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.config.settings import load_config
from src.data.dataset import APTOSDataset, read_records, resolve_image_path
from src.explainability.gradcam import generate_gradcam, overlay_heatmap
from src.inference.predict import InferenceEngine, is_referable
from src.models.model import build_model
from src.preprocessing.fundus_preprocessor import FundusPreprocessor, crop_black_border
from src.quality.quality_assessment import assess_quality


def _synthetic_fundus(size: int = 128) -> Image.Image:
    rng = np.random.default_rng(42)
    yy, xx = np.ogrid[:size, :size]
    circle = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size * 0.43) ** 2
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[circle] = np.clip(rng.normal(125, 35, (int(circle.sum()), 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def test_config_and_nested_dataset_path_resolution() -> None:
    config = load_config("config.yaml")
    assert config.train_images.name == "train_images"
    first_id = pd.read_csv(config.train_csv).iloc[0]["id_code"]
    path = resolve_image_path(config.train_images, str(first_id))
    assert path.is_file()
    records = read_records(config.val_csv, config.val_images)
    assert len(records) == 366


def test_dataset_getitem_is_lazy_and_normalized(tmp_path: Path) -> None:
    image_dir = tmp_path / "nested" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "case-1.png"
    _synthetic_fundus(48).save(image_path)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"id_code": ["case-1"], "diagnosis": [2]}).to_csv(csv_path, index=False)
    dataset = APTOSDataset(csv_path, image_dir, image_size=32, preprocessor=FundusPreprocessor(32, apply_clahe=False))
    tensor, label = dataset[0]
    assert tuple(tensor.shape) == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert label == 2


def test_preprocessing_crops_black_border() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[4:16, 5:25] = 120
    cropped = crop_black_border(image)
    assert cropped.shape[:2] == (12, 20)
    processed = FundusPreprocessor(size=24, apply_clahe=False)(Image.fromarray(image))
    assert processed.size == (24, 24)


def test_quality_gate_rejects_invalid_and_accepts_reasonable_image() -> None:
    bad = assess_quality(np.zeros((64, 64, 3), dtype=np.uint8))
    assert bad.quality_status == "REJECT"
    assert bad.quality_score >= 0
    good = assess_quality(_synthetic_fundus())
    assert 0 <= good.quality_score <= 100
    assert good.image_integrity


def test_model_output_shape_and_gradcam() -> None:
    model = build_model(num_classes=5, pretrained=False).eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        logits = model(image)
    assert tuple(logits.shape) == (1, 5)
    explanation = generate_gradcam(model, image)
    assert explanation["heatmap"].shape == (64, 64)
    assert 0 <= explanation["predicted_class"] < 5
    assert 0 <= explanation["confidence"] <= 1


def test_referable_grade_rule() -> None:
    assert not is_referable(0)
    assert not is_referable(1)
    assert is_referable(2)
    assert is_referable(3)
    assert is_referable(4)


def test_gradcam_overlay_preserves_original_dimensions() -> None:
    image = np.full((30, 40, 3), 120, dtype=np.uint8)
    heatmap = np.zeros((4, 5), dtype=np.float32)
    original, _, overlay = overlay_heatmap(image, heatmap)
    assert original.shape == (30, 40, 3)
    assert overlay.shape == (30, 40, 3)
    assert np.array_equal(original, overlay)


def test_inference_returns_report_and_gradcam(tmp_path: Path) -> None:
    config = load_config("config.yaml")
    checkpoint = tmp_path / "best_model.pt"
    model = build_model(num_classes=5, pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    engine = InferenceEngine(config, checkpoint)
    result = engine.predict(_synthetic_fundus(), case_id="DR-TEST", output_dir=tmp_path / "artifacts")
    assert result["quality_status"] in {"ACCEPT", "REJECT"}
    if result["quality_status"] == "ACCEPT":
        assert set(result["probabilities"]) == {"No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"}
        assert Path(result["gradcam_paths"]["overlay"]).is_file()
        assert Path(result["report_paths"]["html"]).is_file()
