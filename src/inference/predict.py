"""Quality-gated model inference with probabilities, Grad-CAM, and reports."""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config.settings import PreprocessingConfig, ProjectConfig, load_config
from src.data.dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from src.explainability.gradcam import generate_gradcam, save_gradcam_visualization
from src.models.model import build_model, load_checkpoint
from src.preprocessing.fundus_preprocessor import FundusPreprocessor
from src.quality.quality_assessment import QualityAssessment, QualityResult
from src.reporting.report import generate_report as write_report
from src.utils.device import get_device


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a prediction is requested without a local trained checkpoint."""


def is_referable(predicted_class: int, referable_classes: tuple[int, ...] = (2, 3, 4)) -> bool:
    """Apply the documented grade 2/3/4 referable rule."""
    return int(predicted_class) in tuple(int(value) for value in referable_classes)


def _to_pil(image: Image.Image | np.ndarray | str | Path | bytes | io.BytesIO) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, io.BytesIO):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


class InferenceEngine:
    """Reusable, cached model runner. The model is loaded once per engine."""

    def __init__(self, config: ProjectConfig, checkpoint_path: str | Path | None = None) -> None:
        self.config = config
        self.device = get_device()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else config.checkpoint_path
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = config.project_root / self.checkpoint_path
        if not self.checkpoint_path.is_file():
            raise CheckpointNotFoundError(f"Model checkpoint not found. Please train the model first: {self.checkpoint_path}")
        self.model = build_model(config.num_classes, pretrained=False).to(self.device).eval()
        checkpoint = load_checkpoint(self.model, self.checkpoint_path, self.device)
        saved_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        saved_num_classes = saved_config.get("num_classes")
        state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint.get("state_dict", {}))) if isinstance(checkpoint, dict) else {}
        classifier_weight = state.get("classifier.1.weight") if isinstance(state, dict) else None
        checkpoint_num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else saved_num_classes
        if checkpoint_num_classes is not None and int(checkpoint_num_classes) != config.num_classes:
            raise ValueError(f"Checkpoint supports {checkpoint_num_classes} classes but configuration expects {config.num_classes}; refusing incompatible predictions.")
        self.model_input_size = int(saved_config.get("image_size", config.image_size))
        saved_preprocessing = saved_config.get("preprocessing") or vars(config.preprocessing)
        self.preprocessing_config = PreprocessingConfig(**saved_preprocessing)
        self.preprocessor = FundusPreprocessor(size=self.model_input_size, **vars(self.preprocessing_config))
        self.quality_assessment = QualityAssessment(config.quality)
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def predict(self, image: Image.Image | np.ndarray | str | Path | bytes | io.BytesIO, case_id: str | None = None, output_dir: str | Path | None = None, write_report_file: bool = True) -> dict[str, Any]:
        original = _to_pil(image)
        quality: QualityResult = self.quality_assessment.assess(original)
        now = datetime.now(timezone.utc)
        case_id = case_id or f"DR-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        result: dict[str, Any] = {
            "case_id": case_id,
            "timestamp": now.isoformat(),
            "device": str(self.device),
            "model_status": {
                "loaded": True,
                "architecture": "EfficientNet-B0",
                "classes": self.config.num_classes,
                "input_size": self.model_input_size,
                "checkpoint": str(self.checkpoint_path),
            },
            "quality_score": quality.quality_score,
            "quality_status": quality.quality_status,
            "quality_reasons": quality.quality_reasons,
            "quality_metrics": quality.as_dict(),
            "predicted_class": None,
            "predicted_class_name": None,
            "confidence": None,
            "probabilities": {},
            "referable": None,
            "confidence_is_low": None,
            "recommendation": "Recapture recommended before screening." if not quality.accepted else "",
            "gradcam_paths": {},
            "report_paths": {},
        }
        if not quality.accepted:
            result["recommendation"] = "A reliable screening result cannot be produced from this image. Please recapture the fundus image."
            return result
        processed = self.preprocessor(original)
        tensor = self.to_tensor(processed).unsqueeze(0).to(self.device)
        explanation = generate_gradcam(self.model, tensor)
        predicted = int(explanation["predicted_class"])
        confidence = float(explanation["confidence"])
        probabilities = {CLASS_NAMES[index]: float(value) for index, value in enumerate(explanation["probabilities"])}
        referable = is_referable(predicted, self.config.referable_classes)
        low_confidence = confidence < self.config.confidence_threshold
        if referable:
            recommendation = "Specialist ophthalmology review recommended."
        elif low_confidence:
            recommendation = "AI result is uncertain. Specialist review recommended."
        else:
            recommendation = "Routine screening follow-up may be considered; this result does not establish that disease is absent."
        result.update({
            "predicted_class": predicted,
            "predicted_class_name": CLASS_NAMES[predicted],
            "confidence": confidence,
            "probabilities": probabilities,
            "referable": referable,
            "confidence_is_low": low_confidence,
            "recommendation": recommendation,
        })
        if output_dir is None:
            output_root = self.config.project_root / "artifacts"
        else:
            output_root = Path(output_dir)
        gradcam_dir = output_root / "gradcam" / case_id
        result["gradcam_paths"] = save_gradcam_visualization(original, explanation["heatmap"], gradcam_dir)
        if write_report_file:
            result["report_paths"] = write_report(result, output_root / "reports")
        return result


def predict(image: Image.Image | np.ndarray | str | Path | bytes | io.BytesIO, checkpoint_path: str | Path | None = None, config_path: str | Path = "config.yaml", case_id: str | None = None, output_dir: str | Path | None = None, write_report_file: bool = True) -> dict[str, Any]:
    """Run the complete quality-gated screening workflow once."""
    config = load_config(config_path)
    return InferenceEngine(config, checkpoint_path).predict(image, case_id=case_id, output_dir=output_dir, write_report_file=write_report_file)
