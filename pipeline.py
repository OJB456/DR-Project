"""Backward-compatible command-line facade for the RetinaScreen modules.

Preferred commands are the focused scripts in ``scripts/``. This facade keeps
the original imports and train/evaluate entry points usable.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from src.config.settings import load_config
from src.data.dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.evaluate import evaluate_checkpoint
from src.explainability.gradcam import generate_gradcam
from src.models.model import build_model
from src.preprocessing.fundus_preprocessor import preprocess_fundus
from src.quality.quality_assessment import QualityResult, assess_quality
from src.training.train import train_model

SEVERITY_NAMES = list(CLASS_NAMES)


def quality_gate(image: np.ndarray, **kwargs: Any) -> QualityResult:
    return assess_quality(image, **kwargs)


def grad_cam(model: torch.nn.Module, image_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
    result = generate_gradcam(model, image_tensor, target_class)
    return result["heatmap"], int(result["predicted_class"]), float(result["confidence"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sanity")
    train = subparsers.add_parser("train")
    train.add_argument("--config", default="config.yaml")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--learning-rate", type=float, default=None)
    train.add_argument("--image-size", type=int, default=None)
    train.add_argument("--patience", type=int, default=None)
    train.add_argument("--resume", default=None)
    train.add_argument("--max-train-samples", type=int, default=None)
    train.add_argument("--max-val-samples", type=int, default=None)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="config.yaml")
    evaluate.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.command == "sanity":
        from src.training.train import run_sanity_check

        run_sanity_check(load_config("config.yaml"))
    elif args.command == "train":
        config = load_config(args.config)
        train_model(config, args.epochs, args.batch_size, args.learning_rate, args.image_size, args.patience, args.resume, args.max_train_samples, args.max_val_samples)
    else:
        evaluate_checkpoint(load_config(args.config), args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
