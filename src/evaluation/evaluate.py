"""Untouched-test evaluation and artifact generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.config.settings import ProjectConfig, load_config
from src.data.dataset import CLASS_NAMES, build_datasets
from src.models.model import build_model, load_checkpoint
from src.utils.device import get_device


def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device, non_blocking=True)), dim=1)
            all_targets.extend(labels.tolist())
            all_predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
            all_probabilities.append(probabilities.cpu().numpy())
    return np.asarray(all_targets), np.asarray(all_predictions), np.concatenate(all_probabilities, axis=0)


def _binary_metrics(targets: np.ndarray, predictions: np.ndarray, referable_classes: tuple[int, ...]) -> dict[str, float | int]:
    actual = np.isin(targets, referable_classes)
    predicted = np.isin(predictions, referable_classes)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "sensitivity": float(recall_score(actual, predicted, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, config: ProjectConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Evaluate a model and optionally write the requested metrics artifacts."""
    targets, predictions, probabilities = collect_predictions(model, loader, device)
    labels = list(range(config.num_classes))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    per_class_precision = precision_score(targets, predictions, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(targets, predictions, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(targets, predictions, labels=labels, average=None, zero_division=0)
    qwk = cohen_kappa_score(targets, predictions, weights="quadratic")
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision_macro": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(targets, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(targets, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(0.0 if np.isnan(qwk) else qwk),
        "per_class": {
            CLASS_NAMES[label]: {"precision": float(per_class_precision[label]), "recall": float(per_class_recall[label]), "f1": float(per_class_f1[label])}
            for label in labels
        },
        "confusion_matrix": matrix.tolist(),
        "referable_dr": _binary_metrics(targets, predictions, config.referable_classes),
        "referable_definition": {"non_referable": [0, 1], "referable": list(config.referable_classes)},
        "sample_count": int(len(targets)),
        "class_names": list(CLASS_NAMES),
    }
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report = classification_report(targets, predictions, labels=labels, target_names=list(CLASS_NAMES), zero_division=0)
        (directory / "classification_report.txt").write_text(report, encoding="utf-8")
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set(title="APTOS test-set confusion matrix", xlabel="Predicted grade", ylabel="True grade", xticks=labels, yticks=labels)
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
        figure.tight_layout()
        figure.savefig(directory / "confusion_matrix.png", dpi=150)
        plt.close(figure)
    return metrics


def evaluate_checkpoint(config: ProjectConfig, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    device = get_device()
    _, _, test_dataset = build_datasets(config)
    loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=device.type == "cuda")
    model = build_model(config.num_classes, pretrained=False).to(device)
    load_checkpoint(model, checkpoint_path or config.checkpoint_path, device)
    metrics = evaluate_model(model, loader, device, config, config.metrics_path.parent)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    evaluate_checkpoint(load_config(args.config), args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
