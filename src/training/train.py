"""Reproducible EfficientNet-B0 training with weighted loss and checkpointing."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.config.settings import ProjectConfig, load_config
from src.data.dataset import build_datasets
from src.models.model import build_model
from src.utils.device import device_info, get_device, seed_everything

LOGGER = logging.getLogger(__name__)


def compute_class_weights(dataset: Any, num_classes: int = 5) -> torch.Tensor:
    """Compute inverse-frequency weights from training labels only."""
    labels = np.asarray([label for _, label in dataset.records], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training data has no examples for classes: {np.where(counts == 0)[0].tolist()}")
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_sanity_check(config: ProjectConfig, datasets: tuple[Any, Any, Any] | None = None) -> dict[str, Any]:
    """Load one item from each split, one batch, and verify the model output shape."""
    train, validation, test = datasets or build_datasets(config)
    sample_info = []
    for name, dataset in (("train", train), ("validation", validation), ("test", test)):
        image, label = dataset[0]
        sample_info.append({"split": name, "shape": list(image.shape), "dtype": str(image.dtype), "label": int(label), "pixel_range": [float(image.min()), float(image.max())]})
    loader = DataLoader(train, batch_size=min(config.batch_size, 4), shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    device = get_device()
    model = build_model(config.num_classes, pretrained=False).to(device).eval()
    with torch.inference_mode():
        logits = model(images.to(device))
    result = {"samples": sample_info, "batch_shape": list(images.shape), "label_shape": list(labels.shape), "logits_shape": list(logits.shape), "device": str(device)}
    if logits.shape[-1] != config.num_classes:
        raise RuntimeError(f"Expected logits with {config.num_classes} classes, got {tuple(logits.shape)}")
    print(json.dumps(result, indent=2))
    return result


def _loader(dataset: Any, config: ProjectConfig, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=device.type == "cuda", persistent_workers=config.num_workers > 0)


def _validation(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            losses.append(float(criterion(logits, labels.to(device)).item()))
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    qwk = cohen_kappa_score(targets, predictions, weights="quadratic")
    return {"loss": float(np.mean(losses) if losses else 0.0), "accuracy": float(accuracy_score(targets, predictions)), "qwk": float(0.0 if np.isnan(qwk) else qwk)}


def _save_training_visualizations(history: dict[str, Any], output_dir: Path, labels: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(labels, minlength=5)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(range(5), counts, color="#177e89")
    axis.set(title="Training class distribution", xlabel="DR grade", ylabel="Images", xticks=range(5))
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(figure)
    epochs = history.get("epochs", [])
    if not epochs:
        return
    x = [item["epoch"] for item in epochs]
    train_loss = [item["train_loss"] for item in epochs]
    val_loss = [item["validation"]["loss"] for item in epochs]
    train_accuracy = [item["train_accuracy"] for item in epochs]
    val_accuracy = [item["validation"]["accuracy"] for item in epochs]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, train_loss, marker="o", label="Train loss")
    axes[0].plot(x, val_loss, marker="o", label="Validation loss")
    axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(x, train_accuracy, marker="o", label="Train accuracy")
    axes[1].plot(x, val_accuracy, marker="o", label="Validation accuracy")
    axes[1].set(title="Training and validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def train_model(config: ProjectConfig, epochs: int | None = None, batch_size: int | None = None, learning_rate: float | None = None, image_size: int | None = None, patience: int | None = None, resume: str | Path | None = None, max_train_samples: int | None = None, max_val_samples: int | None = None) -> dict[str, Any]:
    """Train using official train/validation splits and save best/last checkpoints."""
    if any(value is not None for value in (batch_size, image_size)):
        from dataclasses import replace

        config = replace(config, batch_size=batch_size or config.batch_size, image_size=image_size or config.image_size)
    seed_everything(config.seed)
    device = get_device()
    train_dataset, validation_dataset, _ = build_datasets(config)
    if max_train_samples:
        train_dataset = Subset(train_dataset, list(range(min(max_train_samples, len(train_dataset)))))
    if max_val_samples:
        validation_dataset = Subset(validation_dataset, list(range(min(max_val_samples, len(validation_dataset)))))
    train_loader = _loader(train_dataset, config, True, device)
    validation_loader = _loader(validation_dataset, config, False, device)
    base_train = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    weights = compute_class_weights(base_train, config.num_classes).to(device)
    print("Training class weights:", weights.detach().cpu().tolist())
    model = build_model(config.num_classes, pretrained=config.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate or config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs or config.epochs, 1))
    criterion = nn.CrossEntropyLoss(weight=weights)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    start_epoch = 0
    best_score = -float("inf")
    history: dict[str, Any] = {"config": config.as_dict(), "device": device_info(), "class_weights": weights.detach().cpu().tolist(), "epochs": []}
    config_file = config.project_root / "artifacts" / "training_config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"config": config.as_dict(), "device": device_info(), "requested_epochs": epochs or config.epochs}, indent=2), encoding="utf-8")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_score = float(checkpoint.get("best_score", best_score))
    total_epochs = epochs or config.epochs
    best_path = config.checkpoint_path
    last_path = config.last_checkpoint_path
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            running_loss += float(loss.item())
            running_correct += int((logits.detach().argmax(1) == labels).sum().item())
            running_count += int(labels.numel())
        scheduler.step()
        validation = _validation(model, validation_loader, device, criterion)
        epoch_record = {"epoch": epoch + 1, "train_loss": running_loss / max(len(train_loader), 1), "train_accuracy": running_correct / max(running_count, 1), "learning_rate": optimizer.param_groups[0]["lr"], "validation": validation}
        history["epochs"].append(epoch_record)
        checkpoint = {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_score": max(best_score, validation["qwk"]), "metrics": validation, "config": config.as_dict()}
        torch.save(checkpoint, last_path)
        if validation["qwk"] > best_score:
            best_score = validation["qwk"]
            torch.save(checkpoint, best_path)
        history["best_validation_qwk"] = best_score
        config.training_history_path.parent.mkdir(parents=True, exist_ok=True)
        config.training_history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        _save_training_visualizations(history, config.project_root / "artifacts" / "visualizations", [label for _, label in base_train.records])
        print(f"epoch={epoch + 1}/{total_epochs} train_loss={epoch_record['train_loss']:.4f} val_loss={validation['loss']:.4f} val_acc={validation['accuracy']:.4f} val_qwk={validation['qwk']:.4f}")
        if len(history["epochs"]) > (patience or config.patience) and all(item["validation"]["qwk"] <= best_score for item in history["epochs"][-(patience or config.patience):]):
            LOGGER.info("Early stopping after epoch %s", epoch + 1)
            break
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional bounded smoke run; omit for full training")
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.num_workers is not None:
        from dataclasses import replace

        config = replace(config, num_workers=args.num_workers)
    train_model(config, args.epochs, args.batch_size, args.learning_rate, args.image_size, args.patience, args.resume, args.max_train_samples, args.max_val_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
