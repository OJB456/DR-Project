"""Runtime and reproducibility helpers."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info() -> dict[str, Any]:
    device = get_device()
    return {
        "device": str(device),
        "python": os.sys.version.split()[0],
        "pytorch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
