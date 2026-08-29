"""Train the local EfficientNet-B0 model; never downloads the dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import main

if __name__ == "__main__":
    raise SystemExit(main())
