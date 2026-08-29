"""Evaluate a local checkpoint once on the untouched test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
